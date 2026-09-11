from __future__ import annotations

import io
import math
import time
import zipfile
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import fitz
import numpy as np

from app.config import settings
from app.ocr_service import ocr_service


# ================================================================
# Document Quality Service
# ================================================================


class DocumentQualityService:
    """
    Main document-quality validation service.

    Scope:
        - image quality
        - readability
        - scan quality
        - OCR readability
        - PDF page quality
        - DOCX embedded-image quality

    This service does NOT:
        - validate PAN/Aadhaar values
        - validate names
        - validate dates
        - validate business rules
        - verify authenticity
        - detect fraud
        - determine document completeness
    """

    # ============================================================
    # Public entry point
    # ============================================================

    def analyze_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:

        start_time = time.perf_counter()

        self._validate_file_size(file_bytes)

        extension = Path(filename).suffix.lower()

        if extension in settings.allowed_image_extensions:

            result = self._analyze_image_file(
                file_bytes,
                filename,
            )

        elif extension == ".pdf":

            result = self._analyze_pdf(
                file_bytes,
                filename,
            )

        elif extension == ".docx":

            result = self._analyze_docx(
                file_bytes,
                filename,
            )

        else:
            raise ValueError(
                f"Unsupported file type: "
                f"{extension or 'unknown'}"
            )

        processing_time = (
            time.perf_counter()
            - start_time
        )

        result["processing"] = {
            "time_seconds": round(
                processing_time,
                2,
            )
        }

        result["content_type"] = (
            content_type
            or (
                "application/pdf"
                if extension == ".pdf"
                else (
                    "image/png"
                    if extension == ".png"
                    else "image/jpeg"
                )
            )
        )

        return result

    # ============================================================
    # File validation
    # ============================================================

    @staticmethod
    def _validate_file_size(
        file_bytes: bytes,
    ) -> None:

        size_mb = len(file_bytes) / (
            1024 * 1024
        )

        if size_mb > settings.max_file_size_mb:

            raise ValueError(
                f"File size {size_mb:.2f} MB "
                f"exceeds maximum allowed size "
                f"of {settings.max_file_size_mb} MB"
            )

    # ============================================================
    # Image processing
    # ============================================================

    def _analyze_image_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> dict[str, Any]:

        image = self._decode_image(file_bytes)

        page_result = self._analyze_page_image(
            image=image,
            page_number=1,
            run_ocr=True,
        )

        return self._build_document_result(
            filename=filename,
            document_type="IMAGE",
            page_details=[page_result],
            composition_counts={
                "machine_readable": 0,
                "scanned": 0,
                "ocr_layered": 0,
                "blank": (
                    1
                    if page_result["content_type"] == "BLANK"
                    else 0
                ),
                "low_content": (
                    1
                    if page_result["content_type"]
                    == "LOW_CONTENT"
                    else 0
                ),
            },
            ocr_results=(
                {0: page_result["ocr"]}
                if page_result.get("ocr") is not None
                else {}
            ),
        )

    @staticmethod
    def _decode_image(
        file_bytes: bytes,
    ) -> np.ndarray:

        array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Unable to decode image"
            )

        return image

    # ============================================================
    # Image quality analysis
    # ============================================================

    def _analyze_page_image(
        self,
        image: np.ndarray,
        page_number: int,
        run_ocr: bool = True,
    ) -> dict[str, Any]:

        metrics = self._image_metrics(image)

        content_type = self._classify_content(
            metrics
        )

        # --------------------------------------------------------
        # Blank
        # --------------------------------------------------------

        if content_type == "BLANK":

            return {
                "page": page_number,
                "status": "PASS",
                "content_type": "BLANK",
                "reason": None,
                "quality": metrics,
                "ocr": None,
            }

        # --------------------------------------------------------
        # Low content
        # --------------------------------------------------------

        if content_type == "LOW_CONTENT":

            return {
                "page": page_number,
                "status": "PASS",
                "content_type": "LOW_CONTENT",
                "reason": None,
                "quality": metrics,
                "ocr": None,
            }

        # --------------------------------------------------------
        # Quality flags
        # --------------------------------------------------------

        quality_flags = self._quality_flags(
            metrics
        )

        critical_failure = any(
            flag["severity"] == "CRITICAL"
            for flag in quality_flags
        )

        # --------------------------------------------------------
        # Catastrophic visual failure
        # --------------------------------------------------------

        if (
            critical_failure
            and not self._ocr_may_recover(metrics)
        ):

            return {
                "page": page_number,
                "status": "FAIL",
                "content_type": "MEANINGFUL",
                "reason": self._primary_reason(
                    quality_flags
                ),
                "quality": metrics,
                "ocr": None,
            }

        # --------------------------------------------------------
        # OCR
        # --------------------------------------------------------

        ocr_result = None

        if run_ocr:

            ocr_result = (
                self._run_ocr_with_recovery(
                    image
                )
            )

        # --------------------------------------------------------
        # OCR result
        # --------------------------------------------------------

        if ocr_result is not None:

            if self._ocr_result_is_readable(
                ocr_result
            ):

                return {
                    "page": page_number,
                    "status": "PASS",
                    "content_type": "MEANINGFUL",
                    "reason": self._warning_reason(
                        quality_flags
                    ),
                    "quality": metrics,
                    "ocr": ocr_result,
                }

            return {
                "page": page_number,
                "status": "FAIL",
                "content_type": "MEANINGFUL",
                "reason": self._ocr_failure_reason(
                    quality_flags
                ),
                "quality": metrics,
                "ocr": ocr_result,
            }

        # --------------------------------------------------------
        # No OCR requested
        # --------------------------------------------------------

        if critical_failure:

            return {
                "page": page_number,
                "status": "FAIL",
                "content_type": "MEANINGFUL",
                "reason": self._primary_reason(
                    quality_flags
                ),
                "quality": metrics,
                "ocr": None,
            }

        return {
            "page": page_number,
            "status": "PASS",
            "content_type": "MEANINGFUL",
            "reason": self._warning_reason(
                quality_flags
            ),
            "quality": metrics,
            "ocr": None,
        }

    # ============================================================
    # Image metrics
    # ============================================================

    @staticmethod
    def _image_metrics(
        image: np.ndarray,
    ) -> dict[str, Any]:

        height, width = image.shape[:2]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        mean_brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        dark_ratio = float(
            np.mean(gray < 30)
        )

        bright_ratio = float(
            np.mean(gray > 245)
        )

        # --------------------------------------------------------
        # Foreground estimate
        # --------------------------------------------------------

        _, binary = cv2.threshold(
            gray,
            245,
            255,
            cv2.THRESH_BINARY_INV,
        )

        foreground_ratio = float(
            np.mean(
                binary > 0
            )
        )

        # --------------------------------------------------------
        # Edge density
        # --------------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        edge_density = float(
            np.mean(
                edges > 0
            )
        )

        # --------------------------------------------------------
        # Noise
        # --------------------------------------------------------

        denoised = cv2.medianBlur(
            gray,
            3,
        )

        noise = cv2.absdiff(
            gray,
            denoised,
        )

        noise_ratio = float(
            np.mean(
                noise > 20
            )
        )

        noise_value = float(
            np.mean(noise)
        )

        # --------------------------------------------------------
        # Skew
        # --------------------------------------------------------

        skew = (
            DocumentQualityService
            ._estimate_skew(gray)
        )

        return {
            "width": int(width),
            "height": int(height),
            "pixels": int(
                width * height
            ),

            "brightness_mean": round(
                mean_brightness,
                2,
            ),

            "contrast_std": round(
                contrast,
                2,
            ),

            "blur_score": round(
                blur_score,
                2,
            ),

            "dark_ratio": round(
                dark_ratio,
                4,
            ),

            "bright_ratio": round(
                bright_ratio,
                4,
            ),

            "foreground_ratio": round(
                foreground_ratio,
                4,
            ),

            "edge_density": round(
                edge_density,
                4,
            ),

            "noise_ratio": round(
                noise_ratio,
                4,
            ),

            "noise_value": round(
                noise_value,
                2,
            ),

            "skew_degrees": round(
                skew,
                2,
            ),
        }

    # ============================================================
    # Content classification
    # ============================================================

    @staticmethod
    def _classify_content(
        metrics: dict[str, Any],
    ) -> str:

        foreground = metrics[
            "foreground_ratio"
        ]

        dark_ratio = metrics[
            "dark_ratio"
        ]

        bright_ratio = metrics[
            "bright_ratio"
        ]

        if (
            foreground < 0.005
            and (
                bright_ratio > 0.995
                or dark_ratio > 0.995
            )
        ):
            return "BLANK"

        if foreground < 0.01:
            return "LOW_CONTENT"

        return "MEANINGFUL"

    # ============================================================
    # Quality flags
    # ============================================================

    @staticmethod
    def _quality_flags(
        metrics: dict[str, Any],
    ) -> list[dict[str, str]]:

        flags: list[
            dict[str, str]
        ] = []

        width = metrics["width"]
        height = metrics["height"]

        blur = metrics["blur_score"]
        contrast = metrics["contrast_std"]

        dark_ratio = metrics[
            "dark_ratio"
        ]

        bright_ratio = metrics[
            "bright_ratio"
        ]

        noise_ratio = metrics[
            "noise_ratio"
        ]

        # --------------------------------------------------------
        # Resolution
        # --------------------------------------------------------

        if (
            width < settings.critical_min_width
            or height < settings.critical_min_height
        ):

            flags.append({
                "code":
                    "CRITICAL_LOW_RESOLUTION",
                "severity":
                    "CRITICAL",
            })

        elif (
            width < settings.min_width
            or height < settings.min_height
        ):

            flags.append({
                "code":
                    "LOW_RESOLUTION",
                "severity":
                    "WARNING",
            })

        # --------------------------------------------------------
        # Blur
        # --------------------------------------------------------

        if (
            blur
            < settings.severe_blur_threshold
        ):

            flags.append({
                "code":
                    "SEVERE_BLUR",
                "severity":
                    "CRITICAL",
            })

        elif (
            blur
            < settings.blur_threshold
        ):

            flags.append({
                "code":
                    "BLUR",
                "severity":
                    "WARNING",
            })

        # --------------------------------------------------------
        # Contrast
        # --------------------------------------------------------

        if (
            contrast
            < settings.severe_contrast_threshold
        ):

            flags.append({
                "code":
                    "SEVERE_LOW_CONTRAST",
                "severity":
                    "CRITICAL",
            })

        elif (
            contrast
            < settings.contrast_threshold
        ):

            flags.append({
                "code":
                    "LOW_CONTRAST",
                "severity":
                    "WARNING",
            })

        # --------------------------------------------------------
        # Nearly black
        # --------------------------------------------------------

        if (
            dark_ratio
            >= settings.dark_ratio_threshold
        ):

            flags.append({
                "code":
                    "NEARLY_BLACK",
                "severity":
                    "CRITICAL",
            })

        # --------------------------------------------------------
        # Nearly white
        # --------------------------------------------------------

        if (
            bright_ratio
            >= settings.bright_ratio_threshold
        ):

            flags.append({
                "code":
                    "NEARLY_WHITE",
                "severity":
                    "CRITICAL",
            })

        # --------------------------------------------------------
        # Noise
        # --------------------------------------------------------

        if (
            noise_ratio
            >= settings.noise_ratio_threshold
        ):

            flags.append({
                "code":
                    "HIGH_NOISE",
                "severity":
                    "WARNING",
            })

        # --------------------------------------------------------
        # Skew
        # --------------------------------------------------------

        if (
            abs(
                metrics["skew_degrees"]
            ) > 15
        ):

            flags.append({
                "code":
                    "HIGH_SKEW",
                "severity":
                    "WARNING",
            })

        return flags

    # ============================================================
    # OCR recovery / validation
    # ============================================================

    @staticmethod
    def _ocr_may_recover(
        metrics: dict[str, Any],
    ) -> bool:

        return (
            metrics["blur_score"]
            >= settings.severe_blur_threshold
            and
            metrics["contrast_std"]
            >= settings.severe_contrast_threshold
            and
            metrics["foreground_ratio"]
            > 0.01
        )

    @staticmethod
    def _orientation_recovery_allowed(
        metrics: dict[str, Any],
    ) -> bool:
        """
        Decide whether it is worth trying rotated OCR.

        Rotation can help orientation problems, but it cannot repair
        a genuinely destroyed image.

        Therefore:
            - severe blur -> do not rotate
            - almost completely black/white -> do not rotate
            - insufficient foreground -> do not rotate
            - otherwise -> rotation is worth attempting
        """

        return (
            metrics["blur_score"]
            >= settings.severe_blur_threshold
            and
            metrics["foreground_ratio"]
            > 0.01
            and
            metrics["dark_ratio"]
            < 0.995
            and
            metrics["bright_ratio"]
            < 0.995
        )

    @staticmethod
    def _ocr_result_statistics(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return OCR statistics regardless of whether the OCR service
        places them under 'statistics' or at the top level.
        """

        statistics = result.get(
            "statistics",
            {},
        )

        if not isinstance(
            statistics,
            dict,
        ):
            statistics = {}

        return statistics

    def _ocr_result_is_readable(
        self,
        result: dict[str, Any],
    ) -> bool:
        """
        Final OCR readability decision.

        OCR engine success alone is not enough. The page must have:
            - sufficient detected text boxes
            - sufficient characters
            - sufficient mean confidence
            - sufficient median confidence

        This lets us explicitly use the configured confidence
        thresholds as the readability gate.
        """

        statistics = (
            self._ocr_result_statistics(
                result
            )
        )

        mean_confidence = statistics.get(
            "mean_confidence",
            result.get(
                "mean_confidence"
            ),
        )

        median_confidence = statistics.get(
            "median_confidence",
            result.get(
                "median_confidence"
            ),
        )

        box_count = statistics.get(
            "box_count",
            result.get(
                "box_count",
                0,
            ),
        )

        character_count = statistics.get(
            "character_count",
            result.get(
                "character_count",
                0,
            ),
        )

        if (
            mean_confidence is None
            or median_confidence is None
        ):
            return bool(
                result.get(
                    "ok",
                    False,
                )
            )

        try:
            mean_confidence = float(
                mean_confidence
            )

            median_confidence = float(
                median_confidence
            )

            box_count = int(
                box_count
            )

            character_count = int(
                character_count
            )

        except (
            TypeError,
            ValueError,
        ):
            return bool(
                result.get(
                    "ok",
                    False,
                )
            )

        return (
            mean_confidence
            >= settings.min_ocr_mean_confidence
            and
            median_confidence
            >= settings.min_ocr_median_confidence
            and
            box_count
            >= settings.min_ocr_boxes
            and
            character_count
            >= settings.min_ocr_characters
        )

    def _mark_ocr_readable(
        self,
        result: dict[str, Any],
        rotation: int = 0,
        recovery: bool = False,
    ) -> dict[str, Any]:
        """
        Normalize a successful OCR result.

        This ensures the rest of the document-quality pipeline sees
        the result as readable when the configured OCR criteria are
        satisfied.
        """

        normalized = dict(result)

        normalized["ok"] = True
        normalized["validation_passed"] = True

        if rotation:
            normalized["rotation_applied"] = rotation

        if recovery:
            normalized["recovered"] = True

        return normalized

    def _ocr_result_is_better(
        self,
        candidate: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        """
        Compare OCR results using confidence first and text amount
        second.
        """

        candidate_stats = (
            self._ocr_result_statistics(
                candidate
            )
        )

        current_stats = (
            self._ocr_result_statistics(
                current
            )
        )

        candidate_conf = float(
            candidate_stats.get(
                "mean_confidence",
                candidate.get(
                    "mean_confidence",
                    0.0,
                ),
            )
            or 0.0
        )

        current_conf = float(
            current_stats.get(
                "mean_confidence",
                current.get(
                    "mean_confidence",
                    0.0,
                ),
            )
            or 0.0
        )

        if candidate_conf != current_conf:
            return (
                candidate_conf
                > current_conf
            )

        candidate_chars = int(
            candidate_stats.get(
                "character_count",
                candidate.get(
                    "character_count",
                    0,
                ),
            )
            or 0
        )

        current_chars = int(
            current_stats.get(
                "character_count",
                current.get(
                    "character_count",
                    0,
                ),
            )
            or 0
        )

        return (
            candidate_chars
            > current_chars
        )

    def _run_ocr_with_recovery(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:

        # --------------------------------------------------------
        # 1. Normal OCR
        # --------------------------------------------------------

        result = ocr_service.analyze(
            image,
            recovery=False,
        )

        # Do not rely only on result["ok"].
        # Explicitly check the OCR confidence/readability criteria.
        if self._ocr_result_is_readable(
            result
        ):

            return self._mark_ocr_readable(
                result
            )

        # --------------------------------------------------------
        # 2. Calculate metrics only after OCR fails.
        # --------------------------------------------------------

        metrics = self._image_metrics(
            image
        )

        best_result = result
        best_rotation = 0

        # --------------------------------------------------------
        # 3. Orientation recovery
        #
        # Only run this on pages that are visually reasonable.
        # We do NOT rotate every page.
        # --------------------------------------------------------

        if self._orientation_recovery_allowed(
            metrics
        ):

            rotations = [
                (
                    90,
                    cv2.rotate(
                        image,
                        cv2.ROTATE_90_CLOCKWISE,
                    ),
                ),
                (
                    270,
                    cv2.rotate(
                        image,
                        cv2.ROTATE_90_COUNTERCLOCKWISE,
                    ),
                ),
                (
                    180,
                    cv2.rotate(
                        image,
                        cv2.ROTATE_180,
                    ),
                ),
            ]

            for rotation, rotated_image in rotations:

                rotated_result = (
                    ocr_service.analyze(
                        rotated_image,
                        recovery=True,
                    )
                )

                # If this orientation satisfies the actual
                # readability criteria, stop immediately.
                if self._ocr_result_is_readable(
                    rotated_result
                ):

                    return self._mark_ocr_readable(
                        rotated_result,
                        rotation=rotation,
                        recovery=True,
                    )

                # Keep the strongest failed result in case
                # preprocessing is also unsuccessful.
                if self._ocr_result_is_better(
                    rotated_result,
                    best_result,
                ):

                    best_result = (
                        rotated_result
                    )

                    best_rotation = rotation

        # --------------------------------------------------------
        # 4. Existing preprocessing recovery
        #
        # This is more expensive, so it happens after orientation
        # recovery.
        # --------------------------------------------------------

        if self._ocr_may_recover(
            metrics
        ):

            recovered = (
                self._preprocess_for_ocr(
                    image
                )
            )

            if recovered is not None:

                recovery_result = (
                    ocr_service.analyze(
                        recovered,
                        recovery=True,
                    )
                )

                if self._ocr_result_is_readable(
                    recovery_result
                ):

                    return self._mark_ocr_readable(
                        recovery_result,
                        recovery=True,
                    )

                if self._ocr_result_is_better(
                    recovery_result,
                    best_result,
                ):

                    best_result = (
                        recovery_result
                    )

        # --------------------------------------------------------
        # 5. Nothing passed the readability criteria.
        # --------------------------------------------------------

        if best_rotation:
            best_result = dict(
                best_result
            )
            best_result[
                "best_rotation_attempted"
            ] = best_rotation

        return best_result

    @staticmethod
    def _preprocess_for_ocr(
        image: np.ndarray,
    ) -> np.ndarray | None:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        denoised = (
            cv2.fastNlMeansDenoising(
                gray,
                None,
                h=7,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        )

        thresholded = (
            cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        )

        return cv2.cvtColor(
            thresholded,
            cv2.COLOR_GRAY2BGR,
        )

    # ============================================================
    # Reason helpers
    # ============================================================

    @staticmethod
    def _primary_reason(
        flags: list[dict[str, str]],
    ) -> str:

        if not flags:
            return "QUALITY_CHECK_FAILED"

        critical_flags = [
            flag
            for flag in flags
            if flag["severity"]
            == "CRITICAL"
        ]

        if critical_flags:
            return critical_flags[0][
                "code"
            ]

        return flags[0]["code"]

    @staticmethod
    def _warning_reason(
        flags: list[dict[str, str]],
    ) -> str | None:

        warnings = [
            flag["code"]
            for flag in flags
            if flag["severity"]
            == "WARNING"
        ]

        if not warnings:
            return None

        return ", ".join(
            warnings
        )

    @staticmethod
    def _ocr_failure_reason(
        flags: list[dict[str, str]],
    ) -> str:

        if flags:

            return (
                "OCR_UNREADABLE: "
                + ", ".join(
                    flag["code"]
                    for flag in flags
                )
            )

        return "OCR_UNREADABLE"

    # ============================================================
    # Skew
    # ============================================================

    @staticmethod
    def _estimate_skew(
        gray: np.ndarray,
    ) -> float:

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=max(
                50,
                min(gray.shape) // 4,
            ),
            maxLineGap=20,
        )

        if lines is None:
            return 0.0

        angles: list[float] = []

        for line in lines[:100]:

            x1, y1, x2, y2 = (
                line[0]
            )

            angle = math.degrees(
                math.atan2(
                    y2 - y1,
                    x2 - x1,
                )
            )

            if -45 <= angle <= 45:
                angles.append(
                    angle
                )

        if not angles:
            return 0.0

        return float(
            median(angles)
        )

    # ============================================================
    # PDF
    # ============================================================

    def _analyze_pdf(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> dict[str, Any]:

        try:
            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to open PDF: {exc}"
            ) from exc

        try:
            page_count = document.page_count

            if page_count > settings.max_pdf_pages:
                raise ValueError(
                    f"PDF contains {page_count} pages. "
                    f"Maximum allowed is {settings.max_pdf_pages} pages."
                )

            return self._analyze_pdf_pages(
                document,
                filename,
            )

        finally:
            document.close()

    # ============================================================
    # Unified PDF analysis
    # ============================================================

    def _analyze_pdf_pages(
        self,
        document: fitz.Document,
        filename: str,
    ) -> dict[str, Any]:

        page_count = document.page_count

        page_details: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        composition_counts = {
            "machine_readable": 0,
            "scanned": 0,
            "ocr_layered": 0,
            "blank": 0,
            "low_content": 0,
        }

        # --------------------------------------------------------
        # PASS 1: PDF structure + cheap 50-DPI screening
        # --------------------------------------------------------

        for page_index in range(page_count):

            page = document.load_page(
                page_index
            )

            native_text = (
                page.get_text(
                    "text"
                ).strip()
            )

            images = page.get_images(
                full=True
            )

            has_text = bool(
                native_text
            )

            has_images = bool(
                images
            )

            # True digital page.
            if has_text and not has_images:

                composition_counts[
                    "machine_readable"
                ] += 1

                page_details.append({
                    "page":
                        page_index + 1,
                    "status":
                        "PASS",
                    "content_type":
                        "MACHINE_READABLE",
                    "reason":
                        None,
                    "quality":
                        None,
                    "ocr":
                        None,
                })

                continue

            if has_text and has_images:

                structural_type = (
                    "OCR_LAYERED"
                )

            elif has_images:

                structural_type = (
                    "SCANNED"
                )

            else:

                structural_type = (
                    "UNKNOWN"
                )

            # Cheap render.
            image = self._render_page(
                page,
                dpi=settings.pdf_quality_render_dpi,
            )

            cheap = (
                self._cheap_image_metrics(
                    image
                )
            )

            content_type = (
                self._classify_cheap_content(
                    cheap
                )
            )

            del image

            if content_type == "BLANK":

                composition_counts[
                    "blank"
                ] += 1

                page_details.append({
                    "page":
                        page_index + 1,
                    "status":
                        "PASS",
                    "content_type":
                        "BLANK",
                    "reason":
                        None,
                    "quality":
                        None,
                    "ocr":
                        None,
                })

                continue

            if content_type == "LOW_CONTENT":

                composition_counts[
                    "low_content"
                ] += 1

                page_details.append({
                    "page":
                        page_index + 1,
                    "status":
                        "PASS",
                    "content_type":
                        "LOW_CONTENT",
                    "reason":
                        None,
                    "quality":
                        None,
                    "ocr":
                        None,
                })

                continue

            if structural_type == "OCR_LAYERED":

                composition_counts[
                    "ocr_layered"
                ] += 1

            elif structural_type == "SCANNED":

                composition_counts[
                    "scanned"
                ] += 1

            else:

                composition_counts[
                    "scanned"
                ] += 1

                structural_type = (
                    "SCANNED"
                )

            risk_score = (
                self._calculate_cheap_page_risk(
                    cheap
                )
            )

            candidates.append({
                "page_index":
                    page_index,
                "composition_type":
                    structural_type,
                "cheap_metrics":
                    cheap,
                "risk_score":
                    risk_score,
            })

            page_details.append({
                "page":
                    page_index + 1,
                "status":
                    "PENDING",
                "content_type":
                    structural_type,
                "reason":
                    None,
                "quality":
                    None,
                "ocr":
                    None,
                "risk_score":
                    risk_score,
            })

        # --------------------------------------------------------
        # PASS 2: initial adaptive OCR
        # --------------------------------------------------------

        ocr_budget = (
            self._calculate_ocr_budget(
                page_count
            )
        )

        initial_budget = min(
            ocr_budget,
            settings.max_initial_ocr_pages,
        )

        if len(candidates) <= initial_budget:

            initial_pages = [
                candidate["page_index"]
                for candidate in candidates
            ]

        else:

            initial_pages = (
                self._select_ocr_pages(
                    candidates,
                    budget=initial_budget,
                )
            )

        ocr_results: dict[
            int,
            dict[str, Any]
        ] = {}

        self._ocr_pages(
            document=document,
            page_indices=initial_pages,
            ocr_results=ocr_results,
        )

        # --------------------------------------------------------
        # PASS 3: targeted recovery around failed OCR pages
        # --------------------------------------------------------

        remaining_budget = max(
            0,
            ocr_budget
            - len(ocr_results),
        )

        if remaining_budget > 0:

            failed_pages = [
                page_index
                for page_index, result
                in ocr_results.items()
                if not self._ocr_result_is_readable(
                    result
                )
            ]

            already_checked = set(
                ocr_results
            )

            recovery_candidates: list[
                tuple[float, int]
            ] = []

            candidate_by_index = {
                candidate["page_index"]:
                    candidate
                for candidate in candidates
            }

            for failed_page in failed_pages:

                start = max(
                    0,
                    failed_page
                    - settings.ocr_failure_expansion_radius,
                )

                end = min(
                    page_count - 1,
                    failed_page
                    + settings.ocr_failure_expansion_radius,
                )

                for neighbor in range(
                    start,
                    end + 1,
                ):

                    if neighbor in already_checked:
                        continue

                    candidate = (
                        candidate_by_index.get(
                            neighbor
                        )
                    )

                    if candidate is None:
                        continue

                    recovery_candidates.append(
                        (
                            float(
                                candidate.get(
                                    "risk_score",
                                    0.0,
                                )
                            ),
                            neighbor,
                        )
                    )

            recovery_candidates = sorted(
                set(
                    recovery_candidates
                ),
                key=lambda item:
                    item[0],
                reverse=True,
            )

            recovery_pages = [
                page_index
                for _, page_index
                in recovery_candidates[
                    :remaining_budget
                ]
            ]

            self._ocr_pages(
                document=document,
                page_indices=recovery_pages,
                ocr_results=ocr_results,
            )

        # --------------------------------------------------------
        # PASS 4: detailed visual checks
        # --------------------------------------------------------

        for page_index, result in (
            ocr_results.items()
        ):

            detail = (
                page_details[
                    page_index
                ]
            )

            page = document.load_page(
                page_index
            )

            image = self._render_page(
                page,
                dpi=settings.pdf_render_dpi,
            )

            metrics = self._image_metrics(
                image
            )

            flags = self._quality_flags(
                metrics
            )

            del image

            detail["quality"] = metrics
            detail["quality_flags"] = flags

            statistics = (
                result.get(
                    "statistics",
                    {},
                )
            )

            if not isinstance(
                statistics,
                dict,
            ):
                statistics = {}

            readable = (
                self._ocr_result_is_readable(
                    result
                )
            )

            detail["ocr"] = {
                "readable":
                    readable,

                "recovery":
                    bool(
                        result.get(
                            "recovered",
                            False,
                        )
                    ),

                "rotation_applied":
                    result.get(
                        "rotation_applied",
                        0,
                    ),

                "mean_confidence":
                    statistics.get(
                        "mean_confidence"
                    ),

                "median_confidence":
                    statistics.get(
                        "median_confidence"
                    ),

                "box_count":
                    statistics.get(
                        "box_count"
                    ),

                "character_count":
                    statistics.get(
                        "character_count"
                    ),
            }

            # OCR is the primary readability signal
            # for sampled pages.
            if readable:

                detail["status"] = (
                    "PASS"
                )

                detail["reason"] = (
                    self._warning_reason(
                        flags
                    )
                )

            else:

                detail["status"] = (
                    "FAIL"
                )

                detail["reason"] = (
                    self._ocr_failure_reason(
                        flags
                    )
                )

        # --------------------------------------------------------
        # Critical cheap visual failures on unsampled pages
        # --------------------------------------------------------

        for candidate in candidates:

            page_index = (
                candidate["page_index"]
            )

            if page_index in ocr_results:
                continue

            cheap = (
                candidate["cheap_metrics"]
            )

            if self._cheap_critical_failure(
                cheap
            ):

                detail = (
                    page_details[
                        page_index
                    ]
                )

                detail["status"] = (
                    "FAIL"
                )

                detail["reason"] = (
                    self._cheap_failure_reason(
                        cheap
                    )
                )

        # Remaining candidates are not individually
        # proven bad.
        for detail in page_details:

            if detail["status"] == "PENDING":
                detail["status"] = "PASS"

        document_type = (
            self._determine_pdf_document_type(
                composition_counts
            )
        )

        return self._build_document_result(
            filename=filename,
            document_type=document_type,
            page_details=page_details,
            composition_counts=composition_counts,
            ocr_results=ocr_results,
        )

    # ============================================================
    # PDF helpers
    # ============================================================

    def _ocr_pages(
        self,
        document: fitz.Document,
        page_indices: list[int],
        ocr_results: dict[
            int,
            dict[str, Any]
        ],
    ) -> None:

        for page_index in page_indices:

            if page_index in ocr_results:
                continue

            page = document.load_page(
                page_index
            )

            image = self._render_page(
                page,
                dpi=settings.pdf_render_dpi,
            )

            try:

                ocr_results[
                    page_index
                ] = (
                    self._run_ocr_with_recovery(
                        image
                    )
                )

            finally:

                del image

    @staticmethod
    def _cheap_image_metrics(
        image: np.ndarray,
    ) -> dict[str, Any]:

        height, width = image.shape[:2]

        small = cv2.resize(
            image,
            (0, 0),
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY,
        )

        mean_brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        dark_ratio = float(
            np.mean(gray < 30)
        )

        bright_ratio = float(
            np.mean(gray > 245)
        )

        _, binary = cv2.threshold(
            gray,
            245,
            255,
            cv2.THRESH_BINARY_INV,
        )

        foreground_ratio = float(
            np.mean(
                binary > 0
            )
        )

        return {
            "width":
                int(width),

            "height":
                int(height),

            "brightness_mean":
                round(
                    mean_brightness,
                    2,
                ),

            "contrast_std":
                round(
                    contrast,
                    2,
                ),

            "dark_ratio":
                round(
                    dark_ratio,
                    4,
                ),

            "bright_ratio":
                round(
                    bright_ratio,
                    4,
                ),

            "foreground_ratio":
                round(
                    foreground_ratio,
                    4,
                ),
        }

    @staticmethod
    def _classify_cheap_content(
        metrics: dict[str, Any],
    ) -> str:

        foreground = (
            metrics[
                "foreground_ratio"
            ]
        )

        dark_ratio = (
            metrics[
                "dark_ratio"
            ]
        )

        bright_ratio = (
            metrics[
                "bright_ratio"
            ]
        )

        if (
            foreground < 0.005
            and (
                bright_ratio > 0.995
                or dark_ratio > 0.995
            )
        ):

            return "BLANK"

        if foreground < 0.01:
            return "LOW_CONTENT"

        return "MEANINGFUL"

    @staticmethod
    def _calculate_cheap_page_risk(
        metrics: dict[str, Any],
    ) -> float:

        risk = 0.0

        contrast = (
            metrics[
                "contrast_std"
            ]
        )

        dark_ratio = (
            metrics[
                "dark_ratio"
            ]
        )

        bright_ratio = (
            metrics[
                "bright_ratio"
            ]
        )

        if (
            contrast
            < settings.severe_contrast_threshold
        ):

            risk += 5

        elif (
            contrast
            < settings.contrast_threshold
        ):

            risk += 2

        if (
            dark_ratio
            >= settings.dark_ratio_threshold
        ):

            risk += 4

        if (
            bright_ratio
            >= settings.bright_ratio_threshold
        ):

            risk += 3

        return risk

    @staticmethod
    def _cheap_critical_failure(
        metrics: dict[str, Any],
    ) -> bool:

        return (
            metrics["dark_ratio"]
            >= 0.995
            or
            metrics["bright_ratio"]
            >= 0.995
            or
            metrics["contrast_std"]
            < settings.severe_contrast_threshold
        )

    @staticmethod
    def _cheap_failure_reason(
        metrics: dict[str, Any],
    ) -> str:

        if (
            metrics["dark_ratio"]
            >= 0.995
        ):
            return "NEARLY_BLACK"

        if (
            metrics["bright_ratio"]
            >= 0.995
        ):
            return "NEARLY_WHITE"

        return "SEVERE_LOW_CONTRAST"

    # ============================================================
    # PDF document type
    # ============================================================

    @staticmethod
    def _determine_pdf_document_type(
        composition_counts: dict[str, int],
    ) -> str:

        machine = (
            composition_counts[
                "machine_readable"
            ]
        )

        scanned = (
            composition_counts[
                "scanned"
            ]
        )

        layered = (
            composition_counts[
                "ocr_layered"
            ]
        )

        types = []

        if machine > 0:
            types.append(
                "DIGITAL"
            )

        if scanned > 0:
            types.append(
                "SCANNED"
            )

        if layered > 0:
            types.append(
                "OCR_LAYERED"
            )

        if len(types) == 1:

            if types[0] == "DIGITAL":
                return "DIGITAL_PDF"

            if types[0] == "SCANNED":
                return "SCANNED_PDF"

            if types[0] == "OCR_LAYERED":
                return "OCR_LAYERED_PDF"

        if len(types) > 1:
            return "MIXED_PDF"

        return "UNKNOWN_PDF"

    # ============================================================
    # PDF OCR budget
    # ============================================================

    def _calculate_ocr_budget(
        self,
        page_count: int,
    ) -> int:

        if page_count <= 3:

            return page_count

        if page_count <= 10:

            return min(
                6,
                page_count,
            )

        if page_count <= 100:

            return min(
                15,
                page_count,
            )

        if page_count <= 300:

            return min(
                20,
                page_count,
            )

        if page_count <= 500:

            return min(
                25,
                page_count,
            )

        return min(
            settings.max_ocr_pages,
            page_count,
        )

    # ============================================================
    # Adaptive OCR page selection
    # ============================================================

    def _select_ocr_pages(
        self,
        candidates: list[
            dict[str, Any]
        ],
        budget: int | None = None,
    ) -> list[int]:

        if not candidates:
            return []

        if budget is None:

            budget = (
                self._calculate_ocr_budget(
                    len(candidates)
                )
            )

        budget = max(
            0,
            min(
                budget,
                len(candidates),
            ),
        )

        if budget == 0:
            return []

        if len(candidates) <= budget:

            return [
                candidate[
                    "page_index"
                ]
                for candidate in candidates
            ]

        selected: set[int] = set()

        # First/last candidate pages.
        selected.add(
            candidates[0][
                "page_index"
            ]
        )

        selected.add(
            candidates[-1][
                "page_index"
            ]
        )

        # Representative sampling.
        representative_count = min(
            max(
                2,
                budget // 2,
            ),
            budget,
            len(candidates),
        )

        if representative_count > 0:

            positions = np.linspace(
                0,
                len(candidates) - 1,
                representative_count,
                dtype=int,
            )

            for position in positions:

                selected.add(
                    candidates[
                        int(position)
                    ][
                        "page_index"
                    ]
                )

        # Highest-risk pages.
        ranked = sorted(
            candidates,
            key=lambda item:
                item.get(
                    "risk_score",
                    0.0,
                ),
            reverse=True,
        )

        for candidate in ranked:

            if len(selected) >= budget:
                break

            selected.add(
                candidate[
                    "page_index"
                ]
            )

        return sorted(
            selected
        )[:budget]

    # ============================================================
    # PDF rendering
    # ============================================================

    @staticmethod
    def _render_page(
        page: fitz.Page,
        dpi: int | None = None,
    ) -> np.ndarray:

        if dpi is None:
            dpi = settings.pdf_render_dpi

        zoom = dpi / 72.0

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        )

        image = image.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        return image

    # ============================================================
    # DOCX
    # ============================================================

    def _analyze_docx(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> dict[str, Any]:

        native_text = (
            self._extract_docx_text(
                file_bytes
            )
        )

        images = (
            self._extract_docx_images(
                file_bytes
            )
        )

        has_text = bool(
            native_text.strip()
        )

        has_images = bool(
            images
        )

        if has_text and has_images:

            document_type = (
                "MIXED_DOCX"
            )

        elif has_images:

            document_type = (
                "IMAGE_BASED_DOCX"
            )

        else:

            document_type = (
                "TEXT_DOCX"
            )

        page_details: list[
            dict[str, Any]
        ] = []

        composition_counts = {
            "machine_readable": 0,
            "scanned": 0,
            "ocr_layered": 0,
            "blank": 0,
            "low_content": 0,
        }

        ocr_results: dict[
            int,
            dict[str, Any]
        ] = {}

        # --------------------------------------------------------
        # Native text DOCX
        # --------------------------------------------------------

        if has_text:

            words = len(
                native_text.split()
            )

            composition_counts[
                "machine_readable"
            ] += 1

            page_details.append({
                "page":
                    1,
                "status":
                    "PASS",
                "content_type":
                    "MACHINE_READABLE",
                "reason":
                    None,
                "quality":
                    {
                        "native_text_words":
                            words,
                    },
                "ocr":
                    None,
            })

        # --------------------------------------------------------
        # Embedded images
        # --------------------------------------------------------

        for image_index, image_bytes in enumerate(
            images,
            start=1,
        ):

            try:

                image = (
                    self._decode_image(
                        image_bytes
                    )
                )

                result = (
                    self._analyze_page_image(
                        image=image,
                        page_number=image_index,
                        run_ocr=True,
                    )
                )

                result[
                    "source"
                ] = "embedded_image"

                page_details.append(
                    result
                )

                if result.get(
                    "ocr"
                ) is not None:

                    ocr_results[
                        image_index - 1
                    ] = result[
                        "ocr"
                    ]

                if (
                    result[
                        "content_type"
                    ]
                    == "BLANK"
                ):

                    composition_counts[
                        "blank"
                    ] += 1

                elif (
                    result[
                        "content_type"
                    ]
                    == "LOW_CONTENT"
                ):

                    composition_counts[
                        "low_content"
                    ] += 1

                else:

                    composition_counts[
                        "scanned"
                    ] += 1

            except Exception as exc:

                page_details.append({
                    "page":
                        image_index,
                    "status":
                        "FAIL",
                    "content_type":
                        "MEANINGFUL",
                    "reason":
                        (
                            "IMAGE_DECODE_FAILED: "
                            f"{exc}"
                        ),
                    "quality":
                        None,
                    "ocr":
                        None,
                    "source":
                        "embedded_image",
                })

        return self._build_document_result(
            filename=filename,
            document_type=document_type,
            page_details=page_details,
            composition_counts=composition_counts,
            ocr_results=ocr_results,
        )

    # ============================================================
    # DOCX text extraction
    # ============================================================

    @staticmethod
    def _extract_docx_text(
        file_bytes: bytes,
    ) -> str:

        try:

            from docx import Document

            document = Document(
                io.BytesIO(
                    file_bytes
                )
            )

            parts: list[str] = []

            for paragraph in (
                document.paragraphs
            ):

                text = (
                    paragraph.text.strip()
                )

                if text:
                    parts.append(
                        text
                    )

            for table in (
                document.tables
            ):

                for row in table.rows:

                    for cell in row.cells:

                        text = (
                            cell.text.strip()
                        )

                        if text:
                            parts.append(
                                text
                            )

            return "\n".join(
                parts
            )

        except Exception:

            return ""

    # ============================================================
    # DOCX image extraction
    # ============================================================

    @staticmethod
    def _extract_docx_images(
        file_bytes: bytes,
    ) -> list[bytes]:

        images: list[
            bytes
        ] = []

        with zipfile.ZipFile(
            io.BytesIO(
                file_bytes
            ),
            "r",
        ) as archive:

            for name in (
                archive.namelist()
            ):

                if not name.startswith(
                    "word/media/"
                ):
                    continue

                extension = (
                    Path(name)
                    .suffix
                    .lower()
                )

                if extension not in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                ):
                    continue

                images.append(
                    archive.read(
                        name
                    )
                )

        return images

    # ============================================================
    # Overall result
    # ============================================================

    def _build_document_result(
        self,
        filename: str,
        document_type: str,
        page_details: list[
            dict[str, Any]
        ],
        composition_counts: dict[
            str,
            int,
        ] | None = None,
        ocr_results: dict[
            int,
            dict[str, Any],
        ] | None = None,
    ) -> dict[str, Any]:

        if composition_counts is None:

            composition_counts = {
                "machine_readable": 0,
                "scanned": 0,
                "ocr_layered": 0,
                "blank": 0,
                "low_content": 0,
            }

        if ocr_results is None:
            ocr_results = {}

        meaningful_pages = [
            page
            for page in page_details
            if page.get(
                "content_type"
            )
            not in {
                "BLANK",
                "LOW_CONTENT",
            }
        ]

        # --------------------------------------------------------
        # OCR failures
        #
        # IMPORTANT:
        # Use the same readability criteria here as the page-level
        # decision. A result that passed confidence validation is
        # NOT considered an OCR failure even if PaddleOCR itself
        # originally returned ok=False.
        # --------------------------------------------------------

        ocr_failed_pages = [
            page_index
            for page_index, result
            in ocr_results.items()
            if not self._ocr_result_is_readable(
                result
            )
        ]

        ocr_readable_pages = (
            len(ocr_results)
            - len(ocr_failed_pages)
        )

        visual_quality_failed_pages = 0
        visual_warning_pages = 0

        for page in page_details:

            flags = page.get(
                "quality_flags",
                [],
            )

            if any(
                flag.get(
                    "severity"
                ) == "CRITICAL"
                for flag in flags
            ):

                visual_quality_failed_pages += 1

            elif flags:

                visual_warning_pages += 1

        # --------------------------------------------------------
        # OCR failure rate
        # --------------------------------------------------------

        sampled_count = len(
            ocr_results
        )

        ocr_failure_rate = (
            len(ocr_failed_pages)
            / sampled_count
            if sampled_count
            else 0.0
        )

        total_meaningful = len(
            meaningful_pages
        )

        # --------------------------------------------------------
        # Document status
        # --------------------------------------------------------

        if total_meaningful == 0:

            status = "SUCCESS"

        else:

            if (
                total_meaningful
                <= settings.strict_meaningful_page_limit
            ):

                allowed_failures = 0

            else:

                allowed_failures = max(
                    1,
                    math.floor(
                        total_meaningful
                        * settings.meaningful_page_failure_percent
                    ),
                )

            hard_visual_failure = (
                visual_quality_failed_pages
                > allowed_failures
            )

            hard_ocr_failure = (
                sampled_count > 0
                and
                ocr_failure_rate
                > settings.meaningful_page_failure_percent
            )

            if (
                hard_visual_failure
                or hard_ocr_failure
            ):

                status = "FAILURE"

            else:

                status = "SUCCESS"

        # --------------------------------------------------------
        # Average OCR confidence
        # --------------------------------------------------------

        mean_confidences: list[
            float
        ] = []

        for result in (
            ocr_results.values()
        ):

            if not self._ocr_result_is_readable(
                result
            ):
                continue

            statistics = (
                result.get(
                    "statistics",
                    {},
                )
            )

            if not isinstance(
                statistics,
                dict,
            ):
                statistics = {}

            confidence = (
                statistics.get(
                    "mean_confidence"
                )
            )

            if confidence is not None:

                mean_confidences.append(
                    float(
                        confidence
                    )
                )

        average_ocr_confidence = (
            round(
                mean(
                    mean_confidences
                ),
                4,
            )
            if mean_confidences
            else None
        )

        # --------------------------------------------------------
        # Quality score
        # --------------------------------------------------------

        score = 100.0

        if meaningful_pages:

            score -= (
                35.0
                * (
                    visual_quality_failed_pages
                    / total_meaningful
                )
            )

        if sampled_count:

            score -= (
                45.0
                * ocr_failure_rate
            )

        score -= min(
            15.0,
            visual_warning_pages
            * 1.5,
        )

        if average_ocr_confidence is not None:

            score += max(
                -10.0,
                min(
                    10.0,
                    (
                        average_ocr_confidence
                        - 0.90
                    )
                    * 100,
                ),
            )

        score = round(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            2,
        )

        # --------------------------------------------------------
        # Warnings
        # --------------------------------------------------------

        warning_definitions = {

            "BAD_EXPOSURE": (
                "The page is too bright or too dark."
            ),

            "SKEW_DETECTED": (
                "The page appears tilted."
            ),

            "LOW_CONTRAST": (
                "The page has low contrast."
            ),

            "BLUR_DETECTED": (
                "The page appears blurred."
            ),

            "LOW_RESOLUTION": (
                "The page resolution may be too low."
            ),

            "HIGH_NOISE": (
                "The page contains significant image noise."
            ),

            "OCR_UNREADABLE": (
                "Text could not be read reliably on the sampled page."
            ),
        }

        warning_pages: dict[
            str,
            set[int]
        ] = {}

        for page in page_details:

            page_number = int(
                page["page"]
            )

            flags = page.get(
                "quality_flags",
                [],
            )

            for flag in flags:

                code = flag.get(
                    "code",
                    "",
                )

                mapped = {
                    "NEARLY_BLACK":
                        "BAD_EXPOSURE",

                    "NEARLY_WHITE":
                        "BAD_EXPOSURE",

                    "SEVERE_LOW_CONTRAST":
                        "LOW_CONTRAST",

                    "HIGH_SKEW":
                        "SKEW_DETECTED",

                    "BLUR":
                        "BLUR_DETECTED",

                    "SEVERE_BLUR":
                        "BLUR_DETECTED",

                    "LOW_RESOLUTION":
                        "LOW_RESOLUTION",

                    "CRITICAL_LOW_RESOLUTION":
                        "LOW_RESOLUTION",

                    "HIGH_NOISE":
                        "HIGH_NOISE",

                    "LOW_CONTRAST":
                        "LOW_CONTRAST",
                }.get(code)

                if mapped:

                    warning_pages.setdefault(
                        mapped,
                        set(),
                    ).add(
                        page_number
                    )

            # IMPORTANT:
            # Use the SAME OCR readability criteria here.
            # This prevents:
            #
            # status = SUCCESS
            # but OCR_UNREADABLE warning
            #
            if (
                page.get("ocr")
                is not None
                and not page["ocr"].get(
                    "readable",
                    False,
                )
            ):

                warning_pages.setdefault(
                    "OCR_UNREADABLE",
                    set(),
                ).add(
                    page_number
                )

        warnings_summary = []

        for code, pages in (
            warning_pages.items()
        ):

            warnings_summary.append({
                "source": (
                    "OCR"
                    if code
                    == "OCR_UNREADABLE"
                    else "IMAGE_QUALITY"
                ),

                "code":
                    code,

                "message":
                    warning_definitions[
                        code
                    ],

                "severity": (
                    "CRITICAL"
                    if (
                        code
                        == "OCR_UNREADABLE"
                        and
                        status
                        == "FAILURE"
                    )
                    else "WARNING"
                ),

                "affected_page_count":
                    len(pages),

                "sample_page_numbers":
                    sorted(pages)[:10],
            })

        # --------------------------------------------------------
        # Page counts
        # --------------------------------------------------------

        total_pages = len(
            page_details
        )

        counted_pages = sum(
            composition_counts.values()
        )

        if counted_pages != total_pages:

            composition_counts = {

                "machine_readable":
                    sum(
                        page.get(
                            "content_type"
                        )
                        == "MACHINE_READABLE"
                        for page
                        in page_details
                    ),

                "scanned":
                    sum(
                        page.get(
                            "content_type"
                        )
                        == "SCANNED"
                        for page
                        in page_details
                    ),

                "ocr_layered":
                    sum(
                        page.get(
                            "content_type"
                        )
                        == "OCR_LAYERED"
                        for page
                        in page_details
                    ),

                "blank":
                    sum(
                        page.get(
                            "content_type"
                        )
                        == "BLANK"
                        for page
                        in page_details
                    ),

                "low_content":
                    sum(
                        page.get(
                            "content_type"
                        )
                        == "LOW_CONTENT"
                        for page
                        in page_details
                    ),
            }

        # --------------------------------------------------------
        # Public contract
        # --------------------------------------------------------

        return {

            "status":
                status,

            "file_name":
                filename,

            "content_type": (
                "application/pdf"
                if filename.lower().endswith(
                    ".pdf"
                )
                else (
                    "image/png"
                    if filename.lower().endswith(
                        ".png"
                    )
                    else "image/jpeg"
                )
            ),

            "document_type":
                document_type,

            "validation_strategy":
                "PDF_PAGE_LEVEL_HYBRID_VALIDATION",

            "document_quality_score":
                score,

            "is_readable":
                status == "SUCCESS",

            "reupload_required":
                status == "FAILURE",

            "message": (
                "Document quality is acceptable for submission."
                if status == "SUCCESS"
                else
                "Document quality is insufficient. "
                "Please review the document and re-upload a clearer copy."
            ),

            "total_pages":
                total_pages,

            "metadata": {

                "page_type_counts": {

                    "MACHINE_READABLE_PAGE":
                        composition_counts[
                            "machine_readable"
                        ],

                    "SCANNED_IMAGE_PAGE":
                        composition_counts[
                            "scanned"
                        ],

                    "OCR_LAYERED_PAGE":
                        composition_counts[
                            "ocr_layered"
                        ],

                    "BLANK_PAGE":
                        composition_counts[
                            "blank"
                        ],

                    "LOW_CONTENT_PAGE":
                        composition_counts[
                            "low_content"
                        ],
                },

                "readability": {

                    "pages_screened":
                        total_pages,

                    "pages_ocr_checked":
                        sampled_count,

                    "ocr_readable_pages":
                        ocr_readable_pages,

                    "ocr_unreadable_pages":
                        len(
                            ocr_failed_pages
                        ),

                    "ocr_failure_rate":
                        round(
                            ocr_failure_rate,
                            4,
                        ),

                    "visual_quality_failed_pages":
                        visual_quality_failed_pages,
                },
            },

            "warnings_summary":
                warnings_summary,
        }

    # ============================================================
    # OCR runtime information
    # ============================================================

    @staticmethod
    def ocr_runtime_info() -> dict[str, Any]:
        return ocr_service.runtime_info()


# ================================================================
# Shared service
# ================================================================

document_quality_service = (
    DocumentQualityService()
)