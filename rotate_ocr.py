import io
import numpy as np
import fitz  # PyMuPDF
import httpx
import cv2
import easyocr

from PIL import Image, ImageOps, ImageFilter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse


app = FastAPI()

# OCR model (load once)
reader = easyocr.Reader(["en", "hi"], gpu=False)

EXTRACTOR_URL = "https://file-extractordev.sidbi.in/extract"
EXTRACTOR_TIMEOUT = 60.0

# ----------------------------
# Small utilities
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def preprocess_for_compress_and_readability(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    return img

def resize_long_edge(pil_img: Image.Image, max_long_edge: int = 1568) -> Image.Image:
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return pil_img
    scale = max_long_edge / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    # Pillow compatible resampling
    if hasattr(Image, "Resampling"):
        return pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return pil_img.resize((new_w, new_h), resample=Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC)

# ----------------------------
# OCR rotation scoring (NEW)
# ----------------------------
def _lanczos_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC

def downscale_for_scoring(pil_img: Image.Image, max_long_edge: int = 1100) -> Image.Image:
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return pil_img
    scale = max_long_edge / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil_img.resize((new_w, new_h), resample=_lanczos_filter())

def rotate_pil(pil_img: Image.Image, angle: int) -> Image.Image:
    if angle % 360 == 0:
        return pil_img
    return pil_img.rotate(angle, expand=True, resample=Image.BICUBIC)

def score_easyocr_results(results, conf_th: float = 0.0):
    confs = []
    total_chars = 0
    for bbox, text, conf in results:
        if conf is None or conf < conf_th:
            continue
        confs.append(float(conf))
        if isinstance(text, str):
            total_chars += len(text)

    median_conf = float(np.median(confs)) if confs else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    n_boxes = len(confs)

    # Robust score: confidence + amount of detected text
    score = (median_conf * 2.0) + (min(n_boxes, 40) * 0.05) + (min(total_chars, 400) * 0.0025)

    return score, {
        "median_conf": median_conf,
        "mean_conf": mean_conf,
        "n_boxes": n_boxes,
        "total_chars": total_chars,
    }

def best_rotation_by_easyocr(
    pil_img: Image.Image,
    angles=(0, 90, 180, 270),
    max_long_edge: int = 1100,
    conf_th: float = 0.0
):
    """
    Cheap orientation selection:
    - Run OCR on downscaled image for each angle
    - Choose best score
    - Rotate full-res image by best angle
    """
    small = downscale_for_scoring(pil_img, max_long_edge=max_long_edge)

    best_score = float("-inf")
    best_angle = 0
    best_stats = None
    per_angle = []

    for a in angles:
        test = rotate_pil(small, a)
        res = reader.readtext(np.array(test))
        s, st = score_easyocr_results(res, conf_th=conf_th)
        per_angle.append({"angle": a, "score": s, **st})
        if s > best_score:
            best_score = s
            best_angle = a
            best_stats = st

    rotated_full = rotate_pil(pil_img, best_angle)
    return rotated_full, best_angle, best_stats, per_angle

# ----------------------------
# Card detection (your logic)
# ----------------------------
def union_bbox_from_easyocr(results, conf_th=0.35):
    xs, ys = [], []
    for bbox, _, conf in results:
        if conf is None or conf < conf_th:
            continue
        for x, y in bbox:
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)

def fallback_bbox_nonwhite(gray, thr=245):
    mask = gray < thr
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def detect_card_bbox_connected_components(gray, thr=235, min_fill=0.55):
    try:
        import scipy.ndimage as ndi
    except Exception:
        return fallback_bbox_nonwhite(gray)

    H, W = gray.shape
    mask = gray < thr
    if not mask.any():
        return None

    bw = ndi.binary_opening(mask, iterations=1)
    bw = ndi.binary_closing(bw, iterations=1)

    labels, n = ndi.label(bw)  # type: ignore
    if n == 0:
        return None

    objs = ndi.find_objects(labels)
    best, best_score = None, -1e18

    for idx, sl in enumerate(objs, start=1):
        if sl is None:
            continue

        ys, xs = sl
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop

        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue

        fill = (labels[sl] == idx).sum() / area
        ar = (x1 - x0) / max(1, (y1 - y0))
        inv = max(ar, 1 / max(ar, 1e-6))
        area_ratio = area / (H * W)

        if not (0.008 <= area_ratio <= 0.55):
            continue
        if not (1.1 <= inv <= 3.2):
            continue
        if fill < min_fill:
            continue

        score = 3 * fill - abs(inv - 1.6) - abs(area_ratio - 0.1)
        if score > best_score:
            best_score = score
            best = (float(x0), float(y0), float(x1), float(y1))

    return best

def detect_card_bbox_any_rotation(gray):
    H, W = gray.shape
    b = detect_card_bbox_connected_components(gray)
    if b:
        return b

    for k in (1, 2, 3):
        g = np.rot90(gray, k)
        b2 = detect_card_bbox_connected_components(g)
        if not b2:
            continue

        x0, y0, x1, y1 = b2
        if k == 1:
            return W - 1 - y1, x0, W - 1 - y0, x1
        if k == 2:
            return W - 1 - x1, H - 1 - y1, W - 1 - x0, H - 1 - y0
        if k == 3:
            return y0, H - 1 - x1, y1, H - 1 - x0

    return None

def extract_card(pil_img: Image.Image):
    gray = np.array(pil_img.convert("L"))
    work = gray.copy()
    H, W = work.shape

    for _ in range(3):
        b = detect_card_bbox_any_rotation(work)

        if not b:
            ocr = reader.readtext(np.array(pil_img))
            b = union_bbox_from_easyocr(ocr)
            if not b:
                return None

        x0, y0, x1, y1 = map(int, b)

        # clamp to image bounds (important)
        x0 = clamp(x0, 0, W - 1)
        x1 = clamp(x1, 0, W)
        y0 = clamp(y0, 0, H - 1)
        y1 = clamp(y1, 0, H)

        if x1 <= x0 or y1 <= y0:
            return None

        crop = pil_img.crop((x0, y0, x1, y1))
        if crop.width > 200 and crop.height > 120:
            return crop

        work[y0:y1, x0:x1] = 255

    return None

# ----------------------------
# Extractor call
# ----------------------------
async def extractor(file_bytes: bytes, filename: str, content_type: str):
    async with httpx.AsyncClient(timeout=EXTRACTOR_TIMEOUT, verify=False) as client:
        files = {"file": (filename, file_bytes, content_type)}
        resp = await client.post(EXTRACTOR_URL, files=files)

    if resp.status_code < 200 or resp.status_code >= 300:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ExtractorDev error {resp.status_code}: {detail}")

    return resp.json()

# ----------------------------
# Quality checks (your logic)
# ----------------------------
def blur_score(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def edge_density(img):
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160).mean()

def contrast(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()

def glare(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 245).mean()

def quality_check(img):
    h, w = img.shape[:2]
    if h < 300 or w < 300:
        return False, (
            "Image is too small/low resolution. Please re-upload a clearer photo (min 300×300) "
            "and make sure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    ocr = reader.readtext(img)
    if not ocr:
        return False, (
            "No readable text found. Please upload a clear photo with the full card visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    confs, blurs, edges, conts = [], [], [], []
    for b, _, c in ocr:
        (x0, y0), (x1, y1) = b[0], b[2]
        x0i, x1i = sorted([int(x0), int(x1)])
        y0i, y1i = sorted([int(y0), int(y1)])

        x0i = clamp(x0i, 0, w - 1)
        x1i = clamp(x1i, 0, w)
        y0i = clamp(y0i, 0, h - 1)
        y1i = clamp(y1i, 0, h)

        roi = img[y0i:y1i, x0i:x1i]
        if roi.size == 0:
            continue

        confs.append(float(c) if c is not None else 0.0)
        blurs.append(blur_score(roi))
        edges.append(edge_density(roi))
        conts.append(contrast(roi))

    if not confs:
        return False, (
            "No readable text found. Please upload a clear photo with the full card visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    if float(np.median(confs)) < 0.25:
        return False, (
            "Whole Text is not clear enough to read. Please retake in better lighting and avoid shadows. "
            "Also ensure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    if float(np.median(blurs)) < 15:
        return False, (
            "Text is not clear enough to read. Please retake in better lighting and avoid shadows. "
            "Also ensure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    if float(np.median(edges)) < 3:
        return False, (
            "Image is blurry. Hold steady, tap to focus on the text, and retake the photo. "
            "Also ensure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    if float(np.median(conts)) < 22:
        return False, (
            "Text contrast is too low. Please use better lighting and avoid dark/colored backgrounds. "
            "Also ensure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    if glare(img) > 0.20:
        return False, (
            "Glare detected. Tilt the camera slightly or move away from direct light and retake. "
            "Also ensure the full card is visible (all edges). "
            "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare."
        )

    return True, None

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/process")
async def process(file: UploadFile = File(...)):
    ext = (file.filename.split(".")[-1].lower() if file.filename else "")
    raw = await file.read()

    if ext not in ["pdf", "jpg", "jpeg", "png"]:
        return JSONResponse({
            "status": "failure",
            "reason": "Invalid file type. Please upload either one of pdf, jpg, jpeg or png"
        })

    # ---- Load image or pass-through multipage PDFs ----
    if ext == "pdf":
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
        except Exception:
            return JSONResponse({"status": "failure", "reason": "Decoding failed. Please retry again."})

        if len(doc) > 1:
            # multipage -> forward to extractor (original behavior)
            try:
                extractor_json = await extractor(raw, file.filename or "file.pdf", "application/pdf")
                return JSONResponse({"status": "success", "data": extractor_json})
            except Exception as e:
                return JSONResponse({"status": "failure", "reason": "extractor_failed", "detail": str(e)})

        # single-page pdf -> render page to image
        page = doc[0]
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        try:
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception:
            return JSONResponse({"status": "failure", "reason": "Decoding failed. Please retry again."})

    else:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return JSONResponse({"status": "failure", "reason": "Decoding failed. Please retry again."})

    # ---- NEW: Rotation selection for BOTH PDFs and images ----
    img, best_angle, best_stats, per_angle = best_rotation_by_easyocr(
        img,
        angles=(0, 90, 180, 270),
        max_long_edge=1100,   # keep it cheap (tune if needed: 900-1500)
        conf_th=0.0
    )

    # ---- Your existing card detection flow (now on rotated image) ----
    card = extract_card(img)
    if not card:
        return JSONResponse({
            "status": "failure",
            "reason": "Card not detected. Please make sure all edges of the cards are visible. "
                      "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare.",
            "rotation": {"best_angle": best_angle, "best_stats": best_stats, "per_angle": per_angle}
        })

    card = preprocess_for_compress_and_readability(card)
    card = resize_long_edge(card)

    cv_img = cv2.cvtColor(np.array(card), cv2.COLOR_GRAY2BGR)
    ok, reason = quality_check(cv_img)
    if not ok:
        return JSONResponse({
            "status": "failure",
            "reason": reason,
            "rotation": {"best_angle": best_angle, "best_stats": best_stats, "per_angle": per_angle}
        })

    img_bytes = pil_to_png_bytes(card)

    try:
        extractor_json = await extractor(img_bytes, "card.png", "image/png")
        return JSONResponse({
            "status": "success",
            "rotation": {"best_angle": best_angle, "best_stats": best_stats, "per_angle": per_angle},
            "data": extractor_json
        })
    except Exception as e:
        return JSONResponse({
            "status": "failure",
            "reason": "extractor_failed",
            "detail": str(e),
            "rotation": {"best_angle": best_angle, "best_stats": best_stats, "per_angle": per_angle}
        })
