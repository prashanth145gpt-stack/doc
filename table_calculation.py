from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

Number = Union[int, float]

app = FastAPI(title="IPO Issue Simulation Engine", version="1.0")

# -----------------------------
def to_int(x: Any, field: str) -> int:
    if x is None:
        raise ValueError(f"Missing required field: {field}")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"{field} must be integer-like, got {x}")
        return int(x)
    if isinstance(x, str):
        s = x.strip().upper()
        if s in {"", "NIL", "-", "NA", "N/A","Cannot be extracted"}:
            return 0
        s = s.replace(",", "").replace(" ", "")
        return int(s)
    raise TypeError(f"{field} must be int/float/str, got {type(x)}")


def to_float(x: Any, field: str) -> float:
    if x is None:
        raise ValueError(f"Missing required field: {field}")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().upper()
        if s in {"", "NIL", "-", "NA", "N/A","Cannot be extracted"}:
            return 0.0
        s = s.replace(",", "").replace(" ", "")
        return float(s)
    raise TypeError(f"{field} must be int/float/str, got {type(x)}")


def lakh_to_rupees(x_lakh: Number) -> float:
    return float(x_lakh) * 100_000.0


def rupees_to_crore(x_rupees: Number) -> float:
    return float(x_rupees) / 10_000_000.0


# Request Model 
class SimulationRequest(BaseModel):
    pre_issue_shares: Any = Field(..., description="Total shares outstanding pre-IPO")
    fresh_issue_shares: Any = Field(..., description="Fresh issue shares")
    ofs_promoter_shares: Any = Field(0, description="OFS shares by promoters; NIL -> 0")
    promoter_pre_issue_shares: Any = Field(..., description="Promoter shares pre-issue")
    pre_issue_networth_lakh: Any = Field(..., description="Net worth pre-issue in ₹ Lakhs")
    issue_price: Any = Field(..., description="Assumed issue price per share (₹)")
    pat_lakh: Dict[str, Any] = Field(..., description="PAT in ₹ Lakhs e.g. {'Mar25':804.16,'Mar24':630.29,'Mar23':292.40}")

    issue_expenses_lakh: Any = Field(0, description="Issue expenses in ₹ Lakhs (optional)")
    weights: Optional[Dict[str, int]] = Field(None, description="Weights for weighted EPS (default Mar25:3 Mar24:2 Mar23:1)")

    @validator("pat_lakh")
    def validate_pat(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("pat_lakh must be a non-empty dict")
        return v


# -----------------------------
# Core Formula Engine
# -----------------------------
def compute_issue_simulation(payload: SimulationRequest) -> Dict[str, Any]:
    # normalize inputs
    pre_issue_shares = to_int(payload.pre_issue_shares, "pre_issue_shares")
    fresh_issue_shares = to_int(payload.fresh_issue_shares, "fresh_issue_shares")
    ofs_promoter_shares = to_int(payload.ofs_promoter_shares, "ofs_promoter_shares")
    promoter_pre_issue_shares = to_int(payload.promoter_pre_issue_shares, "promoter_pre_issue_shares")

    pre_issue_networth_lakh = to_float(payload.pre_issue_networth_lakh, "pre_issue_networth_lakh")
    issue_price = to_float(payload.issue_price, "issue_price")
    issue_expenses_lakh = to_float(payload.issue_expenses_lakh, "issue_expenses_lakh")

    pat_lakh = {k: to_float(v, f"pat_lakh[{k}]") for k, v in payload.pat_lakh.items()}

    weights = payload.weights or {"Mar25": 3, "Mar24": 2, "Mar23": 1}

    # issue shares
    total_issue_shares = fresh_issue_shares + ofs_promoter_shares
    post_issue_shares = pre_issue_shares + total_issue_shares

    # promoter post-issue (assumption: OFS shares sold by promoters)
    promoter_post_issue_shares = promoter_pre_issue_shares - ofs_promoter_shares
    promoter_post_issue_pct = promoter_post_issue_shares / post_issue_shares if post_issue_shares else None

    # issue size & valuation (₹ Cr)
    fresh_proceeds_rupees = fresh_issue_shares * issue_price
    ofs_value_rupees = ofs_promoter_shares * issue_price
    total_issue_value_rupees = total_issue_shares * issue_price
    post_issue_valuation_rupees = post_issue_shares * issue_price

    fresh_issue_size_cr = rupees_to_crore(fresh_proceeds_rupees)
    ofs_issue_size_cr = rupees_to_crore(ofs_value_rupees)
    total_issue_size_cr = rupees_to_crore(total_issue_value_rupees)
    post_issue_valuation_cr = rupees_to_crore(post_issue_valuation_rupees)

    # EPS (pre and post)
    eps = {}
    eps_after_issue = {}
    for period, pat in pat_lakh.items():
        pat_rupees = lakh_to_rupees(pat)
        eps[period] = pat_rupees / pre_issue_shares if pre_issue_shares else None
        eps_after_issue[period] = pat_rupees / post_issue_shares if post_issue_shares else None

    # weighted average EPS
    num = 0.0
    den = 0.0
    for period, e in eps.items():
        w = weights.get(period, 0)
        if e is not None and w > 0:
            num += e * w
            den += w
    wtd_avg_eps = num / den if den else None

    # choose latest period 
    latest_key = "Mar25" if "Mar25" in eps else next(iter(eps.keys()))
    eps_latest = eps.get(latest_key)
    eps_post_latest = eps_after_issue.get(latest_key)

    # PE ratios
    pe_latest = issue_price / eps_latest if eps_latest else None
    pe_post_issue_latest = issue_price / eps_post_latest if eps_post_latest else None
    pe_weighted = issue_price / wtd_avg_eps if wtd_avg_eps else None

    # BV & PBV
    pre_issue_networth_rupees = lakh_to_rupees(pre_issue_networth_lakh)
    bv_pre = pre_issue_networth_rupees / pre_issue_shares if pre_issue_shares else None
    pbv_pre = issue_price / bv_pre if bv_pre else None

    # Post-issue net worth model
    post_issue_networth_lakh = pre_issue_networth_lakh + (fresh_proceeds_rupees / 100_000.0) - issue_expenses_lakh
    post_issue_networth_rupees = lakh_to_rupees(post_issue_networth_lakh)

    bv_post = post_issue_networth_rupees / post_issue_shares if post_issue_shares else None
    pbv_post = issue_price / bv_post if bv_post else None

    return {
        "normalized_inputs": {
            "pre_issue_shares": pre_issue_shares,
            "fresh_issue_shares": fresh_issue_shares,
            "ofs_promoter_shares": ofs_promoter_shares,
            "promoter_pre_issue_shares": promoter_pre_issue_shares,
            "issue_price": issue_price,
            "pre_issue_networth_lakh": pre_issue_networth_lakh,
            "issue_expenses_lakh": issue_expenses_lakh,
            "pat_lakh": pat_lakh,
            "weights": weights
        },
        "share_capital": {
            "total_issue_shares": total_issue_shares,
            "post_issue_shares": post_issue_shares
        },
        "issue_metrics": {
            "fresh_issue_size_cr": fresh_issue_size_cr,
            "ofs_issue_size_cr": ofs_issue_size_cr,
            "total_issue_size_cr": total_issue_size_cr,
            "post_issue_valuation_cr": post_issue_valuation_cr
        },
        "shareholding_metrics": {
            "promoter_post_issue_shares": promoter_post_issue_shares,
            "promoter_post_issue_pct": promoter_post_issue_pct
        },
        "eps_metrics": {
            "eps": eps,
            "eps_after_issue": eps_after_issue,
            "weighted_avg_eps": wtd_avg_eps
        },
        "valuation_metrics": {
            "latest_period": latest_key,
            "pe_latest": pe_latest,
            "pe_post_issue_latest": pe_post_issue_latest,
            "pe_weighted": pe_weighted,
            "bv_pre": bv_pre,
            "pbv_pre": pbv_pre,
            "post_issue_networth_lakh": post_issue_networth_lakh,
            "bv_post": bv_post,
            "pbv_post": pbv_post
        }
    }


# -----------------------------
# API Endpoint (Postman)
# -----------------------------
@app.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        return compute_issue_simulation(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
