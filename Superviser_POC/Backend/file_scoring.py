import json
import os
from datetime import datetime
from andromeda.core.workflow import WorkflowBuilder
from config import COMBINED_INPUT, UNIFIED_CLAIMS_OUT, RISK_METRICS_OUT, LARGE_LOSS_THRESHOLD_PCT, logger

def normalize_submission(loss_runs: list[dict]) -> dict:
    unified_claims = []

    for loss_run in loss_runs:
        policy_year_label = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"

        for claim in loss_run.get("claims", []):
            row = dict(claim)
            row["source_file"] = loss_run.get("source_file")
            row["carrier_name"] = loss_run.get("carrier_name")
            row["policy_year"] = policy_year_label
            row["policy_start"] = loss_run.get("policy_start")
            row["policy_end"] = loss_run.get("policy_end")
            row["annual_premium"] = loss_run.get("annual_premium")
            row["per_occurrence_limit"] = loss_run.get("per_occurrence_limit")
            unified_claims.append(row)

    submission_summary = {
        "insured_name": loss_runs[0].get("insured_name") if loss_runs else None,
        "policy_years_covered": len(loss_runs),
        "total_premium_all_years": sum(lr.get("annual_premium") or 0 for lr in loss_runs),
        "total_claims": len(unified_claims),
        "source_files": [lr.get("source_file") for lr in loss_runs],
    }

    return {
            "submission_summary": submission_summary,
            "unified_claims": unified_claims,
            }

#################### RISK METRICS ##########################3

def compute_year_metrics(claims, loss_runs):
    by_year = {}

    for loss_run in loss_runs:
        year = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"

        year_claims = [c for c in claims if c["policy_year"] == year]

        premium = loss_run.get("annual_premium", 0)
        total_incurred = sum(c.get("total_incurred", 0) for c in year_claims)
        total_reserve = sum(c.get("reserve", 0) for c in year_claims)

        open_claims = [
            c for c in year_claims
            if c.get("status") in ("Open", "Reopened")
        ]

        by_year[year] = {
            "carrier_name": loss_run.get("carrier_name"),
            "premium": premium,
            "claim_count": len(year_claims),
            "total_incurred": round(total_incurred, 2),
            "loss_ratio_pct": round((total_incurred / premium) * 100, 1) if premium else None,
            "avg_severity": round(total_incurred / len(year_claims), 2) if year_claims else 0,
            "open_claim_count": len(open_claims),
            "pct_claims_open": round((len(open_claims) / len(year_claims)) * 100, 1) if year_claims else 0,
            "reserve_share_of_incurred_pct": round((total_reserve / total_incurred) * 100, 1) if total_incurred else 0,
        }

    return by_year


def compute_aggregate_metrics(by_year, claims):
    total_premium = sum(y["premium"] for y in by_year.values())
    total_incurred = sum(y["total_incurred"] for y in by_year.values())

    return {
        "total_premium": round(total_premium, 2),
        "total_incurred": round(total_incurred, 2),
        "overall_loss_ratio_pct": round((total_incurred / total_premium) * 100, 1)
        if total_premium else None,
        "total_claim_count": len(claims),
        "avg_claims_per_year": round(len(claims) / len(by_year), 2)
        if by_year else 0,
    }


def compute_trend(by_year, loss_runs):

    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except (TypeError, ValueError):
            return None

    years = []

    for loss_run in loss_runs:
        year = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"
        metrics = by_year.get(year)

        if metrics and metrics["loss_ratio_pct"] is not None:
            dt = parse_date(loss_run.get("policy_start"))
            if dt:
                years.append((year, metrics, dt))

    years.sort(key=lambda x: x[2])

    if len(years) < 2:
        return None

    oldest = years[0]
    newest = years[-1]

    if newest[1]["loss_ratio_pct"] < oldest[1]["loss_ratio_pct"]:
        direction = "improving"
    elif newest[1]["loss_ratio_pct"] > oldest[1]["loss_ratio_pct"]:
        direction = "worsening"
    else:
        direction = "flat"

    return {
        "oldest_year": oldest[0],
        "oldest_year_loss_ratio_pct": oldest[1]["loss_ratio_pct"],
        "newest_year": newest[0],
        "newest_year_loss_ratio_pct": newest[1]["loss_ratio_pct"],
        "direction": direction,
    }


def compute_open_exposure(claims):
    open_claims = [
        c for c in claims
        if c.get("status") in ("Open", "Reopened")
    ]

    return {
        "open_claim_count": len(open_claims),
        "total_outstanding_reserve": round(
            sum(c.get("reserve", 0) for c in open_claims), 2
        ),
    }


def find_large_losses(claims):
    losses = []

    for c in claims:
        limit = c.get("per_occurrence_limit")

        if limit and c.get("total_incurred", 0) >= limit * LARGE_LOSS_THRESHOLD_PCT:
            losses.append({
                "claim_number": c.get("claim_number"),
                "policy_year": c.get("policy_year"),
                "total_incurred": c.get("total_incurred"),
                "per_occurrence_limit": limit,
                "pct_of_limit": round((c["total_incurred"] / limit) * 100, 1),
            })

    return losses


def compute_top_causes(claims):
    cause_counts = {}

    for claim in claims:
        cause = claim.get("cause_of_loss", "Unknown")
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    top_causes = sorted(
        cause_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return [
        {"cause": cause, "count": count}
        for cause, count in top_causes
    ]


def generate_maturity_caveats(by_year):
    caveats = []

    for year, metrics in by_year.items():
        if metrics["pct_claims_open"] >= 30:
            caveats.append(
                f"{year}: {metrics['pct_claims_open']}% of claims remain open."
            )

    return caveats


def compute_risk_metrics(normalized: dict, loss_runs: list[dict]) -> dict:
    claims = normalized["unified_claims"]

    by_year = compute_year_metrics(claims, loss_runs)
    aggregate = compute_aggregate_metrics(by_year, claims)
    trend = compute_trend(by_year, loss_runs)
    open_exposure = compute_open_exposure(claims)
    large_losses = find_large_losses(claims)
    top_causes = compute_top_causes(claims)
    maturity_caveats = generate_maturity_caveats(by_year)

    return {
        "by_policy_year": by_year,
        "aggregate": aggregate,
        "trend": trend,
        "open_exposure": open_exposure,
        "large_losses": large_losses,
        "top_causes_of_loss": top_causes,
        "maturity_caveats": maturity_caveats,
    }

###################### ANDromeda Pipeline ########################
def step_normalize(state: dict) -> dict:
    normalized = normalize_submission(state["loss_runs"])
    return {"normalized": normalized}


def step_score(state: dict) -> dict:
    risk_metrics = compute_risk_metrics(state["normalized"], state["loss_runs"])
    return {"risk_metrics": risk_metrics}


def build_scoring_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="ScoringPipeline")
    (
        pipeline
        .start("normalize").run(step_normalize)
        .finish("score").run(step_score)
    )
    return pipeline

def func_scoring():
    with open(COMBINED_INPUT) as f:
        loss_runs = json.load(f)

    logger.info("Loaded %d structured loss run(s)", len(loss_runs))

    pipeline = build_scoring_pipeline()
    result = pipeline.execute(state={"loss_runs": loss_runs})

    with open(UNIFIED_CLAIMS_OUT, "w") as f:
        json.dump(result["normalized"], f, indent=2)
    logger.info("Step 5 complete: Unified claims saved")

    with open(RISK_METRICS_OUT, "w") as f:
        json.dump(result["risk_metrics"], f, indent=2)
    logger.info("Step 6 complete: Risk metrics saved")


if __name__ == "__main__":
    func_scoring()