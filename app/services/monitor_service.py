"""
app/services/monitor_service.py
--------------------------------
Computes model monitoring metrics from prediction logs.

PSI (Population Stability Index) — the industry-standard metric for detecting
data drift. Compares the distribution of a feature in live traffic against the
baseline distribution captured at training time.

PSI interpretation:
  < 0.10  -> stable      (distribution hasn't changed)
  0.10-0.20 -> warning   (minor shift, keep an eye on it)
  > 0.20  -> drift       (significant shift — retrain or investigate)

Why PSI and not KS test?
- PSI is the most common choice in financial/ML production because it's
  easy to explain to non-technical stakeholders: "the data looks different
  from what the model was trained on by X%."
- KS test gives a p-value, which requires more statistical context to act on.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import PredictionLog
from app.schemas.monitor import (
    MonitorResponse,
    FeatureDrift,
    PredictionDistribution,
    DailyChurnRate,
)

logger = logging.getLogger(__name__)

# Numeric features to compute PSI for (categorical drift is more complex — future work)
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]


def _compute_psi(baseline_counts: list, baseline_edges: list, live_values: list) -> float:
    """
    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    baseline_counts: histogram counts from training data
    baseline_edges:  bin edges from training data
    live_values:     raw feature values from live predictions
    """
    if len(live_values) < 10:
        return 0.0   # not enough data to compute meaningful PSI

    baseline_counts = np.array(baseline_counts, dtype=float)
    baseline_pct = baseline_counts / baseline_counts.sum()

    # Bin the live data using the same edges as training
    live_counts, _ = np.histogram(live_values, bins=baseline_edges)
    live_counts = live_counts.astype(float)
    live_pct = live_counts / live_counts.sum() if live_counts.sum() > 0 else live_counts

    # Clip to avoid log(0) — PSI is undefined when a bin is completely empty
    eps = 1e-6
    baseline_pct = np.clip(baseline_pct, eps, None)
    live_pct = np.clip(live_pct, eps, None)

    psi = np.sum((live_pct - baseline_pct) * np.log(live_pct / baseline_pct))
    return round(float(psi), 4)


def _psi_status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.20:
        return "warning"
    return "drift"


def _build_churn_rate_series(logs: List[PredictionLog]) -> List[DailyChurnRate]:
    """Create a day-level churn-rate time series for UI trend charts."""
    by_day = {}

    for log in logs:
        ts = log.timestamp
        if ts is None:
            continue
        day = ts.date().isoformat()
        if day not in by_day:
            by_day[day] = {"total": 0, "churn": 0}

        by_day[day]["total"] += 1
        if log.prediction == 1:
            by_day[day]["churn"] += 1

    series = []
    for day in sorted(by_day.keys()):
        total = by_day[day]["total"]
        churn = by_day[day]["churn"]
        rate = round((churn / total), 4) if total > 0 else 0.0
        series.append(
            DailyChurnRate(
                date=day,
                churn_rate=rate,
                churn_count=churn,
                total_predictions=total,
            )
        )

    return series


def get_monitor_report(db: Session, window_days: int = 7) -> MonitorResponse:
    """
    Pull prediction logs for the past `window_days` days and compute:
    - Prediction distribution (churn rate, risk breakdown)
    - Day-level churn-rate trend
    - PSI drift scores for each numeric feature
    """
    # -- Load baseline distributions ------------------------------------------
    metadata_path = Path(settings.METADATA_PATH)
    with open(metadata_path) as f:
        metadata = json.load(f)
    baseline_dists = metadata.get("baseline_distributions", {})
    model_version = metadata.get("model_version", "unknown")

    # -- Query prediction logs for the window ---------------------------------
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    logs: List[PredictionLog] = (
        db.query(PredictionLog)
        .filter(PredictionLog.timestamp >= cutoff)
        .all()
    )

    total = len(logs)

    # -- Prediction distribution ----------------------------------------------
    if total == 0:
        pred_dist = PredictionDistribution(
            total_predictions=0,
            churn_count=0,
            no_churn_count=0,
            avg_probability=0.0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
        )
        return MonitorResponse(
            model_version=model_version,
            window_days=window_days,
            prediction_distribution=pred_dist,
            churn_rate_over_time=[],
            feature_drift=[],
            overall_drift_status="stable",
        )

    churn_count = sum(1 for log in logs if log.prediction == 1)
    avg_prob = round(sum(log.probability for log in logs) / total, 4)
    high_risk = sum(1 for log in logs if log.risk_label == "High")
    medium_risk = sum(1 for log in logs if log.risk_label == "Medium")
    low_risk = sum(1 for log in logs if log.risk_label == "Low")

    pred_dist = PredictionDistribution(
        total_predictions=total,
        churn_count=churn_count,
        no_churn_count=total - churn_count,
        avg_probability=avg_prob,
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
    )

    churn_series = _build_churn_rate_series(logs)

    # -- PSI for each numeric feature -----------------------------------------
    drift_results: List[FeatureDrift] = []
    all_statuses = []

    for feature in NUMERIC_FEATURES:
        if feature not in baseline_dists:
            continue

        # Extract live values from the JSON input_features column
        live_values = []
        for log in logs:
            if log.input_features and feature in log.input_features:
                try:
                    live_values.append(float(log.input_features[feature]))
                except (TypeError, ValueError):
                    pass

        baseline = baseline_dists[feature]
        psi = _compute_psi(
            baseline_counts=baseline["counts"],
            baseline_edges=baseline["bin_edges"],
            live_values=live_values,
        )
        status = _psi_status(psi)
        all_statuses.append(status)

        drift_results.append(
            FeatureDrift(
                feature=feature,
                psi_score=psi,
                status=status,
            )
        )

    # Overall status = worst individual status
    if "drift" in all_statuses:
        overall = "drift"
    elif "warning" in all_statuses:
        overall = "warning"
    else:
        overall = "stable"

    return MonitorResponse(
        model_version=model_version,
        window_days=window_days,
        prediction_distribution=pred_dist,
        churn_rate_over_time=churn_series,
        feature_drift=drift_results,
        overall_drift_status=overall,
    )
