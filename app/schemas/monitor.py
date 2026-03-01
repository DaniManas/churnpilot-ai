from pydantic import BaseModel
from typing import List


class FeatureDrift(BaseModel):
    """PSI (Population Stability Index) score for one feature."""
    feature:    str
    psi_score:  float
    status:     str    # "stable" (<0.1), "warning" (0.1–0.2), "drift" (>0.2)


class PredictionDistribution(BaseModel):
    """How predictions have been split over the monitored window."""
    total_predictions: int
    churn_count:       int      # prediction == 1
    no_churn_count:    int      # prediction == 0
    avg_probability:   float
    high_risk_count:   int      # probability > 0.7
    medium_risk_count: int
    low_risk_count:    int


class DailyChurnRate(BaseModel):
    """Daily churn trend used by the monitoring dashboard line chart."""
    date: str
    churn_rate: float
    churn_count: int
    total_predictions: int


class MonitorResponse(BaseModel):
    model_version: str
    window_days: int  # how many days of data were used
    prediction_distribution: PredictionDistribution
    churn_rate_over_time: List[DailyChurnRate]
    feature_drift: List[FeatureDrift]
    overall_drift_status: str  # "stable" / "warning" / "drift"
