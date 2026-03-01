from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """
    Input: one customer's details.
    Column names and categories match the IBM Telco Customer Churn dataset exactly.
    Pydantic validates every field — wrong type or invalid category returns HTTP 422.
    """
    # Numeric
    tenure:          int   = Field(..., ge=0, le=72,    description="Months as a customer (0–72)")
    monthly_charges: float = Field(..., ge=0.0, le=200.0, description="Monthly bill in USD")
    total_charges:   float = Field(..., ge=0.0,          description="Total amount billed to date")

    # Categorical — Literal enforces exact valid values
    contract:         Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method:   Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

    # Binary
    senior_citizen:    bool
    partner:           bool  # Has a partner?
    dependents:        bool  # Has dependents?
    phone_service:     bool  # Has phone service?
    paperless_billing: bool  # Enrolled in paperless billing?

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 3,
                "monthly_charges": 95.5,
                "total_charges": 286.5,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "senior_citizen": False,
                "partner": False,
                "dependents": False,
                "phone_service": True,
                "paperless_billing": True
            }
        }
    }


class PredictResponse(BaseModel):
    """Output: churn probability + human-readable risk label."""
    request_id:    str
    prediction:    int    # 0 = will NOT churn, 1 = WILL churn
    probability:   float  # e.g. 0.72 = 72% chance of churning
    risk_label:    str    # "Low" (<0.4) | "Medium" (0.4–0.7) | "High" (>0.7)
    model_version: str
    latency_ms:    float
