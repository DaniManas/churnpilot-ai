"""
app/services/ml_service.py
--------------------------
Loads the trained pipeline once at startup and exposes a clean predict() interface.

The pipeline is loaded ONCE at module import — not on every request.
Loading a model file on every request would be extremely slow at any real traffic volume.
"""

import json
import pickle
import logging
import pandas as pd
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

# ── Feature order must exactly match what the pipeline was trained on ─────────
# Any change here must be mirrored in ml/train.py
NUMERIC_FEATURES     = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
BINARY_FEATURES      = ["senior_citizen", "partner", "dependents",
                         "phone_service", "paperless_billing"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


class MLService:
    """
    Wraps the trained sklearn pipeline.
    One instance is created at startup and reused for every request.
    """

    def __init__(self):
        self.pipeline = None
        self.metadata = None
        self._load()

    def _load(self):
        """Load pipeline and metadata from disk. Called once at startup."""
        model_path    = Path(settings.MODEL_PATH)
        metadata_path = Path(settings.METADATA_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run  python ml/train.py  first."
            )

        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(
            f"Model v{self.metadata['model_version']} loaded | "
            f"ROC-AUC: {self.metadata['roc_auc_score']} | "
            f"Source: {self.metadata.get('data_source', 'unknown')}"
        )

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"] if self.metadata else "unknown"

    def predict(self, input_data: dict) -> dict:
        """
        Run a single prediction.

        Args:
            input_data: dict with keys matching ALL_FEATURES
                        (bool values are converted to int for the pipeline)

        Returns:
            dict with prediction, probability, risk_label, model_version
        """
        # Convert bools to int (pipeline expects 0/1, not True/False)
        cleaned = {
            k: int(v) if isinstance(v, bool) else v
            for k, v in input_data.items()
        }

        # Build a single-row DataFrame in the exact column order
        df = pd.DataFrame([cleaned])[ALL_FEATURES]

        probability = float(self.pipeline.predict_proba(df)[0][1])
        prediction  = int(probability >= 0.5)

        return {
            "prediction":    prediction,
            "probability":   round(probability, 4),
            "risk_label":    self._risk_label(probability),
            "model_version": self.model_version,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions on a DataFrame (used by batch_service).
        Appends three columns: churn_prediction, churn_probability, risk_label.
        """
        # Convert any bool columns to int
        bool_cols = [c for c in BINARY_FEATURES if c in df.columns]
        for col in bool_cols:
            df[col] = df[col].map(lambda x: int(x) if isinstance(x, bool) else x)

        feature_df    = df[ALL_FEATURES]
        probabilities = self.pipeline.predict_proba(feature_df)[:, 1]

        result = df.copy()
        result["churn_prediction"]  = (probabilities >= 0.5).astype(int)
        result["churn_probability"] = probabilities.round(4)
        result["risk_label"]        = [self._risk_label(p) for p in probabilities]
        return result

    @staticmethod
    def _risk_label(probability: float) -> str:
        """Translate raw probability into a business-readable risk tier."""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"


# ── Singleton — model loaded once when this module is first imported ──────────
ml_service = MLService()
