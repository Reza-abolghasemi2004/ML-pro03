"""Predicting churn for a single customer using the saved model bundle."""

import logging
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .feature_engineering import add_features
from .preprocessing import clean_dataset

logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    pass


class PredictionService:
    """Wraps the joblib bundle saved at the end of training."""

    def __init__(self, model_path=None):
        self.path = Path(model_path) if model_path else config.MODEL_PATH
        self.bundle = None

    def is_available(self):
        return self.path.exists()

    def load(self):
        if not self.path.exists():
            raise ModelNotFoundError(
                f"No saved model found at {self.path}. Run training first."
            )
        self.bundle = joblib.load(self.path)
        logger.info("Loaded saved model '%s' from %s", self.bundle.get("model_name", "?"), self.path)
        return self.bundle

    def predict(self, record):
        """`record` is a dict with the raw dataset columns for one customer.

        The same cleaning + feature engineering + fitted pipeline used in
        training is applied, so predictions stay consistent.
        """
        if self.bundle is None:
            self.load()

        df = pd.DataFrame([record])
        missing = [col for col in config.REQUIRED_COLUMNS if col not in df.columns and col != config.TARGET_COLUMN]
        if missing:
            raise ValueError("Record is missing field(s): " + ", ".join(missing))

        df = clean_dataset(df)
        df = add_features(df)

        features = self.bundle["numeric_features"] + self.bundle["categorical_features"]
        pipeline = self.bundle["pipeline"]

        proba = pipeline.predict_proba(df[features])[0]
        positive_index = list(pipeline.classes_).index(config.POSITIVE_LABEL)
        churn_probability = float(proba[positive_index])

        will_churn = churn_probability >= 0.5
        if churn_probability >= 0.60:
            risk = "HIGH"
        elif churn_probability >= 0.35:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "churn_probability": churn_probability,
            "predicted_label": "Churn" if will_churn else "No churn",
            "risk": risk,
        }
