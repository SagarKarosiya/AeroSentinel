
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


class AeroSentinelInference:

    def __init__(self):

        print("Loading AeroSentinel models...")

        self.anomaly_models = {}
        self.forecast_model = None

        self.load_anomaly_models()
        self.load_forecast_model()

        print("\nAll models loaded successfully.")

    # --------------------------------------------------
    # LOAD ANOMALY MODELS
    # --------------------------------------------------

    def load_anomaly_models(self):

        for hardware in ["holybro", "speedybee"]:

            model_path = (
                MODEL_DIR / f"{hardware}_isolation_forest.joblib"
            )

            if not model_path.exists():
                raise FileNotFoundError(model_path)

            bundle = joblib.load(model_path)

            self.anomaly_models[hardware] = bundle

            print(f"Loaded {hardware} anomaly model")

    # --------------------------------------------------
    # LOAD REMAINING-TIME FORECAST MODEL
    # --------------------------------------------------

    def load_forecast_model(self):

        model_path = (
            MODEL_DIR / "holybro_future_remaining_time_30s.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.forecast_model = joblib.load(model_path)

        self.forecast_features = list(
            self.forecast_model.feature_names_in_
        )

        print("Loaded Holybro future remaining-time model")
        print("Forecast features:", len(self.forecast_features))

    # --------------------------------------------------
    # PREPARE INPUT FEATURES
    # --------------------------------------------------

    @staticmethod
    def prepare_features(features, feature_columns):

        input_df = pd.DataFrame([features])

        # Match the exact feature order used during training.
        input_df = input_df.reindex(
            columns=feature_columns
        )

        # Handle invalid numeric values.
        input_df = input_df.replace(
            [np.inf, -np.inf], np.nan
        )

        for col in input_df.columns:

            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

        return input_df

    # --------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------

    def detect_anomaly(self, hardware, features):

        hardware = hardware.lower()

        if hardware not in self.anomaly_models:
            raise ValueError(
                f"Unsupported hardware configuration: {hardware}"
            )

        bundle = self.anomaly_models[hardware]

        feature_columns = bundle["feature_columns"]
        imputer = bundle["imputer"]
        model = bundle["model"]
        threshold = float(bundle["anomaly_threshold"])

        input_df = self.prepare_features(
            features,
            feature_columns
        )

        # Same preprocessing used during training.
        X = imputer.transform(input_df)

        # Match training:
        # Higher score = more anomalous.
        anomaly_score = float(
            -model.score_samples(X)[0]
        )

        is_anomaly = anomaly_score > threshold

        return {
            "hardware_config": hardware,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "is_anomaly": bool(is_anomaly),
        }

    # --------------------------------------------------
    # FUTURE REMAINING-TIME FORECAST
    # --------------------------------------------------

    def forecast_remaining_time(self, features):

        input_df = self.prepare_features(
            features,
            self.forecast_features
        )

        predicted_time = float(
            self.forecast_model.predict(input_df)[0]
        )

        return {
            "hardware_config": "holybro",
            "forecast_horizon_seconds": 30,
            "predicted_bms_remaining_time_seconds": predicted_time,
        }

    # --------------------------------------------------
    # COMBINED INFERENCE
    # --------------------------------------------------

    def predict(self, hardware, features):

        hardware = hardware.lower()

        result = {
            "anomaly_detection": self.detect_anomaly(
                hardware,
                features
            )
        }

        # Remaining-time model is currently Holybro-only.
        if hardware == "holybro":

            result["remaining_time_forecast"] = (
                self.forecast_remaining_time(features)
            )

        return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    engine = AeroSentinelInference()

    print("\nAeroSentinel inference engine is ready.")
