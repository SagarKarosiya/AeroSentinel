
from pathlib import Path

import pandas as pd

from inference_engine import AeroSentinelInference


ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "data" / "features"


def test_anomaly_inference(engine, hardware):

    print("\n" + "=" * 65)
    print(f"TESTING {hardware.upper()} ANOMALY INFERENCE")
    print("=" * 65)

    path = FEATURE_DIR / f"{hardware}_window_features.parquet"

    df = pd.read_parquet(path)

    test_df = df[df["split"] == "test"].copy()

    bundle = engine.anomaly_models[hardware]
    feature_columns = bundle["feature_columns"]

    results = []

    # Test five actual windows.
    for index, (_, row) in enumerate(test_df.head(5).iterrows(), 1):

        features = row[feature_columns].to_dict()

        prediction = engine.detect_anomaly(
            hardware,
            features
        )

        results.append(prediction)

        print(f"\nWindow {index}")
        print("Anomaly score:", prediction["anomaly_score"])
        print("Threshold:", prediction["threshold"])
        print("Anomaly:", prediction["is_anomaly"])

    print("\nAnomaly inference test completed.")


def test_forecast_inference(engine):

    print("\n" + "=" * 65)
    print("TESTING HOLYBRO FUTURE REMAINING-TIME FORECAST")
    print("=" * 65)

    path = FEATURE_DIR / "holybro_future_remaining_time_30s.parquet"

    df = pd.read_parquet(path)

    test_df = df[df["split"] == "test"].copy()

    features = test_df.iloc[0][engine.forecast_features].to_dict()

    prediction = engine.forecast_remaining_time(features)

    print("\nForecast result:")
    print(prediction)

    print("\nForecast inference test completed.")


def main():

    engine = AeroSentinelInference()

    test_anomaly_inference(engine, "holybro")
    test_anomaly_inference(engine, "speedybee")

    test_forecast_inference(engine)

    print("\n" + "=" * 65)
    print("ALL INFERENCE TESTS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
