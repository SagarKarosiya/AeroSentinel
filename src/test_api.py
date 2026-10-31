
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8000"

FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean_features(features):

    cleaned = {}

    for key, value in features.items():

        if pd.isna(value) or not np.isfinite(float(value)):
            cleaned[key] = None
        else:
            cleaned[key] = float(value)

    return cleaned


def send_request(endpoint, payload):

    try:

        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=60
        )

        print("HTTP status:", response.status_code)

        try:
            print("Response:", response.json())
        except ValueError:
            print("Response:", response.text)

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        print("\nERROR: Cannot connect to AeroSentinel API.")
        print("Start the server using:")
        print("uvicorn src.api:app --reload")

    except requests.exceptions.RequestException as error:

        print("\nAPI request failed:", error)

    return None


# --------------------------------------------------
# HOLYBRO ANOMALY DETECTION
# --------------------------------------------------

def test_holybro_anomaly():

    print("\n" + "=" * 65)
    print("TESTING HOLYBRO ANOMALY API")
    print("=" * 65)

    path = FEATURE_DIR / "holybro_window_features.parquet"

    df = pd.read_parquet(path)

    test_df = df[df["split"] == "test"]

    bundle = joblib.load(
        MODEL_DIR / "holybro_isolation_forest.joblib"
    )

    feature_columns = bundle["feature_columns"]

    row = test_df.iloc[0]

    features = clean_features(
        row[feature_columns].to_dict()
    )

    payload = {
        "hardware_config": "holybro",
        "features": features
    }

    return send_request(
        "/predict/anomaly",
        payload
    )


# --------------------------------------------------
# SPEEDYBEE ANOMALY DETECTION
# --------------------------------------------------

def test_speedybee_anomaly():

    print("\n" + "=" * 65)
    print("TESTING SPEEDYBEE ANOMALY API")
    print("=" * 65)

    path = FEATURE_DIR / "speedybee_window_features.parquet"

    df = pd.read_parquet(path)

    test_df = df[df["split"] == "test"]

    bundle = joblib.load(
        MODEL_DIR / "speedybee_isolation_forest.joblib"
    )

    feature_columns = bundle["feature_columns"]

    row = test_df.iloc[0]

    features = clean_features(
        row[feature_columns].to_dict()
    )

    payload = {
        "hardware_config": "speedybee",
        "features": features
    }

    return send_request(
        "/predict/anomaly",
        payload
    )


# --------------------------------------------------
# HOLYBRO REMAINING-TIME FORECAST
# --------------------------------------------------

def test_remaining_time():

    print("\n" + "=" * 65)
    print("TESTING HOLYBRO REMAINING-TIME API")
    print("=" * 65)

    model_path = (
        MODEL_DIR / "holybro_future_remaining_time_30s.joblib"
    )

    model = joblib.load(model_path)

    feature_columns = list(model.feature_names_in_)

    dataset_path = (
        FEATURE_DIR / "holybro_future_remaining_time_30s.parquet"
    )

    df = pd.read_parquet(dataset_path)

    test_df = df[df["split"] == "test"]

    row = test_df.iloc[0]

    features = clean_features(
        row[feature_columns].to_dict()
    )

    payload = {
        "hardware_config": "holybro",
        "features": features
    }

    return send_request(
        "/predict/remaining-time",
        payload
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("AEROSENTINEL API TEST SUITE")
    print("=" * 65)

    # Check API availability first.
    try:

        health = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        health.raise_for_status()

        print("\nAPI Health:", health.json())

    except requests.exceptions.RequestException:

        print("\nERROR: AeroSentinel API is not running.")
        print("Start it in another terminal:")
        print("uvicorn src.api:app --reload")
        return

    results = {}

    results["Holybro anomaly"] = test_holybro_anomaly()
    results["SpeedyBee anomaly"] = test_speedybee_anomaly()
    results["Holybro remaining-time"] = test_remaining_time()

    print("\n" + "=" * 65)
    print("AEROSENTINEL API TEST SUMMARY")
    print("=" * 65)

    for name, result in results.items():

        if result is not None:
            print(f"{name}: SUCCESS")
        else:
            print(f"{name}: FAILED")

    successful = sum(
        result is not None
        for result in results.values()
    )

    print(f"\nSuccessful tests: {successful}/3")


if __name__ == "__main__":
    main()
