
from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PREDICTION_DIR = ROOT / "data" / "predictions"

for folder in [MODEL_DIR, REPORT_DIR, PREDICTION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

HARDWARES = ["holybro", "speedybee"]

METADATA_COLUMNS = {
    "hardware_config",
    "flight_id",
    "split",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
}

MISSING_LIMIT = 0.20
VALIDATION_PERCENTILE = 95


def prepare_numeric_data(df, feature_columns):
    X = df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def train_hardware_model(hardware):

    print("\n" + "=" * 70)
    print(f"TRAINING {hardware.upper()} ANOMALY MODEL")
    print("=" * 70)

    input_path = FEATURE_DIR / f"{hardware}_window_features.parquet"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_parquet(input_path)

    print("Dataset shape:", df.shape)
    print("Flights:", df["flight_id"].nunique())

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Train windows:", len(train_df))
    print("Validation windows:", len(val_df))
    print("Test windows:", len(test_df))

    # Only statistical numeric features are model inputs.
    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    X_train_all = prepare_numeric_data(
        train_df, feature_columns
    )

    # Select features using training data only.
    missing_pct = X_train_all.isna().mean()

    selected_features = [
        col for col in X_train_all.columns
        if missing_pct[col] <= MISSING_LIMIT
        and X_train_all[col].nunique(dropna=True) > 1
    ]

    if len(selected_features) == 0:
        raise ValueError("No usable features found.")

    print("\nInitial numeric features:", len(feature_columns))
    print("Selected features:", len(selected_features))
    print("Removed features:", len(feature_columns) - len(selected_features))

    X_train = X_train_all[selected_features]

    X_val = prepare_numeric_data(
        val_df, selected_features
    )

    X_test = prepare_numeric_data(
        test_df, selected_features
    )

    # Imputation is fitted only on training data.
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Check finite values.
    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_val).all()
        and np.isfinite(X_test).all()
    ):
        raise ValueError("Non-finite values remain after imputation.")

    # Train only on training windows.
    model = IsolationForest(
        n_estimators=300,
        max_samples=min(256, len(X_train)),
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("\nFitting Isolation Forest...")
    model.fit(X_train)

    # Higher score means more anomalous.
    train_scores = -model.score_samples(X_train)
    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    # Provisional threshold selected using validation data.
    threshold = float(
        np.percentile(val_scores, VALIDATION_PERCENTILE)
    )

    train_flags = train_scores > threshold
    val_flags = val_scores > threshold
    test_flags = test_scores > threshold

    print("\nANOMALY SCORE SUMMARY")
    print("Validation threshold:", threshold)
    print("Train flagged:", int(train_flags.sum()), "/", len(train_flags))
    print("Validation flagged:", int(val_flags.sum()), "/", len(val_flags))
    print("Test flagged:", int(test_flags.sum()), "/", len(test_flags))

    # Save model and preprocessing together.
    artifact = {
        "hardware_config": hardware,
        "feature_columns": selected_features,
        "imputer": imputer,
        "model": model,
        "anomaly_threshold": threshold,
        "threshold_validation_percentile": VALIDATION_PERCENTILE,
    }

    model_path = MODEL_DIR / f"{hardware}_isolation_forest.joblib"
    joblib.dump(artifact, model_path)

    # Save predictions for all three splits.
    prediction_frames = []

    for split_name, split_df, scores, flags in [
        ("train", train_df, train_scores, train_flags),
        ("val", val_df, val_scores, val_flags),
        ("test", test_df, test_scores, test_flags),
    ]:

        result = split_df[
            [
                "hardware_config",
                "flight_id",
                "split",
                "window_start_s",
                "window_end_s",
            ]
        ].copy()

        result["anomaly_score"] = scores
        result["is_anomaly"] = flags.astype(int)

        prediction_frames.append(result)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    prediction_path = (
        PREDICTION_DIR / f"{hardware}_anomaly_predictions.csv"
    )

    predictions.to_csv(prediction_path, index=False)

    # Per-flight test anomaly rates.
    test_predictions = predictions[
        predictions["split"] == "test"
    ]

    per_flight = test_predictions.groupby("flight_id").agg(
        windows=("is_anomaly", "count"),
        flagged_windows=("is_anomaly", "sum"),
        mean_anomaly_score=("anomaly_score", "mean"),
        max_anomaly_score=("anomaly_score", "max"),
    ).reset_index()

    per_flight["anomaly_rate_pct"] = (
        per_flight["flagged_windows"]
        / per_flight["windows"] * 100
    )

    flight_report_path = (
        REPORT_DIR / f"{hardware}_test_flight_anomaly_report.csv"
    )

    per_flight.to_csv(flight_report_path, index=False)

    metrics = {
        "hardware_config": hardware,
        "training_windows": int(len(train_df)),
        "validation_windows": int(len(val_df)),
        "test_windows": int(len(test_df)),
        "initial_numeric_features": int(len(feature_columns)),
        "selected_features": int(len(selected_features)),
        "anomaly_threshold": threshold,
        "threshold_validation_percentile": VALIDATION_PERCENTILE,
        "train_flagged": int(train_flags.sum()),
        "validation_flagged": int(val_flags.sum()),
        "test_flagged": int(test_flags.sum()),
        "test_anomaly_rate_pct": float(test_flags.mean() * 100),
        "test_flights": int(test_df["flight_id"].nunique()),
    }

    metrics_path = REPORT_DIR / f"{hardware}_anomaly_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSAVED FILES")
    print(model_path)
    print(prediction_path)
    print(flight_report_path)
    print(metrics_path)

    print("\nTop 10 test flights by anomaly rate:")
    print(
        per_flight.sort_values(
            "anomaly_rate_pct", ascending=False
        ).head(10).to_string(index=False)
    )

    print("\nTraining completed for", hardware)

    del df, train_df, val_df, test_df
    del X_train, X_val, X_test
    del predictions, prediction_frames
    gc.collect()

    return metrics


def main():

    all_metrics = []

    for hardware in HARDWARES:
        metrics = train_hardware_model(hardware)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)

    summary_path = REPORT_DIR / "anomaly_model_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("AEROSENTINEL BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nSummary saved:", summary_path)


if __name__ == "__main__":
    main()
