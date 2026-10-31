

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

FUTURE_FILE = ROOT / "data/features/holybro_future_remaining_time_30s.parquet"
CURRENT_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
MODEL_FILE = ROOT / "models/holybro_future_remaining_time_30s.joblib"

SUMMARY_FILE = ROOT / "reports/forecast_generalization_summary.csv"
FLIGHT_FILE = ROOT / "reports/forecast_generalization_per_flight.csv"
PREDICTION_FILE = ROOT / "reports/forecast_generalization_predictions.csv"

TARGET = "future_remaining_time_s"
CURRENT_TARGET = "target_remaining_time_s"

KEYS = ["flight_id", "window_start_s", "window_end_s"]

EXCLUDED = {
    TARGET,
    CURRENT_TARGET,
    "hardware_config",
    "flight_id",
    "split",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "target_observations",
    "actual_forecast_gap_s"
}


def metrics(actual, predicted):
    return {
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


print("Loading datasets and model...")

future = pd.read_parquet(FUTURE_FILE)
current = pd.read_parquet(CURRENT_FILE)
model = joblib.load(MODEL_FILE)

current = current[KEYS + [CURRENT_TARGET]].copy()

df = future.merge(
    current,
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

if len(df) != len(future):
    raise RuntimeError("Some future dataset rows did not match current targets.")

# Recreate the model feature list.
train = df[df["split"] == "train"]

features = [
    col for col in df.columns
    if col not in EXCLUDED
    and pd.api.types.is_numeric_dtype(df[col])
]

features = [
    col for col in features
    if train[col].notna().any()
    and train[col].nunique(dropna=True) > 1
]

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if set(model_features) != set(features):
        raise RuntimeError("Model and dataset feature lists do not match.")
    features = model_features

print("Model features:", len(features))
print("Total samples:", len(df))

# Predict every split using the same saved model.
X = df[features].replace([np.inf, -np.inf], np.nan)
df["model_prediction_s"] = model.predict(X).clip(min=0)
df["persistence_prediction_s"] = df[CURRENT_TARGET]

df["model_absolute_error_s"] = (
    df[TARGET] - df["model_prediction_s"]
).abs()

df["persistence_absolute_error_s"] = (
    df[TARGET] - df["persistence_prediction_s"]
).abs()

df.to_csv(PREDICTION_FILE, index=False)

# --------------------------------------------------
# Overall results by split
# --------------------------------------------------

summary_rows = []

for split_name in ["train", "val", "test"]:

    part = df[df["split"] == split_name]

    if part.empty:
        continue

    model_metrics = metrics(
        part[TARGET],
        part["model_prediction_s"]
    )

    baseline_metrics = metrics(
        part[TARGET],
        part["persistence_prediction_s"]
    )

    improvement = (
        (baseline_metrics["MAE_seconds"] -
         model_metrics["MAE_seconds"])
        / baseline_metrics["MAE_seconds"]
        * 100
    )

    summary_rows.append({
        "split": split_name,
        "windows": len(part),
        "flights": part["flight_id"].nunique(),
        "model_MAE_s": model_metrics["MAE_seconds"],
        "model_RMSE_s": model_metrics["RMSE_seconds"],
        "model_R2": model_metrics["R2"],
        "persistence_MAE_s": baseline_metrics["MAE_seconds"],
        "persistence_RMSE_s": baseline_metrics["RMSE_seconds"],
        "MAE_improvement_pct": improvement
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_FILE, index=False)

# --------------------------------------------------
# Per-flight performance
# --------------------------------------------------

flight_rows = []

for (split_name, flight_id), part in df.groupby(
    ["split", "flight_id"]
):

    model_mae = mean_absolute_error(
        part[TARGET],
        part["model_prediction_s"]
    )

    baseline_mae = mean_absolute_error(
        part[TARGET],
        part["persistence_prediction_s"]
    )

    improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae > 0 else np.nan
    )

    flight_rows.append({
        "split": split_name,
        "flight_id": flight_id,
        "windows": len(part),
        "target_median_s": part[TARGET].median(),
        "model_MAE_s": model_mae,
        "persistence_MAE_s": baseline_mae,
        "improvement_pct": improvement,
        "model_max_error_s": (
            part["model_absolute_error_s"].max()
        )
    })

flight_report = pd.DataFrame(flight_rows)
flight_report = flight_report.sort_values(
    ["split", "model_MAE_s"],
    ascending=[True, False]
)

flight_report.to_csv(FLIGHT_FILE, index=False)

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 65)
print("FORECAST GENERALIZATION SUMMARY")
print("=" * 65)

print(summary.round(3).to_string(index=False))

for split_name in ["train", "val", "test"]:

    part = flight_report[
        flight_report["split"] == split_name
    ]

    if part.empty:
        continue

    beats = (part["improvement_pct"] > 0).sum()

    print(f"\n{split_name.upper()} FLIGHT PERFORMANCE")
    print("-" * 65)
    print("Flights:", len(part))
    print("Flights beating persistence:", beats, "/", len(part))
    print("Worst flight MAE:", round(part["model_MAE_s"].max(), 3))
    print("Median flight MAE:", round(part["model_MAE_s"].median(), 3))

    print("\nWorst 5 flights:")
    print(
        part.head(5)[
            [
                "flight_id",
                "windows",
                "model_MAE_s",
                "persistence_MAE_s",
                "improvement_pct"
            ]
        ].round(3).to_string(index=False)
    )

print("\nSaved reports:")
print(SUMMARY_FILE)
print(FLIGHT_FILE)
print(PREDICTION_FILE)
