
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/features/holybro_future_remaining_time_30s.parquet"
PRED_FILE = ROOT / "reports/future_remaining_time_30s_test_predictions.csv"
MODEL_FILE = ROOT / "models/holybro_future_remaining_time_30s.joblib"

ERROR_REPORT = ROOT / "reports/flight28_forecast_errors.csv"
FEATURE_REPORT = ROOT / "reports/flight28_feature_distribution.csv"

TARGET = "future_remaining_time_s"
CURRENT_TARGET = "target_remaining_time_s"
FLIGHT_ID = 28

print("Loading data and trained model...")

df = pd.read_parquet(DATA_FILE)
pred = pd.read_csv(PRED_FILE)
model = joblib.load(MODEL_FILE)

train = df[df["split"] == "train"].copy()
flight = df[
    (df["flight_id"] == FLIGHT_ID) &
    (df["split"] == "test")
].copy()

if flight.empty:
    raise RuntimeError("Flight 28 was not found in the test dataset.")

# --------------------------------------------------
# 1. Inspect flight 28 prediction errors over time
# --------------------------------------------------

keys = ["flight_id", "window_start_s", "window_end_s"]

flight_pred = pred[pred["flight_id"] == FLIGHT_ID].copy()

flight_pred["error_s"] = (
    flight_pred["model_prediction_s"] -
    flight_pred[TARGET]
)

flight_pred["absolute_error_s"] = flight_pred["error_s"].abs()

flight_pred = flight_pred.sort_values("window_start_s")

flight_pred.to_csv(ERROR_REPORT, index=False)

print("\n" + "=" * 60)
print("FLIGHT 28 FORECAST ERROR ANALYSIS")
print("=" * 60)

print("Windows:", len(flight_pred))
print("MAE:", round(flight_pred["absolute_error_s"].mean(), 3))
print("Maximum error:", round(flight_pred["absolute_error_s"].max(), 3))

print("\nFirst 15 windows:")
print(
    flight_pred[
        [
            "window_start_s",
            "window_end_s",
            TARGET,
            "model_prediction_s",
            "persistence_prediction_s",
            "absolute_error_s"
        ]
    ].head(15).round(3).to_string(index=False)
)

print("\nWorst 10 windows:")
print(
    flight_pred.nlargest(10, "absolute_error_s")[
        [
            "window_start_s",
            TARGET,
            "model_prediction_s",
            "persistence_prediction_s",
            "absolute_error_s"
        ]
    ].round(3).to_string(index=False)
)

# --------------------------------------------------
# 2. Recover the model's feature importance
# --------------------------------------------------

excluded = {
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

features = [
    col for col in df.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(df[col])
]

features = [
    col for col in features
    if train[col].notna().any()
    and train[col].nunique(dropna=True) > 1
]

estimator = model.named_steps["model"]

if not hasattr(estimator, "feature_importances_"):
    raise RuntimeError("The saved model has no feature_importances_.")

importance = pd.DataFrame({
    "feature": features,
    "importance": estimator.feature_importances_
}).sort_values("importance", ascending=False)

# --------------------------------------------------
# 3. Compare flight 28 feature values with training
# --------------------------------------------------

diagnostics = []

for _, row in importance.iterrows():

    col = row["feature"]

    train_values = pd.to_numeric(
        train[col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    flight_values = pd.to_numeric(
        flight[col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(train_values) == 0 or len(flight_values) == 0:
        continue

    low = train_values.quantile(0.01)
    high = train_values.quantile(0.99)

    outside_pct = (
        ((flight_values < low) | (flight_values > high)).mean()
        * 100
    )

    diagnostics.append({
        "feature": col,
        "importance": row["importance"],
        "train_median": train_values.median(),
        "flight28_median": flight_values.median(),
        "train_p01": low,
        "train_p99": high,
        "flight28_min": flight_values.min(),
        "flight28_max": flight_values.max(),
        "flight28_outside_train_p01_p99_pct": outside_pct
    })

diagnostics = pd.DataFrame(diagnostics)

diagnostics = diagnostics.sort_values(
    ["importance", "flight28_outside_train_p01_p99_pct"],
    ascending=[False, False]
)

diagnostics.to_csv(FEATURE_REPORT, index=False)

print("\n" + "=" * 60)
print("TOP 20 IMPORTANT FEATURES: FLIGHT 28 VS TRAINING")
print("=" * 60)

print(
    diagnostics.head(20).round(4).to_string(index=False)
)

print("\nTop 20 important features with greatest distribution shift:")

print(
    diagnostics.sort_values(
        "flight28_outside_train_p01_p99_pct",
        ascending=False
    ).head(20).round(4).to_string(index=False)
)

print("\nReports saved:")
print(ERROR_REPORT)
print(FEATURE_REPORT)
