
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

REPORT_FILE = ROOT / "reports/remaining_time_fallback_comparison.csv"
PREDICTION_FILE = ROOT / "reports/remaining_time_fallback_test_predictions.csv"
METRICS_FILE = ROOT / "reports/remaining_time_fallback_metrics.json"

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


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


print("Loading datasets and model...")

future = pd.read_parquet(FUTURE_FILE)
current = pd.read_parquet(CURRENT_FILE)
model = joblib.load(MODEL_FILE)

# Attach the current BMS estimate for the persistence baseline.
current = current[KEYS + [CURRENT_TARGET]].copy()

df = future.merge(
    current,
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

if len(df) != len(future):
    raise RuntimeError(
        f"Merge mismatch: future={len(future)}, merged={len(df)}"
    )

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

if validation.empty or test.empty:
    raise RuntimeError("Validation or test dataset is empty.")

# Recreate the numeric feature list using training data only.
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

# Preserve the exact feature order used by the saved model, if available.
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if set(model_features) != set(features):
        raise RuntimeError(
            "Feature mismatch between dataset and saved model.\n"
            f"Model features: {len(model_features)}\n"
            f"Dataset features: {len(features)}"
        )
    features = model_features

print("Features:", len(features))
print("Train rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))

def predict_split(data):
    X = data[features].replace([np.inf, -np.inf], np.nan)

    prediction = model.predict(X)

    result = data[
        KEYS + [TARGET, CURRENT_TARGET]
    ].copy()

    result["model_prediction_s"] = prediction
    result["persistence_prediction_s"] = result[CURRENT_TARGET]

    result["model_prediction_s"] = result[
        "model_prediction_s"
    ].clip(lower=0)

    return result


print("\nGenerating validation and test predictions...")

val_pred = predict_split(validation)
test_pred = predict_split(test)

# --------------------------------------------------
# Choose fallback threshold using validation only.
# If ML and persistence disagree by more than the
# threshold, use persistence instead of ML.
# --------------------------------------------------

val_difference = (
    val_pred["model_prediction_s"] -
    val_pred["persistence_prediction_s"]
).abs().to_numpy()

val_actual = val_pred[TARGET].to_numpy()
val_model = val_pred["model_prediction_s"].to_numpy()
val_persistence = val_pred["persistence_prediction_s"].to_numpy()

candidates = np.unique(
    np.concatenate(([0.0], val_difference, [np.inf]))
)

best_threshold = None
best_mae = float("inf")
best_fallback_count = None

for threshold in candidates:

    use_persistence = val_difference > threshold

    hybrid = np.where(
        use_persistence,
        val_persistence,
        val_model
    )

    mae = mean_absolute_error(val_actual, hybrid)
    fallback_count = int(use_persistence.sum())

    # Prefer fewer fallbacks if validation MAE is tied.
    if (
        mae < best_mae - 1e-9
        or (
            abs(mae - best_mae) <= 1e-9
            and (
                best_fallback_count is None
                or fallback_count < best_fallback_count
            )
        )
    ):
        best_mae = float(mae)
        best_threshold = float(threshold)
        best_fallback_count = fallback_count

print("\n" + "=" * 65)
print("VALIDATION-ONLY FALLBACK THRESHOLD")
print("=" * 65)

print("Selected disagreement threshold:", best_threshold)
print("Validation hybrid MAE:", round(best_mae, 3), "seconds")
print("Validation fallback windows:", best_fallback_count)

# --------------------------------------------------
# Evaluate the frozen threshold on test data.
# --------------------------------------------------

test_difference = (
    test_pred["model_prediction_s"] -
    test_pred["persistence_prediction_s"]
).abs()

test_pred["fallback_used"] = (
    test_difference > best_threshold
)

test_pred["hybrid_prediction_s"] = np.where(
    test_pred["fallback_used"],
    test_pred["persistence_prediction_s"],
    test_pred["model_prediction_s"]
)

actual = test_pred[TARGET]

model_metrics = calculate_metrics(
    actual,
    test_pred["model_prediction_s"]
)

persistence_metrics = calculate_metrics(
    actual,
    test_pred["persistence_prediction_s"]
)

hybrid_metrics = calculate_metrics(
    actual,
    test_pred["hybrid_prediction_s"]
)

comparison = pd.DataFrame([
    {"method": "ML model", **model_metrics},
    {"method": "Persistence", **persistence_metrics},
    {"method": "Hybrid fallback", **hybrid_metrics}
])

comparison.to_csv(REPORT_FILE, index=False)
test_pred.to_csv(PREDICTION_FILE, index=False)

metrics = {
    "selected_threshold_seconds": best_threshold,
    "validation_hybrid_mae_seconds": best_mae,
    "validation_fallback_windows": best_fallback_count,
    "test_windows": len(test_pred),
    "test_fallback_windows": int(test_pred["fallback_used"].sum()),
    "test_fallback_percentage": float(
        test_pred["fallback_used"].mean() * 100
    ),
    "test_model": model_metrics,
    "test_persistence": persistence_metrics,
    "test_hybrid": hybrid_metrics
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 65)
print("FINAL TEST COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string(index=False))

print("\nFallback windows:", int(test_pred["fallback_used"].sum()))
print("Total test windows:", len(test_pred))

print("\nFlight 28 comparison:")

flight28 = test_pred[test_pred["flight_id"] == 28].copy()

if not flight28.empty:
    print("Windows:", len(flight28))
    print(
        "ML MAE:",
        round(mean_absolute_error(
            flight28[TARGET],
            flight28["model_prediction_s"]
        ), 3)
    )
    print(
        "Persistence MAE:",
        round(mean_absolute_error(
            flight28[TARGET],
            flight28["persistence_prediction_s"]
        ), 3)
    )
    print(
        "Hybrid MAE:",
        round(mean_absolute_error(
            flight28[TARGET],
            flight28["hybrid_prediction_s"]
        ), 3)
    )
    print(
        "Fallback used:",
        int(flight28["fallback_used"].sum()),
        "/",
        len(flight28)
    )

print("\nSaved reports:")
print(REPORT_FILE)
print(PREDICTION_FILE)
print(METRICS_FILE)
