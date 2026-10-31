
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

FUTURE_FILE = ROOT / "data/features/holybro_future_remaining_time_30s.parquet"
CURRENT_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "future_remaining_time_s"
CURRENT_TARGET = "target_remaining_time_s"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load future dataset and recover current target
# --------------------------------------------------

print("Loading future forecast dataset...")

future = pd.read_parquet(FUTURE_FILE)
current = pd.read_parquet(CURRENT_FILE)

join_keys = [
    "flight_id",
    "window_start_s",
    "window_end_s"
]

current_values = current[
    join_keys + [CURRENT_TARGET]
].copy()

df = future.merge(
    current_values,
    on=join_keys,
    how="left",
    validate="one_to_one"
)

if df[CURRENT_TARGET].isna().any():
    raise RuntimeError(
        "Some current target values could not be matched."
    )

print("Forecast samples:", len(df))
print("Flights:", df["flight_id"].nunique())

# --------------------------------------------------
# 2. Separate train, validation and test
# --------------------------------------------------

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

print("\nSplit sizes:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

y_train = train[TARGET].astype(float)
y_val = val[TARGET].astype(float)
y_test = test[TARGET].astype(float)

# --------------------------------------------------
# 3. Create model inputs
# --------------------------------------------------

excluded_columns = {
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

candidate_features = [
    col for col in df.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

# Remove columns entirely missing or constant in training.
usable_features = []

for col in candidate_features:
    values = train[col].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(values) and values.nunique() > 1:
        usable_features.append(col)

if not usable_features:
    raise RuntimeError("No usable training features found.")

X_train = train[usable_features].replace(
    [np.inf, -np.inf], np.nan
)
X_val = val[usable_features].replace(
    [np.inf, -np.inf], np.nan
)
X_test = test[usable_features].replace(
    [np.inf, -np.inf], np.nan
)

print("\nCandidate numeric features:", len(candidate_features))
print("Usable features:", len(usable_features))

# --------------------------------------------------
# 4. Persistence baseline
# --------------------------------------------------

persistence_val = val[CURRENT_TARGET].to_numpy()
persistence_test = test[CURRENT_TARGET].to_numpy()

persistence_val_mae = mean_absolute_error(
    y_val, persistence_val
)

persistence_test_mae = mean_absolute_error(
    y_test, persistence_test
)

print("\nPERSISTENCE BASELINE")
print("Validation MAE:", round(persistence_val_mae, 3), "seconds")
print("Test MAE:", round(persistence_test_mae, 3), "seconds")

# --------------------------------------------------
# 5. Define candidate models
# --------------------------------------------------

models = {
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=42
    )
}

try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )

    print("\nXGBoost available.")

except ImportError:
    print("\nXGBoost unavailable; skipping.")

# --------------------------------------------------
# 6. Train and compare using validation MAE
# --------------------------------------------------

results = []
trained_models = {}

for name, estimator in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val)

    val_mae = mean_absolute_error(
        y_val, val_predictions
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_predictions)
    )

    val_r2 = r2_score(y_val, val_predictions)

    results.append({
        "model": name,
        "validation_mae_s": float(val_mae),
        "validation_rmse_s": float(val_rmse),
        "validation_r2": float(val_r2)
    })

    trained_models[name] = pipeline

    print("Validation MAE:", round(val_mae, 3), "seconds")
    print("Validation RMSE:", round(val_rmse, 3), "seconds")
    print("Validation R2:", round(val_r2, 5))

# --------------------------------------------------
# 7. Select best model on validation MAE
# --------------------------------------------------

comparison = pd.DataFrame(results).sort_values(
    "validation_mae_s"
)

comparison.to_csv(
    REPORT_DIR / "future_remaining_time_model_comparison.csv",
    index=False
)

best_name = comparison.iloc[0]["model"]
best_model = trained_models[best_name]

print("\nSelected model:", best_name)

# --------------------------------------------------
# 8. Evaluate once on test data
# --------------------------------------------------

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\n" + "=" * 55)
print("30-SECOND FORECAST TEST RESULTS")
print("=" * 55)

print("Selected model:", best_name)
print("Model test MAE:", round(test_mae, 3), "seconds")
print("Model test RMSE:", round(test_rmse, 3), "seconds")
print("Model test R2:", round(test_r2, 5))

print("\nPersistence test MAE:",
      round(persistence_test_mae, 3), "seconds")

if test_mae < persistence_test_mae:
    print("Model beats the persistence baseline.")
else:
    print("Model does not beat the persistence baseline.")

# --------------------------------------------------
# 9. Save model and test predictions
# --------------------------------------------------

model_path = MODEL_DIR / "holybro_future_remaining_time_30s.joblib"

joblib.dump(best_model, model_path)

predictions = test[
    join_keys + [TARGET, CURRENT_TARGET]
].copy()

predictions["model_prediction_s"] = test_predictions
predictions["persistence_prediction_s"] = persistence_test

predictions["model_absolute_error_s"] = np.abs(
    predictions[TARGET] - predictions["model_prediction_s"]
)

predictions["persistence_absolute_error_s"] = np.abs(
    predictions[TARGET] - predictions["persistence_prediction_s"]
)

predictions.to_csv(
    REPORT_DIR / "future_remaining_time_30s_test_predictions.csv",
    index=False
)

# --------------------------------------------------
# 10. Save final metrics
# --------------------------------------------------

metrics = {
    "task": "30-second future battery remaining-time estimation",
    "forecast_horizon_seconds": 30,
    "actual_forecast_gap_seconds": float(
        df["actual_forecast_gap_s"].median()
    ),
    "selected_model": best_name,
    "features_used": len(usable_features),
    "train_windows": len(train),
    "validation_windows": len(val),
    "test_windows": len(test),
    "test_flights": int(test["flight_id"].nunique()),
    "model_test_mae_seconds": float(test_mae),
    "model_test_rmse_seconds": float(test_rmse),
    "model_test_r2": float(test_r2),
    "persistence_test_mae_seconds": float(persistence_test_mae),
    "beats_persistence": bool(test_mae < persistence_test_mae),
    "target_interpretation": (
        "Future battery-management remaining-time estimate, "
        "not actual time until flight termination."
    )
}

with open(
    REPORT_DIR / "future_remaining_time_30s_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nSaved model:", model_path)
print("Saved comparison and test predictions.")
print("Saved metrics.")
print("\nFORECAST TRAINING COMPLETE")
