
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "target_remaining_time_s"
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading remaining-time dataset...")

df = pd.read_parquet(DATA_FILE)

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

print("\nSplit sizes:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# --------------------------------------------------
# 2. Prepare model input columns
# --------------------------------------------------

excluded_columns = {
    TARGET,
    "hardware_config",
    "flight_id",
    "split",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "target_observations"
}

leakage_tokens = [
    "time_remaining",
    "remaining_time"
]

candidate_columns = [
    col for col in df.columns
    if col not in excluded_columns
    and not any(token in col.lower() for token in leakage_tokens)
]

# Keep numeric features only
candidate_columns = [
    col for col in candidate_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

X_train = train[candidate_columns].replace(
    [np.inf, -np.inf], np.nan
)

X_val = val[candidate_columns].replace(
    [np.inf, -np.inf], np.nan
)

X_test = test[candidate_columns].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train[TARGET].astype(float)
y_val = val[TARGET].astype(float)
y_test = test[TARGET].astype(float)

# Remove columns that are entirely missing or constant in training.
usable_columns = []

for col in candidate_columns:
    values = X_train[col].dropna()

    if len(values) > 0 and values.nunique() > 1:
        usable_columns.append(col)

X_train = X_train[usable_columns]
X_val = X_val[usable_columns]
X_test = X_test[usable_columns]

print("\nNumeric candidate features:", len(candidate_columns))
print("Usable training features:", len(usable_columns))

if not usable_columns:
    raise RuntimeError("No usable numeric training features found.")

# --------------------------------------------------
# 3. Define regression models
# --------------------------------------------------

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
}

# Add XGBoost if installed.
try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=4
    )

    print("\nXGBoost is available.")

except ImportError:
    print("\nXGBoost is not installed. Skipping it.")

# --------------------------------------------------
# 4. Baseline: predict training median
# --------------------------------------------------

baseline_prediction = np.full(
    len(y_val), y_train.median()
)

baseline_mae = mean_absolute_error(
    y_val, baseline_prediction
)

print("\nTraining median baseline validation MAE:",
      round(baseline_mae, 2), "seconds")

# --------------------------------------------------
# 5. Train and validate models
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

    mae = mean_absolute_error(y_val, val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    r2 = r2_score(y_val, val_predictions)

    results.append({
        "model": name,
        "validation_mae_seconds": float(mae),
        "validation_rmse_seconds": float(rmse),
        "validation_r2": float(r2)
    })

    trained_models[name] = pipeline

    print("Validation MAE:", round(mae, 2), "seconds")
    print("Validation RMSE:", round(rmse, 2), "seconds")
    print("Validation R2:", round(r2, 4))

# --------------------------------------------------
# 6. Select best model using validation MAE
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "validation_mae_seconds"
)

results_df.to_csv(
    REPORT_DIR / "remaining_time_validation_comparison.csv",
    index=False
)

best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

print("\n" + "=" * 55)
print("BEST MODEL:", best_name)
print("=" * 55)

# --------------------------------------------------
# 7. Final test evaluation
# --------------------------------------------------

print("\nEvaluating selected model on untouched test set...")

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

baseline_test_predictions = np.full(
    len(y_test), y_train.median()
)

baseline_test_mae = mean_absolute_error(
    y_test, baseline_test_predictions
)

print("\nTEST RESULTS")
print("Model:", best_name)
print("MAE:", round(test_mae, 2), "seconds")
print("MAE:", round(test_mae / 60, 2), "minutes")
print("RMSE:", round(test_rmse, 2), "seconds")
print("R2:", round(test_r2, 4))
print("Median baseline MAE:", round(baseline_test_mae, 2), "seconds")

# --------------------------------------------------
# 8. Save model and predictions
# --------------------------------------------------

model_path = MODEL_DIR / "holybro_remaining_time_model.joblib"

joblib.dump(best_model, model_path)

predictions = test[
    [
        "flight_id",
        "window_start_s",
        "window_end_s",
        TARGET
    ]
].copy()

predictions["predicted_remaining_time_s"] = test_predictions
predictions["absolute_error_s"] = np.abs(
    predictions[TARGET] - predictions["predicted_remaining_time_s"]
)

predictions.to_csv(
    REPORT_DIR / "remaining_time_test_predictions.csv",
    index=False
)

# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

fitted_estimator = best_model.named_steps["model"]

if hasattr(fitted_estimator, "feature_importances_"):

    importance = pd.DataFrame({
        "feature": usable_columns,
        "importance": fitted_estimator.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "remaining_time_feature_importance.csv",
        index=False
    )

# --------------------------------------------------
# 10. Save final metrics
# --------------------------------------------------

metrics = {
    "task": "Battery management remaining-time estimation",
    "selected_model": best_name,
    "training_windows": len(train),
    "validation_windows": len(val),
    "test_windows": len(test),
    "training_flights": int(train["flight_id"].nunique()),
    "validation_flights": int(val["flight_id"].nunique()),
    "test_flights": int(test["flight_id"].nunique()),
    "features_used": len(usable_columns),
    "validation_mae_seconds": float(
        results_df.iloc[0]["validation_mae_seconds"]
    ),
    "test_mae_seconds": float(test_mae),
    "test_mae_minutes": float(test_mae / 60),
    "test_rmse_seconds": float(test_rmse),
    "test_r2": float(test_r2),
    "test_median_baseline_mae_seconds": float(baseline_test_mae),
    "interpretation": (
        "Predicts the recorded battery-management remaining-time "
        "estimate, not actual time until flight termination."
    )
}

with open(
    REPORT_DIR / "remaining_time_model_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nSaved model:", model_path)
print("Saved validation comparison.")
print("Saved test predictions.")
print("Saved final metrics.")

print("\nTRAINING COMPLETE")
