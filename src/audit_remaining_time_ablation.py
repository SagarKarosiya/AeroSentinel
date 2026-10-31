
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
REPORT_FILE = ROOT / "reports/remaining_time_ablation_results.csv"

TARGET = "target_remaining_time_s"

df = pd.read_parquet(DATA_FILE)

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

excluded = {
    TARGET, "hardware_config", "flight_id", "split",
    "window_start_s", "window_end_s",
    "window_duration_s", "target_observations"
}

features = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in excluded
    and "time_remaining" not in c.lower()
    and "remaining_time" not in c.lower()
]

# Remove features missing or constant in training.
features = [
    c for c in features
    if train[c].notna().any() and train[c].nunique(dropna=True) > 1
]

# Four feature sets.
feature_sets = {
    "full_model": features,

    "without_remaining": [
        c for c in features
        if "remaining__battery_status_0" not in c
    ],

    "without_remaining_and_discharged": [
        c for c in features
        if "remaining__battery_status_0" not in c
        and "discharged_mah__battery_status_0" not in c
    ],

    "without_all_battery_features": [
        c for c in features
        if "battery_status" not in c.lower()
    ]
}

y_train = train[TARGET].astype(float)
y_val = val[TARGET].astype(float)
y_test = test[TARGET].astype(float)

results = []

for name, cols in feature_sets.items():

    print("\n" + "=" * 60)
    print(name)
    print("Features:", len(cols))
    print("=" * 60)

    if not cols:
        print("No usable features. Skipping.")
        continue

    X_train = train[cols].replace([np.inf, -np.inf], np.nan)
    X_val = val[cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[cols].replace([np.inf, -np.inf], np.nan)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    results.append({
        "feature_set": name,
        "feature_count": len(cols),
        "validation_mae_s": val_mae,
        "test_mae_s": test_mae,
        "test_rmse_s": test_rmse,
        "test_r2": test_r2
    })

    print("Validation MAE:", round(val_mae, 3), "seconds")
    print("Test MAE:", round(test_mae, 3), "seconds")
    print("Test RMSE:", round(test_rmse, 3), "seconds")
    print("Test R2:", round(test_r2, 5))

results_df = pd.DataFrame(results).sort_values(
    "validation_mae_s"
)

results_df.to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 60)
print("FEATURE ABLATION COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

print("\nSaved:", REPORT_FILE)
