
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_flight

ROOT = Path(__file__).resolve().parents[1]

HARDWARE = "holybro"
FLIGHT_ID = 1

# Load trained model
import joblib

model_path = ROOT / "models" / "holybro_isolation_forest.joblib"
artifact = joblib.load(model_path)

model = artifact["model"]
imputer = artifact["imputer"]
feature_columns = artifact["feature_columns"]

# Load engineered windows
feature_path = ROOT / "data" / "features" / "holybro_window_features.parquet"
features = pd.read_parquet(feature_path)

# Select flight windows
flight_data = features[
    (features["flight_id"] == FLIGHT_ID) &
    (features["split"] == "test")
].copy()

# Load anomaly predictions
prediction_path = ROOT / "data" / "predictions" / "holybro_anomaly_predictions.csv"
predictions = pd.read_csv(prediction_path)

flight_predictions = predictions[
    predictions["flight_id"] == FLIGHT_ID
]

# Merge predictions with feature windows
data = flight_data.merge(
    flight_predictions[
        ["flight_id", "window_start_s", "anomaly_score", "is_anomaly"]
    ],
    on=["flight_id", "window_start_s"],
    how="inner"
)

# Load raw flight data
raw_df, metadata = load_flight(HARDWARE, FLIGHT_ID)

print("\nFLIGHT INFORMATION")
print("Flight ID:", FLIGHT_ID)
print("Raw rows:", len(raw_df))
print("Engineered windows:", len(data))
print("Flagged windows:", int(data["is_anomaly"].sum()))

# Calculate feature deviation from training median/IQR
train_data = features[features["split"] == "train"]

diagnostics = []

for feature in feature_columns:
    if feature not in data.columns or feature not in train_data.columns:
        continue

    train_values = train_data[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(train_values) < 10:
        continue

    median = train_values.median()
    q1 = train_values.quantile(0.25)
    q3 = train_values.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        continue

    flagged_values = data.loc[
        data["is_anomaly"] == 1, feature
    ].replace([np.inf, -np.inf], np.nan).dropna()

    normal_values = data.loc[
        data["is_anomaly"] == 0, feature
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if len(flagged_values) == 0:
        continue

    flagged_deviation = abs(flagged_values.median() - median) / iqr

    normal_deviation = (
        abs(normal_values.median() - median) / iqr
        if len(normal_values) else np.nan
    )

    diagnostics.append({
        "feature": feature,
        "training_median": median,
        "flagged_median": flagged_values.median(),
        "normal_median": normal_values.median(),
        "flagged_deviation_iqr": flagged_deviation,
        "normal_deviation_iqr": normal_deviation,
        "flagged_missing_pct": data.loc[
            data["is_anomaly"] == 1, feature
        ].isna().mean() * 100
    })

report = pd.DataFrame(diagnostics)

report = report.sort_values(
    "flagged_deviation_iqr",
    ascending=False
)

output_path = ROOT / "reports" / "holybro_flight1_feature_diagnostics.csv"
report.to_csv(output_path, index=False)

print("\nTOP 25 UNUSUAL FEATURES")
print(report.head(25).to_string(index=False))

print("\nReport saved:", output_path)
