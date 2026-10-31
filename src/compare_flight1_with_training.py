
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

features = pd.read_parquet(
    ROOT / "data/features/holybro_window_features.parquet"
)

diagnostics = pd.read_csv(
    ROOT / "reports/holybro_flight1_feature_diagnostics.csv"
)

predictions = pd.read_csv(
    ROOT / "data/predictions/holybro_anomaly_predictions.csv"
)

flight1 = features[
    (features["flight_id"] == 1) &
    (features["split"] == "test")
].copy()

flight1 = flight1.merge(
    predictions[["flight_id", "window_start_s", "anomaly_score", "is_anomaly"]],
    on=["flight_id", "window_start_s"],
    how="left"
)

train = features[features["split"] == "train"]

# Select top 12 diagnostic signals
top_features = diagnostics.head(12)["feature"].tolist()

results = []

for feature in top_features:
    if feature not in train.columns or feature not in flight1.columns:
        continue

    train_values = pd.to_numeric(
        train[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    test_values = pd.to_numeric(
        flight1[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(train_values) == 0 or len(test_values) == 0:
        continue

    q1 = train_values.quantile(0.25)
    q3 = train_values.quantile(0.75)

    results.append({
        "feature": feature,
        "training_median": train_values.median(),
        "training_q1": q1,
        "training_q3": q3,
        "flight1_median": test_values.median(),
        "flight1_min": test_values.min(),
        "flight1_max": test_values.max(),
        "flight1_outside_training_iqr_pct": (
            ((test_values < q1) | (test_values > q3)).mean() * 100
        )
    })

report = pd.DataFrame(results)

output = ROOT / "reports/holybro_flight1_vs_training.csv"
report.to_csv(output, index=False)

print("\nFLIGHT 1 VS TRAINING DISTRIBUTION")
print(report.to_string(index=False))

print("\nReport saved:", output)

print("\nFLIGHT 1 ANOMALY WINDOWS BY TIME")
print(
    flight1[
        ["window_start_s", "anomaly_score", "is_anomaly"]
    ].to_string(index=False)
)
