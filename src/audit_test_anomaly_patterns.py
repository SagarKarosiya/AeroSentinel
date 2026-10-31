

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

feature_file = ROOT / "data/features/holybro_window_features.parquet"
prediction_file = ROOT / "data/predictions/holybro_anomaly_predictions.csv"

features = pd.read_parquet(feature_file)
predictions = pd.read_csv(prediction_file)

data = features.merge(
    predictions[
        ["hardware_config", "flight_id", "window_start_s",
         "anomaly_score", "is_anomaly"]
    ],
    on=["hardware_config", "flight_id", "window_start_s"],
    how="inner"
)

signals = [
    "remaining__battery_status_0__std",
    "remaining__battery_status_0__range",
    "eph__vehicle_gps_position_0__mean",
    "epv__vehicle_gps_position_0__mean",
    "prediction_error__battery_status_0__mean",
]

train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

results = []

for signal in signals:
    if signal not in data.columns:
        continue

    reference = train[signal].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    flagged = test.loc[
        test["is_anomaly"] == 1, signal
    ].replace([np.inf, -np.inf], np.nan).dropna()

    normal = test.loc[
        test["is_anomaly"] == 0, signal
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if len(reference) == 0:
        continue

    q1 = reference.quantile(0.25)
    q3 = reference.quantile(0.75)
    iqr = q3 - q1

    results.append({
        "feature": signal,
        "training_median": reference.median(),
        "flagged_median": flagged.median(),
        "normal_median": normal.median(),
        "flagged_outside_training_iqr_pct":
            ((flagged < q1) | (flagged > q3)).mean() * 100,
        "normal_outside_training_iqr_pct":
            ((normal < q1) | (normal > q3)).mean() * 100,
        "median_difference_iqr":
            abs(flagged.median() - normal.median()) / iqr
            if iqr > 0 and len(flagged) and len(normal)
            else np.nan
    })

report = pd.DataFrame(results)

output = ROOT / "reports/holybro_test_anomaly_patterns.csv"
report.to_csv(output, index=False)

print("\nHOLYBRO TEST SET: FLAGGED VS NORMAL WINDOWS")
print(report.to_string(index=False))

print("\nANOMALY RATE BY TEST FLIGHT")

flight_report = (
    test.groupby("flight_id")
    .agg(
        windows=("is_anomaly", "size"),
        flagged=("is_anomaly", "sum"),
        mean_score=("anomaly_score", "mean")
    )
    .reset_index()
)

flight_report["anomaly_rate_pct"] = (
    flight_report["flagged"] / flight_report["windows"] * 100
)

print(
    flight_report.sort_values(
        "anomaly_rate_pct", ascending=False
    ).to_string(index=False)
)

print("\nSaved:", output)
