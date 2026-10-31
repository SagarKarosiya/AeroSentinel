
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

features = pd.read_parquet(
    ROOT / "data/features/holybro_window_features.parquet"
)

predictions = pd.read_csv(
    ROOT / "data/predictions/holybro_anomaly_predictions.csv"
)

data = features.merge(
    predictions[
        ["hardware_config", "flight_id", "window_start_s",
         "anomaly_score", "is_anomaly"]
    ],
    on=["hardware_config", "flight_id", "window_start_s"],
    how="inner"
)

test = data[data["split"] == "test"].copy()

signals = {
    "battery_remaining_std":
        "remaining__battery_status_0__std",

    "battery_remaining_range":
        "remaining__battery_status_0__range",

    "gps_eph":
        "eph__vehicle_gps_position_0__mean",

    "gps_epv":
        "epv__vehicle_gps_position_0__mean",
}

results = []

for flight_id, group in test.groupby("flight_id"):

    flagged = group[group["is_anomaly"] == 1]
    normal = group[group["is_anomaly"] == 0]

    if len(flagged) < 1 or len(normal) < 1:
        continue

    row = {
        "flight_id": flight_id,
        "flagged_windows": len(flagged),
        "normal_windows": len(normal),
    }

    for name, column in signals.items():

        if column not in group.columns:
            continue

        flagged_values = pd.to_numeric(
            flagged[column], errors="coerce"
        ).dropna()

        normal_values = pd.to_numeric(
            normal[column], errors="coerce"
        ).dropna()

        if len(flagged_values) == 0 or len(normal_values) == 0:
            row[name + "_flagged_higher"] = np.nan
            continue

        row[name + "_flagged_higher"] = (
            flagged_values.median() > normal_values.median()
        )

        row[name + "_median_ratio"] = (
            flagged_values.median() / normal_values.median()
            if normal_values.median() != 0 else np.nan
        )

    results.append(row)

report = pd.DataFrame(results)

output = ROOT / "reports/holybro_anomaly_patterns_per_flight.csv"
report.to_csv(output, index=False)

print("\nPER-FLIGHT ANOMALY PATTERN CHECK")
print(report.to_string(index=False))

print("\nSUMMARY")

for name in signals:
    col = name + "_flagged_higher"

    if col in report.columns:
        valid = report[col].dropna()

        print(
            f"{name}: flagged median higher in "
            f"{int(valid.sum())}/{len(valid)} flights"
        )

print("\nSaved:", output)
