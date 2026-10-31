
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

features = pd.read_parquet(
    ROOT / "data/features/holybro_window_features.parquet"
)

predictions = pd.read_csv(
    ROOT / "data/predictions/holybro_anomaly_predictions.csv"
)

df = features[
    (features["flight_id"] == 1) &
    (features["split"] == "test")
].copy()

df = df.merge(
    predictions[
        ["flight_id", "window_start_s", "anomaly_score", "is_anomaly"]
    ],
    on=["flight_id", "window_start_s"],
    how="left"
)

signals = {
    "Battery remaining variability":
        "remaining__battery_status_0__std",
    "GPS horizontal uncertainty":
        "eph__vehicle_gps_position_0__mean",
    "GPS vertical uncertainty":
        "epv__vehicle_gps_position_0__mean",
    "Battery prediction error":
        "prediction_error__battery_status_0__mean",
}

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for ax, (title, column) in zip(axes, signals.items()):
    if column not in df.columns:
        ax.set_title(f"{title} - feature unavailable")
        continue

    ax.plot(
        df["window_start_s"],
        df[column],
        marker="o",
        label=title
    )

    anomalous = df[df["is_anomaly"] == 1]

    ax.scatter(
        anomalous["window_start_s"],
        anomalous[column],
        marker="x",
        s=65,
        label="Flagged anomaly"
    )

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Flight time (seconds)")

plt.tight_layout()

output = ROOT / "reports/holybro_flight1_anomaly_timeline.png"
plt.savefig(output, dpi=150)
plt.show()

print("Saved:", output)
