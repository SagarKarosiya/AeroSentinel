
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
OUTPUT_FILE = ROOT / "data/features/holybro_future_remaining_time_30s.parquet"
REPORT_FILE = ROOT / "reports/future_remaining_time_30s_summary.csv"

HORIZON_SECONDS = 30.0
TARGET = "target_remaining_time_s"

df = pd.read_parquet(INPUT_FILE)

required = {
    "flight_id", "split", "window_start_s",
    "window_end_s", TARGET
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

results = []
flight_reports = []

print("Building future remaining-time dataset...")
print("Forecast horizon:", HORIZON_SECONDS, "seconds")

for flight_id, group in df.groupby("flight_id"):

    group = group.sort_values("window_start_s").reset_index(drop=True)

    starts = group["window_start_s"].to_numpy(dtype=float)
    ends = group["window_end_s"].to_numpy(dtype=float)
    targets = group[TARGET].to_numpy(dtype=float)

    future_targets = []
    future_times = []

    for i in range(len(group)):

        required_future_time = ends[i] + HORIZON_SECONDS

        # Find the first future window beginning at or after
        # the requested forecast horizon.
        future_index = np.searchsorted(
            starts,
            required_future_time - 1e-8,
            side="left"
        )

        if future_index >= len(group):
            future_targets.append(np.nan)
            future_times.append(np.nan)
            continue

        future_targets.append(targets[future_index])
        future_times.append(starts[future_index] - ends[i])

    group["future_remaining_time_s"] = future_targets
    group["actual_forecast_gap_s"] = future_times

    valid = group.dropna(
        subset=["future_remaining_time_s"]
    ).copy()

    if len(valid):
        results.append(valid)

    flight_reports.append({
        "flight_id": int(flight_id),
        "split": group["split"].iloc[0],
        "original_windows": len(group),
        "forecast_windows": len(valid),
        "removed_windows": len(group) - len(valid),
        "median_forecast_gap_s": (
            valid["actual_forecast_gap_s"].median()
            if len(valid) else np.nan
        )
    })

if not results:
    raise RuntimeError("No future forecast samples could be generated.")

forecast_df = pd.concat(results, ignore_index=True)

# Remove the current-window target column.
# It is not an input feature for this forecasting experiment.
forecast_df = forecast_df.drop(columns=[TARGET])

forecast_df.to_parquet(OUTPUT_FILE, index=False)

report = pd.DataFrame(flight_reports)
report.to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 60)
print("FUTURE FORECAST DATASET COMPLETE")
print("=" * 60)

print("Forecast horizon requested:", HORIZON_SECONDS, "seconds")
print("Total samples:", len(forecast_df))
print("Flights:", forecast_df["flight_id"].nunique())

print("\nSamples by split:")
print(forecast_df.groupby("split").size().to_string())

print("\nActual forecast gap distribution:")
print(
    forecast_df["actual_forecast_gap_s"]
    .describe()
    .round(2)
    .to_string()
)

print("\nFuture target distribution:")
print(
    forecast_df["future_remaining_time_s"]
    .describe()
    .round(2)
    .to_string()
)

print("\nSaved:")
print(OUTPUT_FILE)
print(REPORT_FILE)
