
import json
import pandas as pd
import numpy as np

from pathlib import Path
from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data/features/holybro_window_features.parquet"
OUTPUT_FILE = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
WINDOW_REPORT = ROOT / "reports/remaining_time_window_summary.csv"
SUMMARY_FILE = ROOT / "reports/remaining_time_dataset_summary.json"

TARGET = "time_remaining_s__battery_status_0"

# --------------------------------------------------
# 1. Load existing engineered windows and manifest
# --------------------------------------------------

print("Loading engineered window features...")

features = pd.read_parquet(FEATURE_FILE)

manifest = load_manifest()
manifest = manifest[
    manifest["hardware_config"] == "holybro"
]

print("Total engineered windows:", len(features))
print("Total Holybro manifest flights:", len(manifest))

# --------------------------------------------------
# 2. Prepare eligible flight IDs
# --------------------------------------------------

eligible_flights = set()

for _, record in manifest.iterrows():

    flight_id = int(record["flight_id"])

    try:
        raw_df, metadata = load_flight("holybro", flight_id)

        if TARGET not in raw_df.columns:
            continue

        target_values = pd.to_numeric(
            raw_df[TARGET], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        if target_values.notna().sum() >= 10:
            eligible_flights.add(flight_id)

    except Exception as exc:
        print(f"Eligibility check failed for flight {flight_id}: {exc}")

print("\nEligible flights:", len(eligible_flights))

# --------------------------------------------------
# 3. Match target observations to each window
# --------------------------------------------------

dataset_parts = []
window_reports = []

eligible_features = features[
    features["flight_id"].isin(eligible_flights)
].copy()

for flight_id, flight_windows in eligible_features.groupby("flight_id"):

    print(f"Processing flight {flight_id}...")

    raw_df, metadata = load_flight("holybro", int(flight_id))

    raw_df = raw_df.copy()

    raw_df["timestamp_s"] = pd.to_numeric(
        raw_df["timestamp_s"], errors="coerce"
    )

    raw_df[TARGET] = pd.to_numeric(
        raw_df[TARGET], errors="coerce"
    )

    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)

    raw_df = raw_df.dropna(subset=["timestamp_s"])

    flight_windows = flight_windows.sort_values(
        "window_start_s"
    ).copy()

    targets = []
    observation_counts = []

    for _, window in flight_windows.iterrows():

        start = float(window["window_start_s"])
        end = float(window["window_end_s"])

        interval = raw_df[
            (raw_df["timestamp_s"] >= start - 1e-8) &
            (raw_df["timestamp_s"] <= end + 1e-8)
        ]

        values = interval[TARGET].dropna()

        if len(values) == 0:
            targets.append(np.nan)
            observation_counts.append(0)
        else:
            targets.append(float(values.median()))
            observation_counts.append(int(len(values)))

    flight_windows["target_remaining_time_s"] = targets
    flight_windows["target_observations"] = observation_counts

    valid_windows = flight_windows.dropna(
        subset=["target_remaining_time_s"]
    ).copy()

    window_reports.append({
        "flight_id": int(flight_id),
        "split": flight_windows["split"].iloc[0],
        "total_windows": len(flight_windows),
        "valid_target_windows": len(valid_windows),
        "excluded_windows": len(flight_windows) - len(valid_windows),
        "median_target_s": (
            valid_windows["target_remaining_time_s"].median()
            if len(valid_windows) else np.nan
        )
    })

    if len(valid_windows):
        dataset_parts.append(valid_windows)

# --------------------------------------------------
# 4. Combine all eligible windows
# --------------------------------------------------

if not dataset_parts:
    raise RuntimeError("No valid target windows were generated.")

dataset = pd.concat(
    dataset_parts,
    ignore_index=True
)

dataset["hardware_config"] = "holybro"

# --------------------------------------------------
# 5. Remove direct target leakage from inputs
# --------------------------------------------------

metadata_columns = {
    "hardware_config",
    "flight_id",
    "split",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "target_remaining_time_s",
    "target_observations"
}

leakage_tokens = [
    "time_remaining",
    "remaining_time"
]

leakage_columns = [
    col for col in dataset.columns
    if any(token in col.lower() for token in leakage_tokens)
    and col not in metadata_columns
]

if leakage_columns:
    print("\nRemoving target-derived input columns:")
    for col in leakage_columns:
        print(" -", col)

    dataset = dataset.drop(columns=leakage_columns)

# --------------------------------------------------
# 6. Save dataset and reports
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
WINDOW_REPORT.parent.mkdir(parents=True, exist_ok=True)

dataset.to_parquet(
    OUTPUT_FILE,
    index=False
)

window_report = pd.DataFrame(window_reports)

window_report.to_csv(
    WINDOW_REPORT,
    index=False
)

split_counts = (
    dataset.groupby("split")
    .size()
    .to_dict()
)

summary = {
    "task": "Holybro battery remaining-time estimation",
    "target": TARGET,
    "eligible_flights": len(eligible_flights),
    "flights_in_dataset": int(dataset["flight_id"].nunique()),
    "total_windows": len(dataset),
    "split_windows": {
        key: int(value)
        for key, value in split_counts.items()
    },
    "target_min_seconds": float(
        dataset["target_remaining_time_s"].min()
    ),
    "target_max_seconds": float(
        dataset["target_remaining_time_s"].max()
    ),
    "target_median_seconds": float(
        dataset["target_remaining_time_s"].median()
    ),
    "removed_leakage_columns": leakage_columns,
    "excluded_windows": int(
        window_report["excluded_windows"].sum()
    )
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# --------------------------------------------------
# 7. Final report
# --------------------------------------------------

print("\n" + "=" * 60)
print("REMAINING-TIME DATASET BUILD COMPLETE")
print("=" * 60)

print("Eligible flights:", summary["eligible_flights"])
print("Flights represented:", summary["flights_in_dataset"])
print("Total windows:", summary["total_windows"])

print("\nWINDOWS BY SPLIT")
for split, count in summary["split_windows"].items():
    print(f"{split}: {count}")

print("\nTARGET DISTRIBUTION")
print("Minimum:", round(summary["target_min_seconds"], 2), "seconds")
print("Maximum:", round(summary["target_max_seconds"], 2), "seconds")
print("Median:", round(summary["target_median_seconds"], 2), "seconds")

print("\nRemoved leakage columns:", len(leakage_columns))
print("Excluded windows:", summary["excluded_windows"])

print("\nSAVED FILES")
print(OUTPUT_FILE)
print(WINDOW_REPORT)
print(SUMMARY_FILE)
