
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

manifest = load_manifest()
results = []

print("=" * 70)
print("AEROSENTINEL TIMESTAMP AND SAMPLING AUDIT")
print("=" * 70)

for _, record in manifest.iterrows():

    hardware = record["hardware_config"]
    flight_id = int(record["flight_id"])
    split = record["split"]

    print(f"Checking {hardware} flight {flight_id} ({split})")

    try:
        df, metadata = load_flight(hardware, flight_id)

        timestamps = pd.to_numeric(
            df["timestamp_s"], errors="coerce"
        ).dropna().to_numpy(dtype=float)

        if len(timestamps) < 2:
            raise ValueError("Insufficient valid timestamps")

        dt = np.diff(timestamps)

        positive_dt = dt[dt > 0]
        median_dt = (
            float(np.median(positive_dt))
            if len(positive_dt) else np.nan
        )

        sampling_hz = (
            1.0 / median_dt
            if np.isfinite(median_dt) and median_dt > 0
            else np.nan
        )

        duration_s = float(timestamps[-1] - timestamps[0])

        gap_threshold = (
            max(3.0 * median_dt, median_dt + 0.1)
            if np.isfinite(median_dt) else np.nan
        )

        results.append({
            "hardware_config": hardware,
            "flight_id": flight_id,
            "split": split,
            "rows": len(df),
            "valid_timestamps": len(timestamps),
            "duration_s": duration_s,
            "median_interval_s": median_dt,
            "sampling_hz": sampling_hz,
            "min_interval_s": (
                float(np.min(positive_dt))
                if len(positive_dt) else np.nan
            ),
            "max_interval_s": (
                float(np.max(dt))
                if len(dt) else np.nan
            ),
            "duplicate_or_nonincreasing_pct": (
                float(np.mean(dt <= 0) * 100)
            ),
            "large_gap_count": (
                int(np.sum(dt > gap_threshold))
                if np.isfinite(gap_threshold) else np.nan
            ),
            "large_gap_pct": (
                float(np.mean(dt > gap_threshold) * 100)
                if np.isfinite(gap_threshold) else np.nan
            ),
            "strictly_increasing": bool(np.all(dt > 0)),
            "status": "OK"
        })

    except Exception as exc:
        results.append({
            "hardware_config": hardware,
            "flight_id": flight_id,
            "split": split,
            "status": f"ERROR: {exc}"
        })

report = pd.DataFrame(results)

output_path = REPORT_DIR / "timestamp_sampling_audit.csv"
report.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("SAMPLING SUMMARY")
print("=" * 70)

valid = report[report["status"] == "OK"]

if not valid.empty:
    summary = valid.groupby("hardware_config").agg(
        flights=("flight_id", "count"),
        median_sampling_hz=("sampling_hz", "median"),
        min_sampling_hz=("sampling_hz", "min"),
        max_sampling_hz=("sampling_hz", "max"),
        median_duration_s=("duration_s", "median"),
        median_large_gap_pct=("large_gap_pct", "median"),
        non_increasing_flights=(
            "strictly_increasing",
            lambda x: int((~x).sum())
        )
    )

    print(summary.to_string())

print("\nTotal flights:", len(report))
print("Successful:", len(valid))
print("Failed:", len(report) - len(valid))
print(f"\nSaved: {output_path}")
