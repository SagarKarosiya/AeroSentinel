
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]

manifest = load_manifest()
manifest = manifest[manifest["hardware_config"] == "holybro"]

TARGET = "time_remaining_s__battery_status_0"

results = []

for _, record in manifest.iterrows():

    flight_id = int(record["flight_id"])

    try:
        df, metadata = load_flight("holybro", flight_id)

        if TARGET not in df.columns:
            continue

        values = pd.to_numeric(
            df[TARGET], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid = values.dropna()

        if len(valid) < 10:
            continue

        timestamp = pd.to_numeric(
            df["timestamp_s"], errors="coerce"
        )

        valid_mask = values.notna() & timestamp.notna()

        temp = pd.DataFrame({
            "time": timestamp[valid_mask],
            "remaining": values[valid_mask]
        })

        correlation = temp["time"].corr(
            temp["remaining"], method="spearman"
        )

        differences = temp["remaining"].diff().dropna()

        results.append({
            "flight_id": flight_id,
            "total_rows": len(df),
            "valid_rows": len(valid),
            "coverage_pct": len(valid) / len(df) * 100,
            "first_value": valid.iloc[0],
            "median_value": valid.median(),
            "last_value": valid.iloc[-1],
            "minimum": valid.min(),
            "maximum": valid.max(),
            "spearman_time_correlation": correlation,
            "decreasing_pct": (
                (differences < 0).mean() * 100
                if len(differences) else np.nan
            ),
            "increasing_pct": (
                (differences > 0).mean() * 100
                if len(differences) else np.nan
            )
        })

    except Exception as exc:
        print(f"Flight {flight_id} failed: {exc}")

report = pd.DataFrame(results)

output = ROOT / "reports/time_remaining_signal_inspection.csv"
report.to_csv(output, index=False)

print("\nTIME REMAINING SIGNAL INSPECTION")
print("Flights inspected:", len(report))

if not report.empty:

    print("\nOVERALL SUMMARY")
    print(
        report[
            [
                "coverage_pct",
                "first_value",
                "last_value",
                "spearman_time_correlation",
                "decreasing_pct",
                "increasing_pct"
            ]
        ].describe().to_string()
    )

    print("\nSAMPLE FLIGHT RESULTS")
    print(report.head(15).to_string(index=False))

    print("\nFLIGHTS WITH POSITIVE TIME CORRELATION")
    print(
        report[
            report["spearman_time_correlation"] > 0
        ][
            ["flight_id", "spearman_time_correlation",
             "first_value", "last_value"]
        ].to_string(index=False)
    )

print("\nSaved:", output)
