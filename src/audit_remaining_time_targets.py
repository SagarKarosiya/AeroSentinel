
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]

manifest = load_manifest()

results = []

for _, record in manifest.iterrows():

    hardware = record["hardware_config"]
    flight_id = int(record["flight_id"])

    print(f"Checking {hardware} flight {flight_id}...")

    try:
        df, metadata = load_flight(hardware, flight_id)

        candidates = [
            col for col in df.columns
            if any(term in col.lower() for term in [
                "time_remaining",
                "remaining_time",
                "flight_time_remaining",
                "battery_remaining",
                "remaining"
            ])
        ]

        for col in candidates:

            values = pd.to_numeric(
                df[col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            results.append({
                "hardware_config": hardware,
                "flight_id": flight_id,
                "feature": col,
                "valid_rows": len(values),
                "total_rows": len(df),
                "coverage_pct": (
                    len(values) / len(df) * 100 if len(df) else 0
                ),
                "minimum": values.min() if len(values) else np.nan,
                "maximum": values.max() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan
            })

    except Exception as exc:
        print(f"WARNING: {hardware} flight {flight_id}: {exc}")

report = pd.DataFrame(results)

output = ROOT / "reports/remaining_time_target_audit.csv"
report.to_csv(output, index=False)

print("\nREMAINING-TIME TARGET AUDIT")

if report.empty:
    print("No matching target-related columns found.")
else:
    summary = (
        report.groupby(["hardware_config", "feature"])
        .agg(
            flights_with_feature=("flight_id", "nunique"),
            total_valid_rows=("valid_rows", "sum"),
            median_coverage_pct=("coverage_pct", "median"),
            overall_min=("minimum", "min"),
            overall_max=("maximum", "max")
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

print("\nSaved:", output)
