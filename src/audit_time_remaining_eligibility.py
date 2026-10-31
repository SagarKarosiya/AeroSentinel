
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]

manifest = load_manifest()
manifest = manifest[
    manifest["hardware_config"] == "holybro"
]

TARGET = "time_remaining_s__battery_status_0"

results = []

for _, record in manifest.iterrows():

    flight_id = int(record["flight_id"])

    try:
        df, metadata = load_flight("holybro", flight_id)

        if TARGET not in df.columns:
            results.append({
                "flight_id": flight_id,
                "status": "COLUMN_MISSING",
                "total_rows": len(df),
                "valid_rows": 0,
                "coverage_pct": 0
            })
            continue

        values = pd.to_numeric(
            df[TARGET], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid = values.dropna()

        timestamp = pd.to_numeric(
            df["timestamp_s"], errors="coerce"
        )

        results.append({
            "flight_id": flight_id,
            "status": (
                "USABLE" if len(valid) >= 10
                else "INSUFFICIENT_VALID_VALUES"
            ),
            "total_rows": len(df),
            "valid_rows": len(valid),
            "coverage_pct": (
                len(valid) / len(df) * 100
                if len(df) else 0
            ),
            "flight_duration_s": (
                timestamp.max() - timestamp.min()
                if timestamp.notna().any() else np.nan
            ),
            "first_estimate_s": (
                valid.iloc[0] if len(valid) else np.nan
            ),
            "last_estimate_s": (
                valid.iloc[-1] if len(valid) else np.nan
            ),
            "median_estimate_s": (
                valid.median() if len(valid) else np.nan
            ),
            "minimum_estimate_s": (
                valid.min() if len(valid) else np.nan
            ),
            "maximum_estimate_s": (
                valid.max() if len(valid) else np.nan
            )
        })

    except Exception as exc:
        results.append({
            "flight_id": flight_id,
            "status": "LOAD_ERROR",
            "error": str(exc)
        })

report = pd.DataFrame(results)

output = ROOT / "reports/time_remaining_eligibility.csv"
report.to_csv(output, index=False)

print("\nTARGET ELIGIBILITY SUMMARY")
print(report["status"].value_counts().to_string())

print("\nFLIGHTS WITHOUT SUFFICIENT VALID VALUES")
print(
    report[
        report["status"] != "USABLE"
    ].to_string(index=False)
)

print("\nUSABLE FLIGHT SUMMARY")
usable = report[report["status"] == "USABLE"]

print(
    usable[
        [
            "coverage_pct",
            "flight_duration_s",
            "first_estimate_s",
            "last_estimate_s",
            "median_estimate_s"
        ]
    ].describe().to_string()
)

print("\nSaved:", output)
