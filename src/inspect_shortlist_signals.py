
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "reports" / "shortlist_signal_inspection.csv"

results = []

manifest = load_manifest()

for hardware in ["holybro", "speedybee"]:

    print(f"\nInspecting {hardware.upper()}")

    shortlist_path = (
        ROOT / "reports" / f"feature_shortlist_{hardware}.csv"
    )
    shortlist = pd.read_csv(shortlist_path)

    hardware_flights = manifest[
        manifest["hardware_config"] == hardware
    ]

    # Select one representative flight from each split.
    sample_flights = (
        hardware_flights
        .sort_values("flight_id")
        .groupby("split", sort=False)
        .head(1)
    )

    print("Sample flights:")
    print(
        sample_flights[
            ["flight_id", "split"]
        ].to_string(index=False)
    )

    for _, flight_record in sample_flights.iterrows():

        flight_id = flight_record["flight_id"]
        split = flight_record["split"]

        df, metadata = load_flight(hardware, flight_id)

        print(
            f"Inspecting {hardware} flight {flight_id} "
            f"({split})"
        )

        for _, feature in shortlist.iterrows():

            column = feature["raw_column"]

            if column not in df.columns:
                results.append({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "concept": feature["concept"],
                    "raw_column": column,
                    "status": "COLUMN_MISSING",
                })
                continue

            series = df[column]
            numeric = pd.to_numeric(
                series, errors="coerce"
            )

            values = numeric.to_numpy(
                dtype=float,
                na_value=np.nan
            )
            finite = values[np.isfinite(values)]

            if len(finite) > 0:
                results.append({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "concept": feature["concept"],
                    "raw_column": column,
                    "status": "NUMERIC",
                    "dtype": str(series.dtype),
                    "rows": len(series),
                    "valid_values": len(finite),
                    "missing_pct": (
                        100 * (len(series) - len(finite))
                        / len(series)
                    ),
                    "unique_values": len(np.unique(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                })

            else:
                examples = (
                    series.dropna()
                    .astype(str)
                    .unique()[:5]
                )

                results.append({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "concept": feature["concept"],
                    "raw_column": column,
                    "status": "NON_NUMERIC_OR_EMPTY",
                    "dtype": str(series.dtype),
                    "rows": len(series),
                    "valid_values": 0,
                    "missing_pct": (
                        100 * series.isna().mean()
                    ),
                    "unique_values": series.nunique(),
                    "examples": " | ".join(examples),
                })

    print(f"Completed {hardware}")

report = pd.DataFrame(results)

report.to_csv(OUTPUT, index=False)

print("\n" + "=" * 65)
print("SIGNAL INSPECTION COMPLETE")
print("=" * 65)

print(f"Inspected records: {len(report)}")
print("\nStatus counts:")
print(report["status"].value_counts().to_string())

print("\nConstant numeric signals:")
numeric_report = report[
    report["status"] == "NUMERIC"
]

constant = numeric_report[
    numeric_report["unique_values"] <= 1
]

print(
    constant[
        [
            "hardware_config",
            "raw_column",
            "flight_id",
            "unique_values",
            "min",
            "max",
        ]
    ].to_string(index=False)
)

print(f"\nSaved report: {OUTPUT}")
