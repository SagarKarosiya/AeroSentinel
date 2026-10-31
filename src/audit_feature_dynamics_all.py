
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parent.parent

manifest = load_manifest()
all_results = []

for hardware in ["holybro", "speedybee"]:

    print("\n" + "=" * 65)
    print(f"FEATURE DYNAMICS AUDIT: {hardware.upper()}")
    print("=" * 65)

    shortlist_path = (
        ROOT / "reports" / f"feature_shortlist_{hardware}.csv"
    )
    shortlist = pd.read_csv(shortlist_path)

    flights = (
        manifest[
            manifest["hardware_config"] == hardware
        ]
        .sort_values("flight_id")
        .reset_index(drop=True)
    )

    stats = {}

    for _, feature in shortlist.iterrows():
        column = feature["raw_column"]

        stats[column] = {
            "hardware_config": hardware,
            "concept": feature["concept"],
            "raw_column": column,
            "flights_with_values": 0,
            "flights_with_variation": 0,
            "flights_constant": 0,
            "total_valid_values": 0,
            "nonzero_values": 0,
            "global_min": np.inf,
            "global_max": -np.inf,
        }

    for position, record in flights.iterrows():

        flight_id = record["flight_id"]

        print(
            f"Auditing {hardware} flight "
            f"{flight_id} ({position + 1}/{len(flights)})"
        )

        df, metadata = load_flight(hardware, flight_id)

        for column, item in stats.items():

            if column not in df.columns:
                continue

            numeric = pd.to_numeric(
                df[column], errors="coerce"
            )

            values = numeric.to_numpy(
                dtype=float,
                na_value=np.nan
            )

            finite = values[np.isfinite(values)]

            if len(finite) == 0:
                continue

            item["flights_with_values"] += 1
            item["total_valid_values"] += len(finite)
            item["nonzero_values"] += int(
                np.count_nonzero(finite)
            )

            item["global_min"] = min(
                item["global_min"], float(np.min(finite))
            )

            item["global_max"] = max(
                item["global_max"], float(np.max(finite))
            )

            if np.unique(finite).size > 1:
                item["flights_with_variation"] += 1
            else:
                item["flights_constant"] += 1

    report = pd.DataFrame(stats.values())

    report["flight_coverage_pct"] = (
        100 * report["flights_with_values"] / len(flights)
    )

    report["dynamic_flight_pct"] = (
        100 * report["flights_with_variation"] / len(flights)
    )

    report["global_range"] = (
        report["global_max"] - report["global_min"]
    )

    report = report.sort_values(
        ["flights_with_variation", "concept", "raw_column"],
        ascending=[False, True, True]
    )

    output = (
        ROOT / "reports" /
        f"feature_dynamics_{hardware}.csv"
    )

    report.to_csv(output, index=False)
    all_results.append(report)

    print("\nDYNAMICS SUMMARY")
    print("-" * 65)

    print(f"Total features: {len(report)}")
    print(
        "Features varying in every flight:",
        int(
            (report["flights_with_variation"] == len(flights))
            .sum()
        )
    )
    print(
        "Features varying in at least one flight:",
        int((report["flights_with_variation"] > 0).sum())
    )
    print(
        "Features never varying within any flight:",
        int((report["flights_with_variation"] == 0).sum())
    )

    print("\nMost consistently dynamic features:")

    print(
        report[
            [
                "concept",
                "raw_column",
                "flights_with_variation",
                "flights_constant",
                "global_min",
                "global_max",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nFeatures with no within-flight variation:")

    print(
        report.loc[
            report["flights_with_variation"] == 0,
            ["concept", "raw_column", "global_min", "global_max"]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output}")

combined = pd.concat(all_results, ignore_index=True)

combined_path = ROOT / "reports" / "feature_dynamics_all.csv"
combined.to_csv(combined_path, index=False)

print("\n" + "=" * 65)
print("ALL-FLIGHT DYNAMICS AUDIT COMPLETE")
print(f"Total features: {len(combined)}")
print(f"Combined report: {combined_path}")
