
from pathlib import Path
import gc
import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)


CATALOGS = {
    "holybro": "feature_channels_final_holybro.csv",
    "speedybee": "feature_channel_candidates_speedybee.csv",
}


def audit_hardware(hardware, catalog_file):

    print("\n" + "=" * 65)
    print(f"FEATURE COVERAGE AUDIT: {hardware.upper()}")
    print("=" * 65)

    catalog = pd.read_csv(PROCESSED / catalog_file)
    manifest = load_manifest()

    flights = manifest[
        manifest["hardware_config"].str.lower() == hardware
    ]

    # Prepare one record per feature
    stats = {}

    for _, feature in catalog.iterrows():

        column = str(feature["raw_column"])

        stats[column] = {
            "concept": feature.get("concept", "unknown"),
            "flights_present": 0,
            "flights_with_values": 0,
            "total_rows": 0,
            "valid_values": 0,
            "non_numeric_flights": 0,
        }

    total_flights = len(flights)

    # Process one flight at a time
    for index, record in flights.iterrows():

        flight_id = record["flight_id"]

        print(
            f"Auditing {hardware} flight "
            f"{flight_id} ({index + 1}/{len(manifest)})"
        )

        df, metadata = load_flight(hardware, flight_id)

        row_count = len(df)

        for column, result in stats.items():

            result["total_rows"] += row_count

            if column not in df.columns:
                continue

            result["flights_present"] += 1

            series = df[column]

            numeric = pd.to_numeric(
                series,
                errors="coerce"
            )

            if series.notna().any() and not numeric.notna().any():
                result["non_numeric_flights"] += 1
                continue

            valid = np.isfinite(
                numeric.to_numpy(dtype=float, na_value=np.nan)
            )

            valid_count = int(valid.sum())

            result["valid_values"] += valid_count

            if valid_count > 0:
                result["flights_with_values"] += 1

        del df
        del metadata
        gc.collect()

    # Build report
    rows = []

    for column, result in stats.items():

        total_rows = result["total_rows"]
        valid_values = result["valid_values"]

        rows.append({
            "hardware_config": hardware,
            "concept": result["concept"],
            "raw_column": column,
            "flights_total": total_flights,
            "flights_present": result["flights_present"],
            "flights_with_values": result["flights_with_values"],
            "flight_coverage_pct": round(
                100 * result["flights_with_values"] / total_flights, 2
            ),
            "total_rows": total_rows,
            "valid_values": valid_values,
            "value_coverage_pct": round(
                100 * valid_values / total_rows, 4
            ) if total_rows else 0,
            "non_numeric_flights": result["non_numeric_flights"],
        })

    report = pd.DataFrame(rows)

    output = REPORTS / f"feature_coverage_{hardware}.csv"
    report.to_csv(output, index=False)

    # Summary
    print("\n" + "-" * 65)
    print(f"{hardware.upper()} COVERAGE SUMMARY")
    print("-" * 65)

    print("Total catalog features:", len(report))

    print(
        "Features with values in every flight:",
        int((report["flights_with_values"] == total_flights).sum())
    )

    print(
        "Features with values in at least one flight:",
        int((report["flights_with_values"] > 0).sum())
    )

    print(
        "Features with zero valid values:",
        int((report["valid_values"] == 0).sum())
    )

    print("\nLowest coverage features:")

    print(
        report.sort_values(
            "flight_coverage_pct"
        )[
            [
                "concept",
                "raw_column",
                "flights_with_values",
                "flight_coverage_pct",
                "value_coverage_pct",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nFeature coverage by concept:")

    print(
        report.groupby("concept").agg(
            features=("raw_column", "count"),
            always_available=(
                "flights_with_values",
                lambda x: (x == total_flights).sum()
            ),
            zero_value_features=(
                "valid_values",
                lambda x: (x == 0).sum()
            )
        ).to_string()
    )

    print("\nReport saved:", output)

    return report


def main():

    all_reports = []

    for hardware, catalog_file in CATALOGS.items():

        report = audit_hardware(hardware, catalog_file)
        all_reports.append(report)

    combined = pd.concat(all_reports, ignore_index=True)

    combined_path = REPORTS / "feature_coverage_all.csv"
    combined.to_csv(combined_path, index=False)

    print("\n" + "=" * 65)
    print("ALL HARDWARE FEATURE AUDIT COMPLETE")
    print("=" * 65)
    print("Combined features:", len(combined))
    print("Combined report:", combined_path)


if __name__ == "__main__":
    main()
