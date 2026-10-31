
from pathlib import Path
import gc
import sys
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "loader_validation.csv"

# Expected usable flight counts from our dataset audit
EXPECTED_COUNTS = {
    "holybro": 118,
    "speedybee": 110,
}

EXPECTED_SPLITS = {
    ("holybro", "train"): 82,
    ("holybro", "val"): 18,
    ("holybro", "test"): 18,
    ("speedybee", "train"): 77,
    ("speedybee", "val"): 16,
    ("speedybee", "test"): 17,
}


def main():

    manifest = load_manifest()

    print("=" * 60)
    print("AEROSENTINEL - FULL DATASET LOADER VALIDATION")
    print("=" * 60)

    print(f"\nManifest entries: {len(manifest)}")

    results = []
    errors = []

    for hardware in EXPECTED_COUNTS:

        hardware_manifest = manifest[
            manifest["hardware_config"].str.lower() == hardware
        ]

        print(f"\n{'-' * 60}")
        print(f"Hardware: {hardware.upper()}")
        print(f"Expected flights: {EXPECTED_COUNTS[hardware]}")
        print(f"Manifest flights: {len(hardware_manifest)}")
        print("-" * 60)

        if len(hardware_manifest) != EXPECTED_COUNTS[hardware]:
            errors.append(
                f"{hardware}: expected "
                f"{EXPECTED_COUNTS[hardware]} manifest flights, "
                f"found {len(hardware_manifest)}"
            )

        for _, record in hardware_manifest.iterrows():

            flight_id = record["flight_id"]
            split = str(record["split"]).lower()

            try:
                # Load only one flight at a time
                df, metadata = load_flight(hardware, flight_id)

                # Basic data validation
                if df.empty:
                    raise ValueError("Empty DataFrame")

                if "timestamp_s" not in df.columns:
                    raise ValueError("timestamp_s missing")

                if df["timestamp_s"].isna().any():
                    raise ValueError("Missing timestamp values")

                if not df.columns.is_unique:
                    raise ValueError("Duplicate column names")

                results.append({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "status": "PASS",
                    "error": ""
                })

                print(
                    f"PASS | {hardware:9} | "
                    f"Flight {str(flight_id):>4} | "
                    f"{split:5} | "
                    f"{len(df):>7} rows | "
                    f"{len(df.columns):>5} columns"
                )

                # Release memory before loading next flight
                del df
                del metadata
                gc.collect()

            except Exception as exc:

                errors.append(
                    f"{hardware} flight {flight_id}: {exc}"
                )

                results.append({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "rows": 0,
                    "columns": 0,
                    "status": "FAIL",
                    "error": str(exc)
                })

                print(
                    f"FAIL | {hardware:9} | "
                    f"Flight {flight_id} | {exc}"
                )

    # Save detailed validation report
    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_PATH, index=False)

    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total flights checked: {len(results)}")
    print(
        f"Passed: "
        f"{sum(r['status'] == 'PASS' for r in results)}"
    )
    print(
        f"Failed: "
        f"{sum(r['status'] == 'FAIL' for r in results)}"
    )

    print("\nSplit distribution:")

    for key, expected in EXPECTED_SPLITS.items():

        hardware, split = key

        actual = len(
            report[
                (report["hardware_config"] == hardware)
                & (report["split"] == split)
                & (report["status"] == "PASS")
            ]
        )

        print(
            f"{hardware:9} | {split:5} | "
            f"{actual}/{expected}"
        )

        if actual != expected:
            errors.append(
                f"{hardware} {split}: expected "
                f"{expected} passing flights, found {actual}"
            )

    print(f"\nReport saved: {OUTPUT_PATH}")

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(" -", error)

        sys.exit(1)

    print("\nALL 228 FLIGHTS VALIDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
