
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


def get_telemetry_files(hardware):
    folder = DATA / hardware

    if hardware == "holybro":
        return sorted(folder.glob("flight_*.parquet"))

    elif hardware == "speedybee":
        return sorted(folder.glob("lap_*.parquet"))

    return []


def get_flight_id(path, hardware):
    prefix = "flight_" if hardware == "holybro" else "lap_"

    try:
        return int(path.stem.replace(prefix, ""))
    except ValueError:
        return None


def main():
    splits_path = DATA / "splits.csv"
    metadata_path = DATA / "flight_metadata.csv"

    splits = pd.read_csv(splits_path)
    metadata = pd.read_csv(metadata_path)

    print("\n=== SPLIT SUMMARY ===")
    print("Rows in splits.csv:", len(splits))

    print("\nFlights per hardware and split:")
    print(
        splits.groupby(["hardware_config", "split"])["flight_id"]
        .nunique()
        .to_string()
    )

    print("\n=== TELEMETRY FILE SUMMARY ===")

    all_ids = set()

    for hardware in ("holybro", "speedybee"):
        files = get_telemetry_files(hardware)

        ids = {
            get_flight_id(path, hardware)
            for path in files
        }

        ids.discard(None)

        all_ids.update(
            (hardware, flight_id)
            for flight_id in ids
        )

        print(f"{hardware}: {len(files)} Parquet flight files")

    split_ids = set(
        zip(
            splits["hardware_config"].astype(str),
            splits["flight_id"].astype(int)
        )
    )

    print("\nFiles without a matching split entry:",
          len(all_ids - split_ids))

    print("Split entries without a matching telemetry file:",
          len(split_ids - all_ids))

    print("\n=== METADATA SUMMARY ===")
    print("Metadata rows:", len(metadata))

    if "exclude" in metadata.columns:
        print("\nExcluded flights:")
        print(
            metadata.groupby("hardware_config")["exclude"]
            .agg(["count", "sum"])
            .to_string()
        )

    if "exclude_reason" in metadata.columns:
        reasons = metadata["exclude_reason"].dropna()

        print("\nExclusion reasons:")
        print(
            reasons.value_counts().head(10).to_string()
            if not reasons.empty
            else "No reasons recorded"
        )

    print("\nAudit finished. No training or source files were modified.")


if __name__ == "__main__":
    main()
