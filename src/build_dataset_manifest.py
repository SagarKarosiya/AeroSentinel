
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

METADATA_FILE = DATA / "flight_metadata.csv"
SPLITS_FILE = DATA / "splits.csv"


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in ("true", "1", "yes")


def normalize_id(value):
    return str(int(float(value)))


def get_telemetry_path(hardware, flight_id):
    flight_id = normalize_id(flight_id)

    if hardware == "holybro":
        filename = f"flight_{flight_id}.parquet"
    elif hardware == "speedybee":
        filename = f"lap_{int(flight_id):03d}.parquet"
    else:
        raise ValueError(f"Unknown hardware: {hardware}")

    return DATA / hardware / filename


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_FILE)
    splits = pd.read_csv(SPLITS_FILE)

    metadata["hardware_config"] = (
        metadata["hardware_config"].str.lower().str.strip()
    )
    splits["hardware_config"] = (
        splits["hardware_config"].str.lower().str.strip()
    )

    metadata["flight_id"] = metadata["flight_id"].apply(normalize_id)
    splits["flight_id"] = splits["flight_id"].apply(normalize_id)

    metadata["exclude"] = metadata["exclude"].apply(normalize_bool)

    keys = ["hardware_config", "flight_id"]

    if metadata.duplicated(keys).any():
        raise ValueError("Duplicate flight IDs in metadata.")

    if splits.duplicated(keys).any():
        raise ValueError("Duplicate flight IDs in splits.")

    merged = metadata.merge(
        splits[keys + ["split"]],
        on=keys,
        how="left",
        validate="one_to_one"
    )

    manifest_rows = []
    excluded_rows = []
    problems = []

    for _, row in merged.iterrows():
        hardware = row["hardware_config"]
        flight_id = row["flight_id"]

        path = get_telemetry_path(hardware, flight_id)

        record = {
            "hardware_config": hardware,
            "flight_id": flight_id,
            "duration_s": row["duration_s"],
            "split": row["split"],
            "gps_available": row.get("gps_available"),
            "telemetry_path": str(path.relative_to(ROOT)),
            "exclude": row["exclude"],
            "exclude_reason": row.get("exclude_reason"),
        }

        if row["exclude"]:
            excluded_rows.append(record)
            continue

        if pd.isna(row["split"]):
            problems.append(f"Missing split: {hardware} flight {flight_id}")
            continue

        if not path.exists():
            problems.append(f"Missing telemetry: {path}")
            continue

        if row["split"] not in ("train", "val", "test"):
            problems.append(
                f"Invalid split: {hardware} flight {flight_id}"
            )
            continue

        manifest_rows.append(record)

    manifest = pd.DataFrame(manifest_rows)
    excluded = pd.DataFrame(excluded_rows)

    if problems:
        print("\nVALIDATION PROBLEMS:")
        for problem in problems:
            print("-", problem)

        raise RuntimeError(
            f"Manifest validation failed with {len(problems)} problems."
        )

    manifest_path = REPORTS / "dataset_manifest.csv"
    excluded_path = REPORTS / "excluded_flights.csv"

    manifest.to_csv(manifest_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    print("\n" + "=" * 55)
    print("AEROSENTINEL DATASET MANIFEST")
    print("=" * 55)

    print("Metadata flights:", len(metadata))
    print("Excluded flights:", len(excluded))
    print("Usable flights:", len(manifest))

    print("\nFLIGHTS BY HARDWARE AND SPLIT:")

    summary = pd.crosstab(
        manifest["hardware_config"],
        manifest["split"],
        margins=True
    )

    print(summary.to_string())

    print("\nExcluded flights:")
    if not excluded.empty:
        print(
            excluded[
                ["hardware_config", "flight_id", "duration_s"]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(manifest_path)
    print(excluded_path)

    print("\nDataset manifest validation PASSED.")


if __name__ == "__main__":
    main()
