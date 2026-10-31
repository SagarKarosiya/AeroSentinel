
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "reports" / "dataset_manifest.csv"


def load_manifest():
    """Load and validate the dataset manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "hardware_config",
        "flight_id",
        "split",
        "telemetry_path",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    if manifest.duplicated(
        ["hardware_config", "flight_id"]
    ).any():
        raise ValueError("Duplicate flight entries found.")

    return manifest


def load_flight(hardware_config, flight_id):
    """Load one flight using its manifest entry."""

    hardware_config = hardware_config.lower().strip()
    flight_id = str(int(float(flight_id)))

    manifest = load_manifest()

    # Match hardware configuration and flight ID
    selected = manifest[
        (manifest["hardware_config"].str.lower() == hardware_config)
        & (manifest["flight_id"].astype(str) == flight_id)
    ]

    if selected.empty:
        raise ValueError(
            f"Flight not found or excluded: "
            f"{hardware_config}, flight {flight_id}"
        )

    if len(selected) > 1:
        raise ValueError("Multiple manifest entries found.")

    record = selected.iloc[0]

    # Resolve telemetry file path
    file_path = Path(record["telemetry_path"])

    if not file_path.is_absolute():
        file_path = ROOT / file_path

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Load one flight
    df = pd.read_parquet(file_path)

    if df.empty:
        raise ValueError(f"Flight contains no data: {file_path}")

    if "timestamp_s" not in df.columns:
        raise ValueError(
            f"timestamp_s missing from {file_path.name}"
        )

    # Normalize Holybro topic names
    if hardware_config == "holybro":

        normalized_columns = [
            col.replace("__10vueltas_", "__")
            for col in df.columns
        ]

        # Prevent accidental duplicate column names
        if len(normalized_columns) != len(set(normalized_columns)):
            raise ValueError(
                f"Duplicate columns after normalization: "
                f"{hardware_config}, flight {flight_id}"
            )

        df.columns = normalized_columns

    return df, record.to_dict()


def load_split(hardware_config, split):
    """Return flight records belonging to a specified split."""

    hardware_config = hardware_config.lower().strip()
    split = split.lower().strip()

    if split not in ("train", "val", "test"):
        raise ValueError("Split must be train, val or test.")

    manifest = load_manifest()

    selected = manifest[
        (manifest["hardware_config"].str.lower() == hardware_config)
        & (manifest["split"].str.lower() == split)
    ]

    return selected.copy()


if __name__ == "__main__":
    print("Data loader module ready.")
    print("Manifest entries:", len(load_manifest()))
