
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

def main():
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))

    if not parquet_files:
        print("No Parquet files found.")
        return

    print("=" * 55)
    print("AEROSENTINEL DATASET SUMMARY")
    print("=" * 55)
    print(f"Parquet files: {len(parquet_files)}")

    # Summarize files by hardware folder.
    hardware = {}
    total_bytes = 0

    for path in parquet_files:
        size = path.stat().st_size
        total_bytes += size
        key = path.parent.name
        hardware.setdefault(key, {"files": 0, "bytes": 0})
        hardware[key]["files"] += 1
        hardware[key]["bytes"] += size

    print(f"Total Parquet size: {total_bytes / (1024**2):.2f} MB")
    print("\nFILES BY HARDWARE:")
    for name, info in sorted(hardware.items()):
        print(
            f"{name}: {info['files']} files, "
            f"{info['bytes'] / (1024**2):.2f} MB"
        )

    # Inspect one flight from each hardware folder.
    print("\nSAMPLE FLIGHT SCHEMAS:")
    seen = set()

    for path in parquet_files:
        hardware_name = path.parent.name
        if hardware_name in seen:
            continue

        df = pd.read_parquet(path)
        print(f"\n{path.relative_to(ROOT)}")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        print("First 20 columns:", df.columns.tolist()[:20])
        print("Numeric columns:", len(df.select_dtypes(include="number").columns))
        print("Missing values in sample flight (top 10):")
        missing = (df.isna().mean() * 100).sort_values(ascending=False)
        print(missing.head(10).round(1).to_dict())
        seen.add(hardware_name)

    # Flight-level metadata and predefined train/test splits.
    print("\nFLIGHT METADATA:")
    metadata_path = DATA_DIR / "flight_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        print(f"Rows: {len(metadata)}")
        print("Columns:", metadata.columns.tolist())
        if "hardware_config" in metadata.columns:
            print("Flights by hardware:")
            print(metadata["hardware_config"].value_counts().to_dict())
        if "exclude" in metadata.columns:
            print("Exclude counts:")
            print(metadata["exclude"].value_counts(dropna=False).to_dict())
    else:
        print("flight_metadata.csv not found.")

    print("\nDATA SPLITS:")
    splits_path = DATA_DIR / "splits.csv"
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        print("Columns:", splits.columns.tolist())
        print("Rows by split:")
        print(splits["split"].value_counts(dropna=False).to_dict())
        if "hardware_config" in splits.columns:
            print("Hardware/split counts:")
            print(
                splits.groupby(["hardware_config", "split"])
                .size().to_dict()
            )
    else:
        print("splits.csv not found.")

    print("\nFEATURE CHANNEL CONCEPTS:")
    for name in ("holybro", "speedybee"):
        path = DATA_DIR / f"feature_channels_final_{name}.csv"
        if path.exists():
            features = pd.read_csv(path)
            print(f"{name}: {len(features)} feature entries")
            if "concept" in features.columns:
                print(features["concept"].value_counts().to_dict())

    print("\nSummary complete.")

if __name__ == "__main__":
    main()
