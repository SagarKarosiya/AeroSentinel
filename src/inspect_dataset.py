
from pathlib import Path
import pandas as pd

# Project root: AeroSentinel/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}")
        print("Place your cleaned CSV files in data/processed/")
        return

    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    for file_path in csv_files:
        print("\n" + "=" * 65)
        print(f"FILE: {file_path.relative_to(PROJECT_ROOT)}")
        print("=" * 65)

        try:
            # Inspect only a small sample, not the full 1 GB file.
            sample = pd.read_csv(file_path, nrows=100)

            print("\nColumns:")
            print(sample.columns.tolist())

            print("\nData types:")
            print(sample.dtypes.to_string())

            print("\nSample shape (up to 100 rows):")
            print(sample.shape)

            print("\nSample memory usage:")
            print(
                f"{sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

            print("\nMissing values in sample:")
            print(sample.isna().sum().to_string())

            print("\nFirst 5 rows:")
            print(sample.head().to_string(index=False))

            print("\nTotal file size:")
            print(f"{file_path.stat().st_size / 1024**2:.2f} MB")

        except Exception as exc:
            print(f"Could not inspect file: {exc}")

if __name__ == "__main__":
    main()
