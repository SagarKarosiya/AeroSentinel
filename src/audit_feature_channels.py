
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

def main():
    for hardware in ("holybro", "speedybee"):
        path = DATA / f"feature_channels_final_{hardware}.csv"

        print(f"\n{'=' * 60}")
        print(f"{hardware.upper()} FEATURE AUDIT")
        print("=" * 60)

        if not path.exists():
            print("Feature file not found:", path)
            continue

        df = pd.read_csv(path)

        print("Total feature entries:", len(df))
        print("Columns:", df.columns.tolist())

        for col in ("concept", "source_topic", "dtype"):
            if col in df.columns:
                print(f"\n{col.upper()} DISTRIBUTION:")
                print(df[col].value_counts(dropna=False).head(20))

        if "raw_column" in df.columns:
            print("\nFIRST 30 RAW FEATURES:")
            print(df["raw_column"].head(30).to_list())

        print("\nMissing values:")
        print(df.isna().sum().to_dict())

if __name__ == "__main__":
    main()
