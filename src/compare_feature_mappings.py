from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

FILES = {
    "Holybro candidates": DATA / "feature_channel_candidates_holybro.csv",
    "SpeedyBee candidates": DATA / "feature_channel_candidates_speedybee.csv",
    "Holybro dictionary": DATA / "data_dictionary_holybro.csv",
    "SpeedyBee dictionary": DATA / "data_dictionary_speedybee.csv",
}

def inspect_file(label, path):
    print(f"\n{'=' * 65}")
    print(label.upper())
    print("=" * 65)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return None

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    for col in ["concept", "source_topic", "dtype"]:
        if col in df.columns:
            print(f"\n{col.upper()} DISTRIBUTION:")
            print(df[col].value_counts(dropna=False).head(15))

    for col in ["raw_column", "base_signal_name"]:
        if col in df.columns:
            print(f"\nFIRST 25 {col.upper()} VALUES:")
            print(df[col].head(25).to_list())

    print("\nMissing values:")
    print(df.isna().sum().to_dict())

    return df

def main():
    results = {}

    for label, path in FILES.items():
        results[label] = inspect_file(label, path)

    holybro = results["Holybro candidates"]
    speedybee = results["SpeedyBee candidates"]

    if holybro is not None and speedybee is not None:
        if "base_signal_name" in holybro.columns and \
           "base_signal_name" in speedybee.columns:

            h = set(holybro["base_signal_name"].dropna().astype(str))
            s = set(speedybee["base_signal_name"].dropna().astype(str))

            print("\n" + "=" * 65)
            print("COMMON FEATURE COMPARISON")
            print("=" * 65)

            print("Holybro unique signals:", len(h))
            print("SpeedyBee unique signals:", len(s))
            print("Common signals:", len(h & s))
            print("Holybro-only signals:", len(h - s))
            print("SpeedyBee-only signals:", len(s - h))

            print("\nCOMMON SIGNALS:")
            print(sorted(h & s))

            print("\nHOLYBRO-ONLY SIGNALS:")
            print(sorted(h - s)[:50])

            print("\nSPEEDYBEE-ONLY SIGNALS:")
            print(sorted(s - h)[:50])

if __name__ == "__main__":
    main()