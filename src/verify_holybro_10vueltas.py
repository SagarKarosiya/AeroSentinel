from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_csv(
    ROOT / "reports" / "dataset_manifest.csv"
)

feature_df = pd.read_csv(
    ROOT / "data" / "processed" /
    "feature_channels_final_holybro.csv"
)

features = set(feature_df["raw_column"].dropna())

affected_ids = [117, 118, 119, 120]

for flight_id in affected_ids:

    row = manifest[
        (manifest["hardware_config"].str.lower() == "holybro") &
        (manifest["flight_id"] == flight_id)
    ].iloc[0]

    path = Path(row["telemetry_path"])

    if not path.is_absolute():
        path = ROOT / path

    df = pd.read_parquet(path)

    columns = set(df.columns)

    renamed = {
        col.replace("__10vueltas_", "__")
        for col in columns
    }

    matched = features & renamed

    print(f"\nFlight {flight_id}")
    print("-" * 40)
    print("Total columns:", len(columns))
    print("Original curated matches:", len(features & columns))
    print("Matches after prefix normalization:", len(matched))
    print("Unmatched curated features:", len(features - renamed))