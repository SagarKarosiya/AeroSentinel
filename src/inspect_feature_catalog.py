
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def inspect_catalog(hardware, catalog_file, flight_id):

    print("\n" + "=" * 65)
    print(f"{hardware.upper()} FEATURE CATALOG")
    print("=" * 65)

    catalog_path = PROCESSED / catalog_file

    if not catalog_path.exists():
        print(f"Catalog not found: {catalog_path}")
        return

    catalog = pd.read_csv(catalog_path)

    print("\nCatalog columns:", list(catalog.columns))
    print("Total catalog entries:", len(catalog))

    df, metadata = __import__(
        "data_loader"
    ).load_flight(hardware, flight_id)

    if "raw_column" not in catalog.columns:
        print("ERROR: raw_column is missing.")
        return

    catalog["raw_column"] = catalog["raw_column"].astype(str)

    print("\nFeature availability in sample flight:")

    for _, row in catalog.iterrows():

        column = row["raw_column"]

        if column not in df.columns:
            status = "NOT FOUND"
        elif not pd.api.types.is_numeric_dtype(df[column]):
            status = "NON-NUMERIC"
        else:
            valid = df[column].notna().sum()
            status = f"NUMERIC | {valid}/{len(df)} valid rows"

        concept = row.get("concept", "unknown")

        print(f"{concept:22} | {column:55} | {status}")

    if "concept" in catalog.columns:
        print("\nFeature count by concept:")
        print(catalog["concept"].value_counts().to_string())

    del df


if __name__ == "__main__":

    inspect_catalog(
        "holybro",
        "feature_channels_final_holybro.csv",
        1
    )

    inspect_catalog(
        "speedybee",
        "feature_channel_candidates_speedybee.csv",
        1
    )
