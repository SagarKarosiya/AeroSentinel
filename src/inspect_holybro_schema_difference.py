from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_csv(ROOT / "reports" / "dataset_manifest.csv")

features = pd.read_csv(
    ROOT / "data" / "processed" / "feature_channels_final_holybro.csv"
)["raw_column"].dropna().tolist()

def get_flight(flight_id):
    row = manifest[
        (manifest["hardware_config"].str.lower() == "holybro")
        & (manifest["flight_id"] == flight_id)
    ].iloc[0]

    path = Path(row["telemetry_path"])

    if not path.is_absolute():
        path = ROOT / path

    return pd.read_parquet(path), path

# Compare one normal flight with one affected flight.
normal, normal_path = get_flight(1)
affected, affected_path = get_flight(117)

normal_columns = set(normal.columns)
affected_columns = set(affected.columns)

print("\n--- SCHEMA COMPARISON ---")
print("Normal flight columns:", len(normal_columns))
print("Affected flight columns:", len(affected_columns))
print("Common columns:", len(normal_columns & affected_columns))

print("\nColumns only in normal flight:")
print(sorted(normal_columns - affected_columns)[:50])

print("\nColumns only in affected flight:")
print(sorted(affected_columns - normal_columns)[:50])

print("\n--- CURATED FEATURE MATCHES ---")
print("Normal flight matches:", len(normal_columns & set(features)))
print("Affected flight matches:", len(affected_columns & set(features)))

print("\n--- SAMPLE COLUMN NAMES ---")
print("Normal:", sorted(normal_columns)[:30])
print("Affected:", sorted(affected_columns)[:30])