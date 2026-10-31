from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

splits = pd.read_csv(DATA / "splits.csv")
metadata = pd.read_csv(DATA / "flight_metadata.csv")

speedy_splits = splits[splits["hardware_config"] == "speedybee"]
speedy_metadata = metadata[metadata["hardware_config"] == "speedybee"]

files = sorted((DATA / "speedybee").glob("*.parquet"))

print("\n=== SPEEDYBEE SPLIT FLIGHT IDs ===")
print("Total:", len(speedy_splits))
print(speedy_splits["flight_id"].head(20).to_list())

print("\n=== SPEEDYBEE METADATA FLIGHT IDs ===")
print("Total:", len(speedy_metadata))
print(speedy_metadata["flight_id"].head(20).to_list())

print("\n=== SPEEDYBEE PARQUET FILENAMES ===")
print("Total:", len(files))
print([f.stem for f in files[:20]])

print("\n=== METADATA COLUMNS ===")
print(speedy_metadata.columns.tolist())

print("\n=== SAMPLE SPEEDYBEE PARQUET SCHEMA ===")
if files:
    df = pd.read_parquet(files[0])
    print("File:", files[0].name)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string())