from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

splits = pd.read_csv(DATA / "splits.csv")
metadata = pd.read_csv(DATA / "flight_metadata.csv")

speedy_splits = splits[
    splits["hardware_config"] == "speedybee"
].copy()

speedy_metadata = metadata[
    metadata["hardware_config"] == "speedybee"
].copy()

speedy_splits["expected_file"] = (
    speedy_splits["flight_id"]
    .astype(int)
    .map(lambda x: f"lap_{x:03d}.parquet")
)

speedy_splits["file_exists"] = speedy_splits[
    "expected_file"
].map(lambda x: (DATA / "speedybee" / x).exists())

print("=== SPEEDYBEE FILE MAPPING ===")
print(speedy_splits[
    ["flight_id", "split", "expected_file", "file_exists"]
].sort_values("flight_id").to_string(index=False))

print("\n=== MAPPING VALIDATION ===")
print("Split entries:", len(speedy_splits))
print("Matching files:", speedy_splits["file_exists"].sum())
print("Missing files:", (~speedy_splits["file_exists"]).sum())

print("\n=== METADATA ID VALIDATION ===")
print(
    "Metadata IDs match split IDs:",
    set(speedy_metadata["flight_id"]) ==
    set(speedy_splits["flight_id"])
)