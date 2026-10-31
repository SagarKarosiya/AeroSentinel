from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_csv(
    ROOT / "reports" / "dataset_manifest.csv"
)

features = pd.read_csv(
    ROOT / "data" / "processed" / "feature_channels_final_holybro.csv"
)["raw_column"].dropna().unique().tolist()

flights = manifest[
    (manifest["hardware_config"].str.lower() == "holybro")
    & (~manifest["exclude"].astype(bool))
]

results = []

for _, flight in flights.iterrows():

    path = Path(flight["telemetry_path"])

    if not path.is_absolute():
        path = ROOT / path

    columns = set(pd.read_parquet(path).columns)

    present = [
        f for f in features
        if f in columns or f"{f}__gps" in columns
    ]

    results.append({
        "flight_id": flight["flight_id"],
        "split": flight["split"],
        "duration_s": flight["duration_s"],
        "features_present": len(present),
        "features_missing": len(features) - len(present),
    })

report = pd.DataFrame(results)

print("\nHOLYBRO FLIGHT FEATURE AVAILABILITY")
print("=" * 65)

print(
    report.sort_values("features_present").to_string(index=False)
)

print("\nSUMMARY BY SPLIT")
print(
    report.groupby("split")["features_present"]
    .agg(["count", "min", "max", "mean"])
    .to_string()
)

output = ROOT / "reports" / "holybro_flight_feature_availability.csv"
report.to_csv(output, index=False)

print("\nSaved:", output)