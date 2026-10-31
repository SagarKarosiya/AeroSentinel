
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "holybro": {
        "flights": 118,
        "allowed_concepts": {
            "battery",
            "attitude",
            "imu",
            "actuator_output",
            "gps_position",
            "gps_receiver",
        },
    },
    "speedybee": {
        "flights": 110,
        "allowed_concepts": {
            "battery",
            "attitude",
            "imu_accel",
            "imu_gyro",
            "actuator_output",
            "altitude",
        },
    },
}

all_shortlists = []

for hardware, settings in CONFIG.items():
    source = ROOT / "reports" / f"feature_coverage_{hardware}.csv"

    df = pd.read_csv(source)

    # Keep only features that have values in every flight.
    selected = df[
        (df["flights_with_values"] == settings["flights"])
        & (df["concept"].isin(settings["allowed_concepts"]))
    ].copy()

    selected["hardware_config"] = hardware

    selected = selected.sort_values(
        ["concept", "raw_column"]
    )

    output = ROOT / "reports" / f"feature_shortlist_{hardware}.csv"
    selected.to_csv(output, index=False)

    all_shortlists.append(selected)

    print("\n" + "=" * 60)
    print(f"{hardware.upper()} STABLE FEATURE SHORTLIST")
    print("=" * 60)

    print(f"Selected features: {len(selected)}")
    print("\nFeatures by concept:")
    print(selected.groupby("concept").size().to_string())

    print("\nSelected raw columns:")
    for concept, group in selected.groupby("concept"):
        print(f"\n[{concept}]")
        for column in group["raw_column"]:
            print(f"  - {column}")

    print(f"\nSaved: {output}")

combined = pd.concat(all_shortlists, ignore_index=True)

combined_path = ROOT / "reports" / "feature_shortlist_all.csv"
combined.to_csv(combined_path, index=False)

print("\n" + "=" * 60)
print("FEATURE SHORTLIST GENERATION COMPLETE")
print(f"Combined features: {len(combined)}")
print(f"Saved: {combined_path}")
