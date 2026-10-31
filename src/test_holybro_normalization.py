from data_loader import load_flight
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

features = set(
    pd.read_csv(
        ROOT / "data" / "processed" /
        "feature_channels_final_holybro.csv"
    )["raw_column"].dropna()
)

for flight_id in [1, 117, 118, 119, 120]:

    df, metadata = load_flight("holybro", flight_id)

    matched = features.intersection(df.columns)

    print(
        f"Flight {flight_id}: "
        f"{len(matched)}/{len(features)} features matched"
    )

    assert len(matched) == 201, (
        f"Feature matching failed for flight {flight_id}"
    )

print("\nAll Holybro normalization tests passed!")