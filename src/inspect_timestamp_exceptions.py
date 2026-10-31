from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "processed" / "holybro"

files = sorted(FOLDER.glob("flight_*.parquet"))

print("Total Holybro files:", len(files))

for file in files:
    try:
        df = pd.read_parquet(file, columns=["timestamp_s"])
        ts = pd.to_numeric(df["timestamp_s"], errors="coerce").dropna()

        if len(ts) < 2:
            print(
                f"CHECK: {file.name} | "
                f"Rows: {len(df)} | Valid timestamps: {len(ts)}"
            )

    except Exception as error:
        print(f"ERROR: {file.name} | {error}")