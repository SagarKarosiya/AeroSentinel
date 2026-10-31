from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

CONFIGS = {
    "holybro": "flight_*.parquet",
    "speedybee": "lap_*.parquet",
}


def audit_hardware(hardware, pattern):
    folder = DATA / hardware
    files = sorted(folder.glob(pattern))

    records = []

    print(f"\n{'=' * 60}")
    print(f"{hardware.upper()} TIMESTAMP AUDIT")
    print("=" * 60)

    for file in files:
        try:
            df = pd.read_parquet(file, columns=["timestamp_s"])

            ts = pd.to_numeric(
                df["timestamp_s"], errors="coerce"
            ).dropna()

            if len(ts) < 2:
                continue

            differences = ts.diff().dropna()
            positive = differences[differences > 0]

            median_interval = (
                positive.median() if len(positive) else np.nan
            )

            records.append({
                "flight": file.name,
                "rows": len(ts),
                "duration_s": ts.max() - ts.min(),
                "median_interval_s": median_interval,
                "estimated_hz": (
                    1 / median_interval
                    if pd.notna(median_interval)
                    and median_interval > 0
                    else np.nan
                ),
                "non_increasing_steps": int(
                    (differences <= 0).sum()
                ),
                "missing_timestamps": int(
                    df["timestamp_s"].isna().sum()
                ),
            })

        except Exception as error:
            print(f"ERROR {file.name}: {error}")

    result = pd.DataFrame(records)

    if result.empty:
        print("No valid timestamp data.")
        return

    print("Flights audited:", len(result))
    print("Median sampling frequency:",
          round(result["estimated_hz"].median(), 3), "Hz")
    print("Minimum frequency:",
          round(result["estimated_hz"].min(), 3), "Hz")
    print("Maximum frequency:",
          round(result["estimated_hz"].max(), 3), "Hz")
    print("Median flight duration:",
          round(result["duration_s"].median(), 2), "seconds")
    print("Flights with non-increasing timestamps:",
          (result["non_increasing_steps"] > 0).sum())
    print("Flights with missing timestamps:",
          (result["missing_timestamps"] > 0).sum())

    print("\nSAMPLING FREQUENCY DISTRIBUTION:")
    print(
        result["estimated_hz"]
        .round(2)
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nFIRST 10 FLIGHTS:")
    print(result.head(10).to_string(index=False))

    output = ROOT / "reports" / f"{hardware}_timestamp_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print("\nFull report saved:", output.relative_to(ROOT))


def main():
    for hardware, pattern in CONFIGS.items():
        audit_hardware(hardware, pattern)


if __name__ == "__main__":
    main()