
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CONFIGS = {
    "holybro": {
        "pattern": "flight_*.parquet",
        "candidate": "feature_channel_candidates_holybro.csv",
    },
    "speedybee": {
        "pattern": "lap_*.parquet",
        "candidate": "feature_channel_candidates_speedybee.csv",
    },
}


def resolve_column(name, columns):
    if name in columns:
        return name

    gps_name = name + "__gps"
    if gps_name in columns:
        return gps_name

    return None


def audit_hardware(hardware, config, manifest):
    print(f"\n{'=' * 60}")
    print(f"{hardware.upper()} FEATURE QUALITY AUDIT")
    print("=" * 60)

    candidates = pd.read_csv(DATA / config["candidate"])
    candidate_names = candidates["raw_column"].drop_duplicates().astype(str)

    stats = {
        name: {
            "feature": name,
            "flights_present": 0,
            "all_null_flights": 0,
            "total_rows": 0,
            "null_values": 0,
            "dtype": None,
        }
        for name in candidate_names
    }

    flights = manifest[
        manifest["hardware_config"] == hardware
    ]

    for _, flight in flights.iterrows():
        path = ROOT / flight["telemetry_path"]
        df = pd.read_parquet(path)
        columns = set(df.columns)

        for name, item in stats.items():
            actual = resolve_column(name, columns)

            if actual is None:
                continue

            values = df[actual]
            null_count = int(values.isna().sum())

            item["flights_present"] += 1
            item["total_rows"] += len(values)
            item["null_values"] += null_count
            item["dtype"] = str(values.dtype)

            if null_count == len(values):
                item["all_null_flights"] += 1

    result = pd.DataFrame(stats.values())

    result["missing_percent"] = (
        result["null_values"] / result["total_rows"].replace(0, 1)
    ) * 100

    result["present_in_all_flights"] = (
        result["flights_present"] == len(flights)
    )

    result = result.sort_values(
        "missing_percent", ascending=False
    )

    REPORTS.mkdir(parents=True, exist_ok=True)

    output = REPORTS / f"{hardware}_feature_quality.csv"
    result.to_csv(output, index=False)

    print("Usable flights:", len(flights))
    print("Candidate features:", len(result))
    print(
        "Features present in every flight:",
        result["present_in_all_flights"].sum()
    )
    print(
        "Features absent from at least one flight:",
        (result["flights_present"] < len(flights)).sum()
    )
    print(
        "Features with 100% missing values overall:",
        (result["missing_percent"] == 100).sum()
    )

    print("\n20 FEATURES WITH HIGHEST MISSING PERCENTAGE:")
    print(
        result[
            [
                "feature",
                "flights_present",
                "all_null_flights",
                "missing_percent",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nReport saved:", output.relative_to(ROOT))


def main():
    manifest = pd.read_csv(
        REPORTS / "dataset_manifest.csv"
    )

    for hardware, config in CONFIGS.items():
        audit_hardware(hardware, config, manifest)


if __name__ == "__main__":
    main()
