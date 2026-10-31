from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "dataset_manifest.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

manifest = pd.read_csv(MANIFEST)

configs = {
    "holybro": PROCESSED / "feature_channels_final_holybro.csv",
    "speedybee": PROCESSED / "feature_channel_candidates_speedybee.csv",
}

for hardware, feature_file in configs.items():

    print("\n" + "=" * 60)
    print(f"{hardware.upper()} CURATED FEATURE AUDIT")
    print("=" * 60)

    features_df = pd.read_csv(feature_file)
    features = features_df["raw_column"].dropna().unique().tolist()

    flights = manifest[
        (manifest["hardware_config"].str.lower() == hardware)
        & (~manifest["exclude"].astype(bool))
    ]

    stats = {
        feature: {
            "flights_present": 0,
            "absent_flights": 0,
            "all_null_flights": 0,
            "total_rows": 0,
            "null_values": 0,
            "dtype": None,
        }
        for feature in features
    }

    for _, flight in flights.iterrows():

        path = Path(flight["telemetry_path"])

        if not path.is_absolute():
            path = ROOT / path

        df = pd.read_parquet(path)
        columns = set(df.columns)

        for feature in features:

            actual = feature

            if actual not in columns and f"{feature}__gps" in columns:
                actual = f"{feature}__gps"

            if actual not in columns:
                stats[feature]["absent_flights"] += 1
                continue

            series = df[actual]

            item = stats[feature]
            item["flights_present"] += 1
            item["total_rows"] += len(series)
            item["null_values"] += int(series.isna().sum())
            item["dtype"] = str(series.dtype)

            if series.isna().all():
                item["all_null_flights"] += 1

    results = []

    for feature, item in stats.items():

        total = item["total_rows"]
        nulls = item["null_values"]

        results.append({
            "feature": feature,
            **item,
            "missing_percent": (
                nulls / total * 100 if total else 100.0
            ),
            "flight_coverage_percent": (
                item["flights_present"] / len(flights) * 100
            ),
            "present_in_all_flights": (
                item["flights_present"] == len(flights)
            ),
        })

    report = pd.DataFrame(results)

    report = report.sort_values(
        ["flight_coverage_percent", "missing_percent"],
        ascending=[False, True]
    )

    output = REPORTS / f"{hardware}_curated_quality_full.csv"
    report.to_csv(output, index=False)

    print("Usable flights:", len(flights))
    print("Curated features:", len(features))
    print(
        "Present in every flight:",
        report["present_in_all_flights"].sum()
    )
    print(
        "Absent from at least one flight:",
        (report["absent_flights"] > 0).sum()
    )
    print(
        "100% missing overall:",
        (report["missing_percent"] == 100).sum()
    )

    print("\nTOP 25 FEATURES BY FLIGHT COVERAGE:")

    print(
        report[
            [
                "feature",
                "flights_present",
                "absent_flights",
                "all_null_flights",
                "missing_percent",
                "flight_coverage_percent",
            ]
        ].head(25).to_string(index=False)
    )

    print("\nSaved:", output)