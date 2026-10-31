from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

configs = {
    "holybro": (
        PROCESSED / "feature_channels_final_holybro.csv",
        REPORTS / "holybro_curated_quality_full.csv",
        118,
    ),
    "speedybee": (
        PROCESSED / "feature_channel_candidates_speedybee.csv",
        REPORTS / "speedybee_curated_quality_full.csv",
        110,
    ),
}

for hardware, (feature_path, quality_path, total_flights) in configs.items():

    features = pd.read_csv(feature_path)
    quality = pd.read_csv(quality_path)

    report = features.merge(
        quality,
        left_on="raw_column",
        right_on="feature",
        how="left"
    )

    report["coverage_percent"] = (
        report["flights_present"] / total_flights * 100
    )

    print("\n" + "=" * 60)
    print(f"{hardware.upper()} COVERAGE BY SENSOR GROUP")
    print("=" * 60)

    summary = report.groupby("concept").agg(
        total_features=("raw_column", "count"),
        available_features=("flights_present", "count"),
        mean_coverage=("coverage_percent", "mean"),
        fully_available=("present_in_all_flights", "sum"),
        zero_missing=("missing_percent", lambda x: (x == 0).sum()),
    ).round(2)

    print(summary.to_string())

    print("\nFEATURES WITH AT LEAST 90% FLIGHT COVERAGE:")

    selected = report[
        report["coverage_percent"] >= 90
    ].sort_values(
        ["concept", "coverage_percent"],
        ascending=[True, False]
    )

    print(
        selected[
            [
                "concept",
                "raw_column",
                "flights_present",
                "coverage_percent",
                "missing_percent",
            ]
        ].to_string(index=False)
    )

    print("\nFEATURES WITH 100% MISSING VALUES:")

    print(
        report.loc[
            report["missing_percent"] == 100,
            ["concept", "raw_column", "flights_present"]
        ].to_string(index=False)
    )