from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

for hardware in ["holybro", "speedybee"]:
    quality_path = REPORTS / f"{hardware}_feature_quality.csv"

    quality = pd.read_csv(quality_path)

    if hardware == "holybro":
        curated_path = (
            PROCESSED / "feature_channels_final_holybro.csv"
        )
    else:
        curated_path = (
            PROCESSED / "feature_channel_candidates_speedybee.csv"
        )

    curated = pd.read_csv(curated_path)

    # Match the curated raw column names against the quality report.
    review = curated.merge(
        quality,
        left_on="raw_column",
        right_on="feature",
        how="left"
    )

    review["flight_coverage_percent"] = (
        review["flights_present"] / (118 if hardware == "holybro" else 110)
    ) * 100

    review = review.sort_values(
        ["concept", "missing_percent"],
        ascending=[True, True]
    )

    output = REPORTS / f"{hardware}_curated_feature_review.csv"
    review.to_csv(output, index=False)

    print("\n" + "=" * 55)
    print(f"{hardware.upper()} CURATED FEATURE REVIEW")
    print("=" * 55)

    print("Curated features:", len(curated))
    print("Matched features:", review["feature"].notna().sum())

    print("\nFEATURE COUNTS BY CONCEPT:")
    print(review.groupby("concept").size().to_string())

    print("\nFEATURES WITH BEST FLIGHT COVERAGE:")
    columns = [
        "concept",
        "raw_column",
        "flights_present",
        "missing_percent",
        "flight_coverage_percent"
    ]

    print(
        review[columns]
        .sort_values(
            ["flight_coverage_percent", "missing_percent"],
            ascending=[False, True]
        )
        .head(30)
        .to_string(index=False)
    )

    print("\nSaved:", output)