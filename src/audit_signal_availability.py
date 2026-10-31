from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

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


def find_column(candidate, actual_columns):
    if candidate in actual_columns:
        return candidate

    # SpeedyBee GPS columns use the __gps suffix.
    gps_candidate = candidate + "__gps"

    if gps_candidate in actual_columns:
        return gps_candidate

    return None


def audit_hardware(hardware, config):
    folder = DATA / hardware
    candidate_file = DATA / config["candidate"]

    print(f"\n{'=' * 55}")
    print(f"{hardware.upper()} SUMMARY")
    print("=" * 55)

    files = sorted(folder.glob(config["pattern"]))
    candidates = pd.read_csv(candidate_file)

    print("Available flights:", len(files))
    print("Candidate entries:", len(candidates))

    sample_files = files[:5]
    results = {}

    for _, row in candidates.iterrows():
        candidate = str(row["raw_column"])
        concept = str(row["concept"])

        results[candidate] = {
            "concept": concept,
            "matched": 0,
            "actual_name": None,
        }

    for file in sample_files:
        df = pd.read_parquet(file)
        columns = set(df.columns)

        for candidate in results:
            actual = find_column(candidate, columns)

            if actual:
                results[candidate]["matched"] += 1
                results[candidate]["actual_name"] = actual

    matched = sum(
        1 for item in results.values()
        if item["matched"] > 0
    )

    print("Flights inspected:", len(sample_files))
    print("Candidates matched:", matched)
    print("Candidates unmatched:", len(results) - matched)

    print("\nMATCHES BY CONCEPT:")

    summary = {}

    for item in results.values():
        concept = item["concept"]
        summary.setdefault(concept, [0, 0])

        if item["matched"] > 0:
            summary[concept][0] += 1

        summary[concept][1] += 1

    for concept, (found, total) in sorted(summary.items()):
        print(f"{concept}: {found}/{total}")

    print("\nUNMATCHED SIGNALS:")

    for name, item in results.items():
        if item["matched"] == 0:
            print(name)

    if sample_files:
        df = pd.read_parquet(sample_files[0])

        print("\nTIMESTAMP INFORMATION:")
        print("First flight:", sample_files[0].name)

        if "timestamp_s" in df.columns:
            print("Timestamp column: timestamp_s")
            print("First values:", df["timestamp_s"].head(5).tolist())
            print("Rows:", len(df))
            print(
                "Duration (seconds):",
                round(
                    df["timestamp_s"].max()
                    - df["timestamp_s"].min(),
                    3
                )
            )


def main():
    for hardware, config in CONFIGS.items():
        audit_hardware(hardware, config)


if __name__ == "__main__":
    main()