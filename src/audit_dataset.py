
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

METADATA_HINTS = (
    "dictionary",
    "flight_stats",
    "flight_summary",
    "missingness",
    "candidates",
    "metadata",
    "manifest",
    "splits",
    "window_index",
)


def main():
    files = sorted(
        DATA_DIR.rglob("*.csv"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {DATA_DIR}"
        )

    print(f"CSV files found: {len(files)}")
    print("\nFiles sorted by size (largest first):")

    for path in files:
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"{size_mb:9.2f} MB  {path.relative_to(ROOT)}")

    print("\n--- Largest-file schema audit ---")

    for path in files[:10]:
        try:
            sample = pd.read_csv(path, nrows=5)
            lower_name = path.name.lower()

            if any(hint in lower_name for hint in METADATA_HINTS):
                kind = "METADATA / SUMMARY"
            else:
                kind = "POTENTIAL TELEMETRY"

            print(f"\nFile: {path.relative_to(ROOT)}")
            print(f"Classification: {kind}")
            print(f"Columns ({len(sample.columns)}):")
            print(sample.columns.tolist())
            print("Sample:")
            print(sample.head(2).to_string(index=False))

        except Exception as exc:
            print(f"\nCould not inspect {path.name}: {exc}")


if __name__ == "__main__":
    main()
