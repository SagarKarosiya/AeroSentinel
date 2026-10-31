
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules"}
EXTENSIONS = {
    ".csv", ".parquet", ".feather", ".json", ".jsonl",
    ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5",
    ".ulg", ".bag", ".zip"
}
HINTS = (
    "dictionary", "flight_stats", "flight_summary",
    "missingness", "candidates", "metadata", "manifest",
    "splits", "window_index"
)

def main():
    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in EXTENSIONS:
            files.append((path.stat().st_size, path))

    files.sort(reverse=True, key=lambda x: x[0])

    lines = [
        f"Project root: {ROOT}",
        f"Data files found: {len(files)}",
        "",
        "TOP 15 LARGEST FILES:"
    ]

    for size, path in files[:15]:
        lines.append(
            f"{size / (1024 ** 2):.2f} MB | {path.relative_to(ROOT)}"
        )

    lines.append("\nSCHEMA PREVIEWS:")

    count = 0
    for size, path in files:
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        if any(h in path.name.lower() for h in HINTS):
            continue

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=2)
            else:
                df = pd.read_parquet(path).head(2)

            lines.extend([
                "",
                f"File: {path.relative_to(ROOT)}",
                f"Size: {size / (1024 ** 2):.2f} MB",
                f"Columns: {df.columns.tolist()[:15]}",
                f"Sample: {df.head(1).to_dict(orient='records')}"
            ])
            count += 1
        except Exception as exc:
            lines.append(f"Could not inspect {path.name}: {exc}")

        if count >= 5:
            break

    report_path = ROOT / "reports" / "telemetry_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:20]))
    print(f"\nFull report saved to: {report_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
