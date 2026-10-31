
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def run_eda(csv_path: str, sample_rows: int = 100_000):
    csv_path = Path(csv_path)

    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    df = pd.read_csv(csv_path, nrows=sample_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Shape of loaded sample:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nStatistics:\n", df.describe(include="all").T)

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        numeric.hist(figsize=(14, 10), bins=30)
        plt.tight_layout()

        output_path = REPORT_DIR / "feature_distributions.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved plot: {output_path}")

    print("EDA completed.")


if __name__ == "__main__":
    files = sorted(DATA_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    run_eda(str(files[0]))
