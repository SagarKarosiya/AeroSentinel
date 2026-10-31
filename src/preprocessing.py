
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Check basic data integrity."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if df.columns.isna().any():
        raise ValueError("Missing column names detected.")

    return df


def prepare_numeric_features(
    df: pd.DataFrame,
    exclude_columns=None,
) -> pd.DataFrame:
    """Select numeric columns, excluding IDs and targets."""
    df = validate_dataframe(df.copy())
    exclude_columns = set(exclude_columns or [])

    features = df.select_dtypes(include="number")
    features = features.drop(
        columns=[
            col for col in exclude_columns
            if col in features.columns
        ],
        errors="ignore",
    )

    if features.empty:
        raise ValueError("No numeric feature columns found.")

    return features


def save_processed_sample(df: pd.DataFrame, filename: str):
    """Save a small or intentionally prepared output dataset."""
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Preprocessing module ready.")
