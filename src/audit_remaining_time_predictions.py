
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "reports/remaining_time_test_predictions.csv"
DATASET = ROOT / "data/features/holybro_remaining_time_dataset.parquet"
OUTPUT = ROOT / "reports/remaining_time_per_flight_audit.csv"
SUMMARY = ROOT / "reports/remaining_time_prediction_audit.txt"

df = pd.read_csv(PREDICTIONS)
dataset = pd.read_parquet(DATASET)

TARGET = "target_remaining_time_s"
PRED = "predicted_remaining_time_s"

df["error_s"] = df[PRED] - df[TARGET]
df["absolute_error_s"] = df["error_s"].abs()

print("=" * 60)
print("REMAINING-TIME TEST PREDICTION AUDIT")
print("=" * 60)

print("\nOverall test performance")
print("Windows:", len(df))
print("Flights:", df["flight_id"].nunique())
print("MAE:", round(df["absolute_error_s"].mean(), 3), "seconds")
print(
    "RMSE:",
    round(np.sqrt(mean_squared_error(df[TARGET], df[PRED])), 3),
    "seconds"
)
print("R2:", round(r2_score(df[TARGET], df[PRED]), 5))

print("\nAbsolute error distribution")
print(df["absolute_error_s"].describe(
    percentiles=[0.5, 0.9, 0.95, 0.99]
).round(3))

# --------------------------------------------------
# Per-flight performance
# --------------------------------------------------

rows = []

for flight_id, group in df.groupby("flight_id"):

    mae = mean_absolute_error(group[TARGET], group[PRED])
    rmse = np.sqrt(mean_squared_error(group[TARGET], group[PRED]))

    rows.append({
        "flight_id": int(flight_id),
        "windows": len(group),
        "target_median_s": group[TARGET].median(),
        "prediction_median_s": group[PRED].median(),
        "mae_s": mae,
        "rmse_s": rmse,
        "max_absolute_error_s": group["absolute_error_s"].max(),
        "error_over_10s_pct": (
            group["absolute_error_s"].gt(10).mean() * 100
        )
    })

per_flight = pd.DataFrame(rows).sort_values("mae_s", ascending=False)

per_flight.to_csv(OUTPUT, index=False)

print("\nPer-flight results:")
print(per_flight.round(3).to_string(index=False))

print("\nWorst 5 flights:")
print(per_flight.head(5).round(3).to_string(index=False))

# --------------------------------------------------
# Check target dependence on elapsed time
# --------------------------------------------------

test_windows = dataset[dataset["split"] == "test"].copy()

print("\nTarget relationship with window timing:")

for col in ["window_start_s", "window_end_s"]:
    if col in test_windows.columns:
        corr = test_windows[col].corr(test_windows[TARGET])
        print(f"{col} correlation: {corr:.5f}")

# --------------------------------------------------
# Check potential feature/target correlations
# --------------------------------------------------

excluded = {
    TARGET, "hardware_config", "flight_id", "split",
    "window_start_s", "window_end_s",
    "window_duration_s", "target_observations"
}

numeric = test_windows.select_dtypes(include=[np.number])

correlations = numeric.corr()[TARGET].drop(TARGET)
correlations = correlations.drop(
    labels=[c for c in excluded if c in correlations.index],
    errors="ignore"
)

print("\nTop 15 absolute feature-target correlations in test:")
print(
    correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    ).head(15).round(5).to_string()
)

# --------------------------------------------------
# Write audit summary
# --------------------------------------------------

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write("REMAINING-TIME PREDICTION AUDIT\n")
    f.write(f"Test windows: {len(df)}\n")
    f.write(f"Test flights: {df['flight_id'].nunique()}\n")
    f.write(f"Overall MAE: {df['absolute_error_s'].mean():.4f} seconds\n")
    f.write(
        f"Overall RMSE: "
        f"{np.sqrt(mean_squared_error(df[TARGET], df[PRED])):.4f} seconds\n"
    )
    f.write(f"R2: {r2_score(df[TARGET], df[PRED]):.6f}\n")
    f.write(f"Flights with MAE above 10 seconds: "
            f"{(per_flight['mae_s'] > 10).sum()}\n")

print("\nReports saved:")
print(OUTPUT)
print(SUMMARY)
