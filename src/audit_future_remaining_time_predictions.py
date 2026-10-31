
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "reports/future_remaining_time_30s_test_predictions.csv"
OUTPUT = ROOT / "reports/future_remaining_time_per_flight_audit.csv"

df = pd.read_csv(FILE)

target = "future_remaining_time_s"
model_col = "model_prediction_s"
baseline_col = "persistence_prediction_s"

df["model_error_s"] = df[model_col] - df[target]
df["baseline_error_s"] = df[baseline_col] - df[target]

df["model_abs_error_s"] = df["model_error_s"].abs()
df["baseline_abs_error_s"] = df["baseline_error_s"].abs()

# --------------------------------------------------
# Overall comparison
# --------------------------------------------------

model_mae = mean_absolute_error(df[target], df[model_col])
baseline_mae = mean_absolute_error(df[target], df[baseline_col])

model_rmse = np.sqrt(
    mean_squared_error(df[target], df[model_col])
)

baseline_rmse = np.sqrt(
    mean_squared_error(df[target], df[baseline_col])
)

improvement = (
    (baseline_mae - model_mae) / baseline_mae
) * 100

print("=" * 65)
print("30-SECOND FORECAST PERFORMANCE AUDIT")
print("=" * 65)

print("\nOverall results")
print("Test windows:", len(df))
print("Test flights:", df["flight_id"].nunique())

print("\nModel")
print("MAE:", round(model_mae, 3), "seconds")
print("RMSE:", round(model_rmse, 3), "seconds")

print("\nPersistence baseline")
print("MAE:", round(baseline_mae, 3), "seconds")
print("RMSE:", round(baseline_rmse, 3), "seconds")

print("\nMAE improvement:", round(improvement, 2), "%")

print("\nModel absolute-error percentiles:")
print(
    df["model_abs_error_s"]
    .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    .round(3)
    .to_string()
)

# --------------------------------------------------
# Per-flight performance
# --------------------------------------------------

rows = []

for flight_id, group in df.groupby("flight_id"):

    model_flight_mae = group["model_abs_error_s"].mean()
    baseline_flight_mae = group["baseline_abs_error_s"].mean()

    rows.append({
        "flight_id": int(flight_id),
        "windows": len(group),
        "target_median_s": group[target].median(),
        "model_mae_s": model_flight_mae,
        "baseline_mae_s": baseline_flight_mae,
        "improvement_pct": (
            (baseline_flight_mae - model_flight_mae)
            / baseline_flight_mae * 100
            if baseline_flight_mae > 0 else np.nan
        ),
        "model_max_error_s": group["model_abs_error_s"].max(),
        "model_error_over_30s_pct": (
            group["model_abs_error_s"].gt(30).mean() * 100
        )
    })

per_flight = pd.DataFrame(rows).sort_values(
    "model_mae_s", ascending=False
)

per_flight.to_csv(OUTPUT, index=False)

print("\nPer-flight performance:")
print(per_flight.round(3).to_string(index=False))

print("\nFlights where model beats persistence:",
      int((per_flight["improvement_pct"] > 0).sum()),
      "/", len(per_flight))

print("\nWorst 5 flights:")
print(per_flight.head(5).round(3).to_string(index=False))

print("\nBest 5 flights:")
print(
    per_flight.tail(5)
    .sort_values("model_mae_s")
    .round(3)
    .to_string(index=False)
)

print("\nSaved:", OUTPUT)
