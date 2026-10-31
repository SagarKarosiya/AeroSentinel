import numpy as np
import pandas as pd

from data_loader import load_flight


def compare_columns(df, col_a, col_b):
    if col_a not in df.columns or col_b not in df.columns:
        print(f"  MISSING: {col_a} or {col_b}")
        return

    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    valid = a.notna() & b.notna()
    a = a[valid]
    b = b[valid]

    if len(a) == 0:
        print(f"  NO VALID PAIRS: {col_a} / {col_b}")
        return

    corr = a.corr(b) if len(a) > 1 else np.nan
    exact_pct = (np.isclose(a, b, rtol=1e-5, atol=1e-8)).mean() * 100

    print(f"\n  A: {col_a}")
    print(f"  B: {col_b}")
    print(f"  Valid paired rows: {len(a)}")
    print(f"  Correlation: {corr:.5f}")
    print(f"  Nearly identical values: {exact_pct:.2f}%")
    print(f"  A range: {a.min():.5f} to {a.max():.5f}")
    print(f"  B range: {b.min():.5f} to {b.max():.5f}")


print("\n" + "=" * 70)
print("HOLYBRO SIGNAL COMPARISON")
print("=" * 70)

df, metadata = load_flight("holybro", 1)

holybro_pairs = [
    (
        "lat__estimator_global_position_0",
        "latitude_deg__vehicle_gps_position_0",
    ),
    (
        "lon__estimator_global_position_0",
        "longitude_deg__vehicle_gps_position_0",
    ),
    (
        "alt__estimator_global_position_0",
        "altitude_msl_m__vehicle_gps_position_0",
    ),
    (
        "eph__estimator_global_position_0",
        "eph__vehicle_gps_position_0",
    ),
    (
        "epv__estimator_global_position_0",
        "epv__vehicle_gps_position_0",
    ),
    (
        "voltage_v__battery_status_0",
        "ocv_estimate__battery_status_0",
    ),
    (
        "voltage_v__battery_status_0",
        "ocv_estimate_filtered__battery_status_0",
    ),
    (
        "voltage_v__battery_status_0",
        "voltage_prediction__battery_status_0",
    ),
]

for a, b in holybro_pairs:
    compare_columns(df, a, b)

del df


print("\n" + "=" * 70)
print("SPEEDYBEE SIGNAL COMPARISON")
print("=" * 70)

df, metadata = load_flight("speedybee", 1)

speedybee_pairs = [
    ("gyroADC[0]", "gyroRaw[0]"),
    ("gyroADC[1]", "gyroRaw[1]"),
    ("gyroADC[2]", "gyroRaw[2]"),
    ("gyroADC[0]", "gyroPeakPitch[0]"),
    ("gyroADC[1]", "gyroPeakRoll[0]"),
    ("gyroADC[2]", "gyroPeakYaw[0]"),
]

for a, b in speedybee_pairs:
    compare_columns(df, a, b)

del df

print("\nSignal comparison completed.")