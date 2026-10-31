import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight


PAIRS = {
    "holybro": [
        ("lat__estimator_global_position_0",
         "latitude_deg__vehicle_gps_position_0"),
        ("lon__estimator_global_position_0",
         "longitude_deg__vehicle_gps_position_0"),
        ("alt__estimator_global_position_0",
         "altitude_msl_m__vehicle_gps_position_0"),
        ("voltage_v__battery_status_0",
         "voltage_prediction__battery_status_0"),
    ],
    "speedybee": [
        ("gyroADC[0]", "gyroRaw[0]"),
        ("gyroADC[1]", "gyroRaw[1]"),
        ("gyroADC[2]", "gyroRaw[2]"),
    ],
}


manifest = load_manifest()

for hardware, pairs in PAIRS.items():

    print("\n" + "=" * 75)
    print(hardware.upper(), "MULTI-FLIGHT COMPARISON")
    print("=" * 75)

    hardware_data = manifest[
        manifest["hardware_config"] == hardware
    ]

    selected = (
        hardware_data.sort_values("flight_id")
        .groupby("split", sort=True)
        .head(2)
    )

    results = []

    for _, record in selected.iterrows():

        flight_id = int(record["flight_id"])
        split = record["split"]

        print(f"Checking {hardware} flight {flight_id} ({split})")

        df, metadata = load_flight(hardware, flight_id)

        for col_a, col_b in pairs:

            if col_a not in df.columns or col_b not in df.columns:
                continue

            a = pd.to_numeric(df[col_a], errors="coerce")
            b = pd.to_numeric(df[col_b], errors="coerce")

            valid = a.notna() & b.notna()
            a = a[valid]
            b = b[valid]

            if len(a) < 2:
                continue

            results.append({
                "hardware": hardware,
                "flight_id": flight_id,
                "split": split,
                "signal_a": col_a,
                "signal_b": col_b,
                "correlation": a.corr(b),
                "near_equal_pct": (
                    np.isclose(a, b, rtol=1e-5, atol=1e-8).mean() * 100
                ),
                "paired_rows": len(a),
            })

        del df

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("No comparisons available.")
        continue

    print("\nPER-FLIGHT RESULTS")
    print(result_df.to_string(index=False))

    print("\nAGGREGATED RESULTS")

    summary = (
        result_df.groupby(["signal_a", "signal_b"])
        .agg(
            flights_checked=("flight_id", "count"),
            median_correlation=("correlation", "median"),
            min_correlation=("correlation", "min"),
            max_correlation=("correlation", "max"),
            median_near_equal_pct=("near_equal_pct", "median"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    output = f"reports/signal_comparison_{hardware}.csv"
    result_df.to_csv(output, index=False)

    print("\nSaved:", output)

print("\nMULTI-FLIGHT COMPARISON COMPLETE")