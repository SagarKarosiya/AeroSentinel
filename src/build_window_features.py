
from pathlib import Path
import gc
import numpy as np
import pandas as pd

from data_loader import load_manifest, load_flight

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FEATURE_DIR = ROOT / "data" / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# Physical-time window configuration
WINDOW_SECONDS = 10
OVERLAP = 0.50

# Initial dynamic signals selected from our earlier audits.
# The script automatically keeps only columns actually present.
FEATURES = {
    "holybro": [
        "output[0]__actuator_outputs_0",
        "q[0]__vehicle_attitude_0",
        "q[1]__vehicle_attitude_0",
        "q[2]__vehicle_attitude_0",
        "q[3]__vehicle_attitude_0",
        "current_a__battery_status_0",
        "discharged_mah__battery_status_0",
        "ocv_estimate__battery_status_0",
        "ocv_estimate_filtered__battery_status_0",
        "prediction_error__battery_status_0",
        "remaining__battery_status_0",
        "scale__battery_status_0",
        "temperature__battery_status_0",
        "volt_based_soc_estimate__battery_status_0",
        "voltage_v__battery_status_0",
        "alt__estimator_global_position_0",
        "eph__estimator_global_position_0",
        "epv__estimator_global_position_0",
        "lat__estimator_global_position_0",
        "lon__estimator_global_position_0",
        "altitude_msl_m__vehicle_gps_position_0",
        "latitude_deg__vehicle_gps_position_0",
        "longitude_deg__vehicle_gps_position_0",
        "eph__vehicle_gps_position_0",
        "epv__vehicle_gps_position_0",
        "hdop__vehicle_gps_position_0",
        "vdop__vehicle_gps_position_0",
        "satellites_used__vehicle_gps_position_0",
        "vel_m_s__vehicle_gps_position_0",
        "vel_n_m_s__vehicle_gps_position_0",
        "vel_e_m_s__vehicle_gps_position_0",
        "vel_d_m_s__vehicle_gps_position_0",
        "cog_rad__vehicle_gps_position_0",
        "pressure__sensor_baro_0",
        "temperature__sensor_baro_0",
    ],

    "speedybee": [
        "motor[0]",
        "servo[1]",
        "servo[2]",
        "servo[3]",
        "servo[4]",
        "BaroAlt (cm)",
        "attitude[0]",
        "attitude[1]",
        "attitude[2]",
        "vbat (V)",
        "amperage (A)",
        "energyCumulative (mAh)",
        "accSmooth[0]",
        "accSmooth[1]",
        "accSmooth[2]",
        "gyroADC[0]",
        "gyroADC[1]",
        "gyroADC[2]",
        "gyroRaw[0]",
        "gyroRaw[1]",
        "gyroRaw[2]",
        "gyroPeakRoll[0]",
        "gyroPeakRoll[1]",
        "gyroPeakRoll[2]",
        "gyroPeakPitch[0]",
        "gyroPeakPitch[1]",
        "gyroPeakPitch[2]",
        "gyroPeakYaw[0]",
        "gyroPeakYaw[1]",
        "gyroPeakYaw[2]",
    ]
}

SAMPLE_RATES = {
    "holybro": 2,
    "speedybee": 50,
}

manifest = load_manifest()
all_reports = []

for hardware in ["holybro", "speedybee"]:

    print("\n" + "=" * 70)
    print(f"BUILDING WINDOWS: {hardware.upper()}")
    print("=" * 70)

    hardware_manifest = manifest[
        manifest["hardware_config"] == hardware
    ]

    window_rows = int(WINDOW_SECONDS * SAMPLE_RATES[hardware])
    stride = max(1, int(window_rows * (1 - OVERLAP)))

    hardware_results = []
    usable_features = None

    for _, record in hardware_manifest.iterrows():

        flight_id = int(record["flight_id"])
        split = record["split"]

        print(f"Flight {flight_id} ({split})")

        try:
            df, metadata = load_flight(hardware, flight_id)

            # Select numeric features that actually exist.
            if usable_features is None:
                available = [
                    col for col in FEATURES[hardware]
                    if col in df.columns
                    and pd.api.types.is_numeric_dtype(df[col])
                ]
                usable_features = available

                print("Available candidate features:", len(available))

            selected = df[usable_features].copy()

            # Convert invalid values to NaN.
            selected = selected.replace(
                [np.inf, -np.inf], np.nan
            )

            # Process one flight at a time.
            selected = selected.astype("float32")

            # Avoid retaining features that are entirely missing
            # or constant in this flight.
            valid_columns = [
                col for col in selected.columns
                if selected[col].notna().any()
                and selected[col].nunique(dropna=True) > 1
            ]

            selected = selected[valid_columns]

            if selected.empty:
                print("  Skipped: no usable dynamic features")
                continue

            timestamps = pd.to_numeric(
                df["timestamp_s"], errors="coerce"
            ).to_numpy(dtype=float)

            if len(timestamps) != len(selected):
                raise ValueError("Timestamp and feature row mismatch")

            flight_windows = []

            for start in range(
                0,
                len(selected) - window_rows + 1,
                stride
            ):

                end = start + window_rows

                block = selected.iloc[start:end]
                block_times = timestamps[start:end]

                if not np.isfinite(block_times).all():
                    continue

                if not np.all(np.diff(block_times) > 0):
                    continue

                # Require a complete window of valid timestamps
                # within the expected physical duration.
                actual_duration = block_times[-1] - block_times[0]

                if actual_duration > WINDOW_SECONDS + (
                    1.0 / SAMPLE_RATES[hardware]
                ):
                    continue

                features = {}

                for col in block.columns:

                    values = block[col].dropna().to_numpy(
                        dtype=np.float32
                    )

                    if len(values) == 0:
                        continue

                    prefix = col

                    features[f"{prefix}__mean"] = float(
                        np.mean(values)
                    )
                    features[f"{prefix}__std"] = float(
                        np.std(values)
                    )
                    features[f"{prefix}__min"] = float(
                        np.min(values)
                    )
                    features[f"{prefix}__max"] = float(
                        np.max(values)
                    )
                    features[f"{prefix}__range"] = float(
                        np.max(values) - np.min(values)
                    )
                    features[f"{prefix}__rms"] = float(
                        np.sqrt(np.mean(values ** 2))
                    )
                    features[f"{prefix}__last"] = float(
                        values[-1]
                    )

                if not features:
                    continue

                features.update({
                    "hardware_config": hardware,
                    "flight_id": flight_id,
                    "split": split,
                    "window_start_s": float(block_times[0]),
                    "window_end_s": float(block_times[-1]),
                    "window_duration_s": float(actual_duration),
                })

                flight_windows.append(features)

            if flight_windows:
                flight_df = pd.DataFrame(flight_windows)
                hardware_results.append(flight_df)

                print(
                    f"  Windows: {len(flight_df)}, "
                    f"features: {len(valid_columns)}"
                )
            else:
                print("  No complete windows generated")

            del df, selected
            gc.collect()

        except Exception as exc:
            print(f"  ERROR: {exc}")

    if hardware_results:

        output = pd.concat(
            hardware_results,
            ignore_index=True
        )

        output_path = (
            FEATURE_DIR / f"{hardware}_window_features.parquet"
        )

        output.to_parquet(output_path, index=False)

        print("\nHARDWARE SUMMARY")
        print("Flights with windows:", output["flight_id"].nunique())
        print("Total windows:", len(output))
        print("Feature columns:", len(output.columns) - 7)
        print("\nSplit distribution:")
        print(output["split"].value_counts().to_string())

        print(f"\nSaved: {output_path}")

        all_reports.append({
            "hardware_config": hardware,
            "flights_with_windows": output["flight_id"].nunique(),
            "total_windows": len(output),
            "feature_columns": len(output.columns) - 7,
            "train_windows": int((output["split"] == "train").sum()),
            "val_windows": int((output["split"] == "val").sum()),
            "test_windows": int((output["split"] == "test").sum()),
        })

        del output

    gc.collect()

if all_reports:
    summary = pd.DataFrame(all_reports)
    summary_path = REPORT_DIR / "window_feature_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
else:
    print("No feature datasets were generated.")
