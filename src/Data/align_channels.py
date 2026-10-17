import os
import glob
import pandas as pd
import numpy as np

def align_flight_telemetry(flight_csv_path: str, output_dir: str, target_freq_hz: int = 10):
    """
    Reads a processed flight CSV, resamples it to a uniform frequency,
    and saves it as an aligned parquet file.
    """
    if not os.path.exists(flight_csv_path):
        print(f"File not found: {flight_csv_path}")
        return

    df = pd.read_csv(flight_csv_path)
    
    if 'timestamp' not in df.columns:
        print(f"Skipping {flight_csv_path}: 'timestamp' column missing.")
        return
        
    # Drop duplicate timestamps and sort
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    
    if pd.isna(min_time) or pd.isna(max_time) or min_time == max_time:
        print(f"Skipping {flight_csv_path}: Invalid timestamp range.")
        return

    # Create uniform timeline
    step = 1.0 / target_freq_hz
    uniform_timeline = np.arange(min_time, max_time, step)
    unified_df = pd.DataFrame({'timestamp': uniform_timeline})

    # Reindex and interpolate features onto the uniform grid
    df = df.set_index('timestamp')
    df_resampled = df.reindex(df.index.union(uniform_timeline)).interpolate(method='linear')
    df_resampled = df_resampled.loc[uniform_timeline].reset_index()

    # Forward fill/backfill remaining NaNs
    unified_df = pd.concat([unified_df, df_resampled.drop(columns=['timestamp'], errors='ignore')], axis=1)
    unified_df = unified_df.ffill().bfill()

    # Save aligned parquet file
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(flight_csv_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_aligned.parquet")
    
    unified_df.to_parquet(out_path, index=False)
    print(f"Successfully aligned and saved: {out_path} with shape {unified_df.shape}")

def batch_align_recursive(processed_dir: str, output_dir: str):
    """
    Recursively scans all subfolders in data/processed for telemetry CSV files.
    """
    # Recursively find all CSV files
    all_csvs = glob.glob(os.path.join(processed_dir, "**/*.csv"), recursive=True)
    
    ignore_keywords = [
        "data_dictionary", "eda_flight_stats", "eda_flight_summary", 
        "eda_missingness", "feature_channel_candidates", "feature_channels_final", 
        "flight_metadata", "ingestion_manifest", "splits", "window_index"
    ]
    
    valid_files = []
    for f in all_csvs:
        fname = os.path.basename(f)
        if any(kw in fname for kw in ignore_keywords):
            continue
        valid_files.append(f)

    print(f"Found {len(valid_files)} valid telemetry CSV files recursively.")
    
    if not valid_files:
        print("No valid telemetry files found. Please verify where your flight CSVs are stored.")
        return

    for f_path in valid_files:
        align_flight_telemetry(f_path, output_dir)

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    ALIGNED_DIR = "data/aligned"
    
    batch_align_recursive(PROCESSED_DIR, ALIGNED_DIR)