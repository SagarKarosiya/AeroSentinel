import os
import glob
import subprocess
import pandas as pd
import numpy as np

def align_all_telemetry(processed_dir="data/processed", output_dir="data/aligned", target_freq_hz=10):
    print("=== STEP 1: Scanning and Aligning Telemetry Data ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Search recursively for all CSV files
    all_csvs = glob.glob(os.path.join(processed_dir, "**/*.csv"), recursive=True)
    
    # Filter out known non-telemetry metadata files
    exclude_keywords = [
        "data_dictionary", "eda_", "feature_channel", "flight_metadata", 
        "ingestion_manifest", "splits", "window_index"
    ]
    
    valid_files = []
    for f in all_csvs:
        fname = os.path.basename(f)
        if any(kw in fname for kw in exclude_keywords):
            continue
        valid_files.append(f)

    print(f"Found {len(valid_files)} potential telemetry files to inspect.")
    
    processed_count = 0
    for f_path in valid_files:
        try:
            df = pd.read_csv(f_path, nrows=5) # Quick check for columns
            if 'timestamp' not in df.columns:
                continue
            
            # Load full dataframe if timestamp exists
            full_df = pd.read_csv(f_path)
            full_df = full_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            min_time = full_df['timestamp'].min()
            max_time = full_df['timestamp'].max()
            
            if pd.isna(min_time) or pd.isna(max_time) or min_time == max_time:
                continue

            # Resample to uniform frequency
            step = 1.0 / target_freq_hz
            uniform_timeline = np.arange(min_time, max_time, step)
            unified_df = pd.DataFrame({'timestamp': uniform_timeline})

            full_df = full_df.set_index('timestamp')
            df_resampled = full_df.reindex(full_df.index.union(uniform_timeline)).interpolate(method='linear')
            df_resampled = df_resampled.loc[uniform_timeline].reset_index()

            unified_df = pd.concat([unified_df, df_resampled.drop(columns=['timestamp'], errors='ignore')], axis=1)
            unified_df = unified_df.ffill().bfill()

            base_name = os.path.splitext(os.path.basename(f_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_aligned.parquet")
            unified_df.to_parquet(out_path, index=False)
            print(f" -> Aligned & Saved: {out_path} (shape: {unified_df.shape})")
            processed_count += 1
            
        except Exception as e:
            print(f"Skipping {f_path} due to error: {e}")

    print(f"Alignment complete. Successfully processed {processed_count} files into {output_dir}.\n")
    return processed_count

def run_autoencoder():
    print("=== STEP 2: Training LSTM Autoencoder ===")
    autoencoder_script = os.path.join("src", "models", "autoencoder.py")
    if not os.path.exists(autoencoder_script):
        print(f"Error: Could not find {autoencoder_script}")
        return
    
    # Run the autoencoder script using the active python interpreter
    result = subprocess.run(["python", autoencoder_script], capture_output=False)
    if result.returncode != 0:
        print("Autoencoder training finished with errors.")
    else:
        print("Autoencoder training completed successfully!")

if __name__ == "__main__":
    count = align_all_telemetry()
    if count > 0:
        run_autoencoder()
    else:
        print("No telemetry files with a 'timestamp' column were found. Let's inspect what CSV headers exist.")
        # Diagnostic fallback: print columns of first 3 CSVs found
        all_csvs = glob.glob(os.path.join("data/processed", "**/*.csv"), recursive=True)[:3]
        for f in all_csvs:
            print(f"File: {f}")
            try:
                print("Columns:", pd.read_csv(f, nrows=1).columns.tolist())
            except Exception as e:
                print("Error reading file:", e)