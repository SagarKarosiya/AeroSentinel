import os
import glob
import pandas as pd
from pyulog import ULog

def parse_ulog_file(file_path: str, output_dir: str):
    """
    Parses a single PX4 ULog file, extracts core telemetry topics, 
    and exports them to CSV format.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        ulog = ULog(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract relevant topics (e.g., attitude, sensor combined, battery status)
        extracted_data = {}
        for data in ulog.data_list:
            if data.name in ['vehicle_attitude', 'sensor_combined', 'battery_status', 'actuator_controls']:
                df = pd.DataFrame(data.data)
                # Normalize timestamp column if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = df['timestamp'] / 1e6  # Convert microseconds to seconds
                extracted_data[data.name] = df

        # Save individual topics or merge them
        os.makedirs(output_dir, exist_ok=True)
        for topic_name, df in extracted_data.items():
            out_path = os.path.join(output_dir, f"{base_name}_{topic_name}.csv")
            df.to_csv(out_path, index=False)
            
        print(f"Successfully parsed and saved telemetry for: {base_name}")
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

def batch_parse_logs(raw_data_dir: str, output_dir: str):
    """
    Finds and parses all .ulg files in the raw dataset directory.
    """
    ulog_files = glob.glob(os.path.join(raw_data_dir, "**/*.ulg"), recursive=True)
    if not ulog_files:
        print(f"No .ulg files found in {raw_data_dir}. Please ensure your dataset is unzipped correctly.")
        return

    print(f"Found {len(ulog_files)} flight log files. Beginning parsing...")
    for file_path in ulog_files:
        parse_ulog_file(file_path, output_dir)

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    batch_parse_logs(RAW_DIR, PROCESSED_DIR)