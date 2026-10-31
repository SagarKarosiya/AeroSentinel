
from prediction_logger import (
    initialize_database,
    log_prediction,
    get_predictions,
    get_prediction_count
)

print("Initializing AeroSentinel prediction database...")

initialize_database()

print("Saving sample anomaly prediction...")

log_prediction(
    prediction_type="anomaly",
    hardware_config="holybro",
    flight_id=1,
    window_index=0,
    anomaly_score=0.5425,
    threshold=0.5030,
    is_anomaly=True
)

print("Saving sample forecast prediction...")

log_prediction(
    prediction_type="remaining_time",
    hardware_config="holybro",
    flight_id=1,
    window_index=0,
    forecast_horizon_seconds=30,
    predicted_bms_remaining_time_seconds=1178.83
)

print("\nTotal records:", get_prediction_count())

print("\nLatest prediction records:")

for record in get_predictions(limit=5):
    print(record)

print("\nDATABASE TEST COMPLETED")
