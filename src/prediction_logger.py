
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "monitoring"
DB_PATH = DB_DIR / "prediction_history.db"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                hardware_config TEXT NOT NULL,
                flight_id TEXT,
                window_index INTEGER,
                anomaly_score REAL,
                threshold REAL,
                is_anomaly INTEGER,
                forecast_horizon_seconds REAL,
                predicted_bms_remaining_time_seconds REAL,
                features_json TEXT
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_hardware
            ON predictions(hardware_config)
        """)


def log_prediction(
    prediction_type,
    hardware_config,
    flight_id=None,
    window_index=None,
    anomaly_score=None,
    threshold=None,
    is_anomaly=None,
    forecast_horizon_seconds=None,
    predicted_bms_remaining_time_seconds=None,
    features=None
):
    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO predictions (
                timestamp,
                prediction_type,
                hardware_config,
                flight_id,
                window_index,
                anomaly_score,
                threshold,
                is_anomaly,
                forecast_horizon_seconds,
                predicted_bms_remaining_time_seconds,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            prediction_type,
            hardware_config,
            str(flight_id) if flight_id is not None else None,
            window_index,
            anomaly_score,
            threshold,
            int(is_anomaly) if is_anomaly is not None else None,
            forecast_horizon_seconds,
            predicted_bms_remaining_time_seconds,
            json.dumps(features, allow_nan=False) if features is not None else None
        ))

        return cursor.lastrowid


def get_predictions(limit=1000):
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]


def get_prediction_count():
    initialize_database()

    with get_connection() as connection:
        result = connection.execute(
            "SELECT COUNT(*) AS total FROM predictions"
        ).fetchone()

    return result["total"]


def clear_prediction_history():
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM predictions")
