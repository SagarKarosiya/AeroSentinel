
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

try:
    from .inference_engine import AeroSentinelInference
    from .prediction_logger import log_prediction
except ImportError:
    from inference_engine import AeroSentinelInference
    from prediction_logger import log_prediction


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AeroSentinel")


# --------------------------------------------------
# INITIALIZE API AND INFERENCE ENGINE
# --------------------------------------------------

app = FastAPI(
    title="AeroSentinel API",
    description="UAV anomaly detection and remaining-time forecasting",
    version="1.1.0"
)

engine = AeroSentinelInference()


# --------------------------------------------------
# REQUEST SCHEMAS
# --------------------------------------------------

class PredictionRequest(BaseModel):

    hardware_config: str = Field(
        description="Supported values: holybro or speedybee"
    )

    features: dict[str, Optional[float]]

    flight_id: Optional[str] = None
    window_index: Optional[int] = None


# --------------------------------------------------
# SAFE DATABASE LOGGING
# --------------------------------------------------

def save_prediction(**kwargs):
    """
    Save prediction without interrupting the API response
    if the database is temporarily unavailable.
    """

    try:
        record_id = log_prediction(**kwargs)
        logger.info("Prediction logged successfully. Record ID: %s", record_id)
        return record_id

    except Exception:
        logger.exception("Prediction logging failed.")
        return None


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "AeroSentinel",
        "status": "running",
        "version": "1.1.0",
        "supported_hardware": ["holybro", "speedybee"],
        "endpoints": [
            "/health",
            "/predict/anomaly",
            "/predict/remaining-time",
            "/predict",
            "/monitoring/history",
            "/monitoring/count"
        ]
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "models_loaded": {
            "holybro_anomaly": "holybro" in engine.anomaly_models,
            "speedybee_anomaly": "speedybee" in engine.anomaly_models,
            "holybro_forecast": engine.forecast_model is not None
        }
    }


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

@app.post("/predict/anomaly")
def predict_anomaly(request: PredictionRequest):

    hardware = request.hardware_config.lower()

    try:

        result = engine.detect_anomaly(
            hardware,
            request.features
        )

        record_id = save_prediction(
            prediction_type="anomaly",
            hardware_config=hardware,
            flight_id=request.flight_id,
            window_index=request.window_index,
            anomaly_score=result["anomaly_score"],
            threshold=result["threshold"],
            is_anomaly=result["is_anomaly"],
            features=request.features
        )

        return {
            "status": "success",
            "prediction": result,
            "record_id": record_id
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        logger.exception("Anomaly prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# REMAINING-TIME FORECAST
# --------------------------------------------------

@app.post("/predict/remaining-time")
def predict_remaining_time(request: PredictionRequest):

    hardware = request.hardware_config.lower()

    if hardware != "holybro":

        raise HTTPException(
            status_code=400,
            detail="Remaining-time forecasting currently supports Holybro only."
        )

    try:

        result = engine.forecast_remaining_time(
            request.features
        )

        record_id = save_prediction(
            prediction_type="remaining_time",
            hardware_config=hardware,
            flight_id=request.flight_id,
            window_index=request.window_index,
            forecast_horizon_seconds=result[
                "forecast_horizon_seconds"
            ],
            predicted_bms_remaining_time_seconds=result[
                "predicted_bms_remaining_time_seconds"
            ],
            features=request.features
        )

        return {
            "status": "success",
            "prediction": result,
            "record_id": record_id
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        logger.exception("Remaining-time forecast failed.")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# COMBINED PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict_all(request: PredictionRequest):

    hardware = request.hardware_config.lower()

    try:

        result = engine.predict(
            hardware,
            request.features
        )

        record_ids = {}

        # Log anomaly result if available
        anomaly_result = result.get("anomaly")

        if isinstance(anomaly_result, dict):

            record_ids["anomaly"] = save_prediction(
                prediction_type="anomaly",
                hardware_config=hardware,
                flight_id=request.flight_id,
                window_index=request.window_index,
                anomaly_score=anomaly_result.get("anomaly_score"),
                threshold=anomaly_result.get("threshold"),
                is_anomaly=anomaly_result.get("is_anomaly"),
                features=request.features
            )

        # Log forecast result if available
        forecast_result = result.get("remaining_time")

        if isinstance(forecast_result, dict):

            record_ids["remaining_time"] = save_prediction(
                prediction_type="remaining_time",
                hardware_config=hardware,
                flight_id=request.flight_id,
                window_index=request.window_index,
                forecast_horizon_seconds=forecast_result.get(
                    "forecast_horizon_seconds"
                ),
                predicted_bms_remaining_time_seconds=forecast_result.get(
                    "predicted_bms_remaining_time_seconds"
                ),
                features=request.features
            )

        return {
            "status": "success",
            "predictions": result,
            "record_ids": record_ids
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        logger.exception("Combined prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# MONITORING: PREDICTION HISTORY
# --------------------------------------------------

@app.get("/monitoring/history")
def prediction_history(
    limit: int = 100,
    hardware_config: Optional[str] = None,
    prediction_type: Optional[str] = None
):

    try:

        from .prediction_logger import get_predictions

    except ImportError:

        from prediction_logger import get_predictions

    try:

        limit = max(1, min(limit, 5000))

        records = get_predictions(limit=limit)

        if hardware_config:
            records = [
                record for record in records
                if record["hardware_config"] == hardware_config.lower()
            ]

        if prediction_type:
            records = [
                record for record in records
                if record["prediction_type"] == prediction_type.lower()
            ]

        return {
            "status": "success",
            "total_returned": len(records),
            "predictions": records
        }

    except Exception as error:

        logger.exception("Could not retrieve prediction history.")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# MONITORING: TOTAL PREDICTION COUNT
# --------------------------------------------------

@app.get("/monitoring/count")
def prediction_count():

    try:

        try:
            from .prediction_logger import get_prediction_count
        except ImportError:
            from prediction_logger import get_prediction_count

        return {
            "status": "success",
            "total_predictions": get_prediction_count()
        }

    except Exception as error:

        logger.exception("Could not retrieve prediction count.")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
