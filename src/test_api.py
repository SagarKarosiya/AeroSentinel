import api
import pytest
from fastapi.testclient import TestClient

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def disable_database_logging(monkeypatch):
    monkeypatch.setattr(api, "save_prediction", lambda **kwargs: None)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.parametrize("hardware", ["holybro", "speedybee"])
def test_anomaly_endpoint(hardware):
    bundle = api.engine.anomaly_models[hardware]
    feature_columns = bundle["feature_columns"]
    features = {column: 0.0 for column in feature_columns}

    response = client.post(
        "/predict/anomaly",
        json={
            "hardware_config": hardware,
            "features": features
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data


def test_remaining_time_endpoint():
    features = {
        column: 0.0
        for column in api.engine.forecast_features
    }

    response = client.post(
        "/predict/remaining-time",
        json={
            "hardware_config": "holybro",
            "features": features
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data


def test_remaining_time_rejects_speedybee():
    response = client.post(
        "/predict/remaining-time",
        json={
            "hardware_config": "speedybee",
            "features": {}
        }
    )

    assert response.status_code == 400
