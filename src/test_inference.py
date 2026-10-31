def test_anomaly_inference(engine, hardware):
    bundle = engine.anomaly_models[hardware]
    feature_columns = bundle["feature_columns"]

    features = {
        column: 0.0
        for column in feature_columns
    }

    prediction = engine.detect_anomaly(hardware, features)

    assert prediction is not None
    assert isinstance(prediction, dict)


def test_forecast_inference(engine):
    features = {
        column: 0.0
        for column in engine.forecast_features
    }

    prediction = engine.forecast_remaining_time(features)

    assert prediction is not None
    assert isinstance(prediction, dict)
