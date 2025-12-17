import os
import tempfile

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.main import app, get_model
from app.pipeline import add_velocity_features_offline, build_preprocessor, engineer_features


def build_dummy_model(model_path: str):
    raw = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 5000,
                "nameOrig": "C0001",
                "oldbalanceOrg": 15000,
                "newbalanceOrig": 10000,
                "nameDest": "C0002",
                "oldbalanceDest": 0,
                "newbalanceDest": 5000,
                "isFlaggedFraud": 0,
                "isFraud": 0,
            },
            {
                "step": 5,
                "type": "CASH_OUT",
                "amount": 25000,
                "nameOrig": "C0003",
                "oldbalanceOrg": 30000,
                "newbalanceOrig": 5000,
                "nameDest": "C0004",
                "oldbalanceDest": 1000,
                "newbalanceDest": 26000,
                "isFlaggedFraud": 0,
                "isFraud": 1,
            },
        ]
    )
    engineered = engineer_features(raw)
    engineered = add_velocity_features_offline(engineered)
    X = engineered.drop(columns=["isFraud"])
    y = engineered["isFraud"]
    preprocessor, _, _ = build_preprocessor(X)

    model = LogisticRegression(max_iter=100)
    from sklearn.pipeline import Pipeline

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    clf.fit(X, y)
    joblib.dump(clf, model_path)


@pytest.fixture(scope="module")
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        build_dummy_model(model_path)
        os.environ["RISKY_MODEL_PATH"] = model_path
        get_model.cache_clear()
        yield TestClient(app)
        get_model.cache_clear()


def test_predict_endpoint(client):
    payload = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 12000,
        "nameOrig": "C9000",
        "oldbalanceOrg": 15000,
        "newbalanceOrig": 3000,
        "nameDest": "M5000",
        "oldbalanceDest": 0,
        "newbalanceDest": 12000,
        "isFlaggedFraud": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert {"fraud_probability", "is_fraud", "reason_code"} <= set(data.keys())
    assert 0.0 <= data["fraud_probability"] <= 1.0
