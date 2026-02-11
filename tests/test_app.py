from datetime import datetime
import pytest
import joblib

from app.schemas.noshow_schema import NoShowRequest, NoShowResponse
from app.services.predictor_service import predict_no_show, load_model

# -------------------
# Dummy model for testing
# -------------------
class DummyModel:
    def predict(self, X):
        return [0.75]


# -------------------
# Schema Tests
# -------------------
def test_noshow_request_schema():
    data = NoShowRequest(
        age=25,
        gender="Male",
        scholorship=0,
        diabetes=0,
        alcoholism=0,
        sms_received=1,
        neighbourhood="Jardim Botânico",
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 5),
    )
    assert data.age == 25
    assert data.gender == "Male"


def test_noshow_response_schema():
    response = NoShowResponse(no_show_probability=0.9)
    assert response.no_show_probability == 0.9


# -------------------
# Predictor Service Tests
# -------------------
def test_predict_no_show_service_branches():
    request = NoShowRequest(
        age=30,
        gender="Female",  # branch for False
        scholorship=1,
        diabetes=0,
        alcoholism=0,
        sms_received=1,
        neighbourhood="Other",  # branch for False
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 3),
    )
    result = predict_no_show(DummyModel(), request)
    assert result.no_show_probability == 0.75


def test_predict_no_show_zero_days():
    request = NoShowRequest(
        age=20,
        gender="Male",
        scholorship=0,
        diabetes=0,
        alcoholism=0,
        sms_received=0,
        neighbourhood="Other",
        handicap=0,
        scheduled_day=datetime(2024, 1, 5),
        appointment_day=datetime(2024, 1, 5),
    )
    result = predict_no_show(DummyModel(), request)
    assert result.no_show_probability == 0.75


# -------------------
# load_model Tests
# -------------------
def test_load_model_success(tmp_path):
    model_file = tmp_path / "model.pkl"
    joblib.dump(DummyModel(), model_file)
    loaded_model = load_model(str(model_file))
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")


def test_load_model_failure():
    result = load_model("non_existent_file.pkl")
    assert result is None


# -------------------
# API Endpoint Tests (offline)
# -------------------
@pytest.mark.skipif(True, reason="Skip if you don't want endpoint tests")
def test_endpoints_offline(monkeypatch):
    # Import app only here to avoid circular import during module load
    from fastapi.testclient import TestClient
    from app.main import app

    # Patch the model
    from app import main
    monkeypatch.setattr(main, "model", DummyModel())

    client = TestClient(app)

    # Health check
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Predict endpoint
    payload = {
        "age": 25,
        "gender": "Male",
        "scholorship": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "sms_received": 1,
        "neighbourhood": "Jardim Botânico",
        "handicap": 0,
        "scheduled_day": "2024-01-01T00:00:00",
        "appointment_day": "2024-01-03T00:00:00"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["no_show_probability"] == 0.75
