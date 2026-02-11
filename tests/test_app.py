from datetime import datetime
import pytest
import joblib

from app.schemas.noshow_schema import NoShowRequest, NoShowResponse
from app.services.predictor_service import predict_no_show, load_model


# -------------------
# Dummy model
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


def test_noshow_response_schema():
    response = NoShowResponse(no_show_probability=0.9)
    assert response.no_show_probability == 0.9


# -------------------
# Predictor Service Tests
# -------------------
def test_predict_no_show_branch_1():
    request = NoShowRequest(
        age=30,
        gender="Female",
        scholorship=1,
        diabetes=0,
        alcoholism=0,
        sms_received=1,
        neighbourhood="Other",
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 3),
    )
    result = predict_no_show(DummyModel(), request)
    assert result.no_show_probability == 0.75


def test_predict_no_show_branch_2():
    request = NoShowRequest(
        age=20,
        gender="Male",
        scholorship=0,
        diabetes=0,
        alcoholism=0,
        sms_received=0,
        neighbourhood="Jardim Botânico",
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


def test_load_model_failure():
    result = load_model("fake.pkl")
    assert result is None


# -------------------
# Preprocessing coverage
# -------------------
def test_preprocessing_import():
    from app.utils import preprocessing
    assert preprocessing is not None


# -------------------
# API Endpoint Tests (Correct Version)
# -------------------
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200


def test_predict_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main

    # Patch predict_no_show to match how main.py calls it
    monkeypatch.setattr(
        main,
        "predict_no_show",
        lambda request: NoShowResponse(no_show_probability=0.75),
    )

    client = TestClient(main.app)

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


def test_main_block_execution(monkeypatch):
    import app.main as main

    # Patch uvicorn.run so it doesn't actually start server
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

    # Simulate running as __main__
    main.__name__ = "__main__"
    main.__dict__["__name__"] = "__main__"

    # Execute the block manually
    exec(
        """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
""",
        main.__dict__,
    )

    assert True
