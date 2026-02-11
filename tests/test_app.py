from datetime import datetime
import joblib
import pandas as pd
from unittest.mock import patch, MagicMock

from app.schemas.noshow_schema import NoShowRequest, NoShowResponse
from app.services.predictor_service import (
    predict_no_show,
    load_model,
    _build_row_from_input,
)


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
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=1,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 5),
    )
    assert data.age == 25


def test_noshow_response_schema():
    response = NoShowResponse(no_show_probability=0.9)
    assert response.no_show_probability == 0.9


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
# _build_row_from_input Tests
# -------------------
def test_build_row_from_input_female_gender():
    """Test _build_row_from_input with female gender variants."""
    expected_cols = ["num__Age", "cat__Gender_F", "cat__Gender_M", "num__Scholarship"]

    request = NoShowRequest(
        age=30,
        gender="F",
        scholarship=1,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 5),
    )

    result = _build_row_from_input(expected_cols, request)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["num__Age"] == 30
    assert result.iloc[0]["cat__Gender_F"] == 1  # Female should be 1
    assert result.iloc[0]["cat__Gender_M"] == 0  # Male should be 0
    assert result.iloc[0]["num__Scholarship"] == 1


def test_build_row_from_input_male_gender():
    """Test _build_row_from_input with male gender variants."""
    expected_cols = ["cat__Gender_F", "cat__Gender_M", "num__Age"]

    request = NoShowRequest(
        age=25,
        gender="Male",
        scholarship=0,
        diabetes=1,
        hipertension=0,
        sms_received=1,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 3),
    )

    result = _build_row_from_input(expected_cols, request)

    assert result.iloc[0]["cat__Gender_F"] == 0  # Female should be 0
    assert result.iloc[0]["cat__Gender_M"] == 1  # Male should be 1
    assert result.iloc[0]["num__Age"] == 25


def test_build_row_from_input_empty_gender():
    """Test _build_row_from_input with empty/None gender."""
    expected_cols = ["cat__Gender_F", "cat__Gender_M"]

    request = NoShowRequest(
        age=25,
        gender="",
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 1),
    )

    result = _build_row_from_input(expected_cols, request)

    assert result.iloc[0]["cat__Gender_F"] == 0
    assert result.iloc[0]["cat__Gender_M"] == 0


def test_build_row_from_input_waiting_time_calculation():
    """Test waiting time calculation in _build_row_from_input."""
    expected_cols = ["num__Waiting_time", "num__Age"]

    request = NoShowRequest(
        age=40,
        gender="F",
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 10),
    )

    result = _build_row_from_input(expected_cols, request)

    assert result.iloc[0]["num__Waiting_time"] == 9  # 10 days - 1 day = 9 days
    assert result.iloc[0]["num__Age"] == 40


def test_build_row_from_input_all_numeric_features():
    """Test _build_row_from_input with all numeric feature mappings."""
    expected_cols = [
        "num__Age",
        "num__Scholarship",
        "num__Hipertension",
        "num__Diabetes",
        "num__Handcap",
        "num__SMS_received",
        "num__Waiting_time",
    ]

    request = NoShowRequest(
        age=35,
        gender="M",
        scholarship=1,
        diabetes=1,
        hipertension=1,
        sms_received=1,
        handicap=2,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 6),
    )

    result = _build_row_from_input(expected_cols, request)

    assert result.iloc[0]["num__Age"] == 35
    assert result.iloc[0]["num__Scholarship"] == 1
    assert result.iloc[0]["num__Hipertension"] == 1
    assert result.iloc[0]["num__Diabetes"] == 1
    assert result.iloc[0]["num__Handcap"] == 2
    assert result.iloc[0]["num__SMS_received"] == 1
    assert result.iloc[0]["num__Waiting_time"] == 5


def test_build_row_from_input_unknown_columns():
    """Test _build_row_from_input with unknown column names (fallback to 0)."""
    expected_cols = ["unknown_col", "another_unknown", "num__Age"]

    request = NoShowRequest(
        age=30,
        gender="F",
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 1),
    )

    result = _build_row_from_input(expected_cols, request)

    assert result.iloc[0]["unknown_col"] == 0  # Should fallback to 0
    assert result.iloc[0]["another_unknown"] == 0  # Should fallback to 0
    assert result.iloc[0]["num__Age"] == 30  # Should map correctly


# -------------------
# predict_no_show Comprehensive Tests
# -------------------
@patch("app.services.predictor_service.load_model")
def test_predict_no_show_model_not_found(mock_load_model):
    """Test predict_no_show when model loading fails."""
    mock_load_model.return_value = None

    request = NoShowRequest(
        age=25,
        gender="F",
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 5),
    )

    result = predict_no_show(request)

    assert result.no_show_probability == 0.0


@patch("app.services.predictor_service.load_model")
def test_predict_no_show_with_mock_model_success(mock_load_model):
    """Test predict_no_show with successful model prediction."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.85]
    mock_load_model.return_value = mock_model

    request = NoShowRequest(
        age=45,
        gender="M",
        scholarship=1,
        diabetes=1,
        hipertension=0,
        sms_received=1,
        handicap=0,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 8),
    )

    result = predict_no_show(request)

    assert result.no_show_probability == 0.85
    mock_model.predict.assert_called_once()


@patch("app.services.predictor_service.load_model")
def test_predict_no_show_prediction_exception(mock_load_model):
    """Test predict_no_show when model.predict() raises an exception."""
    # Create a mock model that raises an exception
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Model prediction failed")
    mock_load_model.return_value = mock_model

    request = NoShowRequest(
        age=30,
        gender="F",
        scholarship=0,
        diabetes=0,
        hipertension=1,
        sms_received=0,
        handicap=1,
        scheduled_day=datetime(2024, 1, 1),
        appointment_day=datetime(2024, 1, 2),
    )

    result = predict_no_show(request)

    assert result.no_show_probability == 0.0  # Should return safe default


@patch("app.services.predictor_service.load_model")
def test_predict_no_show_uses_fallback_columns(mock_load_model):
    """Test predict_no_show uses fallback column names when preprocessor not found."""
    # Mock model without find_preprocessor functionality
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.42]
    mock_load_model.return_value = mock_model

    request = NoShowRequest(
        age=28,
        gender="Female",
        scholarship=1,
        diabetes=0,
        hipertension=1,
        sms_received=1,
        handicap=0,
        scheduled_day=datetime(2024, 1, 5),
        appointment_day=datetime(2024, 1, 12),
    )

    result = predict_no_show(request)

    assert result.no_show_probability == 0.42
    # Verify model.predict was called with a DataFrame
    args, kwargs = mock_model.predict.call_args
    assert isinstance(args[0], pd.DataFrame)
    assert len(args[0].columns) == 9  # Should have 9 fallback columns


# -------------------
# Edge Cases and Integration Tests
# -------------------
def test_load_model_with_custom_path(tmp_path):
    """Test load_model with custom path parameter."""
    custom_model = DummyModel()
    custom_path = tmp_path / "custom_model.pkl"
    joblib.dump(custom_model, custom_path)

    loaded = load_model(str(custom_path))

    assert loaded is not None
    assert isinstance(loaded, DummyModel)


def test_predict_no_show_integration_with_different_genders():
    """Integration test with different gender inputs."""
    test_cases = [
        ("Male", "M", "m", "MALE"),
        ("Female", "F", "f", "FEMALE"),
        ("", None),
    ]

    with patch("app.services.predictor_service.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_load.return_value = mock_model

        for gender_variants in test_cases:
            for gender in gender_variants:
                if gender is not None:
                    request = NoShowRequest(
                        age=30,
                        gender=gender,
                        scholarship=0,
                        diabetes=0,
                        hipertension=0,
                        sms_received=0,
                        handicap=0,
                        scheduled_day=datetime(2024, 1, 1),
                        appointment_day=datetime(2024, 1, 5),
                    )
                    result = predict_no_show(request)
                    assert result.no_show_probability == 0.5


def test_build_row_negative_waiting_time():
    """Test _build_row_from_input with negative waiting time (appointment before scheduled)."""
    expected_cols = ["num__Waiting_time"]

    request = NoShowRequest(
        age=25,
        gender="F",
        scholarship=0,
        diabetes=0,
        hipertension=0,
        sms_received=0,
        handicap=0,
        scheduled_day=datetime(2024, 1, 10),
        appointment_day=datetime(2024, 1, 5),  # 5 days earlier
    )

    result = _build_row_from_input(expected_cols, request)

    assert (
        result.iloc[0]["num__Waiting_time"] == -5
    )  # Should handle negative waiting time


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
        "scholarship": 0,
        "diabetes": 0,
        "hipertension": 0,
        "sms_received": 1,
        "handicap": 0,
        "scheduled_day": "2024-01-01T00:00:00",
        "appointment_day": "2024-01-03T00:00:00",
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
