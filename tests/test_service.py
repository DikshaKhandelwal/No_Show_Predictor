from app.services import predictor_service


def test_predict_service_no_model():
    result = predictor_service.predict({})
    assert isinstance(result, dict)
