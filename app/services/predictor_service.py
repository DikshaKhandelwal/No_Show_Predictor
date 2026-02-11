import joblib
from app.schemas.noshow_schema import NoShowRequest, NoShowResponse

MODEL_PATH = "model/model.pkl"


def load_model(path: str = MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_no_show(model, inputs: NoShowRequest) -> NoShowResponse:
    """Predict no-show probability based on input features using the loaded model."""
    features = [
        inputs.age,
        inputs.gender == "Male",
        inputs.scholorship,
        inputs.diabetes,
        inputs.alcoholism,
        inputs.sms_received,
        inputs.neighbourhood == "Jardim Botânico",
        inputs.handicap,
        (inputs.appointment_day - inputs.scheduled_day).days,
    ]
    prediction = model.predict([features])[0]
    return NoShowResponse(no_show_probability=prediction)
