import joblib
from app.schemas.noshow_schema import NoShowRequest, NoShowResponse
from app.utils.paths import BASE_DIR

MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def load_model(path: str = MODEL_PATH):
    try:
        model = joblib.load(path)
        print("Loaded model")
        return model
    except Exception:
        return None


def predict_no_show(inputs: NoShowRequest) -> NoShowResponse:
    """Predict no-show probability based on input features using the loaded model."""
    if inputs.gender.lower() == "m":
        male = 1
        female = 0
    else:
        male = 0
        female = 1
    features = [
        inputs.age,
        inputs.scholarship,
        inputs.hipertension,
        inputs.diabetes,
        inputs.handicap,
        inputs.sms_received,
        (inputs.appointment_day - inputs.scheduled_day).days,
        female,
        male,
    ]
    model = load_model(MODEL_PATH)
    if not model:
        print("Model not found, returning default probability")
        return NoShowResponse(no_show_probability=0.0)
    prediction = model.predict([features])[0]
    return NoShowResponse(no_show_probability=prediction)
