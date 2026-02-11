import pickle
from typing import Dict

MODEL_PATH = "model/model.pkl"


def load_model(path: str = MODEL_PATH):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict(features: Dict) -> Dict:
    model = load_model()
    if model is None:
        return {"prediction": None, "error": "model_not_available"}
    # placeholder: real code would transform features and call model.predict
    return {"prediction": 0}
