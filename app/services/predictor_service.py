import joblib
from app.schemas.noshow_schema import NoShowRequest, NoShowResponse
from app.utils.paths import BASE_DIR
import pandas as pd
from typing import List


try:

    from model.inspect_model_inputs import (
        find_preprocessor,
        get_feature_names_from_ct,
    )
except ImportError:
    # Fallback functions if model inspection is not available
    def find_preprocessor(model):
        return None

    def get_feature_names_from_ct(ct):
        return []


MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def load_model(path: str = MODEL_PATH):
    try:
        model = joblib.load(path)
        print("Loaded model")
        return model
    except Exception:
        return None


def _build_row_from_input(
    expected_cols: List[str], inputs: NoShowRequest
) -> pd.DataFrame:
    waiting_time = (inputs.appointment_day - inputs.scheduled_day).days

    token_map = {
        "Age": inputs.age,
        "Scholarship": inputs.scholarship,
        "Hipertension": inputs.hipertension,
        "Diabetes": inputs.diabetes,
        "Handcap": inputs.handicap,
        "SMS_received": inputs.sms_received,
        "Waiting_time": waiting_time,
    }

    row = []
    gender_normalized = inputs.gender.strip().upper() if inputs.gender else ""
    for col in expected_cols:
        # categorical gender columns like 'cat__Gender_F' or 'cat__Gender_M'
        if "Gender" in col:
            # suffix after last underscore
            suffix = col.split("_")[-1].upper()
            if suffix and gender_normalized and gender_normalized[0] == suffix[0]:
                row.append(1)
            else:
                row.append(0)
            continue

        # numeric token mapping (look for token names inside the column)
        placed = False
        for tok, val in token_map.items():
            if tok in col:
                row.append(val)
                placed = True
                break
        if not placed:
            # fallback: 0
            row.append(0)

    return pd.DataFrame([row], columns=expected_cols)


def predict_no_show(inputs: NoShowRequest) -> NoShowResponse:
    """Predict no-show probability based on input features using the loaded model.

    Builds a single-row DataFrame matching the preprocessor's expected column names
    so the ColumnTransformer receives a correctly shaped/named input.
    """
    model = load_model(MODEL_PATH)
    if not model:
        print("Model not found, returning default probability")
        return NoShowResponse(no_show_probability=0.0)

    pre = find_preprocessor(model)
    expected_cols: List[str] = []
    if pre is not None and hasattr(pre, "transformers_"):
        expected_cols = get_feature_names_from_ct(pre)

    if not expected_cols:
        # Fallback to the original order if we couldn't read expected names
        expected_cols = [
            "num__Age",
            "num__Scholarship",
            "num__Hipertension",
            "num__Diabetes",
            "num__Handcap",
            "num__SMS_received",
            "num__Waiting_time",
            "cat__Gender_F",
            "cat__Gender_M",
        ]

    X = _build_row_from_input(expected_cols, inputs)

    try:
        prediction = model.predict(X)[0]
    except Exception:
        # If predict fails, return a safe default
        return NoShowResponse(no_show_probability=0.0)

    return NoShowResponse(no_show_probability=float(prediction))
