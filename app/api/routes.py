from fastapi import APIRouter
from ..main import app

router = APIRouter()


@router.get("/predict")
def predict_dummy():
    return {"prediction": "not_implemented"}


app.include_router(router)
