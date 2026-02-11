from fastapi import FastAPI
from app.services.predictor_service import predict_no_show
from app.schemas.noshow_schema import NoShowRequest, NoShowResponse

app = FastAPI(title="No-Show Predictor")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=NoShowResponse, status_code=200)
def predict(request: NoShowRequest):
    status = predict_no_show(request)
    return status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
