from fastapi import FastAPI

app = FastAPI(title="No-Show Predictor")


@app.get("/health")
def health_check():
    return {"status": "ok"}
