from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import time
from prometheus_client import start_http_server, generate_latest
from fastapi.responses import Response
from ml_system.metrics import (
    PREDICTIONS,
    MODEL_READY,
    MODEL_ACCURACY,
    RETRAIN_COUNT,
    RESPONSE_DELAY
)

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

MODEL_READY.set(1)

app = FastAPI(title="Inference API")


class Record(BaseModel):
    features: list


@app.post("/predict")
def predict(record: Record):
    start_time = time.time()
    X = scaler.transform(np.array(record.features).reshape(1, -1))
    pred = model.predict(X)[0]
    PREDICTIONS.inc()
    RESPONSE_DELAY.set(time.time() - start_time)
    return {"prediction": int(pred)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    start_http_server(8000)
