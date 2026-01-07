from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_client import start_http_server, Counter, Gauge, generate_latest
from fastapi.responses import Response

MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Inference API")

PREDICTIONS = Counter("predictions_total", "Total predictions served")
MODEL_READY = Gauge("model_ready", "1 if model loaded")

MODEL_READY.set(1)


class Record(BaseModel):
    features: list


@app.post("/predict")
def predict(record: Record):
    X = scaler.transform(np.array(record.features).reshape(1, -1))
    pred = model.predict(X)[0]
    PREDICTIONS.inc()
    return {"prediction": int(pred)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    start_http_server(8000)
