# ml_system/api/main.py
import time
import json
import joblib
import logging
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------
RESPONSE_DELAY = Histogram(
    "response_delay_seconds",
    "Model inference response latency in seconds",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Latest trained model accuracy"
)

RETRAIN_COUNT = Counter(
    "retrain_count_total",
    "Number of times retraining has occurred"
)

DATALAKE_UNAVAILABLE = Counter(
    "datalake_unavailable",
    "Number of times datalake returned 503"
)

RECORDS_PROCESSED = Counter(
    "records_processed_total",
    "Total records processed from ingestion"
)

FEATURE_ADDED = Counter(
    "feature_added",
    "New feature added to schema"
)

FEATURE_REMOVED = Counter(
    "feature_removed",
    "Feature removed from schema"
)

DRIFT_DETECTED = Gauge(
    "distribution_drift_detected",
    "1 if drift detected, else 0"
)


# ---------------------------------------------------------
# Load Model Artifacts
# ---------------------------------------------------------
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    logger.error(f"Model or scaler not found: {e}")
    raise RuntimeError("Model or scaler missing. Train the model first.")

try:
    with open("feature_list.json") as f:
        FEATURE_LIST = json.load(f)
except:
    FEATURE_LIST = None
    logger.warning("feature_list.json missing–continuing without feature validation")


# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(
    title="ML Inference API",
    description="Provides prediction, metrics, and health endpoints.",
    version="1.0.0"
)


# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    features: List[float]


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):

    # Validate feature count (if schema saved)
    if FEATURE_LIST is not None and len(req.features) != len(FEATURE_LIST):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_LIST)} features, got {len(req.features)}"
        )

    start = time.time()

    # Convert to numpy and scale
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Make prediction
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled).max()

    # Update latency metric
    elapsed = time.time() - start
    RESPONSE_DELAY.observe(elapsed)

    # Log
    logger.info(f"Prediction: {pred[0]}, probability: {prob:.3f}, latency={elapsed:.4f}")

    return {
        "prediction": int(pred[0]),
        "probability": float(prob),
        "latency": elapsed
    }


# ---------------------------------------------------------
# Prometheus Metrics Endpoint
# ---------------------------------------------------------
@app.get("/metrics")
def metrics():
    """
    Exposes all Prometheus metrics for scraping.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
