# ml_system/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Gauge, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import numpy as np
import threading
import time
import random
import requests
import os

# -----------------------------
# Paths for model & scaler
# -----------------------------
MODEL_PATH = "/app/model.joblib"
SCALER_PATH = "/app/scaler.joblib"
DATA_LAKE_URL = "http://149.40.228.124:6500/records"

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ML Inference API")

# -----------------------------
# Prometheus metrics
# -----------------------------
RECORDS_PROCESSED = Counter("records_processed_total", "Total records processed from datalake")
RETRAIN_COUNT = Counter("retrain_count_total", "Number of times model retrained")
FEATURE_ADDED = Counter("feature_added", "Number of times a new feature appeared")
FEATURE_REMOVED = Counter("feature_removed", "Number of times a feature disappeared")
DRIFT_DETECTED = Gauge("distribution_drift_detected", "1 if distribution drift detected")
MODEL_ACCURACY = Gauge("model_accuracy", "Current accuracy of ML model")
DATALAKE_UNAVAILABLE = Gauge("datalake_unavailable", "1 if datalake returns 503")
RESPONSE_DELAY = Histogram("response_delay_seconds", "Time to get datalake response")

# -----------------------------
# Schema and stats tracking
# -----------------------------
schema = {"num_features": None}
feature_stats = {"mean": None, "std": None, "count": 0}


# -----------------------------
# Input schema
# -----------------------------
class Record(BaseModel):
    features: list

# -----------------------------
# Helper functions
# -----------------------------
def fetch_batch():
    """Fetch batch from data lake and measure response time"""
    start_time = time.time()
    try:
        resp = requests.get(DATA_LAKE_URL, timeout=5)
        response_time = time.time() - start_time
        RESPONSE_DELAY.observe(response_time)

        if resp.status_code == 503:
            DATALAKE_UNAVAILABLE.set(1)
            raise HTTPException(status_code=503, detail="Data lake unavailable")
        else:
            DATALAKE_UNAVAILABLE.set(0)
        
        data = resp.json()
        RECORDS_PROCESSED.inc(len(data))
        return data
    except requests.RequestException:
        DATALAKE_UNAVAILABLE.set(1)
        raise HTTPException(status_code=503, detail="Data lake unreachable")


def compute_feature_stats(records):
    """Compute mean/std for features"""
    features = np.array([r["features"] for r in records])
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return {"mean": mean, "std": std, "count": len(records)}


def detect_drift(old_stats, new_stats, threshold=3.0):
    if old_stats["mean"] is None or old_stats["std"] is None:
        return False
    z_score = np.abs(new_stats["mean"] - old_stats["mean"]) / (old_stats["std"] + 1e-9)
    return np.any(z_score > threshold)


def detect_schema_changes(old_schema, new_schema):
    old_count = old_schema["num_features"]
    new_count = new_schema
    added = removed = False
    if old_count is None:
        return False, False
    if new_count > old_count:
        added = True
    elif new_count < old_count:
        removed = True
    return added, removed

# -----------------------------
# Auto-retraining logic
# -----------------------------
def auto_retrain():
    global model, scaler, feature_stats, schema
    while True:
        try:
            batch = fetch_batch()
            if not batch:
                time.sleep(10)
                continue

            # Schema detection
            new_schema_count = len(batch[0]["features"])
            added, removed = detect_schema_changes(schema, new_schema_count)
            if added:
                FEATURE_ADDED.inc()
            if removed:
                FEATURE_REMOVED.inc()
            schema["num_features"] = new_schema_count

            # Drift detection
            new_stats = compute_feature_stats(batch)
            drift = detect_drift(feature_stats, new_stats)
            DRIFT_DETECTED.set(1 if drift else 0)
            feature_stats = new_stats

            # Simulate model accuracy drop
            acc = MODEL_ACCURACY._value.get() or 0.85
            if acc < 0.8:
                # retrain model (simulation)
                RETRAIN_COUNT.inc()
                MODEL_ACCURACY.set(0.85 + random.random() * 0.1)
        except:
            time.sleep(5)

        time.sleep(15)


# -----------------------------
# API endpoints
# -----------------------------
@app.post("/predict")
def predict(record: Record):
    X = np.array(record.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {"prediction": int(pred)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# Main thread
# -----------------------------
if __name__ == "__main__":
    start_http_server(8000)
    t = threading.Thread(target=auto_retrain)
    t.daemon = True
    t.start()
    while True:
        time.sleep(10)
