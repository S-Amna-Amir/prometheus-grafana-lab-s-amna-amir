# ml_system/training/train_model.py
import os
import json
import joblib
import logging
import numpy as np
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from prometheus_client import Gauge, Counter, CollectorRegistry, push_to_gateway

# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------
registry = CollectorRegistry()

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Accuracy of the latest trained model',
    registry=registry
)

RETRAIN_COUNT = Counter(
    'retrain_count_total',
    'Number of times the model was retrained',
    registry=registry
)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data/raw")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(os.path.dirname(__file__), "..", "feature_list.json")


MIN_ACCURACY = 0.80
MAX_RETRIES = 5

PUSHGATEWAY = "http://localhost:9091"  # optional; can remove if not using pushgateway


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
def load_batches() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all JSON batches in data/raw/ and returns X, y.
    """
    X = []
    y = []

    files = sorted(os.listdir(RAW_DATA_DIR))

    if not files:
        raise RuntimeError("No ingestion data found in data/raw/. Cannot train model.")

    for f in files:
        if not f.endswith(".json"):
            continue

        path = os.path.join(RAW_DATA_DIR, f)
        with open(path, "r") as infile:
            records = json.load(infile)

        for rec in records:
            X.append(rec["features"])
            y.append(rec["label"])

    X = np.array(X)
    y = np.array(y)

    return X, y


# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------
def preprocess(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardizes the features and returns scaled version + fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train_model() -> float:
    """
    Trains a RandomForest classifier until accuracy >= MIN_ACCURACY.
    Saves model, scaler, and feature list.
    Returns final accuracy.
    """
    logger.info("Loading data...")
    X, y = load_batches()

    logger.info(f"Loaded dataset: {len(X)} samples, {X.shape[1]} features.")

    # Track feature list
    feature_list = list(range(X.shape[1]))
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_list, f, indent=2)

    retries = 0
    accuracy = 0.0

    # Retrain loop
    while accuracy < MIN_ACCURACY and retries < MAX_RETRIES:
        logger.info(f"Training attempt {retries + 1}...")

        # Preprocess
        X_scaled, scaler = preprocess(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Model
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        logger.info(f"Accuracy: {accuracy:.4f}")

        retries += 1

        if accuracy >= MIN_ACCURACY:
            break

    # ---------------------------------------------------------
    # Save Artifacts
    # ---------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Model saved to model.joblib")
    logger.info("Scaler saved to scaler.joblib")

    # ---------------------------------------------------------
    # Update Prometheus Metrics
    # ---------------------------------------------------------
    RETRAIN_COUNT.inc()
    MODEL_ACCURACY.set(accuracy)

    try:
        push_to_gateway(PUSHGATEWAY, job='ml_training', registry=registry)
        logger.info("Pushed metrics to Pushgateway.")
    except Exception as e:
        logger.warning(f"Could not push metrics to Pushgateway: {e}")

    logger.info(f"Final accuracy: {accuracy:.4f}")
    return accuracy


# ---------------------------------------------------------
# CLI Run
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting model training...")
    acc = train_model()
    print(f"Training complete. Final accuracy = {acc:.4f}")
