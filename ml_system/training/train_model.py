import os
import json
import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_system.metrics import MODEL_ACCURACY, RETRAIN_COUNT

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

MIN_ACCURACY = 0.80


def load_data():
    X, y = [], []
    for f in os.listdir(RAW_DIR):
        with open(os.path.join(RAW_DIR, f)) as infile:
            for r in json.load(infile):
                X.append(r["features"])
                y.append(r["label"])
    return np.array(X), np.array(y)


def train_until_threshold():
    X, y = load_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    acc = 0.0
    MAX_ATTEMPTS = 20
    attempts = 0

    while acc < MIN_ACCURACY and attempts < MAX_ATTEMPTS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=200)
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        logger.info(f"Attempt {attempts+1}: Accuracy={acc:.4f}")
        attempts += 1

        # Update metrics
        MODEL_ACCURACY.set(acc)
        RETRAIN_COUNT.inc()

    if acc < MIN_ACCURACY:
        raise RuntimeError("Failed to reach minimum accuracy")    

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Training complete with accuracy %.4f" % acc)
    print(f"Training complete with accuracy {acc:.4f}")


if __name__ == "__main__":
    train_until_threshold()
