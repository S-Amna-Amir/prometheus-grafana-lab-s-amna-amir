# ml_system/ingestion/ingestion_loop.py
import os
import json
import time
import logging
import numpy as np
from typing import List, Dict

from prometheus_client import Counter, Gauge, start_http_server
from datalake_client import DataLakeClient

# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------
RECORDS_PROCESSED = Counter(
    "records_processed_total",
    "Total number of individual records processed from datalake"
)

FEATURE_ADDED = Counter(
    "feature_added",
    "Number of times a new feature appeared"
)

FEATURE_REMOVED = Counter(
    "feature_removed",
    "Number of times a feature disappeared"
)

DRIFT_DETECTED = Gauge(
    "distribution_drift_detected",
    "1 if drift detected, otherwise 0"
)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    filename="ingestion_loop.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
DATA_DIR = "data/raw"
SCHEMA_FILE = "schema_tracker.json"
STATS_FILE = "feature_stats.json"

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------
# Helpers for schema and stats
# ---------------------------------------------------------
def load_json_or_default(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return default


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------
# Drift Detection (Basic Version)
# ---------------------------------------------------------
def compute_feature_stats(records: List[Dict]) -> Dict:
    """
    Computes simple distribution stats for features.
    Features look like: {"features": [...], "label": x}
    """
    features = [r["features"] for r in records]
    features = np.array(features)

    return {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "count": len(records)
    }


def detect_drift(old_stats: Dict, new_stats: Dict, threshold: float = 3.0) -> bool:
    """
    Detects drift using a simple z-score rule:
    |new_mean - old_mean| > threshold * old_std
    """
    if "mean" not in old_stats or "std" not in old_stats:
        return False  # no baseline yet

    old_mean = np.array(old_stats["mean"])
    old_std = np.array(old_stats["std"]) + 1e-9  # avoid divide by zero
    new_mean = np.array(new_stats["mean"])

    z_score = np.abs(new_mean - old_mean) / old_std
    drift = np.any(z_score > threshold)

    return bool(drift)


# ---------------------------------------------------------
# Schema Change Detection
# ---------------------------------------------------------
def detect_schema_changes(old_schema: Dict, new_schema: Dict):
    """
    Detects if number of features changed.
    Example datalake responses:
    {"features":[...],"label":0}
    """
    old_count = old_schema.get("num_features")
    new_count = new_schema.get("num_features")

    added = removed = False

    if old_count is None:
        return False, False  # initial state

    if new_count > old_count:
        added = True
    elif new_count < old_count:
        removed = True

    return added, removed


# ---------------------------------------------------------
# Main Ingestion Loop
# ---------------------------------------------------------
def run_ingestion_loop(sleep_time: float = 5.0):
    """
    Continuously fetches data batches, saves them, updates metrics,
    detects schema changes, and updates drift statistics.
    """

    # Start Prometheus exporter for this ingestion service
    start_http_server(9100)
    logger.info("Ingestion exporter running on :9100")

    client = DataLakeClient()

    # Load stored schema + stats
    saved_schema = load_json_or_default(SCHEMA_FILE, default={})
    saved_stats = load_json_or_default(STATS_FILE, default={})

    batch_id = 0

    while True:
        try:
            # Fetch from datalake
            records, schema = client.fetch_batch()

            # -------------------------------------------------------------
            # 1. Save raw batch
            # -------------------------------------------------------------
            batch_path = os.path.join(DATA_DIR, f"batch_{batch_id}.json")
            with open(batch_path, "w") as f:
                json.dump(records, f)

            logger.info(f"Saved batch {batch_id} with {len(records)} records.")

            # Metrics
            RECORDS_PROCESSED.inc(len(records))

            # -------------------------------------------------------------
            # 2. Schema tracking
            # -------------------------------------------------------------
            new_schema = {"num_features": len(records[0]["features"])}

            added, removed = detect_schema_changes(saved_schema, new_schema)

            if added:
                FEATURE_ADDED.inc()
                logger.warning("Feature added detected!")
            if removed:
                FEATURE_REMOVED.inc()
                logger.warning("Feature removed detected!")

            # Save schema always
            save_json(SCHEMA_FILE, new_schema)
            saved_schema = new_schema

            # -------------------------------------------------------------
            # 3. Drift detection
            # -------------------------------------------------------------
            new_stats = compute_feature_stats(records)

            drift = detect_drift(saved_stats, new_stats)

            if drift:
                logger.warning("Distribution drift detected!")
                DRIFT_DETECTED.set(1)
            else:
                DRIFT_DETECTED.set(0)

            # Update stats baseline
            save_json(STATS_FILE, new_stats)
            saved_stats = new_stats

            batch_id += 1

        except Exception as e:
            logger.error(f"Error during ingestion loop: {e}")

        # Sleep before next iteration
        time.sleep(sleep_time)


# ---------------------------------------------------------
# Script Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting ingestion loop...")
    run_ingestion_loop()
