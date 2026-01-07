import os
import json
import time
import logging
import numpy as np
from prometheus_client import start_http_server
from ml_system.ingestion.datalake_client import DataLakeClient
from ml_system.metrics import (
    RECORDS_PROCESSED,
    FEATURE_ADDED,
    FEATURE_REMOVED,
    DRIFT_DETECTED,
    DATALAKE_UNAVAILABLE
)

logging.basicConfig(
    filename="ingestion_loop.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/raw"
SCHEMA_FILE = "schema.json"
STATS_FILE = "stats.json"
os.makedirs(DATA_DIR, exist_ok=True)


def compute_stats(records):
    X = np.array([r["features"] for r in records])
    return {
        "mean": X.mean(axis=0).tolist(),
        "std": X.std(axis=0).tolist(),
        "count": len(records)
    }


def detect_drift(old, new, threshold=3.0):
    if not old:
        return False
    old_mean = np.array(old["mean"])
    old_std = np.array(old["std"]) + 1e-9
    new_mean = np.array(new["mean"])
    return bool(np.any(np.abs(new_mean - old_mean) / old_std > threshold))


def run():
    start_http_server(9100)
    client = DataLakeClient()
    schema = {}
    stats = {}
    batch_id = 0

    while True:
        try:
            records = client.fetch_batch()
            RECORDS_PROCESSED.inc(len(records))

            if not records:
                logger.warning("Empty batch received")
                time.sleep(5)
                continue

            with open(f"{DATA_DIR}/batch_{batch_id}.json", "w") as f:
                json.dump(records, f)

            new_schema = {"num_features": len(records[0]["features"])}
            if schema:
                if new_schema["num_features"] > schema["num_features"]:
                    FEATURE_ADDED.inc()
                elif new_schema["num_features"] < schema["num_features"]:
                    FEATURE_REMOVED.inc()

            new_stats = compute_stats(records)
            drift = detect_drift(stats, new_stats)
            DRIFT_DETECTED.set(1 if drift else 0)

            schema, stats = new_schema, new_stats
            json.dump(schema, open(SCHEMA_FILE, "w"))
            json.dump(stats, open(STATS_FILE, "w"))

            batch_id += 1

        except ConnectionError as e:
            DATALAKE_UNAVAILABLE.inc()
            logger.critical(f"Fatal ingestion error: {e}")
            print(f"Fatal ingestion error: {e}")
            break

        except Exception as e:
            logger.error(f"Non-fatal ingestion error: {e}")
            print(f"Non-fatal ingestion error: {e}")

        time.sleep(5)


if __name__ == "__main__":
    run()
