from prometheus_client import Counter, Gauge

# Ingestion metrics
RECORDS_PROCESSED = Counter("records_processed_total", "Records ingested")
FEATURE_ADDED = Counter("feature_added_total", "Feature additions")
FEATURE_REMOVED = Counter("feature_removed_total", "Feature removals")
DRIFT_DETECTED = Gauge("distribution_drift_detected", "1 if drift detected")
DATALAKE_UNAVAILABLE = Counter("datalake_unavailable", "Number of times datalake is unavailable")

# API / model metrics
PREDICTIONS = Counter("predictions_total", "Total predictions served")
MODEL_READY = Gauge("model_ready", "1 if model loaded")
MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy")
RETRAIN_COUNT = Counter("retrain_count_total", "Number of retraining events")
RESPONSE_DELAY = Gauge("response_delay_seconds", "Time to serve a prediction")
