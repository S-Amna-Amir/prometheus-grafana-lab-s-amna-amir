from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time, random, threading

# -----------------------------
# Metrics
# -----------------------------
REQUESTS = Counter("records_processed_total", "Total records processed from datalake")
RETRAIN_COUNT = Counter("retrain_count_total", "Number of times model was retrained")
FEATURE_ADDED = Counter("feature_added", "Number of times a new feature appeared")
FEATURE_REMOVED = Counter("feature_removed", "Number of times a feature disappeared")
DRIFT_DETECTED = Gauge("distribution_drift_detected", "1 if distribution drift detected")
MODEL_ACCURACY = Gauge("model_accuracy", "Current accuracy of ML model")
DATALAKE_UNAVAILABLE = Gauge("datalake_unavailable", "1 if datalake returns 503")
RESPONSE_DELAY = Histogram("response_delay_seconds", "Time to get datalake response")

# -----------------------------
# Simulate ML / Data Loop
# -----------------------------
def simulate_metrics():
    while True:
        # Simulate processing batch
        REQUESTS.inc(random.randint(5, 20))
        RETRAIN_COUNT.inc(random.randint(0, 1))
        FEATURE_ADDED.inc(random.randint(0, 1))
        FEATURE_REMOVED.inc(random.randint(0, 1))
        DRIFT_DETECTED.set(random.choice([0, 1]))
        MODEL_ACCURACY.set(random.uniform(0.75, 0.95))
        DATALAKE_UNAVAILABLE.set(random.choice([0, 1]))
        RESPONSE_DELAY.observe(random.gauss(0.2, 0.1))
        time.sleep(5)

if __name__ == '__main__':
    # Start Prometheus server
    start_http_server(8000)
    t = threading.Thread(target=simulate_metrics)
    t.daemon = True
    t.start()

    # Keep main thread alive
    while True:
        time.sleep(10)
