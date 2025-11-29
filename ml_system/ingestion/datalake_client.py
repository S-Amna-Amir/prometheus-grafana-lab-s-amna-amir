# ml_system/ingestion/datalake_client.py
import time
import json
import logging
import requests
from prometheus_client import Counter

# -------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------
DATALAKE_UNAVAILABLE = Counter(
    "datalake_unavailable",
    "Number of times the datalake returned 503"
)

# -------------------------------------------------
# Logger Setup
# -------------------------------------------------
logging.basicConfig(
    filename="datalake_client.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DataLakeClient:
    """
    Client responsible for fetching batches from the datalake endpoint.
    Includes retry logic, 503 handling, and schema/record parsing.
    """

    def __init__(self, base_url: str = "http://149.40.228.124:6500", timeout: int = 10):
        self.endpoint = f"{base_url}/records"
        self.timeout = timeout

    # -------------------------------------------------
    # Fetch Batch
    # -------------------------------------------------
    def fetch_batch(self, max_retries: int = 5, backoff_factor: float = 1.5):
        """
        Fetch a batch of records from the datalake.

        Returns:
            tuple: (records: list[dict], schema: dict)

        Raises:
            ConnectionError if max retries are exceeded.
        """
        attempt = 0

        while attempt < max_retries:
            try:
                response = requests.get(self.endpoint, timeout=self.timeout)

                # ------------------------
                # Handle 503 explicitly
                # ------------------------
                if response.status_code == 503:
                    DATALAKE_UNAVAILABLE.inc()
                    logger.warning("Datalake unavailable (503). Retrying...")
                    time.sleep(backoff_factor ** attempt)
                    attempt += 1
                    continue

                # ------------------------
                # Other non-200 errors
                # ------------------------
                if response.status_code != 200:
                    logger.error(
                        f"Unexpected status code {response.status_code}: {response.text}"
                    )
                    time.sleep(backoff_factor ** attempt)
                    attempt += 1
                    continue

                # ------------------------
                # Successful response
                # ------------------------
                data = response.json()

                if "records" not in data or "schema" not in data:
                    logger.error("Invalid response format. Expected keys: records, schema.")
                    raise ValueError("Invalid datalake response format.")

                logger.info(
                    f"Fetched {len(data['records'])} records from datalake."
                )
                return data["records"], data["schema"]

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error while fetching datalake batch: {e}")
                time.sleep(backoff_factor ** attempt)
                attempt += 1

        # After retries exhausted:
        logger.critical("Max retries exceeded. Datalake still unavailable.")
        raise ConnectionError("Unable to contact datalake after repeated attempts.")


# -------------------------------------------------
# Standalone Test Mode
# -------------------------------------------------
if __name__ == "__main__":
    client = DataLakeClient()

    try:
        records, schema = client.fetch_batch()
        print("Received schema:", json.dumps(schema, indent=2))
        print(f"Received {len(records)} records.")
    except Exception as e:
        print("Error:", e)
