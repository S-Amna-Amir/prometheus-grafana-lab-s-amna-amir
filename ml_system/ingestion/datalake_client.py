import time
import logging
import requests
from prometheus_client import Counter

DATALAKE_503 = Counter(
    "datalake_503_total",
    "Number of times datalake returned HTTP 503"
)

logging.basicConfig(
    filename="datalake_client.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DataLakeClient:
    """
    Single, enforced ingestion client.
    ONLY allowed endpoint: /records
    """

    ENDPOINT = "http://149.40.228.124:6500/records"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def fetch_batch(self, retries: int = 5, backoff: float = 2.0):
        for attempt in range(retries):
            try:
                resp = requests.get(self.ENDPOINT, timeout=self.timeout)

                if resp.status_code == 503:
                    DATALAKE_503.inc()
                    logger.warning("503 from datalake — retrying")
                    time.sleep(backoff ** attempt)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                logger.error(f"Datalake request failed: {e}")
                time.sleep(backoff ** attempt)

        logger.critical("Datalake unavailable after retries")
        raise ConnectionError("Datalake unreachable")
