"""Bound retries to transient network errors while loading models and datasets."""

import time

from aiohttp import ClientConnectionError, ClientPayloadError
from httpx import HTTPStatusError, TransportError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout


def load_with_retries(operation, description: str, max_attempts: int = 3, delay: float = 10):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except (
            ConnectionError,
            TimeoutError,
            RequestsConnectionError,
            Timeout,
            HTTPError,
            HTTPStatusError,
            TransportError,
            ClientConnectionError,
            ClientPayloadError,
        ) as error:
            response = getattr(error, "response", None)
            if response is not None:
                status = response.status_code
                if status not in (408, 429) and not 500 <= status < 600:
                    raise
            if attempt == max_attempts:
                raise
            print(f"Error {description} (attempt {attempt}/{max_attempts}): {error}")
            time.sleep(delay)
