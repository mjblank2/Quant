from __future__ import annotations
import time, random, requests, logging

RETRY_STATUS = {429, 500, 502, 503, 504}

def get_json(url: str, params: dict | None = None, timeout: int = 30, max_tries: int = 5, backoff_base: float = 0.5):
    params = params or {}
    last = None
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            r = requests.get(url, params=params, timeout=timeout)
            elapsed = time.time() - t0
            if r.status_code == 200:
                return r.json()
            last = (r.status_code, r.text[:200])
            logging.warning("get_json non-200 url=%s status=%s elapsed=%.3fs body=%s", url, r.status_code, elapsed, (r.text or '')[:180])
            if r.status_code in RETRY_STATUS:
                sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                time.sleep(min(30, sleep))
                continue
            return {}
        except Exception as e:
            last = repr(e)
            logging.warning("get_json exception url=%s attempt=%d: %s", url, attempt, e)
            sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            time.sleep(min(30, sleep))
    return {}

