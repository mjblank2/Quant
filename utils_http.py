from __future__ import annotations
import time
import random
import requests
import logging
from typing import Any, Dict, Optional

RETRY_STATUS = {429, 500, 502, 503, 504}

def get_json(url: str, params: Optional[Dict[str, Any]] = None, max_tries: int = 5, backoff_base: float = 0.5, timeout: float = 30.0) -> Dict[str, Any]:
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            r = requests.get(url, params=params, timeout=timeout)
            elapsed = time.time() - t0
            if r.status_code == 200:
                return r.json()
            logging.warning("get_json non-200 url=%s status=%s elapsed=%.3fs body=%s", url, r.status_code, elapsed, (r.text or '')[:180])
            if r.status_code in RETRY_STATUS:
                sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                time.sleep(min(30, sleep))
                continue
            return {}
        except Exception as e:
            logging.warning("get_json exception url=%s attempt=%d: %s", url, attempt, e)
            sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            time.sleep(min(30, sleep))
    return {}

# Async variant using httpx
async def get_json_async(url: str, params: Optional[Dict[str, Any]] = None, max_tries: int = 5, backoff_base: float = 0.5, timeout: float = 30.0):
    import asyncio
    import httpx
    for attempt in range(1, max_tries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUS:
                sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                await asyncio.sleep(min(30, sleep))
                continue
            return {}
        except Exception as e:
            logging.warning("get_json_async exception url=%s attempt=%d: %s", url, attempt, e)
            sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            await asyncio.sleep(min(30, sleep))
    return {}
