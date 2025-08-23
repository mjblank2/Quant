from __future__ import annotations
import asyncio, random, logging
import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}

async def get_json_async(client: httpx.AsyncClient, url: str, params: dict | None = None,
                         max_tries: int = 6, backoff_base: float = 0.5, timeout: float = 30.0) -> dict:
    params = params or {}
    for attempt in range(1, max_tries + 1):
        try:
            r = await client.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logging.warning("get_json_async non-200 url=%s status=%s body=%s", url, r.status_code, (r.text or "")[:200])
            if r.status_code in RETRY_STATUS:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        sleep = float(ra)
                    except Exception:
                        sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                else:
                    sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                await asyncio.sleep(min(60, sleep))
                continue
            return {}
        except Exception as e:
            logging.warning("get_json_async exception url=%s attempt=%d: %s", url, attempt, e)
            sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            await asyncio.sleep(min(60, sleep))
    return {}
