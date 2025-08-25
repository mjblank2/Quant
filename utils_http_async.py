from __future__ import annotations
import asyncio
import random
import logging
import aiohttp

RETRY_STATUS = {429, 500, 502, 503, 504}

async def get_json_async(session: aiohttp.ClientSession, url: str, params: dict | None = None, timeout: float = 20.0, max_tries: int = 4, backoff_base: float = 0.4) -> dict:
    for attempt in range(1, max_tries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                txt = await resp.text()
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception:
                        import json
                        return json.loads(txt)
                logging.warning("get_json_async non-200 url=%s status=%s body=%s", url, resp.status, txt[:180])
                if resp.status in RETRY_STATUS:
                    sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
                    await asyncio.sleep(min(30, sleep))
                else:
                    return {}
        except Exception as e:
            logging.warning("get_json_async exception url=%s attempt=%d: %s", url, attempt, e)
            sleep = backoff_base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            await asyncio.sleep(min(30, sleep))
    return {}
