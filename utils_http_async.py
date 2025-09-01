from __future__ import annotations
import asyncio
import logging
import random
import time
import aiohttp
from utils_logging import structured_logger, generate_request_id

RETRY_STATUS = {429, 500, 502, 503, 504}


async def get_json_async(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
    timeout: float = 20.0,
    max_tries: int = 4,
    backoff_base: float = 0.4,
    request_id: str | None = None,
) -> dict:
    if request_id is None:
        request_id = generate_request_id()
        
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            async with session.get(
                url, params=params, timeout=timeout
            ) as resp:
                txt = await resp.text()
                elapsed_ms = (time.time() - t0) * 1000
                
                # Get response size
                bytes_received = len(txt.encode('utf-8')) if txt else 0
                
                # Log the HTTP call
                structured_logger.log_http_call(
                    method="GET",
                    url=url,
                    status_code=resp.status,
                    response_time_ms=elapsed_ms,
                    bytes_received=bytes_received,
                    attempt=attempt,
                    request_id=request_id,
                    params=params
                )
                
                if resp.status == 200:
                    try:
                        return await resp.json()
                    except Exception:
                        import json
                        return json.loads(txt)
                        
                logging.warning(
                    "get_json_async non-200 url=%s status=%s body=%s",
                    url,
                    resp.status,
                    txt[:180],
                )
                
                if resp.status in RETRY_STATUS:
                    sleep = (
                        backoff_base
                        * (2 ** (attempt - 1))
                        * (1 + random.random() * 0.25)
                    )
                    await asyncio.sleep(min(30, sleep))
                else:
                    return {}
                    
        except Exception as e:
            # Log exception with structured logging
            structured_logger.log_http_call(
                method="GET",
                url=url,
                status_code=0,
                response_time_ms=0,
                bytes_received=0,
                attempt=attempt,
                request_id=request_id,
                error=str(e),
                params=params
            )
            
            logging.warning(
                "get_json_async exception url=%s attempt=%d: %s",
                url,
                attempt,
                e,
            )
            sleep = (
                backoff_base
                * (2 ** (attempt - 1))
                * (1 + random.random() * 0.25)
            )
            await asyncio.sleep(min(30, sleep))
    return {}
