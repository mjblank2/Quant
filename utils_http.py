from __future__ import annotations
import logging
import random
import time
import requests
from typing import Any, Dict, Optional
from utils_logging import structured_logger, generate_request_id

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_tries: int = 5,
    backoff_base: float = 0.5,
    timeout: float = 30.0,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    if request_id is None:
        request_id = generate_request_id()
        
    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            r = requests.get(url, params=params, timeout=timeout)
            elapsed_ms = (time.time() - t0) * 1000
            
            # Get response size
            bytes_received = len(r.content) if r.content else 0
            
            # Log the HTTP call
            structured_logger.log_http_call(
                method="GET",
                url=url,
                status_code=r.status_code,
                response_time_ms=elapsed_ms,
                bytes_received=bytes_received,
                attempt=attempt,
                request_id=request_id,
                params=params
            )
            
            if r.status_code == 200:
                return r.json()
                
            # Legacy warning for non-200 responses
            logging.warning(
                "get_json non-200 url=%s status=%s elapsed=%.3fs body=%s",
                url,
                r.status_code,
                elapsed_ms / 1000,
                (r.text or "")[:180],
            )
            
            if r.status_code in RETRY_STATUS:
                sleep = (
                    backoff_base
                    * (2 ** (attempt - 1))
                    * (1 + random.random() * 0.25)
                )
                time.sleep(min(30, sleep))
                continue
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
                "get_json exception url=%s attempt=%d: %s", url, attempt, e
            )
            sleep = (
                backoff_base
                * (2 ** (attempt - 1))
                * (1 + random.random() * 0.25)
            )
            time.sleep(min(30, sleep))
    return {}


# Async variant using httpx
async def get_json_async(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_tries: int = 5,
    backoff_base: float = 0.5,
    timeout: float = 30.0,
    request_id: Optional[str] = None,
):
    import asyncio
    import httpx
    
    if request_id is None:
        request_id = generate_request_id()

    for attempt in range(1, max_tries + 1):
        try:
            t0 = time.time()
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, params=params)
            elapsed_ms = (time.time() - t0) * 1000
            
            # Get response size
            bytes_received = len(r.content) if r.content else 0
            
            # Log the HTTP call
            structured_logger.log_http_call(
                method="GET",
                url=url,
                status_code=r.status_code,
                response_time_ms=elapsed_ms,
                bytes_received=bytes_received,
                attempt=attempt,
                request_id=request_id,
                params=params
            )
            
            if r.status_code == 200:
                return r.json()
                
            if r.status_code in RETRY_STATUS:
                sleep = (
                    backoff_base
                    * (2 ** (attempt - 1))
                    * (1 + random.random() * 0.25)
                )
                await asyncio.sleep(min(30, sleep))
                continue
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
