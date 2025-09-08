"""
Logging utilities for request tracking and observability
"""
from __future__ import annotations
import time
import logging
import json
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
import uuid

# Warm up the UUID generator to avoid cold-start latency
_ = uuid.uuid4()


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())


class StructuredLogger:
    """Structured logger for request/response observability."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def log_request(self,
                   method: str,
                   path: str,
                   request_id: str,
                   user_agent: Optional[str] = None,
                   remote_addr: Optional[str] = None,
                   referer: Optional[str] = None,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   **kwargs) -> None:
        """Log incoming request with structured metadata."""
        if not self.logger.isEnabledFor(logging.INFO):
            return

        log_data = {
            "event": "request_start",
            "request_id": request_id,
            "method": method,
            "path": path,
            "timestamp": datetime.utcnow().isoformat(),
            "user_agent": user_agent,
            "remote_addr": remote_addr,
            "referer": referer,
            "user_id": user_id,
            "session_id": session_id,
            **kwargs,
        }

        self.logger.info(json.dumps(log_data))
    
    def log_response(self,
                    request_id: str,
                    status_code: int,
                    response_time_ms: float,
                    bytes_sent: int,
                    method: str,
                    path: str,
                    cache_status: Optional[str] = None,
                    upstream_time_ms: Optional[float] = None,
                    **kwargs) -> None:
        """Log response with comprehensive metrics."""
        if not self.logger.isEnabledFor(logging.INFO):
            return

        log_data = {
            "event": "request_complete",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "bytes_sent": bytes_sent,
            "timestamp": datetime.utcnow().isoformat(),
            "cache_status": cache_status,
            "upstream_time_ms": upstream_time_ms,
            **kwargs,
        }

        # Flag potential timeout issues
        if response_time_ms > 10000:  # 10+ seconds
            log_data["timeout_risk"] = True
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def log_http_call(self,
                     method: str,
                     url: str, 
                     status_code: int,
                     response_time_ms: float,
                     bytes_received: int,
                     attempt: int = 1,
                     request_id: Optional[str] = None,
                     **kwargs) -> None:
        """Log outbound HTTP calls with metrics."""
        if not self.logger.isEnabledFor(logging.INFO):
            return

        log_data = {
            "event": "http_call",
            "request_id": request_id or generate_request_id(),
            "method": method,
            "url": url,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "bytes_received": bytes_received,
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }

        # Flag issues
        if response_time_ms > 10000:
            log_data["timeout_risk"] = True
        if bytes_received == 0:
            log_data["empty_response"] = True
        if bytes_received < 100:  # Tiny response - possible polling/heartbeat
            log_data["tiny_response"] = True

        if status_code >= 400 or response_time_ms > 10000:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))


@contextmanager
def request_timer():
    """Context manager for timing requests."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        return (end_time - start_time) * 1000  # Return milliseconds


# Global structured logger instance
structured_logger = StructuredLogger("quant.requests")
