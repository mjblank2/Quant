"""
Health check and status API for Render deployment monitoring
"""
from __future__ import annotations
import os
import logging
import time
from datetime import datetime
from typing import Optional
from collections import defaultdict, deque

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from sqlalchemy import text
    from utils_logging import generate_request_id, structured_logger
    import config
    FASTAPI_AVAILABLE = True
    
    # Try to import db engine, but handle gracefully if DB not configured
    try:
        from db import engine
        DB_AVAILABLE = True
    except Exception:
        DB_AVAILABLE = False
        engine = None
        
except ImportError:
    FASTAPI_AVAILABLE = False
    DB_AVAILABLE = False
    engine = None
    # Define dummy classes for when FastAPI is not available
    class BaseHTTPMiddleware:
        pass

logger = logging.getLogger(__name__)

# In-memory metrics storage (for simple monitoring)
class MetricsStore:
    def __init__(self):
        self.requests = deque(maxlen=1000)  # Keep last 1000 requests
        self.request_counts = defaultdict(int)
        self.slow_requests = deque(maxlen=100)  # Keep last 100 slow requests
        self.tiny_responses = deque(maxlen=100)  # Keep last 100 tiny responses
        
    def add_request(self, request_data: Dict[str, Any]):
        request_data['timestamp'] = time.time()
        self.requests.append(request_data)
        
        # Track request counts
        path = request_data.get('path', 'unknown')
        self.request_counts[path] += 1
        
        # Track slow requests (>10s)
        if request_data.get('response_time_ms', 0) > 10000:
            self.slow_requests.append(request_data)
            
        # Track tiny responses (<100 bytes)
        if 0 < request_data.get('bytes_sent', 0) < 100:
            self.tiny_responses.append(request_data)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        now = time.time()
        recent_cutoff = now - 300  # Last 5 minutes
        
        recent_requests = [r for r in self.requests if r.get('timestamp', 0) > recent_cutoff]
        
        return {
            "total_requests": len(self.requests),
            "recent_requests_5min": len(recent_requests),
            "slow_requests_count": len(self.slow_requests),
            "tiny_responses_count": len(self.tiny_responses),
            "request_counts_by_path": dict(self.request_counts),
            "avg_response_time_ms": sum(r.get('response_time_ms', 0) for r in recent_requests) / len(recent_requests) if recent_requests else 0,
        }

# Global metrics store - always available
metrics_store = MetricsStore()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all FastAPI requests with comprehensive metrics."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = generate_request_id()
        
        # Extract request metadata
        start_time = time.time()
        method = request.method
        path = str(request.url.path)
        user_agent = request.headers.get("user-agent")
        remote_addr = request.client.host if request.client else None
        referer = request.headers.get("referer")
        
        # Log request start
        structured_logger.log_request(
            method=method,
            path=path,
            request_id=request_id,
            user_agent=user_agent,
            remote_addr=remote_addr,
            referer=referer,
            query_params=dict(request.query_params)
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Get response size (estimate from headers or default)
        bytes_sent = int(response.headers.get("content-length", 0))
        if bytes_sent == 0 and hasattr(response, 'body'):
            # Estimate size for responses without content-length
            bytes_sent = len(getattr(response, 'body', b''))
        
        # Log response
        structured_logger.log_response(
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            bytes_sent=bytes_sent,
            method=method,
            path=path,
            cache_status=response.headers.get("x-cache", None)
        )
        
        # Store metrics
        metrics_store.add_request({
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "response_time_ms": response_time_ms,
            "bytes_sent": bytes_sent,
            "user_agent": user_agent,
            "remote_addr": remote_addr
        })
        
        # Add request ID to response headers
        response.headers["x-request-id"] = request_id
        
        return response

# Global variables that are always defined
app = None

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Small-Cap Quant System API",
        description="Health checks and status monitoring for Render deployment",
        version="1.0.0"
    )
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint for Render monitoring.
        Returns 200 OK if all critical services are operational.
        """
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": os.getenv("SERVICE", "web"),
            "app_mode": os.getenv("APP_MODE", "streamlit"),
            "checks": {}
        }
        
        # Check database connectivity
        try:
            if DB_AVAILABLE and engine:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1")).scalar()
                    status["checks"]["database"] = "healthy" if result == 1 else "unhealthy"
            else:
                status["checks"]["database"] = "not_configured"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            status["checks"]["database"] = "unhealthy"
            status["status"] = "unhealthy"
        
        # Check Redis connectivity (if configured)
        try:
            redis_url = getattr(config, 'REDIS_URL', None)
            if redis_url:
                import redis
                r = redis.from_url(redis_url, socket_timeout=5)
                r.ping()
                status["checks"]["redis"] = "healthy"
            else:
                status["checks"]["redis"] = "not_configured"
        except ImportError:
            status["checks"]["redis"] = "not_available"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            status["checks"]["redis"] = "unhealthy"
            status["status"] = "unhealthy"
        
        # Return appropriate HTTP status code
        http_status = 200 if status["status"] == "healthy" else 503
        return JSONResponse(content=status, status_code=http_status)
    
    @app.get("/status")
    async def system_status() -> Dict[str, Any]:
        """
        Detailed system status including data freshness and configuration.
        """
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "service": os.getenv("SERVICE", "web"),
                "app_mode": os.getenv("APP_MODE", "streamlit"),
                "python_path": os.getenv("PYTHONPATH", ""),
            },
            "data_status": {},
            "configuration": {}
        }
        
        # Check data freshness
        allowed_tables = ["daily_bars", "features", "predictions", "universe", "trades"]
        
        try:
            if DB_AVAILABLE and engine:
                with engine.connect() as conn:
                    for table in allowed_tables:
                        try:
                            result = conn.execute(text(f"SELECT MAX(ts) FROM {table}")).scalar()
                            status["data_status"][table] = {
                                "latest_timestamp": result.isoformat() if result else None,
                                "available": result is not None
                            }
                        except Exception as e:
                            status["data_status"][table] = {
                                "latest_timestamp": None,
                                "available": False,
                                "error": str(e)
                            }
            else:
                status["data_status"]["error"] = "Database not configured"
        except Exception as e:
            status["data_status"]["error"] = f"Database query failed: {e}"
        
        # Configuration summary (non-sensitive)
        try:
            status["configuration"] = {
                "allow_shorts": getattr(config, 'ALLOW_SHORTS', False),
                "top_n": getattr(config, 'TOP_N', 25),
                "gross_leverage": getattr(config, 'GROSS_LEVERAGE', 1.0),
                "preferred_model": getattr(config, 'PREFERRED_MODEL', 'blend_v1'),
                "backtest_start": getattr(config, 'BACKTEST_START', '2019-01-01'),
                "task_queue_available": bool(getattr(config, 'REDIS_URL', None)),
            }
        except Exception as e:
            status["configuration"]["error"] = f"Configuration read failed: {e}"
        
        return status
    
    @app.get("/")
    async def root():
        """Root endpoint redirecting to health check."""
        return {"message": "Small-Cap Quant System API", "health_check": "/health", "status": "/status", "metrics": "/metrics"}
    
    @app.get("/metrics")
    async def get_metrics() -> Dict[str, Any]:
        """
        Monitoring metrics endpoint for Grafana/Datadog dashboards.
        """
        metrics = metrics_store.get_metrics_summary()
        
        # Add system metrics
        metrics.update({
            "timestamp": datetime.utcnow().isoformat(),
            "service": os.getenv("SERVICE", "web"),
            "app_mode": os.getenv("APP_MODE", "streamlit"),
        })
        
        return metrics
    
    @app.get("/metrics/slow-requests")
    async def get_slow_requests() -> Dict[str, Any]:
        """
        Get recent slow requests (>10s) for timeout analysis.
        """
        slow_requests = list(metrics_store.slow_requests)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "slow_requests_count": len(slow_requests),
            "slow_requests": [
                {
                    "request_id": req.get("request_id"),
                    "method": req.get("method"),
                    "path": req.get("path"),
                    "response_time_ms": req.get("response_time_ms"),
                    "status_code": req.get("status_code"),
                    "user_agent": req.get("user_agent"),
                    "timestamp": datetime.fromtimestamp(req.get("timestamp", 0)).isoformat()
                }
                for req in slow_requests
            ]
        }
    
    @app.get("/metrics/tiny-responses")
    async def get_tiny_responses() -> Dict[str, Any]:
        """
        Get recent tiny responses (<100 bytes) for polling/heartbeat analysis.
        """
        tiny_responses = list(metrics_store.tiny_responses)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tiny_responses_count": len(tiny_responses),
            "tiny_responses": [
                {
                    "request_id": req.get("request_id"),
                    "method": req.get("method"),
                    "path": req.get("path"),
                    "bytes_sent": req.get("bytes_sent"),
                    "response_time_ms": req.get("response_time_ms"),
                    "user_agent": req.get("user_agent"),
                    "timestamp": datetime.fromtimestamp(req.get("timestamp", 0)).isoformat()
                }
                for req in tiny_responses
            ]
        }
    
    @app.get("/metrics/client-breakdown")
    async def get_client_breakdown() -> Dict[str, Any]:
        """
        Get per-client breakdown by user agent, IP for traffic analysis.
        """
        now = time.time()
        recent_cutoff = now - 3600  # Last hour
        recent_requests = [r for r in metrics_store.requests if r.get('timestamp', 0) > recent_cutoff]
        
        # Group by user agent
        user_agent_stats = defaultdict(lambda: {"count": 0, "avg_response_time": 0, "total_bytes": 0})
        ip_stats = defaultdict(lambda: {"count": 0, "avg_response_time": 0, "total_bytes": 0})
        
        for req in recent_requests:
            ua = req.get("user_agent", "unknown")
            ip = req.get("remote_addr", "unknown")
            response_time = req.get("response_time_ms", 0)
            bytes_sent = req.get("bytes_sent", 0)
            
            # User agent stats
            user_agent_stats[ua]["count"] += 1
            user_agent_stats[ua]["total_bytes"] += bytes_sent
            user_agent_stats[ua]["avg_response_time"] = (
                user_agent_stats[ua]["avg_response_time"] * (user_agent_stats[ua]["count"] - 1) + response_time
            ) / user_agent_stats[ua]["count"]
            
            # IP stats
            ip_stats[ip]["count"] += 1
            ip_stats[ip]["total_bytes"] += bytes_sent
            ip_stats[ip]["avg_response_time"] = (
                ip_stats[ip]["avg_response_time"] * (ip_stats[ip]["count"] - 1) + response_time
            ) / ip_stats[ip]["count"]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_range_hours": 1,
            "total_requests": len(recent_requests),
            "user_agent_breakdown": dict(user_agent_stats),
            "ip_breakdown": dict(ip_stats),
        }

else:
    # Fallback for when FastAPI is not available
    def create_health_check_response():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": os.getenv("SERVICE", "web"),
            "note": "FastAPI not available, basic health check only"
        }