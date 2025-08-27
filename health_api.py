"""
Health check and status API for Render deployment monitoring
"""
from __future__ import annotations
import os
import logging
from datetime import datetime
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from sqlalchemy import text
    from db import engine
    import config
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Small-Cap Quant System API",
        description="Health checks and status monitoring for Render deployment",
        version="1.0.0"
    )

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
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                status["checks"]["database"] = "healthy" if result == 1 else "unhealthy"
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
        return {"message": "Small-Cap Quant System API", "health_check": "/health", "status": "/status"}

else:
    # Fallback for when FastAPI is not available
    def create_health_check_response():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": os.getenv("SERVICE", "web"),
            "note": "FastAPI not available, basic health check only"
        }