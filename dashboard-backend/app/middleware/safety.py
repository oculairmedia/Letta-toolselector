from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
from typing import Callable

from config.settings import settings

logger = logging.getLogger(__name__)

class SafetyMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce read-only mode and other safety measures."""
    
    # Methods that modify data
    WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    
    # Endpoints that are allowed even in read-only mode
    READ_ONLY_ALLOWED_PATHS = {
        "/",
        f"{settings.API_V1_STR}/info",
        f"{settings.API_V1_STR}/health",
        f"{settings.API_V1_STR}/config",
        f"{settings.API_V1_STR}/search",
        f"{settings.API_V1_STR}/rerank",
        f"{settings.API_V1_STR}/docs",
        f"{settings.API_V1_STR}/openapi.json",
        f"{settings.API_V1_STR}/redoc"
    }
    
    # Dangerous operations that require explicit enablement
    DANGEROUS_PATHS = {
        f"{settings.API_V1_STR}/tools/attach",
        f"{settings.API_V1_STR}/tools/detach", 
        f"{settings.API_V1_STR}/tools/delete",
        f"{settings.API_V1_STR}/agents/modify",
        f"{settings.API_V1_STR}/agents/delete"
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through safety checks."""
        
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        try:
            # Check read-only mode
            if settings.READ_ONLY_MODE:
                if method in self.WRITE_METHODS:
                    # Allow specific read-only endpoints
                    if not any(path.startswith(allowed) for allowed in self.READ_ONLY_ALLOWED_PATHS):
                        logger.warning(f"Blocked {method} {path} - read-only mode enabled")
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": "Write operations disabled in read-only mode",
                                "detail": f"{method} {path} not allowed",
                                "read_only_mode": True,
                                "timestamp": time.time()
                            }
                        )
            
            # Check dangerous operations
            if not settings.ENABLE_DANGEROUS_OPERATIONS:
                if any(path.startswith(dangerous) for dangerous in self.DANGEROUS_PATHS):
                    logger.warning(f"Blocked {method} {path} - dangerous operations disabled")
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Dangerous operations disabled for safety",
                            "detail": f"{method} {path} requires ENABLE_DANGEROUS_OPERATIONS=true",
                            "dangerous_operation": True,
                            "timestamp": time.time()
                        }
                    )
            
            # Add safety headers
            response = await call_next(request)
            
            # Add safety information to response headers
            response.headers["X-Read-Only-Mode"] = str(settings.READ_ONLY_MODE)
            response.headers["X-Dangerous-Operations"] = str(settings.ENABLE_DANGEROUS_OPERATIONS)
            response.headers["X-Process-Time"] = str(time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Safety middleware error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal safety middleware error",
                    "timestamp": time.time()
                }
            )