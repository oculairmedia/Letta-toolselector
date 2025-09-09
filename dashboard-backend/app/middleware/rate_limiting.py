from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import time
import logging
from typing import Dict, Callable
from collections import defaultdict, deque

from config.settings import settings

logger = logging.getLogger(__name__)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app):
        super().__init__(app)
        # In-memory rate limiting storage (use Redis in production)
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use client IP as identifier
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        async with self.lock:
            now = time.time()
            window_start = now - settings.RATE_LIMIT_WINDOW
            
            # Get or create request queue for client
            requests = self.request_counts[client_id]
            
            # Remove old requests outside the time window
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # Check if rate limit exceeded
            if len(requests) >= settings.RATE_LIMIT_REQUESTS:
                return True
            
            # Add current request
            requests.append(now)
            return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting."""
        
        # Skip rate limiting for health checks and static content
        if request.url.path in ["/health", f"{settings.API_V1_STR}/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        try:
            if await self._is_rate_limited(client_id):
                logger.warning(f"Rate limit exceeded for client {client_id}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Maximum {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds",
                        "retry_after": settings.RATE_LIMIT_WINDOW,
                        "timestamp": time.time()
                    },
                    headers={
                        "Retry-After": str(settings.RATE_LIMIT_WINDOW),
                        "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
                        "X-RateLimit-Window": str(settings.RATE_LIMIT_WINDOW)
                    }
                )
            
            response = await call_next(request)
            
            # Add rate limit headers to response
            async with self.lock:
                remaining = max(0, settings.RATE_LIMIT_REQUESTS - len(self.request_counts[client_id]))
            
            response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Window"] = str(settings.RATE_LIMIT_WINDOW)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}", exc_info=True)
            return await call_next(request)