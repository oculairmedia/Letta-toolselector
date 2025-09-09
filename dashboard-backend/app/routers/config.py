from fastapi import APIRouter, Depends
from typing import Dict, Any
import time

from config.settings import settings

router = APIRouter(tags=["configuration"])

@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """Get current backend configuration (safe subset)."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_version": settings.API_V1_STR,
        "environment": settings.ENVIRONMENT,
        "configuration": {
            "read_only_mode": settings.READ_ONLY_MODE,
            "dangerous_operations_enabled": settings.ENABLE_DANGEROUS_OPERATIONS,
            "rate_limiting": {
                "enabled": settings.ENABLE_RATE_LIMITING,
                "requests_per_window": settings.RATE_LIMIT_REQUESTS,
                "window_seconds": settings.RATE_LIMIT_WINDOW
            },
            "search": {
                "default_limit": settings.DEFAULT_SEARCH_LIMIT,
                "max_limit": settings.MAX_SEARCH_LIMIT,
                "query_expansion_enabled": settings.ENABLE_QUERY_EXPANSION
            },
            "reranking": {
                "enabled": settings.ENABLE_RERANKING,
                "model": settings.RERANKER_MODEL,
                "timeout_seconds": settings.RERANKER_TIMEOUT
            },
            "cors": {
                "allowed_origins": settings.BACKEND_CORS_ORIGINS
            }
        },
        "endpoints": {
            "health": f"{settings.API_V1_STR}/health",
            "search": f"{settings.API_V1_STR}/search",
            "rerank": f"{settings.API_V1_STR}/rerank",
            "docs": f"{settings.API_V1_STR}/docs"
        },
        "timestamp": time.time()
    }

@router.get("/config/limits")
async def get_limits() -> Dict[str, Any]:
    """Get API limits and constraints."""
    return {
        "rate_limiting": {
            "enabled": settings.ENABLE_RATE_LIMITING,
            "requests_per_window": settings.RATE_LIMIT_REQUESTS,
            "window_seconds": settings.RATE_LIMIT_WINDOW
        },
        "search_limits": {
            "default_limit": settings.DEFAULT_SEARCH_LIMIT,
            "max_limit": settings.MAX_SEARCH_LIMIT,
            "min_score": 0.0,
            "max_score": 1.0
        },
        "safety": {
            "read_only_mode": settings.READ_ONLY_MODE,
            "dangerous_operations": settings.ENABLE_DANGEROUS_OPERATIONS
        },
        "timeouts": {
            "reranker_timeout": settings.RERANKER_TIMEOUT,
            "default_request_timeout": 30
        },
        "timestamp": time.time()
    }