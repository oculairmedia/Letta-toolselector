from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
import time
import logging
import aiohttp
from fastapi.responses import Response

from config.settings import settings
from app.services.ldts_client import LDTSClient

router = APIRouter(tags=["configuration"])
logger = logging.getLogger(__name__)

async def get_ldts_client() -> LDTSClient:
    """Dependency to get LDTS client."""
    from app.main import ldts_client
    if ldts_client is None:
        raise HTTPException(status_code=503, detail="LDTS client not initialized")
    return ldts_client

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

# Proxy endpoints for LDTS API compatibility
@router.get("/config/reranker")
async def proxy_get_reranker_config(ldts_client: LDTSClient = Depends(get_ldts_client)):
    """Proxy reranker config requests to LDTS API."""
    try:
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        async with ldts_client.session.get(f"{ldts_client.api_url}/api/v1/config/reranker") as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
    except Exception as e:
        logger.error(f"Proxy reranker config failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")

@router.get("/config/embedding")
async def proxy_get_embedding_config(ldts_client: LDTSClient = Depends(get_ldts_client)):
    """Proxy embedding config requests to LDTS API."""
    try:
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        async with ldts_client.session.get(f"{ldts_client.api_url}/api/v1/config/embedding") as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
    except Exception as e:
        logger.error(f"Proxy embedding config failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")

@router.get("/models/embedding")
async def proxy_get_embedding_models(ldts_client: LDTSClient = Depends(get_ldts_client)):
    """Proxy embedding models requests to LDTS API."""
    try:
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        async with ldts_client.session.get(f"{ldts_client.api_url}/api/v1/models/embedding") as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
    except Exception as e:
        logger.error(f"Proxy embedding models failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")

@router.get("/models/reranker")
async def proxy_get_reranker_models(ldts_client: LDTSClient = Depends(get_ldts_client)):
    """Proxy reranker models requests to LDTS API."""
    try:
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        async with ldts_client.session.get(f"{ldts_client.api_url}/api/v1/models/reranker") as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
    except Exception as e:
        logger.error(f"Proxy reranker models failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")

@router.get("/config/presets")
async def get_configuration_presets() -> Dict[str, Any]:
    """Get configuration presets for different use cases."""
    return {
        "presets": [
            {
                "id": "default",
                "name": "Default Configuration",
                "description": "Standard configuration for most use cases",
                "config": {
                    "search_limit": settings.DEFAULT_SEARCH_LIMIT,
                    "enable_reranking": settings.ENABLE_RERANKING,
                    "min_score": 0.0,
                    "reranker_model": settings.RERANKER_MODEL
                }
            },
            {
                "id": "high_precision",
                "name": "High Precision",
                "description": "Optimized for accuracy over speed",
                "config": {
                    "search_limit": 5,
                    "enable_reranking": True,
                    "min_score": 0.7,
                    "reranker_model": settings.RERANKER_MODEL
                }
            },
            {
                "id": "high_recall",
                "name": "High Recall",
                "description": "Optimized for finding more relevant tools",
                "config": {
                    "search_limit": settings.MAX_SEARCH_LIMIT,
                    "enable_reranking": True,
                    "min_score": 0.1,
                    "reranker_model": settings.RERANKER_MODEL
                }
            }
        ],
        "timestamp": time.time()
    }

@router.post("/config/reranker/test")
async def proxy_test_reranker_connection(
    request_data: Dict[str, Any],
    ldts_client: LDTSClient = Depends(get_ldts_client)
):
    """Proxy reranker connection test requests to LDTS API."""
    try:
        if not ldts_client.session:
            raise HTTPException(status_code=503, detail="LDTS client not initialized")
        
        async with ldts_client.session.post(
            f"{ldts_client.api_url}/api/v1/config/reranker/test",
            json=request_data
        ) as response:
            content = await response.read()
            return Response(
                content=content,
                status_code=response.status,
                headers=dict(response.headers),
                media_type="application/json"
            )
    except Exception as e:
        logger.error(f"Proxy reranker test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy request failed: {str(e)}")