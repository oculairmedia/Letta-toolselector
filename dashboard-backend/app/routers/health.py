from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import time
import asyncio

from app.services.ldts_client import LDTSClient
from config.settings import settings

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check(ldts_client: LDTSClient = Depends(lambda: None)) -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "read_only_mode": settings.READ_ONLY_MODE
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including LDTS services."""
    from app.main import ldts_client
    
    if not ldts_client:
        raise HTTPException(status_code=503, detail="LDTS client not available")
    
    try:
        ldts_health = await ldts_client.check_health()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.VERSION,
            "read_only_mode": settings.READ_ONLY_MODE,
            "services": ldts_health,
            "configuration": {
                "rate_limiting": settings.ENABLE_RATE_LIMITING,
                "reranking": settings.ENABLE_RERANKING,
                "dangerous_operations": settings.ENABLE_DANGEROUS_OPERATIONS,
                "environment": settings.ENVIRONMENT
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )