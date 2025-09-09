from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import time
import logging

from app.services.ldts_client import LDTSClient
from app.models.rerank import RerankRequest, RerankResponse, RerankError
from config.settings import settings

router = APIRouter(tags=["reranking"])
logger = logging.getLogger(__name__)

async def get_ldts_client() -> LDTSClient:
    """Dependency to get LDTS client."""
    from app.main import ldts_client
    if ldts_client is None:
        raise HTTPException(status_code=503, detail="LDTS client not initialized")
    return ldts_client

@router.post("/rerank", response_model=RerankResponse)
async def rerank_tools(
    request: RerankRequest,
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> RerankResponse:
    """Rerank a list of tools based on relevance to query."""
    
    if not settings.ENABLE_RERANKING:
        raise HTTPException(
            status_code=400,
            detail="Reranking is disabled"
        )
    
    start_time = time.time()
    original_count = len(request.tools)
    
    try:
        logger.info(f"Reranking {original_count} tools for query: '{request.query}'")
        
        # Call LDTS reranking
        reranked_tools = await ldts_client.rerank_tools(
            query=request.query,
            tools=request.tools,
            model=request.model
        )
        
        # Add rerank scores if requested and available
        if not request.include_scores:
            for tool in reranked_tools:
                tool.pop("rerank_score", None)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = RerankResponse(
            query=request.query,
            reranked_tools=reranked_tools,
            model_used=request.model or settings.RERANKER_MODEL,
            original_count=original_count,
            reranked_count=len(reranked_tools),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Reranking completed: {len(reranked_tools)} results in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}"
        )

@router.get("/rerank/models")
async def get_available_models() -> Dict[str, Any]:
    """Get available reranking models."""
    return {
        "models": [
            {
                "name": "bge-reranker-large",
                "description": "BGE Large Reranker - High quality general purpose reranker",
                "default": True
            },
            {
                "name": "bge-reranker-base", 
                "description": "BGE Base Reranker - Faster, smaller reranker",
                "default": False
            }
        ],
        "default_model": settings.RERANKER_MODEL,
        "reranking_enabled": settings.ENABLE_RERANKING,
        "reranker_url": settings.RERANKER_URL if settings.ENABLE_RERANKING else None,
        "timestamp": time.time()
    }

@router.get("/rerank/config")
async def get_rerank_config() -> Dict[str, Any]:
    """Get reranking configuration."""
    return {
        "enabled": settings.ENABLE_RERANKING,
        "default_model": settings.RERANKER_MODEL,
        "timeout_seconds": settings.RERANKER_TIMEOUT,
        "reranker_url": settings.RERANKER_URL if settings.ENABLE_RERANKING else None,
        "max_tools_per_request": 100,
        "supported_models": ["bge-reranker-large", "bge-reranker-base"],
        "timestamp": time.time()
    }