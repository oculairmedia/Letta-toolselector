from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
import time
import logging

from app.services.ldts_client import LDTSClient
from app.models.search import SearchRequest, SearchResponse, ToolResult, SearchError
from config.settings import settings

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)

async def get_ldts_client() -> LDTSClient:
    """Dependency to get LDTS client."""
    from app.main import ldts_client
    if ldts_client is None:
        raise HTTPException(status_code=503, detail="LDTS client not initialized")
    return ldts_client

@router.post("/search", response_model=SearchResponse)
async def search_tools(
    request: SearchRequest,
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> SearchResponse:
    """Search for tools using LDTS semantic search."""
    start_time = time.time()
    
    try:
        logger.info(f"Searching tools: query='{request.query}', limit={request.limit}")
        
        # Call LDTS search
        search_result = await ldts_client.search_tools(
            query=request.query,
            agent_id=request.agent_id,
            limit=request.limit,
            enable_reranking=request.enable_reranking,
            min_score=request.min_score
        )
        
        # Parse results
        tools_data = search_result.get("tools", [])
        reranking_applied = search_result.get("reranking_applied", False)
        
        # Convert to ToolResult objects
        results = []
        for tool in tools_data:
            tool_result = ToolResult(
                id=tool.get("id", "unknown"),
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                score=tool.get("score", 0.0),
                rerank_score=tool.get("rerank_score"),
                category=tool.get("category"),
                source=tool.get("source"),
                metadata=tool.get("metadata") if request.include_metadata else None
            )
            results.append(tool_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            limit=request.limit,
            enable_reranking=request.enable_reranking,
            reranking_applied=reranking_applied,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Search completed: found {len(results)} results in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/search", response_model=SearchResponse)
async def search_tools_get(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    agent_id: Optional[str] = Query(None, description="Agent ID for context"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    enable_reranking: bool = Query(True, description="Enable reranking of results"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum relevance score"),
    include_metadata: bool = Query(True, description="Include tool metadata"),
    ldts_client: LDTSClient = Depends(get_ldts_client)
) -> SearchResponse:
    """Search for tools using GET parameters (alternative to POST)."""
    
    # Convert GET parameters to SearchRequest
    request = SearchRequest(
        query=query,
        agent_id=agent_id,
        limit=limit,
        enable_reranking=enable_reranking,
        min_score=min_score,
        include_metadata=include_metadata
    )
    
    return await search_tools(request, ldts_client)

@router.get("/search/suggestions")
async def get_search_suggestions(
    partial: str = Query(..., min_length=1, max_length=100, description="Partial query"),
    limit: int = Query(5, ge=1, le=20, description="Number of suggestions")
) -> Dict[str, Any]:
    """Get search query suggestions based on partial input."""
    
    # Common tool categories and operations for suggestions
    common_queries = [
        "data analysis", "file operations", "web scraping", "database",
        "api integration", "text processing", "image processing", "email",
        "scheduling", "monitoring", "backup", "security", "testing",
        "documentation", "deployment", "logging", "configuration"
    ]
    
    # Simple prefix matching for suggestions
    suggestions = [
        query for query in common_queries 
        if query.lower().startswith(partial.lower())
    ][:limit]
    
    return {
        "partial": partial,
        "suggestions": suggestions,
        "count": len(suggestions),
        "timestamp": time.time()
    }