"""
LDTS-27: Search testing API endpoints

Implements the search and reranking endpoints expected by the frontend.
All operations are read-only and safe for testing.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import time
import logging
from safety import safety_check, SafetyLevel

logger = logging.getLogger(__name__)

# Create router for search endpoints
router = APIRouter(prefix="/api/v1/tools", tags=["search"])

# Request/Response models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    category: Optional[str] = Field(None, description="Filter by category")
    source: Optional[str] = Field(None, description="Filter by source")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    name: str
    description: str
    category: str
    source: str
    tags: List[str]
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int
    reranked: bool = False
    reranker_config: Optional[Dict[str, Any]] = None

class RerankRequest(BaseModel):
    """Rerank request model"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    category: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    reranker_config: Optional[Dict[str, Any]] = None

class RerankResponse(BaseModel):
    """Rerank response with comparison"""
    query: str
    original_results: List[SearchResult]
    reranked_results: List[SearchResult]
    original_search_time_ms: int
    rerank_time_ms: int
    total_time_ms: int
    reranker_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# LDTS-27: Standard tool search endpoint
@router.post("/search", response_model=SearchResponse)
@safety_check("tool_search", SafetyLevel.READ_ONLY)
async def search_tools(search_request: SearchRequest, request: Request):
    """
    Standard tool search without reranking
    
    This endpoint provides the baseline search functionality using 
    the existing LDTS Weaviate search system.
    """
    start_time = time.time()
    
    # Import rate limiting functions
    from rate_limiting import check_rate_limit, get_request_context
    
    # Check rate limit for search operations
    await check_rate_limit(request, "search")
    
    try:
        # Use request context manager for proper resource tracking
        async with get_request_context(request, "search"):
            # Import ML resources (initialized at startup)
            from main import ml_resources
            weaviate_search = ml_resources.get("weaviate_search")
            
            if not weaviate_search:
                raise HTTPException(status_code=503, detail="Search service not available")
            
            # Perform search using existing LDTS system
            search_params = {
                "query": search_request.query,
                "limit": search_request.limit,
                "category": search_request.category,
                "source": search_request.source,
                "tags": search_request.tags or []
            }
            
            # Call existing search functionality
            raw_results = await perform_standard_search(weaviate_search, search_params)
            
            # Transform to response format
            results = [
                SearchResult(
                    id=result.get("id", "unknown"),
                    name=result.get("name", "Unknown Tool"),
                    description=result.get("description", "No description"),
                    category=result.get("category", "uncategorized"),
                    source=result.get("source", "unknown"),
                    tags=result.get("tags", []),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                )
                for result in raw_results
            ]
            
            search_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Standard search completed: '{search_request.query}' -> {len(results)} results in {search_time}ms")
            
            return SearchResponse(
                query=search_request.query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
                reranked=False
            )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# LDTS-27: Search with reranking endpoint
@router.post("/search/rerank", response_model=RerankResponse)
@safety_check("tool_search_rerank", SafetyLevel.READ_ONLY)
async def search_tools_with_reranking(rerank_request: RerankRequest, request: Request):
    """
    Search with reranking for comparison testing
    
    Performs both standard search and reranked search to enable
    side-by-side comparison in the frontend.
    """
    total_start_time = time.time()
    
    # Check rate limit for reranking operations (more expensive)
    await check_rate_limit(request, "rerank")
    
    try:
        # Use request context manager for proper resource tracking
        async with get_request_context(request, "rerank"):
            from main import ml_resources
            weaviate_search = ml_resources.get("weaviate_search")
            
            if not weaviate_search:
                raise HTTPException(status_code=503, detail="Search service not available")
            
            # Step 1: Perform standard search
            original_start = time.time()
            search_params = {
                "query": rerank_request.query,
                "limit": rerank_request.limit * 2,  # Get more results for reranking
                "category": rerank_request.category,
                "source": rerank_request.source,
                "tags": rerank_request.tags or []
            }
            
            original_raw_results = await perform_standard_search(weaviate_search, search_params)
            original_search_time = int((time.time() - original_start) * 1000)
            
            # Step 2: Perform reranking
            rerank_start = time.time()
            reranked_raw_results = await perform_reranked_search(
                weaviate_search, 
                search_params,
                rerank_request.reranker_config or {}
            )
            rerank_time = int((time.time() - rerank_start) * 1000)
            
            # Transform results
            original_results = [
                SearchResult(
                    id=result.get("id", "unknown"),
                    name=result.get("name", "Unknown Tool"),
                    description=result.get("description", "No description"),
                    category=result.get("category", "uncategorized"),
                    source=result.get("source", "unknown"),
                    tags=result.get("tags", []),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                )
                for result in original_raw_results[:rerank_request.limit]
            ]
            
            reranked_results = [
                SearchResult(
                    id=result.get("id", "unknown"),
                    name=result.get("name", "Unknown Tool"),
                    description=result.get("description", "No description"),
                    category=result.get("category", "uncategorized"),
                    source=result.get("source", "unknown"),
                    tags=result.get("tags", []),
                    score=result.get("rerank_score", result.get("score", 0.0)),
                    metadata={**result.get("metadata", {}), "rerank_score": result.get("rerank_score")}
                )
                for result in reranked_raw_results[:rerank_request.limit]
            ]
            
            total_time = int((time.time() - total_start_time) * 1000)
            
            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics(original_results, reranked_results)
            
            logger.info(f"Rerank search completed: '{rerank_request.query}' -> {len(reranked_results)} results")
            
            return RerankResponse(
                query=rerank_request.query,
                original_results=original_results,
                reranked_results=reranked_results,
                original_search_time_ms=original_search_time,
                rerank_time_ms=rerank_time,
                total_time_ms=total_time,
                reranker_config=rerank_request.reranker_config or get_default_reranker_config(),
                performance_metrics=performance_metrics
            )
        
    except Exception as e:
        logger.error(f"Rerank search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rerank search failed: {str(e)}")

# Helper functions
async def perform_standard_search(weaviate_search, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Perform standard Weaviate search"""
    try:
        # Use existing LDTS search functionality
        results = await weaviate_search.search_tools_with_query_expansion(
            query=params["query"],
            limit=params["limit"],
            category_filter=params.get("category"),
            source_filter=params.get("source"),
            tags_filter=params.get("tags", [])
        )
        return results
    except Exception as e:
        logger.error(f"Standard search error: {e}")
        # Return mock results for development
        return generate_mock_results(params["query"], params["limit"])

async def perform_reranked_search(weaviate_search, params: Dict[str, Any], reranker_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Perform search with reranking"""
    try:
        # Use existing LDTS reranking functionality
        results = await weaviate_search.search_and_rerank_tools(
            query=params["query"],
            limit=params["limit"],
            category_filter=params.get("category"),
            source_filter=params.get("source"),
            tags_filter=params.get("tags", []),
            reranker_config=reranker_config
        )
        return results
    except Exception as e:
        logger.error(f"Reranked search error: {e}")
        # Return mock results for development
        mock_results = generate_mock_results(params["query"], params["limit"])
        # Add rerank scores to mock results
        for i, result in enumerate(mock_results):
            result["rerank_score"] = max(0.0, result["score"] * (0.9 + 0.2 * (i % 3)) / (i + 1))
        return mock_results

def generate_mock_results(query: str, limit: int) -> List[Dict[str, Any]]:
    """Generate mock results for development"""
    mock_tools = [
        {"name": "Social Media Scheduler", "category": "social", "source": "postiz"},
        {"name": "Content Generator", "category": "content", "source": "openai"},
        {"name": "Data Analyzer", "category": "analytics", "source": "pandas"},
        {"name": "File Manager", "category": "filesystem", "source": "internal"},
        {"name": "Email Sender", "category": "communication", "source": "smtp"},
        {"name": "Calendar Integration", "category": "scheduling", "source": "google"},
        {"name": "Task Manager", "category": "productivity", "source": "todoist"},
        {"name": "Weather API", "category": "api", "source": "openweather"},
    ]
    
    results = []
    for i in range(min(limit, len(mock_tools))):
        tool = mock_tools[i % len(mock_tools)]
        results.append({
            "id": f"tool_{i}_{tool['name'].lower().replace(' ', '_')}",
            "name": tool["name"],
            "description": f"A {tool['category']} tool for {query} related tasks",
            "category": tool["category"],
            "source": tool["source"],
            "tags": [tool["category"], "testing"],
            "score": max(0.1, 1.0 - (i * 0.1)),
            "metadata": {"mock": True, "generated_for": query}
        })
    
    return results

def calculate_performance_metrics(original: List[SearchResult], reranked: List[SearchResult]) -> Dict[str, Any]:
    """Calculate performance metrics for comparison"""
    if not original or not reranked:
        return {}
    
    # Calculate ranking changes
    rank_changes = []
    for i, reranked_item in enumerate(reranked):
        original_rank = next((j for j, orig in enumerate(original) if orig.id == reranked_item.id), -1)
        if original_rank >= 0:
            rank_changes.append(original_rank - i)  # Positive = improved ranking
    
    # Calculate score improvements
    score_improvements = []
    for reranked_item in reranked:
        original_item = next((orig for orig in original if orig.id == reranked_item.id), None)
        if original_item:
            score_improvements.append(reranked_item.score - original_item.score)
    
    return {
        "total_rank_changes": len([x for x in rank_changes if x != 0]),
        "improved_rankings": len([x for x in rank_changes if x > 0]),
        "declined_rankings": len([x for x in rank_changes if x < 0]),
        "average_rank_change": sum(rank_changes) / len(rank_changes) if rank_changes else 0,
        "average_score_improvement": sum(score_improvements) / len(score_improvements) if score_improvements else 0,
        "max_score_improvement": max(score_improvements) if score_improvements else 0,
        "reranking_effectiveness": len([x for x in rank_changes if x > 0]) / len(rank_changes) if rank_changes else 0
    }

def get_default_reranker_config() -> Dict[str, Any]:
    """Get default reranker configuration"""
    return {
        "provider": "ollama",
        "model": "qwen3-reranker-4b",
        "top_k": 10,
        "threshold": 0.5,
        "batch_size": 16
    }