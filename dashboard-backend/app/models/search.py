from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import time

class SearchRequest(BaseModel):
    """Request model for tool search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    agent_id: Optional[str] = Field(None, description="Agent ID for context")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    enable_reranking: bool = Field(True, description="Enable reranking of results")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    include_metadata: bool = Field(True, description="Include tool metadata in results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class ToolResult(BaseModel):
    """Individual tool search result."""
    id: str = Field(..., description="Tool ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    rerank_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Reranking score")
    category: Optional[str] = Field(None, description="Tool category")
    source: Optional[str] = Field(None, description="Tool source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Response model for tool search."""
    query: str = Field(..., description="Original search query")
    results: List[ToolResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total number of results found")
    limit: int = Field(..., ge=1, description="Result limit applied")
    enable_reranking: bool = Field(..., description="Whether reranking was enabled")
    reranking_applied: bool = Field(..., description="Whether reranking was actually applied")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    
class SearchError(BaseModel):
    """Error response for search operations."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    query: Optional[str] = Field(None, description="Original query that caused error")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")