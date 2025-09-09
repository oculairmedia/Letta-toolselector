from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import time

class RerankRequest(BaseModel):
    """Request model for reranking tools."""
    query: str = Field(..., min_length=1, max_length=500, description="Query for reranking")
    tools: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="Tools to rerank")
    model: Optional[str] = Field(None, description="Reranker model to use")
    include_scores: bool = Field(True, description="Include reranking scores in response")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('tools')
    def validate_tools(cls, v):
        if not v:
            raise ValueError('Tools list cannot be empty')
        
        for i, tool in enumerate(v):
            if not isinstance(tool, dict):
                raise ValueError(f'Tool {i} must be a dictionary')
            if 'id' not in tool:
                raise ValueError(f'Tool {i} must have an id field')
            if 'name' not in tool and 'description' not in tool:
                raise ValueError(f'Tool {i} must have either name or description')
        
        return v

class RerankResponse(BaseModel):
    """Response model for reranking."""
    query: str = Field(..., description="Original query")
    reranked_tools: List[Dict[str, Any]] = Field(..., description="Reranked tools")
    model_used: Optional[str] = Field(None, description="Reranker model used")
    original_count: int = Field(..., ge=0, description="Number of input tools")
    reranked_count: int = Field(..., ge=0, description="Number of output tools")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")

class RerankError(BaseModel):
    """Error response for reranking operations."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    query: Optional[str] = Field(None, description="Original query that caused error")
    model: Optional[str] = Field(None, description="Model that failed")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")