from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
import time

class Tool(BaseModel):
    """Base tool model."""
    id: str = Field(..., description="Tool ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    source: str = Field(..., description="Tool source")
    category: Optional[str] = Field(None, description="Tool category")
    tags: Optional[List[str]] = Field(None, description="Tool tags")

class ExtendedTool(Tool):
    """Extended tool model with additional metadata."""
    mcp_server_name: Optional[str] = Field(None, description="MCP server name")
    last_updated: Optional[str] = Field(None, description="Last updated timestamp")
    registered_in_letta: Optional[bool] = Field(None, description="Whether registered in Letta")
    embedding_id: Optional[str] = Field(None, description="Embedding ID")

class ToolDetailResponse(ExtendedTool):
    """Detailed tool information response."""
    json_schema: Optional[Any] = Field(None, description="Tool JSON schema")
    parameters: Optional[Any] = Field(None, description="Tool parameters")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ToolBrowseRequest(BaseModel):
    """Request parameters for tool browsing."""
    page: int = Field(0, ge=0, description="Page number (0-based)")
    limit: int = Field(25, ge=1, le=100, description="Items per page")
    search: Optional[str] = Field(None, min_length=1, max_length=500, description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    source: Optional[str] = Field(None, description="Filter by source")
    mcp_server: Optional[str] = Field(None, description="Filter by MCP server")
    sort: Literal['name', 'category', 'updated', 'relevance'] = Field('name', description="Sort field")
    order: Literal['asc', 'desc'] = Field('asc', description="Sort order")

    @validator('search')
    def validate_search(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

class ToolBrowseResponse(BaseModel):
    """Response model for tool browsing."""
    tools: List[ExtendedTool] = Field(..., description="List of tools")
    total: int = Field(..., ge=0, description="Total number of tools")
    page: int = Field(..., ge=0, description="Current page number")
    limit: int = Field(..., ge=1, description="Items per page")
    has_more: bool = Field(..., description="Whether there are more pages")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")

class ExportRequest(BaseModel):
    """Request parameters for tool export."""
    format: Literal['json', 'csv'] = Field('json', description="Export format")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters to apply")

class RefreshResponse(BaseModel):
    """Response model for tool refresh operations."""
    success: bool = Field(..., description="Whether refresh was successful")
    tools_refreshed: int = Field(..., ge=0, description="Number of tools refreshed")
    timestamp: float = Field(default_factory=time.time, description="Refresh timestamp")
    message: str = Field(..., description="Status message")

class CategoryListResponse(BaseModel):
    """Response model for tool categories."""
    categories: List[str] = Field(..., description="List of available categories")
    count: int = Field(..., ge=0, description="Number of categories")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")

class SourceListResponse(BaseModel):
    """Response model for tool sources."""
    sources: List[str] = Field(..., description="List of available sources")
    count: int = Field(..., ge=0, description="Number of sources")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")