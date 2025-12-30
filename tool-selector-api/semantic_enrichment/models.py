"""
Data Models for Semantic Enrichment System

Pydantic models for MCP server profiles, enriched tools, and enrichment results.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MCPServerProfile(BaseModel):
    """
    Semantic profile for an MCP server.
    
    Contains domain context, capabilities, and semantic keywords that 
    provide context for all tools from this server.
    """
    server_name: str = Field(..., description="MCP server name (e.g., 'huly', 'bookstack')")
    domain: str = Field(..., description="Primary domain (e.g., 'project management')")
    primary_capabilities: List[str] = Field(
        default_factory=list,
        description="Main capabilities (e.g., ['issue tracking', 'sprint planning'])"
    )
    entity_types: List[str] = Field(
        default_factory=list,
        description="Entity types handled (e.g., ['issue', 'project', 'sprint'])"
    )
    action_verbs: List[str] = Field(
        default_factory=list,
        description="Common actions (e.g., ['create', 'update', 'query'])"
    )
    integration_context: str = Field(
        "",
        description="How this server fits into workflows"
    )
    semantic_keywords: List[str] = Field(
        default_factory=list,
        description="Related search terms users might use"
    )
    profile_hash: str = Field(
        "",
        description="Hash of tool list for change detection"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this profile was last updated"
    )
    tool_count: int = Field(0, description="Number of tools in this server")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EnrichedTool(BaseModel):
    """
    Semantically enriched tool with enhanced search fields.
    """
    tool_id: str = Field(..., description="Unique tool identifier")
    name: str = Field(..., description="Tool name")
    original_description: str = Field("", description="Original tool description")
    mcp_server_name: Optional[str] = Field(None, description="Source MCP server")
    
    # Enrichment fields
    enhanced_description: str = Field(
        "",
        description="LLM-generated rich description (200-400 words)"
    )
    action_entities: List[str] = Field(
        default_factory=list,
        description="Action-entity pairs (e.g., ['create issue', 'update project'])"
    )
    semantic_keywords: List[str] = Field(
        default_factory=list,
        description="Search terms from server + tool enrichment"
    )
    use_cases: List[str] = Field(
        default_factory=list,
        description="Natural language use case scenarios"
    )
    server_domain: str = Field(
        "",
        description="Domain from MCP server profile"
    )
    
    # Metadata
    enrichment_hash: str = Field(
        "",
        description="Hash of tool content for change detection"
    )
    last_enriched: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this tool was last enriched"
    )
    enrichment_model: str = Field(
        "",
        description="Model used for enrichment (e.g., 'claude-sonnet-4-20250514')"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EnrichmentResult(BaseModel):
    """
    Result of an enrichment operation.
    """
    success: bool = Field(..., description="Whether enrichment succeeded")
    tool_id: str = Field(..., description="Tool that was enriched")
    tool_name: str = Field("", description="Tool name")
    
    # Enrichment output (if successful)
    enhanced_description: Optional[str] = None
    action_entities: List[str] = Field(default_factory=list)
    semantic_keywords: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    
    # Error info (if failed)
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Timing
    duration_ms: float = Field(0, description="Time taken in milliseconds")
    tokens_used: int = Field(0, description="API tokens consumed")
    
    @classmethod
    def from_error(cls, tool_id: str, tool_name: str, error: Exception) -> "EnrichmentResult":
        """Create a failed result from an exception."""
        return cls(
            success=False,
            tool_id=tool_id,
            tool_name=tool_name,
            error=str(error),
            error_type=type(error).__name__,
            duration_ms=0,
            tokens_used=0
        )


class ServerProfileResult(BaseModel):
    """
    Result of profiling an MCP server.
    """
    success: bool
    server_name: str
    profile: Optional[MCPServerProfile] = None
    error: Optional[str] = None
    duration_ms: float = 0
    tokens_used: int = 0
    tools_analyzed: int = 0
    
    @classmethod
    def from_error(cls, server_name: str, error: Exception) -> "ServerProfileResult":
        """Create a failed result from an exception."""
        return cls(
            success=False,
            server_name=server_name,
            error=str(error),
            duration_ms=0,
            tokens_used=0
        )


class EnrichmentStats(BaseModel):
    """
    Statistics for an enrichment run.
    """
    total_servers: int = 0
    servers_profiled: int = 0
    servers_cached: int = 0
    servers_failed: int = 0
    
    total_tools: int = 0
    tools_enriched: int = 0
    tools_cached: int = 0
    tools_failed: int = 0
    
    total_duration_ms: float = 0
    total_tokens_used: int = 0
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
