"""
Data models for the Letta Tool Selector API.

This module contains Pydantic models and TypedDicts for:
- Tool representations
- API request/response structures
- Configuration objects
- Operation results

These models provide type safety and validation for the API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class ToolType(str, Enum):
    """Tool type classification."""
    EXTERNAL_MCP = "external_mcp"
    LETTA_CORE = "letta_core"
    LETTA_MEMORY_CORE = "letta_memory_core"
    LETTA_MULTI_AGENT_CORE = "letta_multi_agent_core"
    LETTA_SLEEPTIME_CORE = "letta_sleeptime_core"
    LETTA_VOICE_SLEEPTIME_CORE = "letta_voice_sleeptime_core"
    LETTA_FILES_CORE = "letta_files_core"
    LETTA_BUILTIN = "letta_builtin"
    CUSTOM = "custom"


class OperationStatus(str, Enum):
    """Generic operation status."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


# ============================================================================
# Tool Models
# ============================================================================

class Tool(BaseModel):
    """Represents a tool in the system."""
    id: Optional[str] = Field(None, description="Tool unique identifier")
    tool_id: Optional[str] = Field(None, description="Alternative tool ID field")
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    tool_type: Optional[str] = Field(None, description="Tool type classification")
    source_type: Optional[str] = Field(None, description="Source type (python, etc.)")
    mcp_server_name: Optional[str] = Field(None, description="MCP server name if external_mcp")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tool tags")
    
    @property
    def effective_id(self) -> Optional[str]:
        """Get the effective tool ID (id or tool_id)."""
        return self.id or self.tool_id
    
    def is_letta_core(self) -> bool:
        """Check if this is a Letta core tool."""
        letta_tool_types = [
            'letta_core', 'letta_voice_sleeptime_core', 'letta_sleeptime_core', 
            'letta_memory_core', 'letta_files_core', 'letta_builtin', 'letta_multi_agent_core'
        ]
        
        if self.tool_type in letta_tool_types:
            return True
        
        core_tool_names = [
            'send_message', 'conversation_search', 'archival_memory_insert', 
            'archival_memory_search', 'core_memory_append', 'core_memory_replace', 
            'pause_heartbeats', 'find_attach_tools'
        ]
        
        if self.name in core_tool_names:
            return True
        
        return False

    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class SearchResult(BaseModel):
    """Represents a tool search result with scoring."""
    id: Optional[str] = None
    tool_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    tool_type: Optional[str] = None
    mcp_server_name: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    
    # Scoring fields
    score: Optional[float] = Field(None, description="Relevance score (0-100)")
    distance: Optional[float] = Field(None, description="Vector distance (0-1)")
    rerank_score: Optional[float] = Field(None, description="Reranker score")
    
    @property
    def effective_score(self) -> float:
        """Get effective score, preferring rerank_score > score > distance-based."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.score is not None:
            return self.score
        if self.distance is not None:
            return 100 * (1 - self.distance)
        return 0.0

    class Config:
        extra = "allow"


# ============================================================================
# Operation Result Models
# ============================================================================

class AttachResult(BaseModel):
    """Result of a tool attach operation."""
    success: bool
    tool_id: Optional[str] = None
    name: Optional[str] = None
    match_score: Optional[float] = None
    error: Optional[str] = None


class DetachResult(BaseModel):
    """Result of a tool detach operation."""
    success: bool
    tool_id: str
    warning: Optional[str] = None
    error: Optional[str] = None


class ProcessToolsResult(BaseModel):
    """Result of batch process_tools operation."""
    successful_attachments: List[AttachResult] = Field(default_factory=list)
    failed_attachments: List[AttachResult] = Field(default_factory=list)
    detached_tools: List[str] = Field(default_factory=list)
    skipped_detachments: List[str] = Field(default_factory=list)


class PruningResult(BaseModel):
    """Result of tool pruning operation."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# API Request Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request body for /api/v1/tools/search endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    enable_reranking: bool = Field(default=False, description="DEPRECATED: Use /search/rerank")
    reranker_config: Optional[Dict[str, Any]] = None


class SearchWithRerankingRequest(BaseModel):
    """Request body for /api/v1/tools/search/rerank endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    reranker_top_k: int = Field(default=20, ge=1, le=100, description="Reranker top-k")
    min_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Minimum score threshold")


class AttachToolsRequest(BaseModel):
    """Request body for /api/v1/tools/attach endpoint."""
    agent_id: str = Field(..., description="Target agent ID")
    query: str = Field(..., min_length=1, description="Search query for tools")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum tools to attach")
    min_score: float = Field(default=35.0, ge=0.0, le=100.0, description="Minimum score threshold")
    keep_tools: Optional[List[str]] = Field(default=None, description="Tool IDs to keep")
    request_heartbeat: bool = Field(default=False, description="Request agent heartbeat after attach")
    auto_prune: bool = Field(default=True, description="Auto-prune before attach")
    drop_rate: float = Field(default=0.6, ge=0.0, le=1.0, description="Drop rate for pruning")


class PruneToolsRequest(BaseModel):
    """Request body for /api/v1/tools/prune endpoint."""
    agent_id: str = Field(..., description="Target agent ID")
    user_prompt: str = Field(default="", description="Context for relevance scoring")
    drop_rate: float = Field(default=0.6, ge=0.0, le=1.0, description="Percentage of tools to drop")
    keep_tool_ids: Optional[List[str]] = Field(default=None, description="Tool IDs to preserve")


# ============================================================================
# API Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: Optional[str] = None
    uptime: Optional[float] = None


class SearchResponse(BaseModel):
    """Response for search endpoints - list of results."""
    results: List[SearchResult]
    total: int
    query: str


class AttachToolsResponse(BaseModel):
    """Response for attach tools endpoint."""
    status: str
    message: str
    attached_tools: List[AttachResult] = Field(default_factory=list)
    detached_tools: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    agent_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Configuration Models
# ============================================================================

class RerankerConfig(BaseModel):
    """Reranker configuration."""
    enabled: bool = True
    provider: str = Field(default="vllm", description="Provider: vllm, ollama")
    model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker model")
    base_url: str = Field(default="http://localhost:11435/v1/rerank", description="Reranker API URL")
    top_k: int = Field(default=20, ge=1, le=100, description="Top-k for reranking")


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    provider: str = Field(default="ollama", description="Provider: ollama, openai")
    model: str = Field(default="nomic-embed-text", description="Embedding model")
    dimension: int = Field(default=768, ge=1, description="Embedding dimension")
    base_url: Optional[str] = None


class ToolSelectorConfig(BaseModel):
    """Main tool selector configuration."""
    max_total_tools: int = Field(default=30, ge=1, description="Max total tools on agent")
    max_mcp_tools: int = Field(default=20, ge=1, description="Max MCP tools on agent")
    min_mcp_tools: int = Field(default=7, ge=0, description="Min MCP tools to keep")
    default_drop_rate: float = Field(default=0.6, ge=0.0, le=1.0, description="Default drop rate")
    default_min_score: float = Field(default=35.0, ge=0.0, le=100.0, description="Default min score")
    manage_only_mcp_tools: bool = Field(default=True, description="Only manage MCP tools")
    never_detach_tools: List[str] = Field(default_factory=lambda: ["find_tools"])


# ============================================================================
# Utility Functions for Model Conversion
# ============================================================================

def dict_to_tool(data: Dict[str, Any]) -> Tool:
    """Convert a dictionary to a Tool model."""
    return Tool(**data)


def dict_to_search_result(data: Dict[str, Any]) -> SearchResult:
    """Convert a dictionary to a SearchResult model."""
    return SearchResult(**data)


def attach_result_from_dict(data: Dict[str, Any]) -> AttachResult:
    """Convert a dictionary to an AttachResult model."""
    return AttachResult(**data)


def detach_result_from_dict(data: Dict[str, Any]) -> DetachResult:
    """Convert a dictionary to a DetachResult model."""
    return DetachResult(**data)


# ============================================================================
# Constants - Letta Core Tool Types and Names
# ============================================================================

LETTA_CORE_TOOL_TYPES = frozenset([
    'letta_core', 'letta_voice_sleeptime_core', 'letta_sleeptime_core', 
    'letta_memory_core', 'letta_files_core', 'letta_builtin', 'letta_multi_agent_core'
])

LETTA_CORE_TOOL_NAMES = frozenset([
    'send_message', 'conversation_search', 'archival_memory_insert', 
    'archival_memory_search', 'core_memory_append', 'core_memory_replace', 
    'pause_heartbeats', 'find_attach_tools'
])


def is_letta_core_tool(tool: Union[Dict[str, Any], Tool]) -> bool:
    """
    Determine if a tool is a Letta core tool that should not be managed by auto selection.
    
    Args:
        tool: Tool dictionary or Tool model
        
    Returns:
        True if the tool is a Letta core tool
    """
    if isinstance(tool, Tool):
        return tool.is_letta_core()
    
    # Dictionary path
    tool_type = tool.get('tool_type', '')
    if tool_type in LETTA_CORE_TOOL_TYPES:
        return True
    
    tool_name = tool.get('name', '')
    if tool_name in LETTA_CORE_TOOL_NAMES:
        return True
    
    return False


# ============================================================================
# Configuration Models
# ============================================================================

import os
from dataclasses import dataclass, field


@dataclass
class ToolLimitsConfig:
    """
    Configuration for tool management limits.
    
    This centralizes all tool limit settings that were previously
    scattered across api_server.py global variables.
    """
    max_total_tools: int = 30
    max_mcp_tools: int = 20
    min_mcp_tools: int = 7
    manage_only_mcp_tools: bool = False
    never_detach_tools: List[str] = field(default_factory=lambda: ['find_tools'])
    
    @classmethod
    def from_env(cls) -> 'ToolLimitsConfig':
        """
        Create configuration from environment variables.
        
        Environment variables:
        - MAX_TOTAL_TOOLS: Maximum total tools on agent (default: 30)
        - MAX_MCP_TOOLS: Maximum MCP tools on agent (default: 20)
        - MIN_MCP_TOOLS: Minimum MCP tools to maintain (default: 7)
        - MANAGE_ONLY_MCP_TOOLS: Only manage MCP tools (default: false)
        - NEVER_DETACH_TOOLS / PROTECTED_TOOLS: Comma-separated tool names (default: find_tools)
        """
        # Support both PROTECTED_TOOLS and NEVER_DETACH_TOOLS for compatibility
        protected_tools_env = os.getenv('PROTECTED_TOOLS') or os.getenv('NEVER_DETACH_TOOLS', 'find_tools')
        never_detach = [name.strip() for name in protected_tools_env.split(',') if name.strip()]
        
        return cls(
            max_total_tools=int(os.getenv('MAX_TOTAL_TOOLS', '30')),
            max_mcp_tools=int(os.getenv('MAX_MCP_TOOLS', '20')),
            min_mcp_tools=int(os.getenv('MIN_MCP_TOOLS', '7')),
            manage_only_mcp_tools=os.getenv('MANAGE_ONLY_MCP_TOOLS', 'false').lower() == 'true',
            never_detach_tools=never_detach
        )
    
    def should_protect_tool(self, tool_name: str) -> bool:
        """Check if a tool should be protected from detachment."""
        tool_name_lower = tool_name.lower()
        return any(protected.lower() in tool_name_lower for protected in self.never_detach_tools)
