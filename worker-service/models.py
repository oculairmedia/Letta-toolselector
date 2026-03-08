from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from tool_selector_client import DEFAULT_LIMIT, DEFAULT_MIN_SCORE


class FindToolsRequest(BaseModel):
    """Incoming payload structure for the worker service."""

    query: Optional[str] = Field(
        default=None,
        description="Search query describing the desired tool behaviour",
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Identifier of the requesting agent",
    )
    keep_tools: Optional[str] = Field(
        default=None,
        description="Comma-separated list of tool IDs that must remain attached",
    )
    limit: Optional[int] = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        description="Maximum number of tools to attach (default matches legacy script)",
    )
    min_score: Optional[float] = Field(
        default=DEFAULT_MIN_SCORE,
        ge=0.0,
        le=100.0,
        description="Minimum similarity score threshold (0-100)",
    )
    request_heartbeat: bool = Field(
        default=False,
        description="Whether to request an immediate heartbeat from the agent",
    )


class FindToolsResponse(BaseModel):
    """Standardised response returned to the MCP server."""

    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    service: str


# ── Granular tool management models ──────────────────────────────────────


class ToolIdentifier(BaseModel):
    """Identifies a tool by name or ID."""

    name: Optional[str] = Field(default=None, description="Tool name")
    tool_id: Optional[str] = Field(default=None, description="Tool ID")


class LookupToolRequest(BaseModel):
    """Request to look up a tool by name or ID."""

    tool_name: Optional[str] = Field(default=None, description="Tool name to look up")
    tool_id: Optional[str] = Field(default=None, description="Tool ID to look up")
    fuzzy: bool = Field(default=False, description="Enable fuzzy matching")
    limit: int = Field(default=5, ge=1, le=20, description="Max fuzzy results")


class AttachToolRequest(BaseModel):
    """Request to directly attach tool(s) to an agent."""

    agent_id: str = Field(..., description="Agent ID to attach tools to")
    tools: List[ToolIdentifier] = Field(..., description="Tools to attach")
    pin: bool = Field(default=False, description="Pin tools to survive pruning")


class DetachToolRequest(BaseModel):
    """Request to directly detach tool(s) from an agent."""

    agent_id: str = Field(..., description="Agent ID to detach tools from")
    tools: List[ToolIdentifier] = Field(..., description="Tools to detach")
    unpin: bool = Field(default=False, description="Also unpin if pinned")


class ListAgentToolsRequest(BaseModel):
    """Request to list tools attached to an agent."""

    agent_id: str = Field(..., description="Agent ID")
    filter: str = Field(default="all", description="Filter: all, core, mcp")
    include_schema: bool = Field(default=False, description="Include tool schemas")


class InspectToolRequest(BaseModel):
    """Request to inspect a tool's full metadata."""

    tool_name_or_id: str = Field(..., description="Tool name or ID to inspect")


class GenericToolResponse(BaseModel):
    """Generic pass-through response for granular tool operations."""

    status: str
    data: Optional[Any] = None
    error: Optional[str] = None
