from __future__ import annotations

from typing import Any, Dict, Optional

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
