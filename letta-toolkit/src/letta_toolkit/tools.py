"""Tool management operations for Letta agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from letta_toolkit.client import LettaAPIError, LettaClient
from letta_toolkit.config import LettaConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ToolOperationResult:
    """Result of a tool operation (attach/detach).
    
    Attributes:
        success: Whether the operation succeeded
        tool_id: ID of the tool
        tool_name: Name of the tool (if known)
        error: Error message if failed
    """
    success: bool
    tool_id: str
    tool_name: str | None = None
    error: str | None = None


@dataclass
class BatchOperationResult:
    """Result of a batch tool operation.
    
    Attributes:
        successful: List of successful operations
        failed: List of failed operations
    """
    successful: list[ToolOperationResult] = field(default_factory=list)
    failed: list[ToolOperationResult] = field(default_factory=list)
    
    @property
    def all_success(self) -> bool:
        """True if all operations succeeded."""
        return len(self.failed) == 0
    
    @property
    def total(self) -> int:
        """Total number of operations."""
        return len(self.successful) + len(self.failed)


def list_agent_tools(
    agent_id: str,
    *,
    include_all: bool = True,
    limit: int | None = None,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> list[dict[str, Any]]:
    """List all tools attached to an agent.
    
    Handles pagination automatically to ensure all tools are returned.
    
    Args:
        agent_id: The agent ID
        include_all: If True, fetch all tools (handles pagination)
        limit: Maximum tools to return (default: 500 if include_all, else API default)
        client: Optional LettaClient instance
        config: Optional config (uses global if not provided)
        
    Returns:
        List of tool dictionaries with full metadata
        
    Raises:
        LettaAPIError: If API request fails
        
    Example:
        >>> tools = list_agent_tools("agent-123")
        >>> print(f"Agent has {len(tools)} tools")
    """
    if not agent_id:
        logger.warning("[list_agent_tools] No agent_id provided")
        return []
    
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    try:
        # Use high limit to get all tools in one request
        effective_limit = limit or (cfg.default_page_limit if include_all else None)
        params = {"limit": effective_limit} if effective_limit else {}
        
        logger.debug(f"[list_agent_tools] Fetching tools for agent {agent_id}")
        result = _client.get(f"/agents/{agent_id}/tools", params=params)
        
        tools = result if isinstance(result, list) else []
        logger.info(f"[list_agent_tools] Retrieved {len(tools)} tools for agent {agent_id}")
        
        return tools
        
    finally:
        if should_close:
            _client.close()


def get_tool_by_name(
    tool_name: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> dict[str, Any] | None:
    """Find a tool by name from the global tools list.
    
    Args:
        tool_name: Name of the tool to find
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        Tool dictionary if found, None otherwise
    """
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    try:
        tools = _client.get("/tools", params={"limit": 500})
        
        if not isinstance(tools, list):
            return None
        
        tool_name_lower = tool_name.lower()
        for tool in tools:
            if tool.get("name", "").lower() == tool_name_lower:
                return tool
        
        return None
        
    except LettaAPIError as e:
        logger.error(f"[get_tool_by_name] Error finding tool '{tool_name}': {e}")
        return None
    finally:
        if should_close:
            _client.close()


def attach_tool_to_agent(
    agent_id: str,
    tool_id: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ToolOperationResult:
    """Attach a tool to an agent.
    
    Args:
        agent_id: The agent ID
        tool_id: The tool ID to attach
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        ToolOperationResult indicating success/failure
    """
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    try:
        logger.debug(f"[attach_tool] Attaching {tool_id} to agent {agent_id}")
        _client.patch(f"/agents/{agent_id}/tools/attach/{tool_id}", json={})
        
        logger.info(f"[attach_tool] Successfully attached {tool_id}")
        return ToolOperationResult(success=True, tool_id=tool_id)
        
    except LettaAPIError as e:
        # 409 means already attached - treat as success
        if e.status_code == 409:
            logger.debug(f"[attach_tool] Tool {tool_id} already attached (409)")
            return ToolOperationResult(success=True, tool_id=tool_id)
        
        logger.error(f"[attach_tool] Failed to attach {tool_id}: {e}")
        return ToolOperationResult(success=False, tool_id=tool_id, error=str(e))
        
    finally:
        if should_close:
            _client.close()


def detach_tool_from_agent(
    agent_id: str,
    tool_id: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ToolOperationResult:
    """Detach a tool from an agent.
    
    Args:
        agent_id: The agent ID
        tool_id: The tool ID to detach
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        ToolOperationResult indicating success/failure
    """
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    try:
        logger.debug(f"[detach_tool] Detaching {tool_id} from agent {agent_id}")
        _client.patch(f"/agents/{agent_id}/tools/detach/{tool_id}", json={})
        
        logger.info(f"[detach_tool] Successfully detached {tool_id}")
        return ToolOperationResult(success=True, tool_id=tool_id)
        
    except LettaAPIError as e:
        # 404 might mean already detached - treat as success
        if e.status_code == 404:
            logger.debug(f"[detach_tool] Tool {tool_id} not found (404), treating as success")
            return ToolOperationResult(success=True, tool_id=tool_id)
        
        logger.error(f"[detach_tool] Failed to detach {tool_id}: {e}")
        return ToolOperationResult(success=False, tool_id=tool_id, error=str(e))
        
    finally:
        if should_close:
            _client.close()


def batch_attach_tools(
    agent_id: str,
    tool_ids: list[str],
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> BatchOperationResult:
    """Attach multiple tools to an agent.
    
    Args:
        agent_id: The agent ID
        tool_ids: List of tool IDs to attach
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        BatchOperationResult with successful and failed operations
    """
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    result = BatchOperationResult()
    
    try:
        for tool_id in tool_ids:
            op_result = attach_tool_to_agent(agent_id, tool_id, client=_client, config=cfg)
            if op_result.success:
                result.successful.append(op_result)
            else:
                result.failed.append(op_result)
        
        logger.info(
            f"[batch_attach] Completed: {len(result.successful)} success, "
            f"{len(result.failed)} failed"
        )
        return result
        
    finally:
        if should_close:
            _client.close()


def batch_detach_tools(
    agent_id: str,
    tool_ids: list[str],
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> BatchOperationResult:
    """Detach multiple tools from an agent.
    
    Args:
        agent_id: The agent ID
        tool_ids: List of tool IDs to detach
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        BatchOperationResult with successful and failed operations
    """
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    result = BatchOperationResult()
    
    try:
        for tool_id in tool_ids:
            op_result = detach_tool_from_agent(agent_id, tool_id, client=_client, config=cfg)
            if op_result.success:
                result.successful.append(op_result)
            else:
                result.failed.append(op_result)
        
        logger.info(
            f"[batch_detach] Completed: {len(result.successful)} success, "
            f"{len(result.failed)} failed"
        )
        return result
        
    finally:
        if should_close:
            _client.close()
