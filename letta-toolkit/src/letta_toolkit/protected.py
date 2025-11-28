"""Protected tools management for Letta agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from letta_toolkit.client import LettaClient
from letta_toolkit.config import LettaConfig, get_config
from letta_toolkit.tools import (
    attach_tool_to_agent,
    get_tool_by_name,
    list_agent_tools,
)

logger = logging.getLogger(__name__)


@dataclass
class ProtectedToolsResult:
    """Result of ensure_protected_tools operation.
    
    Attributes:
        success: True if no failures occurred
        attached: List of tools that were attached
        already_present: List of tools that were already attached
        failed: List of tools that failed to attach
        error: Error message if operation failed entirely
    """
    success: bool = True
    attached: list[dict[str, Any]] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def ensure_protected_tools(
    agent_id: str,
    *,
    protected_tools: list[str] | None = None,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ProtectedToolsResult:
    """Ensure protected tools are attached to an agent.
    
    Checks if protected tools are attached to the agent and attaches any
    that are missing. Protected tools are critical tools that should always
    be available to agents.
    
    Args:
        agent_id: The agent ID
        protected_tools: List of tool names to protect (uses config default if not provided)
        client: Optional LettaClient instance
        config: Optional config
        
    Returns:
        ProtectedToolsResult with details of what was attached/present/failed
        
    Example:
        >>> result = ensure_protected_tools("agent-123")
        >>> if result.attached:
        ...     print(f"Attached: {[t['name'] for t in result.attached]}")
    """
    if not agent_id:
        return ProtectedToolsResult(
            success=False,
            error="No agent_id provided"
        )
    
    cfg = config or get_config()
    _client = client or LettaClient(cfg)
    should_close = client is None
    
    # Get protected tool names from config if not provided
    tool_names = protected_tools or cfg.protected_tools
    
    if not tool_names:
        return ProtectedToolsResult(
            success=True,
            error="No protected tools configured"
        )
    
    logger.info(f"[ensure_protected_tools] Checking {len(tool_names)} protected tools for agent {agent_id}")
    
    result = ProtectedToolsResult()
    
    try:
        # Get current agent tools
        current_tools = list_agent_tools(agent_id, client=_client, config=cfg)
        current_tool_names = {
            tool.get("name", "").lower() 
            for tool in current_tools 
            if tool.get("name")
        }
        
        logger.debug(f"[ensure_protected_tools] Agent has {len(current_tool_names)} tools")
        
        for tool_name in tool_names:
            tool_name_lower = tool_name.lower().strip()
            
            if tool_name_lower in current_tool_names:
                logger.debug(f"[ensure_protected_tools] Tool '{tool_name}' already attached")
                result.already_present.append(tool_name)
            else:
                # Need to attach this tool
                logger.info(f"[ensure_protected_tools] Tool '{tool_name}' missing, looking up ID...")
                
                tool_info = get_tool_by_name(tool_name, client=_client, config=cfg)
                
                if tool_info and tool_info.get("id"):
                    tool_id = tool_info["id"]
                    attach_result = attach_tool_to_agent(
                        agent_id, tool_id, client=_client, config=cfg
                    )
                    
                    if attach_result.success:
                        result.attached.append({
                            "name": tool_name,
                            "id": tool_id
                        })
                    else:
                        result.failed.append({
                            "name": tool_name,
                            "id": tool_id,
                            "reason": "attach_failed",
                            "error": attach_result.error
                        })
                else:
                    logger.warning(f"[ensure_protected_tools] Could not find tool ID for '{tool_name}'")
                    result.failed.append({
                        "name": tool_name,
                        "reason": "tool_not_found"
                    })
        
        result.success = len(result.failed) == 0
        
        logger.info(
            f"[ensure_protected_tools] Result: "
            f"{len(result.attached)} attached, "
            f"{len(result.already_present)} present, "
            f"{len(result.failed)} failed"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[ensure_protected_tools] Unexpected error: {e}")
        return ProtectedToolsResult(
            success=False,
            error=str(e)
        )
        
    finally:
        if should_close:
            _client.close()


def is_tool_protected(
    tool_name: str,
    *,
    protected_tools: list[str] | None = None,
    config: LettaConfig | None = None,
) -> bool:
    """Check if a tool name is in the protected tools list.
    
    Args:
        tool_name: Name of the tool to check
        protected_tools: List of protected tool names (uses config if not provided)
        config: Optional config
        
    Returns:
        True if the tool is protected
    """
    cfg = config or get_config()
    tool_names = protected_tools or cfg.protected_tools
    
    return tool_name.lower().strip() in {t.lower().strip() for t in tool_names}
