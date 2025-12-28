"""
Tool Manager Service

This module handles tool attachment, detachment, and batch processing operations.
It provides a clean interface for managing tools on Letta agents.

Key Functions:
- attach_tool: Attach a single tool to an agent
- detach_tool: Detach a single tool from an agent
- process_tools: Batch attach/detach operations with MIN_MCP_TOOLS enforcement
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set

import aiohttp

from models import is_letta_core_tool

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Module State (to be injected)
# ============================================================================

# These will be set by the main application
_http_session: Optional[aiohttp.ClientSession] = None
_letta_url: Optional[str] = None
_headers: Optional[Dict[str, str]] = None
_use_letta_sdk: bool = False
_get_letta_sdk_client = None  # Callable to get SDK client


def configure(
    http_session: Optional[aiohttp.ClientSession] = None,
    letta_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    use_letta_sdk: bool = False,
    get_letta_sdk_client_func=None
):
    """
    Configure the tool manager with required dependencies.
    
    Args:
        http_session: aiohttp ClientSession for API calls
        letta_url: Base URL for Letta API
        headers: HTTP headers for API calls
        use_letta_sdk: Whether to use SDK mode
        get_letta_sdk_client_func: Function to get SDK client
    """
    global _http_session, _letta_url, _headers, _use_letta_sdk, _get_letta_sdk_client
    _http_session = http_session
    _letta_url = letta_url
    _headers = headers
    _use_letta_sdk = use_letta_sdk
    _get_letta_sdk_client = get_letta_sdk_client_func


def get_min_mcp_tools() -> int:
    """Get MIN_MCP_TOOLS from environment."""
    return int(os.getenv('MIN_MCP_TOOLS', '7'))


# ============================================================================
# Core Tool Operations
# ============================================================================

async def detach_tool(agent_id: str, tool_id: str, tool_name: str = None) -> Dict[str, Any]:
    """
    Detach a single tool from an agent.
    
    Args:
        agent_id: The agent ID
        tool_id: The tool ID to detach
        tool_name: Optional tool name for logging
        
    Returns:
        Dict with keys: success, tool_id, and optionally warning or error
    """
    # Use SDK if enabled
    if _use_letta_sdk and _get_letta_sdk_client:
        try:
            sdk_client = _get_letta_sdk_client()
            return await sdk_client.detach_tool(agent_id, tool_id, tool_name)
        except Exception as e:
            logger.error(f"SDK detach_tool failed, error: {e}")
            return {"success": False, "tool_id": tool_id, "error": str(e)}
    
    # Fall back to aiohttp
    if not _http_session:
        logger.error(f"HTTP session not initialized for detach_tool (agent: {agent_id}, tool: {tool_id})")
        return {"success": False, "tool_id": tool_id, "error": "HTTP session not available"}
    
    try:
        detach_url = f"{_letta_url}/agents/{agent_id}/tools/detach/{tool_id}"
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with _http_session.patch(detach_url, headers=_headers, timeout=timeout) as response:
            try:
                response_data = await response.json()
            except aiohttp.ContentTypeError:
                response_text = await response.text()
                logger.warning(f"Non-JSON response from detach endpoint: {response_text}")
                response_data = {"text": response_text}
            
            if response.status == 200:
                return {"success": True, "tool_id": tool_id}
            elif response.status == 404:
                logger.warning(f"Tool {tool_id} not found or already detached (404)")
                return {"success": True, "tool_id": tool_id, "warning": "Tool not found or already detached"}
            else:
                logger.error(f"Failed to detach tool {tool_id}: HTTP {response.status}, Response: {response_data}")
                return {"success": False, "tool_id": tool_id, "error": f"HTTP {response.status}: {str(response_data)}"}
                
    except asyncio.TimeoutError:
        logger.error(f"Timeout while detaching tool {tool_id}")
        return {"success": False, "tool_id": tool_id, "error": "Request timed out"}
    except Exception as e:
        logger.error(f"Error detaching tool {tool_id}: {str(e)}")
        return {"success": False, "tool_id": tool_id, "error": str(e)}


async def attach_tool(agent_id: str, tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach a single tool to an agent.
    
    Args:
        agent_id: The agent ID
        tool: Tool dict with id/tool_id, name, and optionally distance for scoring
        
    Returns:
        Dict with keys: success, tool_id, name, and optionally match_score or error
    """
    tool_name = tool.get('name', 'Unknown')
    tool_id = tool.get('tool_id') or tool.get('id')
    
    if not tool_id:
        logger.error(f"No tool ID found for tool {tool_name}")
        return {"success": False, "tool_id": None, "name": tool_name, "error": "No tool ID available"}
    
    # Use SDK if enabled
    if _use_letta_sdk and _get_letta_sdk_client:
        try:
            sdk_client = _get_letta_sdk_client()
            result = await sdk_client.attach_tool(agent_id, tool_id, tool_name)
            if result.get('success') and 'distance' in tool:
                result['match_score'] = 100 * (1 - tool.get('distance', 0))
            return result
        except Exception as e:
            logger.error(f"SDK attach_tool failed, error: {e}")
            return {"success": False, "tool_id": tool_id, "name": tool_name, "error": str(e)}
    
    # Fall back to aiohttp
    if not _http_session:
        logger.error(f"HTTP session not initialized for attach_tool (agent: {agent_id})")
        return {"success": False, "tool_id": tool_id, "name": tool_name, "error": "HTTP session not available"}
    
    try:
        attach_url = f"{_letta_url}/agents/{agent_id}/tools/attach/{tool_id}"
        async with _http_session.patch(attach_url, headers=_headers) as response:
            if response.status == 200:
                return {
                    "success": True,
                    "tool_id": tool_id,
                    "name": tool.get("name"),
                    "match_score": 100 * (1 - tool.get("distance", 0)) if "distance" in tool else 100
                }
            else:
                logger.error(f"Failed to attach tool {tool_id}: HTTP {response.status}")
                return {"success": False, "tool_id": tool_id, "name": tool.get("name")}
    except Exception as e:
        logger.error(f"Error attaching tool {tool_id}: {str(e)}")
        return {"success": False, "tool_id": tool_id, "name": tool.get("name")}


async def process_tools(
    agent_id: str,
    mcp_tools: List[Dict[str, Any]],
    matching_tools: List[Dict[str, Any]],
    keep_tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process tool detachments and attachments in parallel.
    
    This function:
    1. Determines which tools to detach (current tools not in keep list or new tools)
    2. Enforces MIN_MCP_TOOLS limit
    3. Runs detachments in parallel
    4. Runs attachments in parallel
    
    Args:
        agent_id: The agent ID
        mcp_tools: Current MCP tools on the agent
        matching_tools: New tools to attach
        keep_tools: Tool IDs that must remain attached
        
    Returns:
        Dict with keys: detached_tools, failed_detachments, successful_attachments, failed_attachments
    """
    keep_tools = keep_tools or []
    logger.info(f"Processing tools for agent {agent_id}")
    logger.info(f"Current unique MCP tools: {len(mcp_tools)}")
    logger.info(f"Tools to attach: {len(matching_tools)}")
    logger.info(f"Tools to keep: {len(keep_tools)}")
    
    MIN_MCP_TOOLS = get_min_mcp_tools()
    
    # Build set of tool IDs to keep
    keep_tool_ids: Set[str] = set()
    for tool_id in keep_tools:
        if tool_id:
            keep_tool_ids.add(tool_id)
    for tool in matching_tools:
        tool_id = tool.get("id") or tool.get("tool_id")
        if tool_id:
            keep_tool_ids.add(tool_id)
    
    logger.info(f"Tool IDs to keep: {keep_tool_ids}")
    
    # Check session availability
    if not _http_session:
        logger.error(f"HTTP session not initialized for process_tools (agent: {agent_id})")
        return {
            "detached_tools": [],
            "failed_detachments": [t.get("tool_id") or t.get("id") for t in mcp_tools],
            "successful_attachments": [],
            "failed_attachments": matching_tools,
            "error": "HTTP session not available"
        }
    
    # Calculate detachment needs
    current_mcp_tool_ids: Set[str] = set()
    for tool in mcp_tools:
        tool_id = tool.get("tool_id") or tool.get("id")
        if tool_id:
            current_mcp_tool_ids.add(tool_id)
    
    # Count potential detachments
    potential_detach_count = sum(
        1 for tool in mcp_tools
        if (tool.get("tool_id") or tool.get("id")) not in keep_tool_ids
    )
    
    # Calculate remaining MCP tools after operations
    new_tools_count = len([
        t for t in matching_tools
        if (t.get("id") or t.get("tool_id")) not in current_mcp_tool_ids
    ])
    remaining_mcp_tools = len(mcp_tools) - potential_detach_count + new_tools_count
    
    logger.info(f"MIN_MCP_TOOLS check: current={len(mcp_tools)}, potential_detach={potential_detach_count}, "
                f"new_tools={new_tools_count}, remaining={remaining_mcp_tools}, min_required={MIN_MCP_TOOLS}")
    
    # Build detachment list with MIN_MCP_TOOLS enforcement
    tools_to_detach: List[Dict[str, Any]] = []
    
    if remaining_mcp_tools < MIN_MCP_TOOLS:
        max_detach_allowed = max(0, len(mcp_tools) + new_tools_count - MIN_MCP_TOOLS)
        logger.warning(f"Limiting detachments to preserve MIN_MCP_TOOLS={MIN_MCP_TOOLS}. Max allowed: {max_detach_allowed}")
        
        detach_count = 0
        for tool in mcp_tools:
            tool_id = tool.get("tool_id") or tool.get("id")
            tool_name = tool.get("name", "Unknown")
            
            if tool_id and tool_id not in keep_tool_ids:
                if detach_count < max_detach_allowed:
                    tools_to_detach.append({"id": tool_id, "tool_id": tool_id, "name": tool_name})
                    detach_count += 1
                else:
                    logger.info(f"Preserving tool {tool_name} ({tool_id}) to maintain MIN_MCP_TOOLS")
    else:
        for tool in mcp_tools:
            tool_id = tool.get("tool_id") or tool.get("id")
            tool_name = tool.get("name", "Unknown")
            
            if tool_id and tool_id not in keep_tool_ids:
                tools_to_detach.append({"id": tool_id, "tool_id": tool_id, "name": tool_name})
    
    logger.info(f"Tools to detach: {len(tools_to_detach)}")
    
    # Run detachments in parallel
    detach_results = []
    if tools_to_detach:
        logger.info(f"Executing {len(tools_to_detach)} detach operations in parallel...")
        detach_tasks = [
            detach_tool(agent_id, tool.get("tool_id") or tool.get("id"))
            for tool in tools_to_detach
        ]
        raw_results = await asyncio.gather(*detach_tasks, return_exceptions=True)
        
        for i, result in enumerate(raw_results):
            tool_id_for_error = tools_to_detach[i].get("tool_id") or tools_to_detach[i].get("id")
            if isinstance(result, Exception):
                logger.error(f"Exception during parallel detach for tool ID {tool_id_for_error}: {result}")
                detach_results.append({"success": False, "tool_id": tool_id_for_error, "error": str(result)})
            else:
                detach_results.append(result)
    
    # Process detachment results
    detached = [r["tool_id"] for r in detach_results if r and r.get("success")]
    failed_detach = [r["tool_id"] for r in detach_results if r and not r.get("success")]
    
    # Run attachments in parallel
    attach_tasks = [attach_tool(agent_id, tool) for tool in matching_tools]
    attach_raw_results = await asyncio.gather(*attach_tasks, return_exceptions=True)
    
    # Process attachment results
    successful_attachments = []
    failed_attachments = []
    
    for i, result in enumerate(attach_raw_results):
        tool_info = matching_tools[i]
        tool_id_for_error = tool_info.get("tool_id") or tool_info.get("id")
        tool_name_for_error = tool_info.get("name", "Unknown")
        
        if isinstance(result, Exception):
            logger.error(f"Exception during parallel attach for tool {tool_name_for_error} ({tool_id_for_error}): {result}")
            failed_attachments.append({
                "success": False,
                "tool_id": tool_id_for_error,
                "name": tool_name_for_error,
                "error": str(result)
            })
        elif isinstance(result, dict) and result.get("success"):
            successful_attachments.append(result)
        else:
            logger.warning(f"Failed attach result for tool {tool_name_for_error} ({tool_id_for_error}): {result}")
            failed_attachments.append({
                "success": False,
                "tool_id": tool_id_for_error,
                "name": tool_name_for_error,
                "error": result.get("error", "Unknown attachment failure") if isinstance(result, dict) else "Unexpected result type"
            })
    
    return {
        "detached_tools": detached,
        "failed_detachments": failed_detach,
        "successful_attachments": successful_attachments,
        "failed_attachments": failed_attachments
    }
