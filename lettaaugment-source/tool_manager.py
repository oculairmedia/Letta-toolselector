"""
Tool Manager Service

This module handles tool attachment, detachment, and batch processing operations.
It provides a clean interface for managing tools on Letta agents.

Key Functions:
- attach_tool: Attach a single tool to an agent
- detach_tool: Detach a single tool from an agent
- process_tools: Batch attach/detach operations with MIN_MCP_TOOLS enforcement
- fetch_agent_tools: Get current tools for an agent
- perform_tool_pruning: Prune tools based on relevance
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable, Awaitable

import aiohttp

from models import is_letta_core_tool, ToolLimitsConfig

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
_search_tools_func: Optional[Callable[..., List[Dict[str, Any]]]] = None  # Search callback
_tool_config: Optional[ToolLimitsConfig] = None  # Tool limits configuration


def configure(
    http_session: Optional[aiohttp.ClientSession] = None,
    letta_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    use_letta_sdk: bool = False,
    get_letta_sdk_client_func=None,
    search_tools_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    tool_config: Optional[ToolLimitsConfig] = None
):
    """
    Configure the tool manager with required dependencies.
    
    Args:
        http_session: aiohttp ClientSession for API calls
        letta_url: Base URL for Letta API
        headers: HTTP headers for API calls
        use_letta_sdk: Whether to use SDK mode
        get_letta_sdk_client_func: Function to get SDK client
        search_tools_func: Function to search tools (query, limit) -> List[Dict]
        tool_config: Tool limits configuration (ToolLimitsConfig)
    """
    global _http_session, _letta_url, _headers, _use_letta_sdk, _get_letta_sdk_client
    global _search_tools_func, _tool_config
    
    _http_session = http_session
    _letta_url = letta_url
    _headers = headers
    _use_letta_sdk = use_letta_sdk
    _get_letta_sdk_client = get_letta_sdk_client_func
    _search_tools_func = search_tools_func
    _tool_config = tool_config


def get_tool_config() -> ToolLimitsConfig:
    """Get current tool limits configuration, falling back to env vars if not set."""
    if _tool_config is not None:
        return _tool_config
    return ToolLimitsConfig.from_env()


def get_min_mcp_tools() -> int:
    """Get MIN_MCP_TOOLS from configuration or environment."""
    return get_tool_config().min_mcp_tools


# ============================================================================
# Agent Tool Queries
# ============================================================================

async def fetch_agent_tools(agent_id: str) -> List[Dict[str, Any]]:
    """
    Fetch an agent's current tools.
    
    Args:
        agent_id: The agent ID
        
    Returns:
        List of tool dictionaries with id, name, tool_type, etc.
        
    Raises:
        ConnectionError: If HTTP session not available
        Exception: If API call fails
    """
    # Use SDK if enabled
    if _use_letta_sdk and _get_letta_sdk_client:
        try:
            sdk_client = _get_letta_sdk_client()
            return await sdk_client.list_agent_tools(agent_id)
        except Exception as e:
            logger.error(f"SDK fetch_agent_tools failed: {e}")
            raise
    
    # Fall back to aiohttp
    if not _http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_tools (agent: {agent_id})")
        raise ConnectionError("HTTP session not available")
    
    try:
        url = f"{_letta_url}/agents/{agent_id}/tools"
        async with _http_session.get(url, headers=_headers) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching agent tools for {agent_id}: {e}")
        raise


# ============================================================================
# Core Tool Operations
# ============================================================================

async def detach_tool(agent_id: str, tool_id: str, tool_name: Optional[str] = None) -> Dict[str, Any]:
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
            detach_tool(agent_id, str(tool.get("tool_id") or tool.get("id") or ""))
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


# ============================================================================
# Tool Pruning
# ============================================================================

import math


async def perform_tool_pruning(
    agent_id: str,
    user_prompt: str,
    drop_rate: float,
    keep_tool_ids: Optional[List[str]] = None,
    newly_matched_tool_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prune tools from an agent based on relevance to the user's prompt.
    
    Only prunes MCP tools ('external_mcp'). Core Letta tools are always preserved.
    Keeps a percentage of the most relevant MCP tools from the entire library,
    plus any explicitly kept or newly matched MCP tools.
    
    Args:
        agent_id: The agent ID
        user_prompt: Query to determine tool relevance
        drop_rate: Fraction of tools to drop (0.0 to 1.0)
        keep_tool_ids: Tool IDs that must be kept
        newly_matched_tool_ids: Recently matched tool IDs to prioritize
        
    Returns:
        Dict with success status, message, and details of the pruning operation
    """
    if _search_tools_func is None:
        return {
            "success": False,
            "error": "Search function not configured. Call configure() with search_tools_func."
        }
    
    config = get_tool_config()
    
    requested_keep_tool_ids = set(keep_tool_ids or [])
    requested_newly_matched_tool_ids = set(newly_matched_tool_ids or [])
    
    logger.info(f"Pruning request for agent {agent_id} with prompt: '{user_prompt}', drop_rate: {drop_rate}")
    logger.info(f"Requested to keep (all types): {requested_keep_tool_ids}, Requested newly matched (all types): {requested_newly_matched_tool_ids}")

    try:
        # 1. Retrieve Current Agent Tools and categorize them
        logger.info(f"Fetching current tools for agent {agent_id}...")
        current_agent_tools_list = await fetch_agent_tools(agent_id)
        
        core_tools_on_agent = []
        mcp_tools_on_agent_list = []
        
        for tool in current_agent_tools_list:
            tool_id = tool.get('id') or tool.get('tool_id')
            if not tool_id:
                logger.warning(f"Tool found on agent without an ID: {tool.get('name', 'Unknown')}. Skipping.")
                continue
            
            # Ensure basic structure for ID consistency
            tool['id'] = tool_id 
            tool['tool_id'] = tool_id

            # Enhanced tool categorization based on configuration
            is_core = is_letta_core_tool(tool)
            tool_name = tool.get('name', '').lower()
            
            # Check if this tool should never be detached
            is_never_detach = (
                tool_id in requested_keep_tool_ids or 
                tool_id in requested_newly_matched_tool_ids or
                config.should_protect_tool(tool.get('name', ''))
            )
            
            if is_never_detach or (config.manage_only_mcp_tools and is_core):
                core_tools_on_agent.append(tool)
            elif tool.get("tool_type") == "external_mcp" or (not is_core and tool.get("tool_type") == "custom"):
                mcp_tools_on_agent_list.append(tool)
            else:
                core_tools_on_agent.append(tool)

        current_mcp_tool_ids = {tool['id'] for tool in mcp_tools_on_agent_list}
        current_core_tool_ids = {tool['id'] for tool in core_tools_on_agent}
        
        # Track protected tools for logging
        protected_tool_names = [
            tool.get('name', 'Unknown') for tool in core_tools_on_agent 
            if config.should_protect_tool(tool.get('name', ''))
            or tool.get('id') in requested_keep_tool_ids
            or tool.get('id') in requested_newly_matched_tool_ids
        ]
        
        num_currently_attached_mcp = len(current_mcp_tool_ids)
        num_currently_attached_core = len(current_core_tool_ids)
        num_total_attached = num_currently_attached_mcp + num_currently_attached_core

        logger.info(f"Agent {agent_id} has {num_total_attached} total tools: "
                    f"{num_currently_attached_mcp} MCP tools, {num_currently_attached_core} Core tools.")
        if protected_tool_names:
            logger.info(f"Protected tools (moved to core): {protected_tool_names}")
        logger.debug(f"MCP tools on agent: {current_mcp_tool_ids}")
        logger.debug(f"Core tools on agent: {current_core_tool_ids}")

        # Check minimum MCP tool count requirement
        MIN_MCP_TOOLS = config.min_mcp_tools
        
        if num_currently_attached_mcp == 0:
            logger.info("No MCP tools currently attached to the agent. Nothing to prune among MCP tools.")
            return {
                "success": True, 
                "message": "No MCP tools to prune. Core tools preserved.",
                "remaining_tools": core_tools_on_agent,  # All current tools remain
                "details": {
                    "tools_on_agent_before_total": num_total_attached,
                    "mcp_tools_on_agent_before": 0,
                    "core_tools_preserved_count": num_currently_attached_core,
                    "target_mcp_tools_to_keep": 0,
                    "mcp_tools_detached_count": 0,
                    "final_tool_ids_on_agent": list(current_core_tool_ids),
                }
            }

        if num_currently_attached_mcp <= MIN_MCP_TOOLS:
            logger.info(f"Agent {agent_id} has {num_currently_attached_mcp} MCP tools, which is at or below the minimum required ({MIN_MCP_TOOLS}). Skipping pruning.")
            return {
                "success": True, 
                "message": f"Pruning skipped: Agent has {num_currently_attached_mcp} MCP tools (minimum required: {MIN_MCP_TOOLS})",
                "remaining_tools": core_tools_on_agent + mcp_tools_on_agent_list,  # All current tools remain
                "details": {
                    "tools_on_agent_before_total": num_total_attached,
                    "mcp_tools_on_agent_before": num_currently_attached_mcp,
                    "core_tools_preserved_count": num_currently_attached_core,
                    "target_mcp_tools_to_keep": num_currently_attached_mcp,
                    "mcp_tools_detached_count": 0,
                    "final_tool_ids_on_agent": list(current_core_tool_ids | current_mcp_tool_ids),
                    "minimum_mcp_tools_enforced": MIN_MCP_TOOLS
                }
            }

        # 2. Determine Target Number of MCP Tools to Keep on Agent
        max_mcp_allowed = config.max_mcp_tools
        max_total_allowed = config.max_total_tools - num_currently_attached_core
        
        target_from_drop_rate = math.floor(num_currently_attached_mcp * (1.0 - drop_rate))
        num_mcp_tools_to_keep = min(target_from_drop_rate, max_mcp_allowed, max_total_allowed)
        num_mcp_tools_to_keep = max(num_mcp_tools_to_keep, MIN_MCP_TOOLS)
        
        if num_mcp_tools_to_keep < 0: 
            num_mcp_tools_to_keep = 0
            
        logger.info("Target MCP tools calculation:")
        logger.info(f"  From drop_rate {drop_rate}: {target_from_drop_rate}")
        logger.info(f"  MAX_MCP_TOOLS limit: {max_mcp_allowed}")
        logger.info(f"  Available space (MAX_TOTAL_TOOLS - core tools): {max_total_allowed}")
        logger.info(f"  MIN_MCP_TOOLS requirement: {MIN_MCP_TOOLS}")
        logger.info(f"  Final target MCP tools to keep: {num_mcp_tools_to_keep}")

        # 3. Find Top Relevant Tools from Entire Library using search_tools
        search_limit = max(num_mcp_tools_to_keep + 50, 100) 
        logger.info(f"Searching for top {search_limit} relevant tools from library for prompt: '{user_prompt}'")
        
        # Use search callback (may be sync or async)
        top_library_tools_data = await asyncio.to_thread(_search_tools_func, query=user_prompt, limit=search_limit)
        
        ordered_top_library_tool_info = []
        seen_top_ids: Set[str] = set()
        for tool_data in top_library_tools_data:
            tool_id = tool_data.get('id') or tool_data.get('tool_id')
            if tool_id and tool_id not in seen_top_ids:
                ordered_top_library_tool_info.append(
                    (tool_id, tool_data.get('name', 'Unknown'), tool_data.get('tool_type'))
                )
                seen_top_ids.add(tool_id)
        logger.info(f"Found {len(ordered_top_library_tool_info)} unique, potentially relevant tools from library search.")

        # 4. Determine Final Set of MCP Tools to Keep on Agent
        final_mcp_tool_ids_to_keep: Set[str] = set()
        
        for tool_id in requested_newly_matched_tool_ids:
            if tool_id in current_mcp_tool_ids:
                final_mcp_tool_ids_to_keep.add(tool_id)
        logger.info(f"Initially keeping newly matched MCP tools (if on agent): {len(final_mcp_tool_ids_to_keep)}. Set: {final_mcp_tool_ids_to_keep}")

        for tool_id in requested_keep_tool_ids:
            if tool_id in current_mcp_tool_ids:
                final_mcp_tool_ids_to_keep.add(tool_id)
        
        # Additional safeguard: protect any never-detach tools
        for tool in mcp_tools_on_agent_list:
            tool_name = tool.get('name', '')
            if config.should_protect_tool(tool_name):
                final_mcp_tool_ids_to_keep.add(tool.get('id'))
                logger.warning(f"Never-detach tool '{tool_name}' found in MCP list - protecting from pruning")
        
        logger.info(f"After adding explicitly requested-to-keep MCP tools (if on agent): {len(final_mcp_tool_ids_to_keep)}. Set: {final_mcp_tool_ids_to_keep}")

        # Handle aggressive pruning when must-keep tools exceed target
        if len(final_mcp_tool_ids_to_keep) >= num_mcp_tools_to_keep:
            logger.info(f"Number of must-keep MCP tools ({len(final_mcp_tool_ids_to_keep)}) meets or exceeds target ({num_mcp_tools_to_keep}). Being more aggressive with detachment.")
            aggressive_target = max(1, math.floor(num_currently_attached_mcp * 0.8))
            
            if len(final_mcp_tool_ids_to_keep) > aggressive_target:
                prioritized_keeps: Set[str] = set()
                
                # HIGHEST PRIORITY: Never-detach tools
                for tool in mcp_tools_on_agent_list:
                    tool_name = tool.get('name', '')
                    if config.should_protect_tool(tool_name):
                        prioritized_keeps.add(tool.get('id'))
                        logger.info(f"Aggressive pruning: PROTECTING never-detach tool '{tool_name}' (ID: {tool.get('id')})")
                
                # SECOND PRIORITY: Explicitly requested to keep
                for tool_id in requested_keep_tool_ids:
                    if tool_id in current_mcp_tool_ids and len(prioritized_keeps) < aggressive_target:
                        prioritized_keeps.add(tool_id)
                        logger.debug(f"Aggressive pruning: keeping explicitly requested tool {tool_id}")
                
                # Third priority: newly matched tools
                for tool_id in requested_newly_matched_tool_ids:
                    if tool_id in current_mcp_tool_ids and tool_id not in prioritized_keeps and len(prioritized_keeps) < aggressive_target:
                        prioritized_keeps.add(tool_id)
                        logger.debug(f"Aggressive pruning: keeping newly matched tool {tool_id}")
                
                # Fourth priority: most relevant tools from library search
                for tool_id, _, tool_type in ordered_top_library_tool_info:
                    if (tool_type == "external_mcp" and tool_id in final_mcp_tool_ids_to_keep
                        and tool_id not in prioritized_keeps and len(prioritized_keeps) < aggressive_target):
                        prioritized_keeps.add(tool_id)
                        logger.debug(f"Aggressive pruning: keeping library-relevant tool {tool_id}")
                
                final_mcp_tool_ids_to_keep = prioritized_keeps
                logger.info(f"Applied aggressive pruning: reduced to {len(final_mcp_tool_ids_to_keep)} tools (target was {aggressive_target})")
        else:
            # Fill remaining slots with most relevant attached MCP tools
            potential_additional_keeps = []
            for tool_id, _, tool_type in ordered_top_library_tool_info:
                if tool_type == "external_mcp" and tool_id in current_mcp_tool_ids and tool_id not in final_mcp_tool_ids_to_keep:
                    potential_additional_keeps.append(tool_id)
            
            num_slots_to_fill = num_mcp_tools_to_keep - len(final_mcp_tool_ids_to_keep)
            
            for tool_id in potential_additional_keeps[:num_slots_to_fill]:
                final_mcp_tool_ids_to_keep.add(tool_id)
            
            logger.info(f"After filling remaining slots with other relevant attached MCP tools: {len(final_mcp_tool_ids_to_keep)}. Set: {final_mcp_tool_ids_to_keep}")

        logger.info(f"Final set of {len(final_mcp_tool_ids_to_keep)} MCP tool IDs decided to be kept on agent: {final_mcp_tool_ids_to_keep}")

        # 5. Identify MCP Tools to Detach
        mcp_tools_to_detach_ids = current_mcp_tool_ids - final_mcp_tool_ids_to_keep
        logger.info(f"Identified {len(mcp_tools_to_detach_ids)} MCP tools to detach: {mcp_tools_to_detach_ids}")

        # 6. Detach Identified MCP Tools
        successful_detachments_info = []
        failed_detachments_info = []
        
        if mcp_tools_to_detach_ids:
            tools_to_detach_list = list(mcp_tools_to_detach_ids)
            detach_tasks = [detach_tool(agent_id, tool_id) for tool_id in tools_to_detach_list]
            logger.info(f"Executing {len(detach_tasks)} detach operations for MCP tools in parallel...")
            detach_results = await asyncio.gather(*detach_tasks, return_exceptions=True)

            id_to_name_map = {tool['id']: tool.get('name', 'Unknown') for tool in mcp_tools_on_agent_list}

            for i, result in enumerate(detach_results):
                tool_id_detached = tools_to_detach_list[i] 
                tool_name_detached = id_to_name_map.get(tool_id_detached, "Unknown")

                if isinstance(result, Exception):
                    logger.error(f"Exception during detach for MCP tool {tool_name_detached} ({tool_id_detached}): {result}")
                    failed_detachments_info.append({"tool_id": tool_id_detached, "name": tool_name_detached, "error": str(result)})
                elif isinstance(result, dict) and result.get("success"):
                    successful_detachments_info.append({"tool_id": tool_id_detached, "name": tool_name_detached})
                else:
                    error_msg = result.get("error", "Unknown detachment failure") if isinstance(result, dict) else "Unexpected result type"
                    logger.warning(f"Failed detach result for MCP tool {tool_name_detached} ({tool_id_detached}): {error_msg}")
                    failed_detachments_info.append({"tool_id": tool_id_detached, "name": tool_name_detached, "error": error_msg})
            logger.info(f"Successfully detached {len(successful_detachments_info)} MCP tools, {len(failed_detachments_info)} failed.")
        else:
            logger.info("No MCP tools to detach based on the strategy.")
            
        # 7. Final list of tools on agent
        final_tool_ids_on_agent = current_core_tool_ids.union(final_mcp_tool_ids_to_keep)
        
        # Build remaining_tools list from core + kept MCP tools (avoids re-fetch)
        remaining_tools = []
        remaining_tools.extend(core_tools_on_agent)
        for tool in mcp_tools_on_agent_list:
            if tool.get('id') in final_mcp_tool_ids_to_keep:
                remaining_tools.append(tool)
        
        return {
            "success": True,
            "message": f"Pruning completed for agent {agent_id}. Only MCP tools were considered for pruning.",
            "remaining_tools": remaining_tools,  # Include tool objects to avoid re-fetch
            "details": {
                "tools_on_agent_before_total": num_total_attached,
                "mcp_tools_on_agent_before": num_currently_attached_mcp,
                "core_tools_preserved_count": num_currently_attached_core,
                "target_mcp_tools_to_keep_after_pruning": num_mcp_tools_to_keep,
                "relevant_library_tools_found_count": len(ordered_top_library_tool_info),
                "final_mcp_tool_ids_kept_on_agent": list(final_mcp_tool_ids_to_keep),
                "final_core_tool_ids_on_agent": list(current_core_tool_ids),
                "actual_total_tools_on_agent_after_pruning": len(final_tool_ids_on_agent),
                "mcp_tools_detached_count": len(successful_detachments_info),
                "mcp_tools_failed_detachment_count": len(failed_detachments_info),
                "drop_rate_applied_to_mcp_tools": drop_rate,
                "explicitly_kept_tool_ids_from_request": list(requested_keep_tool_ids),
                "newly_matched_tool_ids_from_request": list(requested_newly_matched_tool_ids),
                "successful_detachments_mcp": successful_detachments_info,
                "failed_detachments_mcp": failed_detachments_info
            }
        }

    except Exception as e:
        logger.error(f"Error during tool pruning for agent {agent_id}: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
