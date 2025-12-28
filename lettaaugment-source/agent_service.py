"""
Agent Service

This module handles communication with Letta agents including:
- Fetching agent info and tools
- Registering MCP tools
- Triggering agent loops after tool attachment
- Webhook notifications to Matrix bridge

Key Functions:
- fetch_agent_info: Get agent name/details
- register_tool: Register MCP tool with Letta
- send_trigger_message: Send trigger message to agent
- trigger_agent_loop: Fire-and-forget loop trigger
- emit_matrix_webhook: Notify Matrix bridge of events
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

import aiohttp

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Module State (to be injected)
# ============================================================================

_http_session: Optional[aiohttp.ClientSession] = None
_letta_url: Optional[str] = None
_headers: Optional[Dict[str, str]] = None
_use_letta_sdk: bool = False
_get_letta_sdk_client: Optional[Callable] = None
_letta_message_base_urls: List[str] = []
_matrix_bridge_webhook_url: Optional[str] = None


def configure(
    http_session: Optional[aiohttp.ClientSession] = None,
    letta_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    use_letta_sdk: bool = False,
    get_letta_sdk_client_func: Optional[Callable] = None,
    letta_message_base_urls: Optional[List[str]] = None,
    matrix_bridge_webhook_url: Optional[str] = None
):
    """
    Configure the agent service with required dependencies.
    
    Args:
        http_session: aiohttp ClientSession for API calls
        letta_url: Base URL for Letta API
        headers: HTTP headers for API calls
        use_letta_sdk: Whether to use SDK mode
        get_letta_sdk_client_func: Function to get SDK client
        letta_message_base_urls: List of Letta message endpoints
        matrix_bridge_webhook_url: URL for Matrix bridge webhook
    """
    global _http_session, _letta_url, _headers, _use_letta_sdk
    global _get_letta_sdk_client, _letta_message_base_urls, _matrix_bridge_webhook_url
    
    _http_session = http_session
    _letta_url = letta_url
    _headers = headers
    _use_letta_sdk = use_letta_sdk
    _get_letta_sdk_client = get_letta_sdk_client_func
    _letta_message_base_urls = letta_message_base_urls or []
    _matrix_bridge_webhook_url = matrix_bridge_webhook_url


# ============================================================================
# Agent Information
# ============================================================================

async def fetch_agent_info(agent_id: str) -> str:
    """
    Fetch agent information (name) from Letta.
    
    Args:
        agent_id: The agent ID
        
    Returns:
        Agent name string
        
    Raises:
        ConnectionError: If HTTP session not available
        Exception: If API call fails
    """
    # Use SDK if enabled
    if _use_letta_sdk and _get_letta_sdk_client:
        try:
            sdk_client = _get_letta_sdk_client()
            return await sdk_client.get_agent_name(agent_id)
        except Exception as e:
            logger.error(f"SDK fetch_agent_info failed: {e}")
            raise
    
    # Fall back to aiohttp
    if not _http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_info (agent: {agent_id})")
        raise ConnectionError("HTTP session not available")
    
    try:
        url = f"{_letta_url}/agents/{agent_id}"
        async with _http_session.get(url, headers=_headers) as response:
            response.raise_for_status()
            agent_data = await response.json()
        return agent_data.get("name", "Unknown Agent")
    except Exception as e:
        logger.error(f"Error fetching agent info for {agent_id}: {e}")
        raise


# ============================================================================
# Tool Registration
# ============================================================================

async def register_tool(tool_name: str, server_name: str) -> Optional[Dict[str, Any]]:
    """
    Register an MCP tool with Letta.
    
    Args:
        tool_name: Name of the tool to register
        server_name: Name of the MCP server
        
    Returns:
        Registered tool dict with id/tool_id, or None if registration failed
        
    Raises:
        ConnectionError: If HTTP session not available
        Exception: If API call fails
    """
    # Use SDK if enabled
    if _use_letta_sdk and _get_letta_sdk_client:
        try:
            sdk_client = _get_letta_sdk_client()
            return await sdk_client.register_mcp_tool(tool_name, server_name)
        except Exception as e:
            logger.error(f"SDK register_tool failed: {e}")
            raise
    
    # Fall back to aiohttp
    if not _http_session:
        logger.error(f"HTTP session not initialized for register_tool (tool: {tool_name}, server: {server_name})")
        raise ConnectionError("HTTP session not available")
    
    try:
        register_url = f"{_letta_url}/tools/mcp/servers/{server_name}/{tool_name}"
        async with _http_session.post(register_url, headers=_headers) as response:
            response.raise_for_status()
            registered_tool = await response.json()
        
        if registered_tool.get('id') or registered_tool.get('tool_id'):
            # Normalize ID fields
            if registered_tool.get('id') and not registered_tool.get('tool_id'):
                registered_tool['tool_id'] = registered_tool['id']
            elif registered_tool.get('tool_id') and not registered_tool.get('id'):
                registered_tool['id'] = registered_tool['tool_id']
            return registered_tool
        return None
    except Exception as e:
        logger.error(f"Error registering tool {tool_name} from {server_name}: {e}")
        raise


# ============================================================================
# Trigger Messages
# ============================================================================

async def send_trigger_message(
    agent_id: str, 
    tool_names: List[str], 
    query: Optional[str] = None
) -> bool:
    """
    Send a trigger message to the agent notifying of new tools.
    
    This is meant to be run as a background task (fire-and-forget).
    
    Args:
        agent_id: The agent ID
        tool_names: List of newly attached tool names
        query: Optional original user query
        
    Returns:
        True if message was sent successfully
    """
    if not _http_session:
        logger.warning("HTTP session not available for trigger message")
        return False
    
    if not _letta_message_base_urls:
        logger.warning("No Letta message endpoints available for trigger")
        return False
    
    # Build the trigger message
    tool_list = ", ".join(tool_names[:5])
    if len(tool_names) > 5:
        tool_list += f" and {len(tool_names) - 5} more"
    
    trigger_message = (
        f"[SYSTEM] New tools attached to your toolkit: {tool_list}. "
        f"These tools are now available. Please proceed with the original request"
    )
    if query:
        trigger_message += f" regarding: {query}"
    trigger_message += "."
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": trigger_message
            }
        ]
    }
    
    last_error = None
    for base_url in _letta_message_base_urls:
        messages_url = f"{base_url}/agents/{agent_id}/messages"
        logger.info(f"[BACKGROUND] Sending trigger message to {agent_id} via {messages_url} ...")
        
        try:
            async with _http_session.post(messages_url, headers=_headers, json=payload) as response:
                if response.status in (200, 201, 202):
                    logger.info(f"[BACKGROUND] Trigger completed for {agent_id} via {messages_url}")
                    
                    # Extract run_id from response and emit webhook
                    new_run_id = None
                    try:
                        response_data = await response.json()
                        messages = response_data.get("messages", [])
                        if messages and len(messages) > 0:
                            new_run_id = messages[0].get("run_id")
                            logger.info(f"[BACKGROUND] New run_id from trigger: {new_run_id}")
                    except Exception as parse_err:
                        logger.warning(f"[BACKGROUND] Could not parse run_id from response: {parse_err}")
                    
                    # Emit webhook to Matrix bridge for cross-run tracking
                    if _matrix_bridge_webhook_url:
                        await emit_matrix_webhook(
                            agent_id=agent_id,
                            new_run_id=new_run_id,
                            tool_names=tool_names,
                            query=query
                        )
                    
                    return True
                    
                text = await response.text()
                last_error = f"HTTP {response.status} - {text[:200]}"
                logger.warning(f"[BACKGROUND] Trigger failed for {agent_id} via {messages_url}: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[BACKGROUND] Error in trigger message for {agent_id} via {messages_url}: {e}")
    
    if last_error:
        logger.warning(f"[BACKGROUND] All trigger attempts failed for {agent_id}: {last_error}")
    
    return False


def trigger_agent_loop(
    agent_id: str, 
    attached_tools: List[Dict[str, Any]], 
    query: Optional[str] = None
) -> bool:
    """
    Fire-and-forget trigger to start a new agent loop with updated tools.
    
    In Letta V1 architecture, tools are passed to the LLM at the start of a request.
    After attaching new tools, we need to trigger a new loop so the agent can use them.
    
    This function spawns a background task and returns immediately - it does NOT wait
    for the agent to process the message.
    
    Args:
        agent_id: The agent ID
        attached_tools: List of attached tool dicts
        query: Optional original user query
        
    Returns:
        True if the background task was successfully created
    """
    if not agent_id or not attached_tools:
        return False
    
    # Build list of attached tool names
    tool_names = []
    for tool in attached_tools:
        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("tool_name", "unknown")
        else:
            name = str(tool)
        tool_names.append(name)
    
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Create a background task - this is TRUE fire-and-forget
        task = loop.create_task(send_trigger_message(agent_id, tool_names, query))
        
        # Add a callback to log when it completes
        def on_complete(t):
            if t.exception():
                logger.warning(f"[BACKGROUND] Trigger task failed with exception: {t.exception()}")
        task.add_done_callback(on_complete)
        
        logger.info(f"Spawned background trigger task for agent {agent_id} with {len(tool_names)} new tools")
        return True
        
    except Exception as e:
        logger.warning(f"Error creating trigger task: {e}")
        return False


# ============================================================================
# Webhook Notifications
# ============================================================================

async def emit_matrix_webhook(
    agent_id: str, 
    new_run_id: Optional[str] = None, 
    tool_names: Optional[List[str]] = None, 
    query: Optional[str] = None
) -> bool:
    """
    Emit webhook to Matrix bridge for cross-run tracking.
    
    This notifies the Matrix bridge that a new run was triggered after tool attachment,
    allowing it to track the conversation across multiple Letta runs.
    
    Args:
        agent_id: The agent ID
        new_run_id: The new run ID from trigger response
        tool_names: List of attached tool names
        query: Original user query
        
    Returns:
        True if webhook was sent successfully
    """
    if not _matrix_bridge_webhook_url:
        return False
    
    if not _http_session:
        logger.warning("[WEBHOOK] HTTP session not available for Matrix bridge webhook")
        return False
    
    webhook_payload = {
        "event": "run_triggered",
        "agent_id": agent_id,
        "new_run_id": new_run_id,
        "trigger_type": "tool_attachment",
        "tools_attached": tool_names or [],
        "query": query,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        async with _http_session.post(
            _matrix_bridge_webhook_url,
            json=webhook_payload,
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                logger.info(f"[WEBHOOK] Notified Matrix bridge of run trigger for {agent_id}, run_id={new_run_id}")
                return True
            else:
                resp_text = await resp.text()
                logger.warning(f"[WEBHOOK] Matrix bridge returned {resp.status}: {resp_text[:200]}")
                return False
    except asyncio.TimeoutError:
        logger.warning(f"[WEBHOOK] Timeout sending webhook to Matrix bridge for {agent_id}")
        return False
    except Exception as e:
        logger.warning(f"[WEBHOOK] Failed to notify Matrix bridge for {agent_id}: {e}")
        return False
