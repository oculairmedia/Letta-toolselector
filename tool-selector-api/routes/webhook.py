"""
Webhook Routes Blueprint

Handles incoming webhook events from Letta for real-time tool synchronization.
Replaces polling-based sync with event-driven updates.

Events handled:
- tool.created: Add new tool to Weaviate index
- tool.updated: Update tool in Weaviate index
- tool.deleted: Remove tool from Weaviate index
"""

import os
import hmac
import hashlib
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable, Any

from quart import Blueprint, request, jsonify
import aiofiles

logger = logging.getLogger(__name__)

webhook_bp = Blueprint('webhook', __name__, url_prefix='/webhook')

# Module-level configuration
_config = {
    'webhook_secret': None,
    'weaviate_client': None,
    'cache_dir': '/app/runtime_cache',
    'tool_cache_file': None,
    'on_tool_change': None,  # Optional callback for additional processing
}


def configure(
    webhook_secret: Optional[str] = None,
    weaviate_client: Any = None,
    cache_dir: str = '/app/runtime_cache',
    on_tool_change: Optional[Callable] = None,
):
    """
    Configure the webhook blueprint with required dependencies.
    
    Args:
        webhook_secret: Secret for HMAC signature verification (optional but recommended)
        weaviate_client: Weaviate client for index updates
        cache_dir: Directory for tool cache files
        on_tool_change: Optional callback invoked after tool changes
    """
    _config['webhook_secret'] = webhook_secret or os.getenv('LETTA_WEBHOOK_SECRET')
    _config['weaviate_client'] = weaviate_client
    _config['cache_dir'] = cache_dir
    _config['tool_cache_file'] = os.path.join(cache_dir, 'tool_cache.json')
    _config['on_tool_change'] = on_tool_change
    logger.info(f"Webhook blueprint configured (secret={'set' if _config['webhook_secret'] else 'not set'})")


def verify_signature(payload: bytes, signature: str) -> bool:
    """
    Verify HMAC-SHA256 signature from Letta webhook.
    
    Args:
        payload: Raw request body bytes
        signature: Signature from X-Letta-Signature header
        
    Returns:
        True if signature is valid or no secret configured, False otherwise
    """
    secret = _config['webhook_secret']
    if not secret:
        logger.warning("Webhook secret not configured - skipping signature verification")
        return True
    
    if not signature:
        logger.warning("No signature provided in webhook request")
        return False
    
    expected = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    # Compare with timing-safe comparison
    return hmac.compare_digest(f"sha256={expected}", signature)


async def read_tool_cache() -> list:
    """Read current tool cache from disk."""
    cache_file = _config['tool_cache_file']
    try:
        if os.path.exists(cache_file):
            async with aiofiles.open(cache_file, 'r') as f:
                content = await f.read()
                return json.loads(content) if content else []
    except Exception as e:
        logger.error(f"Error reading tool cache: {e}")
    return []


async def write_tool_cache(tools: list):
    """Write tool cache to disk."""
    cache_file = _config['tool_cache_file']
    try:
        os.makedirs(_config['cache_dir'], exist_ok=True)
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(tools, indent=2))
        logger.info(f"Tool cache updated: {len(tools)} tools")
    except Exception as e:
        logger.error(f"Error writing tool cache: {e}")


async def add_tool_to_weaviate(tool_data: dict) -> bool:
    """
    Add a single tool to Weaviate index.
    
    Args:
        tool_data: Tool data from webhook payload
        
    Returns:
        True if successful, False otherwise
    """
    client = _config['weaviate_client']
    if not client:
        logger.warning("Weaviate client not configured - skipping index update")
        return False
    
    try:
        collection = await asyncio.to_thread(client.collections.get, "Tool")
        
        # Build properties for Weaviate
        properties = {
            "tool_id": tool_data.get("tool_id", ""),
            "name": tool_data.get("tool_name", ""),
            "description": tool_data.get("description", ""),
            "tool_type": tool_data.get("tool_type", "unknown"),
            "json_schema": json.dumps(tool_data.get("json_schema", {})),
            "tags": tool_data.get("tags", []),
            "source_type": "webhook",
            "last_synced": datetime.now(timezone.utc).isoformat(),
        }
        
        # Generate combined text for embedding
        combined_text = f"{properties['name']} {properties['description']}"
        properties["combined_text"] = combined_text
        
        await asyncio.to_thread(collection.data.insert, properties)
        logger.info(f"Added tool to Weaviate: {properties['name']}")
        return True
        
    except Exception as e:
        logger.error(f"Error adding tool to Weaviate: {e}")
        return False


async def update_tool_in_weaviate(tool_data: dict) -> bool:
    """
    Update a tool in Weaviate index (delete + insert).
    
    Args:
        tool_data: Tool data from webhook payload
        
    Returns:
        True if successful, False otherwise
    """
    # Weaviate doesn't have direct update, so delete then insert
    await delete_tool_from_weaviate(tool_data.get("tool_name", ""))
    return await add_tool_to_weaviate(tool_data)


async def delete_tool_from_weaviate(tool_name: str) -> bool:
    """
    Delete a tool from Weaviate index by name.
    
    Args:
        tool_name: Name of the tool to delete
        
    Returns:
        True if successful, False otherwise
    """
    client = _config['weaviate_client']
    if not client:
        logger.warning("Weaviate client not configured - skipping index update")
        return False
    
    try:
        import weaviate.classes.query as wq
        
        collection = await asyncio.to_thread(client.collections.get, "Tool")
        name_filter = wq.Filter.by_property("name").equal(tool_name)
        
        result = await asyncio.to_thread(
            collection.data.delete_many,
            where=name_filter
        )
        
        deleted = getattr(result, 'successful', 0)
        logger.info(f"Deleted {deleted} object(s) from Weaviate for tool: {tool_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting tool from Weaviate: {e}")
        return False


async def update_cache_for_tool(event_type: str, tool_data: dict):
    """
    Update local tool cache based on event type.
    
    Args:
        event_type: One of 'tool.created', 'tool.updated', 'tool.deleted'
        tool_data: Tool data from webhook payload
    """
    tools = await read_tool_cache()
    tool_name = tool_data.get("tool_name", "")
    tool_id = tool_data.get("tool_id", "")
    
    if event_type == "tool.deleted":
        # Remove tool from cache
        tools = [t for t in tools if t.get("name") != tool_name and t.get("id") != tool_id]
        
    elif event_type == "tool.created":
        # Add tool to cache (check for duplicates first)
        existing = next((t for t in tools if t.get("name") == tool_name), None)
        if not existing:
            tools.append({
                "id": tool_id,
                "name": tool_name,
                "description": tool_data.get("description", ""),
                "tool_type": tool_data.get("tool_type", ""),
                "json_schema": tool_data.get("json_schema", {}),
                "tags": tool_data.get("tags", []),
                "source": "webhook",
            })
            
    elif event_type == "tool.updated":
        # Update existing tool in cache
        for i, t in enumerate(tools):
            if t.get("name") == tool_name or t.get("id") == tool_id:
                tools[i] = {
                    **t,
                    "description": tool_data.get("description", t.get("description", "")),
                    "tool_type": tool_data.get("tool_type", t.get("tool_type", "")),
                    "json_schema": tool_data.get("json_schema", t.get("json_schema", {})),
                    "tags": tool_data.get("tags", t.get("tags", [])),
                    "source": "webhook",
                }
                break
        else:
            # Tool not found, add it
            tools.append({
                "id": tool_id,
                "name": tool_name,
                "description": tool_data.get("description", ""),
                "tool_type": tool_data.get("tool_type", ""),
                "json_schema": tool_data.get("json_schema", {}),
                "tags": tool_data.get("tags", []),
                "source": "webhook",
            })
    
    await write_tool_cache(tools)


@webhook_bp.route('/letta', methods=['POST'])
async def handle_letta_webhook():
    """
    Handle incoming Letta webhook events.
    
    Expected payload structure:
    {
        "event_type": "tool.created" | "tool.updated" | "tool.deleted",
        "timestamp": "ISO8601",
        "data": {
            "tool_id": "...",
            "tool_name": "...",
            "tool_type": "...",
            "description": "...",
            "json_schema": {...},
            "tags": [...]
        }
    }
    """
    # Get raw body for signature verification
    body = await request.get_data()
    signature = request.headers.get('X-Letta-Signature', '')
    
    # Verify signature
    if not verify_signature(body, signature):
        logger.warning("Webhook signature verification failed")
        return jsonify({"error": "Invalid signature"}), 401
    
    # Parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in webhook payload: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    event_type = payload.get("event_type", "")
    tool_data = payload.get("data", {})
    timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
    
    logger.info(f"Received webhook event: {event_type} for tool: {tool_data.get('tool_name', 'unknown')}")
    
    # Handle supported event types
    if event_type == "tool.created":
        weaviate_success = await add_tool_to_weaviate(tool_data)
        await update_cache_for_tool(event_type, tool_data)
        
    elif event_type == "tool.updated":
        weaviate_success = await update_tool_in_weaviate(tool_data)
        await update_cache_for_tool(event_type, tool_data)
        
    elif event_type == "tool.deleted":
        weaviate_success = await delete_tool_from_weaviate(tool_data.get("tool_name", ""))
        await update_cache_for_tool(event_type, tool_data)
        
    else:
        logger.info(f"Ignoring unhandled event type: {event_type}")
        return jsonify({"status": "ignored", "event_type": event_type}), 200
    
    # Invoke optional callback
    if _config['on_tool_change']:
        try:
            await _config['on_tool_change'](event_type, tool_data)
        except Exception as e:
            logger.error(f"Error in on_tool_change callback: {e}")
    
    return jsonify({
        "status": "processed",
        "event_type": event_type,
        "tool_name": tool_data.get("tool_name"),
        "weaviate_updated": weaviate_success if 'weaviate_success' in dir() else False,
        "timestamp": timestamp,
    }), 200


@webhook_bp.route('/health', methods=['GET'])
async def webhook_health():
    """Health check for webhook endpoint."""
    return jsonify({
        "status": "healthy",
        "endpoint": "/webhook/letta",
        "secret_configured": bool(_config['webhook_secret']),
        "weaviate_configured": bool(_config['weaviate_client']),
    }), 200
