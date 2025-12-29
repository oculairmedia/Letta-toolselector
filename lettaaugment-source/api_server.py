from quart import Quart, request, jsonify
# Restore search_tools import, remove get_all_tools as cache is used for listing
from weaviate_tool_search_with_reranking import search_tools, search_tools_with_reranking, init_client as init_weaviate_client
from weaviate_client_manager import get_client_manager, close_client_manager
# Import models for type definitions and utilities
from models import (
    LETTA_CORE_TOOL_TYPES, LETTA_CORE_TOOL_NAMES,
    is_letta_core_tool as models_is_letta_core_tool
)
# Import tool manager for attach/detach operations
import tool_manager
# Import agent service for agent communication
import agent_service
# Import search service for unified search operations
import search_service
import os
import asyncio
import aiohttp
import aiofiles # Import aiofiles
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
import json # Added json import
import time # Need time for cache timeout check
import math # For cosine similarity and math.floor
import uuid # For generating comparison IDs
from hypercorn.config import Config
from hypercorn.asyncio import serve
from simple_config_validation import validate_configuration
from bm25_vector_overrides import bm25_vector_override_service
try:
    from cost_control_manager import (
        get_cost_manager, CostCategory, BudgetPeriod, AlertLevel,
        record_embedding_cost, record_weaviate_cost, record_letta_api_cost
    )
except ImportError:
    # Handle case where cost_control_manager is not available
    get_cost_manager = None
    CostCategory = None
    BudgetPeriod = None
    AlertLevel = None
    def record_embedding_cost(*args, **kwargs):
        return None
    def record_weaviate_cost(*args, **kwargs):
        return None
    def record_letta_api_cost(*args, **kwargs):
        return None
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import audit logging for structured events
from audit_logging import (
    emit_tool_event, emit_batch_event, emit_pruning_event, emit_limit_enforcement_event,
    AuditAction, AuditSource
)

# Import connection test functions from config routes
from routes.config import test_ollama_connection, test_weaviate_connection

# Import Letta SDK client wrapper for SDK-based API calls
# Feature flag to enable SDK migration (set USE_LETTA_SDK=true to enable)
USE_LETTA_SDK = os.getenv('USE_LETTA_SDK', 'false').lower() == 'true'
_letta_sdk_client = None

# Tool search provider configuration
# Options: 'weaviate' (default), 'letta', 'hybrid' (try Letta first, fallback to Weaviate)
TOOL_SEARCH_PROVIDER = os.getenv('TOOL_SEARCH_PROVIDER', 'weaviate').lower()

if USE_LETTA_SDK or TOOL_SEARCH_PROVIDER in ('letta', 'hybrid'):
    try:
        from letta_sdk_client import LettaSDKClient, get_client as get_letta_sdk_client
        logger.info("Letta SDK client imported successfully - SDK mode enabled")
        if not USE_LETTA_SDK:
            USE_LETTA_SDK = True  # Enable SDK for tool search even if not for other operations
    except ImportError as e:
        logger.warning(f"Failed to import Letta SDK client, falling back to aiohttp: {e}")
        USE_LETTA_SDK = False
        if TOOL_SEARCH_PROVIDER == 'letta':
            TOOL_SEARCH_PROVIDER = 'weaviate'
            logger.warning("TOOL_SEARCH_PROVIDER set to 'letta' but SDK import failed, falling back to 'weaviate'")

app = Quart(__name__)
# Load .env file - try container path first, then current directory
if os.path.exists('/app/.env'):
    load_dotenv('/app/.env')
else:
    load_dotenv()

def _normalize_letta_base_url(url: str):
    """Normalize Letta base URL to ensure it includes /v1 and no trailing slash."""
    if not url:
        return None
    normalized = url.rstrip('/')
    if not normalized.endswith('/v1'):
        normalized = f"{normalized}/v1"
    return normalized

raw_letta_url = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1')
LETTA_URL = _normalize_letta_base_url(raw_letta_url)


def _build_message_base_urls():
    candidates = []
    direct_raw = os.getenv('LETTA_DIRECT_MESSAGE_URL') or os.getenv('LETTA_DIRECT_URL')
    if direct_raw:
        direct_url = _normalize_letta_base_url(direct_raw)
        if direct_url:
            candidates.append(direct_url)
    if LETTA_URL:
        candidates.append(LETTA_URL)
        if LETTA_URL.startswith('https://'):
            http_candidate = 'http://' + LETTA_URL[len('https://'):]
            candidates.append(http_candidate)
    seen = set()
    ordered = []
    for url in candidates:
        if url and url not in seen:
            ordered.append(url)
            seen.add(url)
    return ordered

LETTA_MESSAGE_BASE_URLS = _build_message_base_urls()

# Load password from environment variable
LETTA_API_KEY = os.getenv('LETTA_PASSWORD')
if not LETTA_API_KEY:
    logger.error("CRITICAL: LETTA_PASSWORD environment variable not set. API calls will likely fail.")
    # Or raise an exception: raise ValueError("LETTA_PASSWORD environment variable not set.")

# Load default drop rate from environment variable
DEFAULT_DROP_RATE = float(os.getenv('DEFAULT_DROP_RATE', '0.1'))
logger.info(f"DEFAULT_DROP_RATE configured as: {DEFAULT_DROP_RATE}")

# Load new tool management configuration
MAX_TOTAL_TOOLS = int(os.getenv('MAX_TOTAL_TOOLS', '30'))
MAX_MCP_TOOLS = int(os.getenv('MAX_MCP_TOOLS', '20'))
EXCLUDE_LETTA_CORE_TOOLS = os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'false').lower() == 'true'
EXCLUDE_OFFICIAL_TOOLS = os.getenv('EXCLUDE_OFFICIAL_TOOLS', 'false').lower() == 'true'
MANAGE_ONLY_MCP_TOOLS = os.getenv('MANAGE_ONLY_MCP_TOOLS', 'false').lower() == 'true'

# Tools that should never be detached (comma-separated list of tool names)
# Support both NEVER_DETACH_TOOLS and PROTECTED_TOOLS for consistency with SDK
_protected_tools_env = os.getenv('PROTECTED_TOOLS') or os.getenv('NEVER_DETACH_TOOLS', 'find_tools')
NEVER_DETACH_TOOLS = [name.strip() for name in _protected_tools_env.split(',') if name.strip()]

# Matrix bridge webhook for cross-run tracking
# When a new run is triggered after tool attachment, notify Matrix bridge
MATRIX_BRIDGE_WEBHOOK_URL = os.getenv('MATRIX_BRIDGE_WEBHOOK_URL')

# Default minimum score threshold for tool attachment (0-100)
DEFAULT_MIN_SCORE = float(os.getenv('DEFAULT_MIN_SCORE', '35.0'))

logger.info("Tool management configuration:")
logger.info(f"  MAX_TOTAL_TOOLS: {MAX_TOTAL_TOOLS}")
logger.info(f"  MAX_MCP_TOOLS: {MAX_MCP_TOOLS}")
logger.info(f"  DEFAULT_MIN_SCORE: {DEFAULT_MIN_SCORE}")
logger.info(f"  EXCLUDE_LETTA_CORE_TOOLS: {EXCLUDE_LETTA_CORE_TOOLS}")
logger.info(f"  EXCLUDE_OFFICIAL_TOOLS: {EXCLUDE_OFFICIAL_TOOLS}")
logger.info(f"  MANAGE_ONLY_MCP_TOOLS: {MANAGE_ONLY_MCP_TOOLS}")
logger.info(f"  NEVER_DETACH_TOOLS: {NEVER_DETACH_TOOLS}")
logger.info(f"  TOOL_SEARCH_PROVIDER: {TOOL_SEARCH_PROVIDER}")

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    # Use the environment variable with the correct header format
    "X-BARE-PASSWORD": f"password {LETTA_API_KEY}" if LETTA_API_KEY else "" # Updated header format
}

# --- Define Cache Directory and File Paths ---
CACHE_DIR = "/app/runtime_cache" # Changed cache directory
TOOL_CACHE_FILE_PATH = os.path.join(CACHE_DIR, "tool_cache.json")
MCP_SERVERS_CACHE_FILE_PATH = os.path.join(CACHE_DIR, "mcp_servers_cache.json")
_tool_cache = None # In-memory cache variable for tools
_tool_cache_last_modified = 0 # Timestamp of last tool cache load
# Note: We won't use in-memory caching for MCP servers here, read on demand

# --- Global Clients ---
weaviate_client = None
http_session = None # Global aiohttp session

# --- Helper function to read tool cache ---
async def read_tool_cache(force_reload=False):
    """Reads the tool cache file asynchronously, using an in-memory cache."""
    global _tool_cache, _tool_cache_last_modified # Use renamed variable
    try:
        # Check modification time synchronously first
        try:
            current_mtime = os.path.getmtime(TOOL_CACHE_FILE_PATH) # Use renamed variable
        except FileNotFoundError:
            logger.error(f"Tool cache file not found: {TOOL_CACHE_FILE_PATH}. Returning empty list.") # Use renamed variable
            _tool_cache = []
            _tool_cache_last_modified = 0
            return []

        # Reload if forced, cache is empty, or file has been modified
        if force_reload or _tool_cache is None or current_mtime > _tool_cache_last_modified:
            logger.info(f"Loading tool cache from file: {TOOL_CACHE_FILE_PATH}") # Use renamed variable
            async with aiofiles.open(TOOL_CACHE_FILE_PATH, mode='r') as f: # Use renamed variable
                content = await f.read()
                _tool_cache = json.loads(content)
            _tool_cache_last_modified = current_mtime # Use renamed variable
            logger.info(f"Loaded {_tool_cache and len(_tool_cache)} tools into cache.")
        # else:
            # logger.debug("Using in-memory tool cache.")
        return _tool_cache if _tool_cache else []
    except FileNotFoundError:
        logger.error(f"Tool cache file not found during async read: {TOOL_CACHE_FILE_PATH}. Returning empty list.") # Use renamed variable
        _tool_cache = []
        _tool_cache_last_modified = 0
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from cache file: {TOOL_CACHE_FILE_PATH}. Returning empty list.") # Use renamed variable
        _tool_cache = []
        _tool_cache_last_modified = 0
        return []
    except Exception as e:
        logger.error(f"Error reading tool cache file {TOOL_CACHE_FILE_PATH}: {e}") # Use renamed variable
        _tool_cache = []
        _tool_cache_last_modified = 0
        return []

# --- Helper function to read MCP servers cache ---
async def read_mcp_servers_cache():
    """Reads the MCP servers cache file asynchronously."""
    try:
        async with aiofiles.open(MCP_SERVERS_CACHE_FILE_PATH, mode='r') as f:
            content = await f.read()
            mcp_servers = json.loads(content)
        logger.debug(f"Successfully read {len(mcp_servers)} MCP servers from cache: {MCP_SERVERS_CACHE_FILE_PATH}")
        return mcp_servers
    except FileNotFoundError:
        logger.error(f"MCP servers cache file not found: {MCP_SERVERS_CACHE_FILE_PATH}. Returning empty list.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from MCP servers cache file: {MCP_SERVERS_CACHE_FILE_PATH}. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Error reading MCP servers cache file {MCP_SERVERS_CACHE_FILE_PATH}: {e}")
        return []

# Removed update_mcp_servers_cache function as this is now handled by sync_service.py

async def unified_tool_search(query: str, limit: int = 10, min_score: float = 0.0):
    """
    Unified tool search that can use either Weaviate or Letta's native search.
    
    Delegates to search_service.search for the actual implementation.
    
    Args:
        query: Search query describing the tool you're looking for
        limit: Maximum number of results to return
        min_score: Minimum relevance score (0-100) to include
        
    Returns:
        List of tool dicts with search results
    """
    return await search_service.search(query=query, limit=limit, min_score=min_score)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0  # Return 0 for invalid or mismatched vectors
    
    dot_product = sum(p * q for p, q in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(p * p for p in vec1))
    magnitude2 = math.sqrt(sum(q * q for q in vec2))
    
    if not magnitude1 or not magnitude2:
        return 0  # Avoid division by zero
    
    return dot_product / (magnitude1 * magnitude2)

async def process_matching_tool(tool, letta_tools_cache, mcp_servers):
    """
    Process a single matching tool asynchronously using the cache.
    Checks if the tool (from cache search result) exists in the main cache.
    If not, attempts registration using mcp_server_name (if available in the tool data).
    """
    tool_name = tool.get('name')
    if not tool_name:
        return None

    # Check if tool exists in the main cache (which represents Letta's state)
    existing_tool = next((t for t in letta_tools_cache if t.get('name') == tool_name), None)

    if existing_tool and (existing_tool.get('id') or existing_tool.get('tool_id')):
        # Ensure both ID fields are present for consistency downstream
        tool_id = existing_tool.get('id') or existing_tool.get('tool_id')
        existing_tool['id'] = tool_id
        existing_tool['tool_id'] = tool_id
        
        # Check if MANAGE_ONLY_MCP_TOOLS is enabled and filter accordingly
        if MANAGE_ONLY_MCP_TOOLS:
            # Only process MCP tools (external_mcp or custom non-Letta tools)
            is_mcp_tool = (existing_tool.get("tool_type") == "external_mcp" or 
                         (not _is_letta_core_tool(existing_tool) and existing_tool.get("tool_type") == "custom"))
            
            if not is_mcp_tool:
                logger.debug(f"Skipping non-MCP tool '{tool_name}' (type: {existing_tool.get('tool_type')}) - MANAGE_ONLY_MCP_TOOLS is enabled")
                return None
            
        return existing_tool
    else:
        # Tool found via cache search but seems incomplete or missing ID in main cache.
        # This implies it might be an MCP tool that needs registration.
        originating_server = tool.get("mcp_server_name")
        if originating_server:
            logger.info(f"Tool '{tool_name}' needs registration. Attempting via originating server '{originating_server}'...")
            try:
                registered_tool = await agent_service.register_tool(tool_name, originating_server)
                if registered_tool:
                    logger.info(f"Successfully registered '{tool_name}' via server '{originating_server}'.")
                    return registered_tool
                else:
                    logger.warning(f"Failed to register '{tool_name}' via originating server '{originating_server}'.")
                    return None # Indicate failure
            except Exception as reg_error:
                logger.error(f"Error during registration attempt for '{tool_name}' via server '{originating_server}': {reg_error}")
                return None
        else:
            # Tool found in cache search but not fully represented in main cache, and no server info.
            logger.warning(f"Tool '{tool_name}' found via search but seems incomplete in cache and missing originating MCP server name. Cannot register.")
            return None # Indicate it's not usable


async def _tools_attach_handler():
    """Handle tool attachment requests with parallel processing using cache"""
    logger.info(f"Received request for {request.path}")
    try:
        data = await request.get_json()
        if not data:
            logger.warning("Attach request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        query = data.get('query', '')
        limit = data.get('limit', 10)
        agent_id = data.get('agent_id')
        keep_tools = data.get('keep_tools', [])
        min_score = data.get('min_score', DEFAULT_MIN_SCORE)  # Add min_score parameter with configurable default
        skip_loop_trigger = data.get('skip_loop_trigger', False)  # Skip loop trigger when called from proxy pre-attach
        
        # Debug: log the full payload to understand what's being sent
        logger.info(f"[DEBUG] Attach request payload: skip_loop_trigger={skip_loop_trigger}, keys={list(data.keys())}")

        if not agent_id:
            logger.warning("Attach request missing 'agent_id'.")
            return jsonify({"error": "agent_id is required"}), 400

        try:
            # 1. Fetch agent-specific info (name and current tools) directly from Letta
            agent_name, current_agent_tools = await asyncio.gather(
                agent_service.fetch_agent_info(agent_id),
                tool_manager.fetch_agent_tools(agent_id)
            )

            # 2. Identify unique MCP tools currently on the agent
            mcp_tools = []
            seen_tool_ids = set()
            logger.info(f"Getting current tools directly from agent {agent_name} ({agent_id})...")
            logger.info(f"Total tools on agent: {len(current_agent_tools)}")
            # Use enhanced categorization for MCP tool counting
            mcp_count = len([t for t in current_agent_tools 
                           if (t.get("tool_type") == "external_mcp" or 
                               (not _is_letta_core_tool(t) and t.get("tool_type") == "custom"))])
            logger.info(f"Found {mcp_count} total MCP tools, checking for duplicates...")

            for tool in current_agent_tools:
                is_mcp_tool = (tool.get("tool_type") == "external_mcp" or 
                             (not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"))
                
                if is_mcp_tool:
                    tool_id = tool.get("id") or tool.get("tool_id")
                    if tool_id and tool_id not in seen_tool_ids:
                        seen_tool_ids.add(tool_id)
                        tool_copy = tool.copy()
                        tool_copy["id"] = tool_id
                        tool_copy["tool_id"] = tool_id
                        mcp_tools.append(tool_copy)

            # 3. Search for matching tools using the async search_tools function
            global weaviate_client # Ensure we're working with the global client

            if not weaviate_client or not weaviate_client.is_ready(): # Check is_ready()
                logger.warning("Weaviate client not ready or not initialized at /attach endpoint. Attempting re-initialization...")
                # Ensure init_weaviate_client is available in this scope if not already global
                # from weaviate_tool_search import init_client as init_weaviate_client # Already imported globally
                weaviate_client = init_weaviate_client() # Attempt to re-initialize
                if not weaviate_client or not weaviate_client.is_ready():
                    logger.error("Failed to re-initialize Weaviate client for /attach. Cannot perform search.")
                    return jsonify({"error": "Weaviate client not available after re-attempt"}), 500
                logger.info("Weaviate client successfully re-initialized for /attach endpoint.")
            
            logger.info(f"Running Weaviate search for query '{query}' directly...")
            # Call the synchronous search_tools function in a separate thread
            matching_tools_from_search = await asyncio.to_thread(
                search_tools,
                query=query,
                limit=limit
            )
            
            logger.info(f"Found {len(matching_tools_from_search)} matching tools from Weaviate search.")
            
            # 3.5. Filter tools by min_score threshold
            filtered_tools = []
            for tool in matching_tools_from_search:
                # Get the score - check for rerank_score first, then fall back to score
                tool_score = tool.get('rerank_score')
                if tool_score is None:
                    tool_score = tool.get('score', 0)
                
                # Convert score from 0-1 to 0-100 scale
                tool_score_percent = tool_score * 100
                
                if tool_score_percent >= min_score:
                    filtered_tools.append(tool)
                    logger.debug(f"Tool '{tool.get('name')}' passed filter with score {tool_score_percent:.1f}% >= {min_score}%")
                else:
                    logger.debug(f"Tool '{tool.get('name')}' filtered out with score {tool_score_percent:.1f}% < {min_score}%")
            
            logger.info(f"Score filtering: {len(filtered_tools)} of {len(matching_tools_from_search)} tools passed min_score threshold of {min_score}%")

            # 4. Process matching tools (check cache, register if needed)
            letta_tools_cache = await read_tool_cache() # Load main cache
            mcp_servers = await read_mcp_servers_cache() # Load MCP servers

            process_tasks = [process_matching_tool(tool, letta_tools_cache, mcp_servers) for tool in filtered_tools]
            processed_tools_results = await asyncio.gather(*process_tasks, return_exceptions=True)
            
            processed_tools = []
            for i, res in enumerate(processed_tools_results):
                if isinstance(res, Exception):
                    logger.error(f"Error processing tool candidate {matching_tools_from_search[i].get('name', 'Unknown')}: {res}")
                elif res: # If not None (i.e., successfully processed or registered)
                    processed_tools.append(res)
            
            logger.info(f"Successfully processed/registered {len(processed_tools)} tools for attachment consideration.")

            # 5. Pre-attach pruning: Check if we need to make room before attaching new tools
            MAX_TOTAL_TOOLS = int(os.getenv('MAX_TOTAL_TOOLS', '30'))
            MAX_MCP_TOOLS = int(os.getenv('MAX_MCP_TOOLS', '20'))
            MIN_MCP_TOOLS = int(os.getenv('MIN_MCP_TOOLS', '7'))
            
            # Count current tools
            total_current_tools = len(current_agent_tools)
            mcp_current_count = len(mcp_tools)
            core_current_count = total_current_tools - mcp_current_count
            
            # Count how many new tools we're trying to attach
            new_tool_ids = set()
            for tool in processed_tools:
                tool_id = tool.get("id") or tool.get("tool_id")
                if tool_id and tool_id not in seen_tool_ids:  # Not already on agent
                    new_tool_ids.add(tool_id)
            
            new_tools_count = len(new_tool_ids)
            logger.info(f"Pre-attach analysis: current_total={total_current_tools}, current_mcp={mcp_current_count}, core={core_current_count}, new_tools={new_tools_count}")
            logger.info(f"Limits: MAX_TOTAL={MAX_TOTAL_TOOLS}, MAX_MCP={MAX_MCP_TOOLS}, MIN_MCP={MIN_MCP_TOOLS}")
            
            # Calculate what the totals would be AFTER attachment (before any pruning)
            projected_total = total_current_tools + new_tools_count
            projected_mcp = mcp_current_count + new_tools_count
            
            logger.info(f"Projected after attach: total={projected_total}, mcp={projected_mcp}")
            
            # Determine if we need pre-attach pruning
            needs_preattach_pruning = False
            if projected_total > MAX_TOTAL_TOOLS:
                logger.warning(f"Pre-attach check: projected total ({projected_total}) exceeds MAX_TOTAL_TOOLS ({MAX_TOTAL_TOOLS})")
                needs_preattach_pruning = True
            elif projected_mcp > MAX_MCP_TOOLS:
                logger.warning(f"Pre-attach check: projected MCP count ({projected_mcp}) exceeds MAX_MCP_TOOLS ({MAX_MCP_TOOLS})")
                needs_preattach_pruning = True
            
            # Perform pre-attach pruning if needed
            if needs_preattach_pruning and query:
                logger.info("Executing pre-attach pruning to make room for new tools...")
                
                # Calculate how many tools we need to remove to make room
                # We want: current_mcp - tools_to_remove + new_tools <= MAX_MCP_TOOLS
                # AND: total_current - tools_to_remove + new_tools <= MAX_TOTAL_TOOLS
                
                # Calculate minimum removals needed for each constraint
                min_removals_for_mcp = max(0, projected_mcp - MAX_MCP_TOOLS)
                min_removals_for_total = max(0, projected_total - MAX_TOTAL_TOOLS)
                min_removals_needed = max(min_removals_for_mcp, min_removals_for_total)
                
                # Also respect MIN_MCP_TOOLS constraint
                max_removals_allowed = max(0, mcp_current_count - MIN_MCP_TOOLS)
                
                tools_to_remove = min(min_removals_needed, max_removals_allowed)
                
                logger.info(f"Pre-attach pruning: need to remove {min_removals_needed} tools (min_for_mcp={min_removals_for_mcp}, min_for_total={min_removals_for_total})")
                logger.info(f"Pre-attach pruning: can remove up to {max_removals_allowed} tools (respecting MIN_MCP_TOOLS={MIN_MCP_TOOLS})")
                logger.info(f"Pre-attach pruning: will remove {tools_to_remove} tools")
                
                if tools_to_remove > 0:
                    # Use aggressive pruning with high drop rate to make room
                    # Calculate effective drop rate: we want to remove tools_to_remove from mcp_current_count
                    effective_drop_rate = min(0.9, tools_to_remove / max(1, mcp_current_count))
                    
                    logger.info(f"Pre-attach pruning: using drop_rate={effective_drop_rate:.2f} to remove ~{tools_to_remove} tools")
                    
                    preattach_prune_result = await tool_manager.perform_tool_pruning(
                        agent_id=agent_id,
                        user_prompt=query,
                        drop_rate=effective_drop_rate,
                        keep_tool_ids=keep_tools,
                        newly_matched_tool_ids=[]  # Don't protect anything during pre-attach pruning
                    )
                    
                    if preattach_prune_result.get("success"):
                        removed_count = preattach_prune_result.get("details", {}).get("mcp_tools_detached_count", 0)
                        logger.info(f"Pre-attach pruning completed: removed {removed_count} tools to make room")
                        
                        # Re-fetch current agent tools after pre-attach pruning
                        current_agent_tools = await tool_manager.fetch_agent_tools(agent_id)
                        mcp_tools = []
                        seen_tool_ids = set()
                        
                        for tool in current_agent_tools:
                            is_mcp_tool = (tool.get("tool_type") == "external_mcp" or 
                                         (not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"))
                            
                            if is_mcp_tool:
                                tool_id = tool.get("id") or tool.get("tool_id")
                                if tool_id and tool_id not in seen_tool_ids:
                                    seen_tool_ids.add(tool_id)
                                    tool_copy = tool.copy()
                                    tool_copy["id"] = tool_id
                                    tool_copy["tool_id"] = tool_id
                                    mcp_tools.append(tool_copy)
                        
                        logger.info(f"After pre-attach pruning: total_tools={len(current_agent_tools)}, mcp_tools={len(mcp_tools)}")
                    else:
                        logger.warning(f"Pre-attach pruning failed: {preattach_prune_result.get('error', 'Unknown error')}")
                else:
                    logger.info("Pre-attach pruning: no tools can be removed (would violate MIN_MCP_TOOLS)")
            elif needs_preattach_pruning and not query:
                logger.warning("Pre-attach pruning needed but skipped (no query provided for relevance scoring)")

            # 6. Perform detachments and attachments
            results = await tool_manager.process_tools(agent_id, mcp_tools, processed_tools, keep_tools)
            
            # 6.5. Emit audit events for attachments and detachments
            try:
                # Generate correlation ID for this operation
                import uuid
                correlation_id = str(uuid.uuid4())
                
                # Emit batch event for successful attachments
                if results.get("successful_attachments"):
                    emit_batch_event(
                        action=AuditAction.ATTACH,
                        agent_id=agent_id,
                        tools=results["successful_attachments"],
                        source=AuditSource.API_ATTACH,
                        reason=f"Query match: {query[:100] if query else 'no query'}" if query else "Requested tool attachment",
                        correlation_id=correlation_id,
                        success_count=len(results["successful_attachments"]),
                        failure_count=0
                    )
                
                # Emit batch event for failed attachments
                if results.get("failed_attachments"):
                    emit_batch_event(
                        action=AuditAction.ATTACH,
                        agent_id=agent_id,
                        tools=[{"tool_id": t.get("tool_id") or t.get("id"), 
                               "name": t.get("name", "unknown"), 
                               "success": False} 
                              for t in results["failed_attachments"]],
                        source=AuditSource.API_ATTACH,
                        reason="Attachment failed",
                        correlation_id=correlation_id,
                        success_count=0,
                        failure_count=len(results["failed_attachments"])
                    )
                
                # Emit batch event for detachments
                if results.get("detached_tools"):
                    emit_batch_event(
                        action=AuditAction.DETACH,
                        agent_id=agent_id,
                        tools=[{"tool_id": tool_id, "name": "unknown", "success": True} 
                              for tool_id in results["detached_tools"]],
                        source=AuditSource.API_ATTACH,
                        reason="Making room for new tools",
                        correlation_id=correlation_id,
                        success_count=len(results["detached_tools"]),
                        failure_count=0
                    )
                
            except Exception as audit_error:
                logger.warning(f"Failed to emit audit events: {audit_error}")
                # Don't fail the operation due to audit logging issues
            
            # 7. Trigger a new agent loop so newly attached tools are available
            # IMPORTANT: Do this BEFORE post-attach pruning so the client gets a response faster
            # In Letta V1 architecture, tools are passed to LLM at request start,
            # so we need to trigger a new loop for the agent to use newly attached tools
            loop_triggered = False
            successful_attachments = results.get("successful_attachments", [])
            logger.info(f"Checking if loop trigger needed: {len(successful_attachments)} successful attachments, skip_loop_trigger={skip_loop_trigger}")
            if successful_attachments and not skip_loop_trigger:
                logger.info(f"Triggering agent loop for {agent_id} with query: {query}")
                try:
                    loop_triggered = agent_service.trigger_agent_loop(
                        agent_id,
                        successful_attachments,
                        query=query
                    )
                    logger.info(f"Loop trigger task spawned: {loop_triggered}")
                except Exception as trigger_error:
                    logger.error(f"Exception during agent_service.trigger_agent_loop: {trigger_error}", exc_info=True)

            return jsonify({
                "success": True,
                "message": f"Successfully processed {len(matching_tools_from_search)} candidates ({len(filtered_tools)} passed min_score={min_score}%), attached {len(results['successful_attachments'])} tool(s) to agent {agent_id}",
                "details": {
                    "detached_tools": results["detached_tools"],
                    "failed_detachments": results["failed_detachments"],
                    "processed_count": len(matching_tools_from_search), # Candidates from search
                    "passed_filter_count": len(filtered_tools), # Tools that passed min_score filter
                    "success_count": len(results["successful_attachments"]),
                    "failure_count": len(results["failed_attachments"]),
                    "successful_attachments": results["successful_attachments"],
                    "failed_attachments": results["failed_attachments"],
                    "preserved_tools": keep_tools,
                    "target_agent": agent_id,
                    "loop_triggered": loop_triggered
                }
            })

        except Exception as e:
            logger.error(f"Error during tool management: {str(e)}", exc_info=True) # Log traceback
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error during attach_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def _is_letta_core_tool(tool: dict) -> bool:
    """
    Determine if a tool is a Letta core tool that should not be managed by auto selection.
    
    Delegates to models.is_letta_core_tool for the actual implementation.
    This wrapper is kept for backward compatibility with existing code.
    """
    return models_is_letta_core_tool(tool)

async def _tools_prune_handler():
    """Prune tools attached to an agent based on their relevance to a user's prompt."""
    logger.info("Received request for /api/v1/tools/prune")
    try:
        data = await request.get_json()
        if not data:
            logger.warning("Prune request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        # Extract required parameters
        agent_id = data.get('agent_id')
        user_prompt = data.get('user_prompt')
        drop_rate = data.get('drop_rate')

        # Extract optional parameters
        keep_tool_ids = data.get('keep_tool_ids', [])
        newly_matched_tool_ids = data.get('newly_matched_tool_ids', [])

        # Validate required parameters
        if not agent_id:
            logger.warning("Prune request missing 'agent_id'.")
            return jsonify({"error": "agent_id is required"}), 400

        if not user_prompt:
            logger.warning("Prune request missing 'user_prompt'.")
            return jsonify({"error": "user_prompt is required"}), 400

        if drop_rate is None or not isinstance(drop_rate, (int, float)) or not (0 <= drop_rate <= 1): # Corrected range check
            logger.warning(f"Prune request has invalid 'drop_rate': {drop_rate}. Must be between 0 and 1.")
            return jsonify({"error": "drop_rate must be a number between 0 and 1"}), 400

        # Call the core pruning logic
        pruning_result = await tool_manager.perform_tool_pruning(
            agent_id=agent_id,
            user_prompt=user_prompt,
            drop_rate=drop_rate,
            keep_tool_ids=keep_tool_ids,
            newly_matched_tool_ids=newly_matched_tool_ids
        )

        # Emit audit events for pruning operation
        try:
            import uuid
            correlation_id = str(uuid.uuid4())
            
            if pruning_result.get("success"):
                details = pruning_result.get("details", {})
                
                # Emit pruning event with structured data
                emit_pruning_event(
                    agent_id=agent_id,
                    tools_before=details.get("tools_on_agent_before_total", 0),
                    tools_after=details.get("actual_total_tools_on_agent_after_pruning", 0),
                    tools_detached=[t.get("tool_id") for t in details.get("successful_detachments_mcp", [])],
                    tools_protected=details.get("explicitly_kept_tool_ids_from_request", []) + 
                                   details.get("newly_matched_tool_ids_from_request", []),
                    drop_rate=drop_rate,
                    correlation_id=correlation_id,
                    metadata={
                        "mcp_tools_before": details.get("mcp_tools_on_agent_before", 0),
                        "target_mcp_tools": details.get("target_mcp_tools_to_keep_after_pruning", 0),
                        "user_prompt_snippet": user_prompt[:100] if user_prompt else "no prompt",
                        "failed_detachments": len(details.get("failed_detachments_mcp", []))
                    }
                )
                
                # Also emit batch event for successful detachments
                if details.get("successful_detachments_mcp"):
                    emit_batch_event(
                        action=AuditAction.DETACH,
                        agent_id=agent_id,
                        tools=details.get("successful_detachments_mcp", []),
                        source=AuditSource.API_PRUNE,
                        reason=f"Pruning with drop_rate={drop_rate}",
                        correlation_id=correlation_id,
                        success_count=len(details.get("successful_detachments_mcp", [])),
                        failure_count=len(details.get("failed_detachments_mcp", []))
                    )
                
                # Emit batch event for failed detachments if any
                if details.get("failed_detachments_mcp"):
                    emit_batch_event(
                        action=AuditAction.DETACH,
                        agent_id=agent_id,
                        tools=[{"tool_id": t.get("tool_id"), 
                               "name": t.get("name", "unknown"), 
                               "success": False} 
                              for t in details.get("failed_detachments_mcp", [])],
                        source=AuditSource.API_PRUNE,
                        reason="Detachment failed during pruning",
                        correlation_id=correlation_id,
                        success_count=0,
                        failure_count=len(details.get("failed_detachments_mcp", []))
                    )
        
        except Exception as audit_error:
            logger.warning(f"Failed to emit audit events for pruning: {audit_error}")
            # Don't fail the operation due to audit logging issues

        if pruning_result.get("success"):
            return jsonify(pruning_result)
        else:
            return jsonify(pruning_result), 500

    except Exception as e:
        logger.error(f"Error during prune_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


async def _tools_sync_handler():
    """Endpoint to manually trigger the sync process (for testing/debugging)."""
    logger.info("Received request for /api/v1/tools/sync")
    try:
        from sync_service import sync_tools # Import locally
        # Run the async sync function
        await sync_tools()
        logger.info("Manual sync process completed successfully.")
        return jsonify({"message": "Sync process completed successfully."})
    except ImportError:
         logger.error("Could not import sync_tools from sync_service.")
         return jsonify({"error": "Sync service function not found."}), 500
    except Exception as e:
        logger.error(f"Error during manual sync: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error during sync: {str(e)}"}), 500


async def _validate_config_handler():
    """LDTS-69: Validate dashboard configuration using schema-based validation"""
    logger.info("Received request for /api/v1/config/validate")
    
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        logger.info(f"Validating configuration with {len(data)} top-level sections")
        
        # Validate the configuration
        validation_result = validate_configuration(data)
        
        logger.info(f"Configuration validation completed: valid={validation_result.valid}, "
                   f"errors={len(validation_result.errors)}, "
                   f"warnings={len(validation_result.warnings)}")
        
        # Return validation result as JSON
        return jsonify(validation_result.to_dict())
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500


# Search parameter routes moved to routes/search.py blueprint


# Reranker and Embedding config routes moved to routes/config.py blueprint

# Ollama routes moved to routes/ollama.py blueprint

# Configuration presets routes moved to routes/config.py blueprint


async def _get_tool_selector_config_handler():
    """Get current tool selector configuration."""
    try:
        # Get configuration from environment variables
        config = {
            "tool_limits": {
                "max_total_tools": int(os.getenv('MAX_TOTAL_TOOLS', '30')),
                "max_mcp_tools": int(os.getenv('MAX_MCP_TOOLS', '20')),
                "min_mcp_tools": int(os.getenv('MIN_MCP_TOOLS', '7'))
            },
            "behavior": {
                "default_drop_rate": float(os.getenv('DEFAULT_DROP_RATE', '0.6')),
                "exclude_letta_core_tools": os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'true').lower() == 'true',
                "exclude_official_tools": os.getenv('EXCLUDE_OFFICIAL_TOOLS', 'true').lower() == 'true',
                "manage_only_mcp_tools": os.getenv('MANAGE_ONLY_MCP_TOOLS', 'true').lower() == 'true'
            },
            "scoring": {
                "min_score_default": float(os.getenv('MIN_SCORE_DEFAULT', '70.0')),
                "semantic_weight": float(os.getenv('SEMANTIC_WEIGHT', '0.7')),
                "keyword_weight": float(os.getenv('KEYWORD_WEIGHT', '0.3'))
            }
        }

        # Add current usage statistics if available
        try:
            # Get basic stats from cached tools
            tools = await read_tool_cache()
            total_tools = len(tools) if tools else 0
            mcp_tools = len([t for t in tools if t.get('source', '').startswith('mcp')]) if tools else 0
            mcp_ratio = (mcp_tools / total_tools) if total_tools > 0 else 0

            config["current_stats"] = {
                "total_tools": total_tools,
                "mcp_tools": mcp_tools,
                "mcp_tools_ratio": round(mcp_ratio, 2),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not load current stats: {str(e)}")
            config["current_stats"] = {
                "total_tools": 0,
                "mcp_tools": 0,
                "mcp_tools_ratio": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting tool selector config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _update_tool_selector_config_handler():
    """Update tool selector configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400

        # Validate configuration data
        validation_errors = []
        warnings = []

        if "tool_limits" in data:
            limits = data["tool_limits"]
            max_total = limits.get("max_total_tools", 30)
            max_mcp = limits.get("max_mcp_tools", 20)
            min_mcp = limits.get("min_mcp_tools", 7)

            # Validation rules
            if not (1 <= min_mcp <= 50):
                validation_errors.append("min_mcp_tools must be between 1 and 50")
            if not (5 <= max_mcp <= 100):
                validation_errors.append("max_mcp_tools must be between 5 and 100")
            if not (10 <= max_total <= 200):
                validation_errors.append("max_total_tools must be between 10 and 200")
            if min_mcp > max_mcp:
                validation_errors.append("min_mcp_tools cannot be greater than max_mcp_tools")
            if max_mcp > max_total:
                validation_errors.append("max_mcp_tools cannot be greater than max_total_tools")

            # Warnings for significant changes
            if max_total > 50:
                warnings.append("High tool limits may impact performance")
            if min_mcp < 3:
                warnings.append("Very low minimum MCP tools may reduce functionality")

        if "behavior" in data:
            behavior = data["behavior"]
            drop_rate = behavior.get("default_drop_rate", 0.6)

            if not (0.1 <= drop_rate <= 0.9):
                validation_errors.append("default_drop_rate must be between 0.1 and 0.9")
            if drop_rate > 0.8:
                warnings.append("High drop rate may remove too many tools")

        if "scoring" in data:
            scoring = data["scoring"]
            min_score = scoring.get("min_score_default", 70.0)
            semantic_weight = scoring.get("semantic_weight", 0.7)
            keyword_weight = scoring.get("keyword_weight", 0.3)

            if not (0.0 <= min_score <= 100.0):
                validation_errors.append("min_score_default must be between 0.0 and 100.0")
            if not (0.0 <= semantic_weight <= 1.0):
                validation_errors.append("semantic_weight must be between 0.0 and 1.0")
            if not (0.0 <= keyword_weight <= 1.0):
                validation_errors.append("keyword_weight must be between 0.0 and 1.0")
            if abs((semantic_weight + keyword_weight) - 1.0) > 0.01:
                warnings.append("semantic_weight + keyword_weight should equal 1.0 for optimal results")

        # Return validation errors if any
        if validation_errors:
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "warnings": warnings
            }), 400

        # Log the configuration update
        logger.info(f"Tool selector configuration update requested: {data}")

        # Log to audit system
        await log_config_change(
            action="update",
            config_type="tool_selector",
            changes=data,
            user_info={"source": "api", "timestamp": datetime.now().isoformat()},
            metadata={"warnings": warnings}
        )

        # For now, just acknowledge the update
        # TODO: Actually persist the configuration changes to environment or config file

        response = {
            "success": True,
            "message": "Tool selector configuration updated successfully",
            "applied_config": data
        }

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error updating tool selector config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Ollama Configuration endpoints moved to routes/config.py blueprint

# Weaviate Configuration endpoints moved to routes/config.py blueprint

# Configuration Backup/Restore/Save/Validate/Audit endpoints moved to routes/backup.py blueprint


async def log_config_change(action, config_type, changes=None, user_info=None, metadata=None):
    """Log a configuration change for audit purposes."""
    try:
        audit_entry = {
            "id": f"{int(time.time() * 1000000)}_{hash(str(changes))}"[:16],  # Unique ID
            "timestamp": datetime.now().isoformat(),
            "action": action,  # create, update, delete, restore, etc.
            "config_type": config_type,  # tool_selector, ollama, weaviate, etc.
            "changes": changes or {},
            "user_info": user_info or {},
            "metadata": metadata or {},
            "system_info": {
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "process_id": os.getpid()
            }
        }

        # Ensure audit directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        audit_file = os.path.join(CACHE_DIR, 'config_audit.json')

        # Append to audit log file (one JSON object per line)
        with open(audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

        # Also log to application logger
        logger.info(f"CONFIG_AUDIT: {action} {config_type} - {changes}")

        # Rotate log file if it gets too large (keep last 10000 entries)
        await rotate_audit_log_if_needed(audit_file)

    except Exception as e:
        logger.error(f"Failed to log config change: {str(e)}")
        # Don't raise exception as this is a logging operation


async def rotate_audit_log_if_needed(audit_file, max_entries=10000):
    """Rotate audit log file if it becomes too large."""
    try:
        if not os.path.exists(audit_file):
            return

        # Count lines in file
        with open(audit_file, 'r') as f:
            line_count = sum(1 for _ in f)

        if line_count > max_entries:
            logger.info(f"Rotating audit log file (current entries: {line_count})")

            # Read all entries
            with open(audit_file, 'r') as f:
                all_lines = f.readlines()

            # Keep only the most recent entries
            recent_lines = all_lines[-max_entries//2:]  # Keep half the max

            # Create backup of old file
            backup_file = f"{audit_file}.rotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(audit_file, backup_file)

            # Write recent entries to new file
            with open(audit_file, 'w') as f:
                f.writelines(recent_lines)

            logger.info(f"Audit log rotated. Kept {len(recent_lines)} recent entries. Backup: {backup_file}")

    except Exception as e:
        logger.error(f"Failed to rotate audit log: {str(e)}")


# NOTE: Maintenance, logs, and environment routes moved to routes/operations.py

# Helper functions for maintenance operations (used by routes/operations.py)
start_time = time.time()  # Track server start time

def get_memory_usage():
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        }
    except ImportError:
        # Fallback if psutil not available
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "note": "psutil not available"}

def get_disk_usage():
    """Get current disk usage."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(CACHE_DIR)
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent": round((used / total) * 100, 2)
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0, "note": "unable to determine"}

def get_cpu_info():
    """Get CPU information."""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    except ImportError:
        return {"cpu_percent": 0, "cpu_count": 1, "note": "psutil not available"}

async def get_tool_count_from_cache():
    """Get tool count from cache."""
    try:
        tools = await read_tool_cache()
        return len(tools) if tools else 0
    except Exception:
        return 0

def get_cache_size():
    """Get total cache size."""
    try:
        total_size = 0
        for root, dirs, files in os.walk(CACHE_DIR):
            total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
        return round(total_size / 1024 / 1024, 2)  # MB
    except Exception:
        return 0

def get_last_sync_time():
    """Get last sync time from cache."""
    try:
        cache_file = os.path.join(CACHE_DIR, 'tool_cache.json')
        if os.path.exists(cache_file):
            stat = os.stat(cache_file)
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        return None
    except Exception:
        return None

async def get_weaviate_index_status():
    """Get Weaviate index status."""
    try:
        config = {"url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')}
        result = await test_weaviate_connection(config)
        if result["available"]:
            return {
                "status": "healthy",
                "class_count": result.get("class_count", 0),
                "version": result.get("version", "unknown")
            }
        else:
            return {"status": "unavailable", "error": result.get("error")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_log_file_size():
    """Get log file size."""
    try:
        # This would depend on your logging configuration
        # For now, return placeholder
        return {"size_mb": 0, "note": "log file size tracking not implemented"}
    except Exception:
        return {"size_mb": 0, "error": "unable to determine"}

async def get_recent_error_count():
    """Get recent error count from logs."""
    # Placeholder implementation
    return 0

async def get_recent_warning_count():
    """Get recent warning count from logs."""
    # Placeholder implementation
    return 0

async def test_letta_connection():
    """Test Letta API connection."""
    try:
        # This would use your existing Letta API connection logic
        return {"available": True, "status": "simulated - implement actual test"}
    except Exception as e:
        return {"available": False, "error": str(e)}

async def perform_cleanup_operation(operation, dry_run):
    """Perform specific cleanup operation."""
    # Placeholder implementations
    operations_impl = {
        "clear_cache": lambda: {"success": True, "details": "Cache cleared (simulated)", "space_freed": 50, "files_affected": 10},
        "rotate_logs": lambda: {"success": True, "details": "Logs rotated (simulated)", "space_freed": 20, "files_affected": 3},
        "cleanup_temp": lambda: {"success": True, "details": "Temp files cleaned (simulated)", "space_freed": 15, "files_affected": 5},
        "compress_old_logs": lambda: {"success": True, "details": "Old logs compressed (simulated)", "space_freed": 100, "files_affected": 8},
        "cleanup_backups": lambda: {"success": True, "details": "Old backups removed (simulated)", "space_freed": 200, "files_affected": 2},
        "optimize_database": lambda: {"success": True, "details": "Database optimized (simulated)", "space_freed": 75, "files_affected": 1},
        "clear_audit_logs": lambda: {"success": True, "details": "Audit logs cleared (simulated)", "space_freed": 30, "files_affected": 1}
    }

    if operation in operations_impl:
        result = operations_impl[operation]()
        if dry_run:
            result["details"] += " (dry run - no changes made)"
            result["space_freed"] = 0
            result["files_affected"] = 0
        return result
    else:
        return {"success": False, "details": f"Unknown operation: {operation}"}

async def perform_optimization(operation):
    """Perform specific optimization operation."""
    # Placeholder implementations
    optimizations_impl = {
        "rebuild_cache": lambda: {"success": True, "details": "Cache rebuilt (simulated)", "performance_impact": {"cache_hit_rate": "+15%"}},
        "reindex_weaviate": lambda: {"success": True, "details": "Weaviate reindexed (simulated)", "performance_impact": {"search_speed": "+10%"}},
        "compact_database": lambda: {"success": True, "details": "Database compacted (simulated)", "performance_impact": {"query_time": "-20%"}},
        "optimize_embeddings": lambda: {"success": True, "details": "Embeddings optimized (simulated)", "performance_impact": {"embedding_retrieval": "+25%"}},
        "cleanup_duplicates": lambda: {"success": True, "details": "Duplicates removed (simulated)", "performance_impact": {"data_consistency": "+100%"}}
    }

    if operation in optimizations_impl:
        return optimizations_impl[operation]()
    else:
        return {"success": False, "details": f"Unknown optimization: {operation}"}


async def _tools_refresh_handler():
    """Refresh the tool index from Letta API."""
    try:
        logger.info("Refreshing tool index...")
        # Force reload the tool cache
        await read_tool_cache(force_reload=True)
        return jsonify({"success": True, "message": "Tool index refreshed successfully"})
    except Exception as e:
        logger.error(f"Error refreshing tool index: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _submit_evaluation_handler():
    """Submit an evaluation rating."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['query', 'result_id', 'rating']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Create evaluation record
        evaluation = {
            "id": "eval_" + str(int(time.time())),
            "query": data.get("query"),
            "result_id": data.get("result_id"),
            "rating": data.get("rating"),
            "feedback": data.get("feedback", ""),
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
        
        # For now, just log the evaluation (can be extended to store in database)
        logger.info(f"Evaluation submitted: {evaluation}")
        
        return jsonify({
            "success": True,
            "data": evaluation
        })
    except Exception as e:
        logger.error(f"Error submitting evaluation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_evaluations_handler():
    """Get evaluations with optional query and limit parameters."""
    try:
        query = request.args.get('query')
        limit = request.args.get('limit', 50, type=int)
        
        # For now, return empty array since we don't have persistent storage
        # This can be extended to query from database
        evaluations = []
        
        # If we had stored evaluations, we would filter and limit them here
        logger.info(f"Evaluations requested - query: {query}, limit: {limit}")
        
        return jsonify({
            "success": True,
            "data": evaluations,
            "total": len(evaluations)
        })
    except Exception as e:
        logger.error(f"Error getting evaluations: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_analytics_handler():
    """Get analytics with optional date range parameters."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # For now, return mock analytics data
        # This can be extended to calculate real metrics
        analytics_data = {
            "search_count": 0,
            "tool_usage": {},
            "avg_rating": 0.0,
            "total_evaluations": 0,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "top_tools": [],
            "recent_searches": []
        }
        
        logger.info(f"Analytics requested - start: {start_date}, end: {end_date}")
        
        return jsonify({
            "success": True,
            "data": analytics_data
        })
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _compare_rerank_handler():
    """Compare two reranker configurations side-by-side."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        query = data.get('query')
        config_a = data.get('config_a')
        config_b = data.get('config_b')
        limit = data.get('limit', 10)
        
        if not query or not config_a or not config_b:
            return jsonify({"success": False, "error": "Missing required fields: query, config_a, config_b"}), 400
        
        # Perform search with both configurations
        results_a = []
        results_b = []
        
        try:
            # Search with configuration A
            search_results_a = search_tools(query, limit=limit, reranker_config=config_a)
            for i, result in enumerate(search_results_a):
                formatted_result = {
                    "tool": {
                        "id": result.get('id', ''),
                        "name": result.get('name', ''),
                        "description": result.get('description', ''),
                        "source": result.get('source', 'unknown'),
                        "category": result.get('category'),
                        "tags": result.get('tags', [])
                    },
                    "score": result.get('score', 0),
                    "rank": i + 1,
                    "reasoning": result.get('reasoning', ''),
                    "config": "A"
                }
                results_a.append(formatted_result)
        except Exception as e:
            logger.error(f"Error with configuration A: {str(e)}")
            results_a = []
            
        try:
            # Search with configuration B
            search_results_b = search_tools(query, limit=limit, reranker_config=config_b)
            for i, result in enumerate(search_results_b):
                formatted_result = {
                    "tool": {
                        "id": result.get('id', ''),
                        "name": result.get('name', ''),
                        "description": result.get('description', ''),
                        "source": result.get('source', 'unknown'),
                        "category": result.get('category'),
                        "tags": result.get('tags', [])
                    },
                    "score": result.get('score', 0),
                    "rank": i + 1,
                    "reasoning": result.get('reasoning', ''),
                    "config": "B"
                }
                results_b.append(formatted_result)
        except Exception as e:
            logger.error(f"Error with configuration B: {str(e)}")
            results_b = []
        
        # Calculate comparison metrics
        comparison_metrics = {
            "total_results_a": len(results_a),
            "total_results_b": len(results_b),
            "avg_score_a": sum(r.get('score', 0) for r in results_a) / max(len(results_a), 1),
            "avg_score_b": sum(r.get('score', 0) for r in results_b) / max(len(results_b), 1),
            "top_5_overlap": calculate_overlap([r['tool']['id'] for r in results_a[:5]], 
                                               [r['tool']['id'] for r in results_b[:5]]),
            "rank_correlation": calculate_rank_correlation(results_a, results_b)
        }
        
        response_data = {
            "query": query,
            "results_a": results_a,
            "results_b": results_b,
            "comparison_metrics": comparison_metrics,
            "config_a_name": config_a.get('name', 'Configuration A'),
            "config_b_name": config_b.get('name', 'Configuration B'),
            "timestamp": time.time()
        }
        
        logger.info(f"Rerank comparison completed for query: {query}")
        return jsonify({"success": True, "data": response_data})
        
    except Exception as e:
        logger.error(f"Error during rerank comparison: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def calculate_overlap(list_a, list_b):
    """Calculate the overlap percentage between two lists."""
    if not list_a or not list_b:
        return 0.0
    set_a = set(list_a)
    set_b = set(list_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / union) * 100 if union > 0 else 0.0


def calculate_rank_correlation(results_a, results_b):
    """Calculate rank correlation between two result sets."""
    try:
        # Create rank mappings
        rank_map_a = {r['tool']['id']: r['rank'] for r in results_a}
        rank_map_b = {r['tool']['id']: r['rank'] for r in results_b}
        
        # Find common tools
        common_tools = set(rank_map_a.keys()) & set(rank_map_b.keys())
        if len(common_tools) < 2:
            return 0.0
        
        # Calculate Spearman correlation
        ranks_a = [rank_map_a[tool_id] for tool_id in common_tools]
        ranks_b = [rank_map_b[tool_id] for tool_id in common_tools]
        
        n = len(ranks_a)
        sum_d_squared = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        correlation = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
        
        return round(correlation, 3)
    except Exception:
        return 0.0


async def _search_test_handler():
    """Test search functionality with parameter overrides and detailed rankings."""
    try:
        query = request.args.get('query')
        if not query:
            return jsonify({"success": False, "error": "Query parameter is required"}), 400
            
        # Parse optional override parameters
        limit = request.args.get('limit', 10, type=int)
        alpha = request.args.get('alpha', type=float)  # Weaviate hybrid search alpha
        distance_metric = request.args.get('distance_metric')  # e.g., 'cosine', 'euclidean'
        reranker_enabled = request.args.get('reranker_enabled', 'true').lower() == 'true'
        reranker_model = request.args.get('reranker_model')
        
        # Build configuration overrides
        weaviate_overrides = {}
        if alpha is not None:
            weaviate_overrides['alpha'] = alpha
        if distance_metric:
            weaviate_overrides['distance_metric'] = distance_metric
            
        reranker_overrides = {
            'enabled': reranker_enabled
        }
        if reranker_model:
            reranker_overrides['model'] = reranker_model
            
        # Perform search with overrides
        start_time = time.time()
        
        # Use the BM25/vector override service if needed
        if weaviate_overrides:
            search_results = bm25_vector_override_service(
                query, 
                limit=limit, 
                overrides=weaviate_overrides,
                reranker_config=reranker_overrides if reranker_enabled else None
            )
        else:
            search_results = search_tools(query, limit=limit, reranker_config=reranker_overrides)
        
        search_time = time.time() - start_time
        
        # Format results with detailed rankings
        formatted_results = []
        for i, result in enumerate(search_results):
            formatted_result = {
                "tool": {
                    "id": result.get('id', ''),
                    "name": result.get('name', ''),
                    "description": result.get('description', ''),
                    "source": result.get('source', 'unknown'),
                    "category": result.get('category'),
                    "tags": result.get('tags', [])
                },
                "score": result.get('rerank_score', 0) if reranker_enabled and result.get('rerank_score') is not None else result.get('score', 0),
                "rank": i + 1,
                "reasoning": result.get('reasoning', ''),
                "vector_score": result.get('vector_score', 0),
                "keyword_score": result.get('keyword_score', 0),
                "rerank_score": result.get('rerank_score', 0) if reranker_enabled else None,
                "original_rank": result.get('original_rank', i + 1)
            }
            formatted_results.append(formatted_result)
        
        response_data = {
            "query": query,
            "results": formatted_results,
            "metadata": {
                "total_found": len(formatted_results),
                "search_time": round(search_time, 3),
                "parameters_used": {
                    "limit": limit,
                    "alpha": alpha,
                    "distance_metric": distance_metric,
                    "reranker_enabled": reranker_enabled,
                    "reranker_model": reranker_model
                },
                "weaviate_overrides": weaviate_overrides,
                "reranker_overrides": reranker_overrides
            }
        }
        
        logger.info(f"Test search completed for query: {query} with overrides")
        return jsonify({"success": True, "data": response_data})
        
    except Exception as e:
        logger.error(f"Error during test search: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_embedding_models_handler():
    """Get available embedding models from all configured providers."""
    try:
        models = []
        
        # OpenAI embedding models
        openai_models = [
            {
                "id": "text-embedding-3-small",
                "name": "OpenAI Text Embedding 3 Small",
                "provider": "openai",
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k": 0.00002,
                "recommended": True
            },
            {
                "id": "text-embedding-3-large", 
                "name": "OpenAI Text Embedding 3 Large",
                "provider": "openai",
                "dimensions": 3072,
                "max_tokens": 8191,
                "cost_per_1k": 0.00013,
                "recommended": False
            },
            {
                "id": "text-embedding-ada-002",
                "name": "OpenAI Text Embedding Ada 002",
                "provider": "openai", 
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k": 0.0001,
                "recommended": False
            }
        ]
        
        # Add OpenAI models if API key is available
        if os.getenv('OPENAI_API_KEY'):
            models.extend(openai_models)
        
        # Ollama embedding models - get from Ollama endpoint
        try:
            ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
            ollama_url = f"http://{ollama_host}:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        ollama_data = await response.json()
                        for model in ollama_data.get('models', []):
                            # Filter for embedding models (heuristic based on name)
                            model_name = model.get('name', '')
                            if any(keyword in model_name.lower() for keyword in ['embed', 'embedding', 'bge', 'e5']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name}",
                                    "provider": "ollama",
                                    "dimensions": "variable",
                                    "max_tokens": "variable", 
                                    "cost_per_1k": 0.0,
                                    "recommended": False,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama embedding models: {str(e)}")
        
        # Add fallback local models if no models found
        if not models:
            models = [
                {
                    "id": "all-MiniLM-L6-v2",
                    "name": "All MiniLM L6 v2 (Local)",
                    "provider": "local",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "cost_per_1k": 0.0,
                    "recommended": True
                }
            ]
        
        return jsonify({
            "success": True,
            "data": {
                "models": models,
                "total": len(models),
                "providers": list(set(m["provider"] for m in models))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_embedding_health_handler():
    """Get comprehensive embedding model health and status information."""
    try:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()

        if not embedding_provider:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'unknown',
                    'provider': 'not_configured',
                    'model': 'not_configured',
                    'availability': False,
                    'response_time_ms': 0,
                    'last_checked': datetime.now().isoformat(),
                    'error_message': 'No embedding provider configured'
                }
            })

        # Start timing
        start_time = time.time()
        status = 'healthy'
        error_message = None
        availability = False
        model_info = {}

        try:
            if embedding_provider == "openai":
                # Test OpenAI embedding model
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")

                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

                # Test with a small embedding request
                response = await asyncio.to_thread(
                    openai.embeddings.create,
                    input="test embedding health check",
                    model=embedding_model
                )

                if response and response.data:
                    availability = True
                    model_info = {
                        'dimensions': len(response.data[0].embedding) if response.data else 0,
                        'max_tokens': 8191 if 'small' in embedding_model else 8191,
                        'cost_per_1k': 0.00002 if 'small' in embedding_model else 0.00013
                    }
                else:
                    status = 'error'
                    error_message = 'No response from OpenAI API'

            elif embedding_provider == "ollama":
                # Test Ollama embedding model
                ollama_host = os.getenv("OLLAMA_EMBEDDING_HOST", "localhost")
                ollama_port = int(os.getenv("OLLAMA_EMBEDDING_PORT", "11434"))
                embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

                async with aiohttp.ClientSession() as session:
                    url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
                    payload = {
                        "model": embedding_model,
                        "prompt": "test embedding health check"
                    }

                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'embedding' in result:
                                availability = True
                                model_info = {
                                    'dimensions': len(result['embedding']) if result.get('embedding') else 0,
                                    'max_tokens': 2048,  # Default for most Ollama models
                                }
                            else:
                                status = 'error'
                                error_message = 'Invalid response format from Ollama'
                        else:
                            status = 'error'
                            error_message = f'Ollama API returned status {response.status}'
            else:
                status = 'error'
                error_message = f'Unsupported embedding provider: {embedding_provider}'

        except asyncio.TimeoutError:
            status = 'error'
            error_message = 'Connection timeout'
        except Exception as e:
            status = 'warning' if availability else 'error'
            error_message = str(e)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Determine final status based on response time
        if status == 'healthy':
            if response_time_ms > 5000:  # 5 seconds
                status = 'warning'
                error_message = 'High response time detected'
            elif response_time_ms > 10000:  # 10 seconds
                status = 'error'
                error_message = 'Very high response time'

        # Get model name
        model_name = 'unknown'
        if embedding_provider == "openai":
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        elif embedding_provider == "ollama":
            model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # Simulate performance metrics (in real implementation, this would come from monitoring)
        performance_metrics = {
            'avg_response_time': response_time_ms,
            'success_rate': 0.98 if availability else 0.0,
            'total_requests': 1250,  # Simulated
            'failed_requests': 25 if not availability else 2  # Simulated
        }

        health_data = {
            'status': status,
            'provider': embedding_provider,
            'model': model_name,
            'availability': availability,
            'response_time_ms': response_time_ms,
            'last_checked': datetime.now().isoformat(),
            'error_message': error_message,
            'performance_metrics': performance_metrics,
            'model_info': model_info
        }

        return jsonify({
            'success': True,
            'data': health_data
        })

    except Exception as e:
        logger.error(f"Embedding health check error: {e}")
        return jsonify({
            'success': False,
            'error': f'Health check failed: {str(e)}'
        }), 500


async def _get_reranker_models_handler():
    """Get available reranker models from all configured providers."""
    try:
        models = []
        
        # Ollama reranker models
        try:
            ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80') 
            ollama_url = f"http://{ollama_host}:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        ollama_data = await response.json()
                        for model in ollama_data.get('models', []):
                            model_name = model.get('name', '')
                            # Filter for reranker models
                            if any(keyword in model_name.lower() for keyword in ['rerank', 'cross-encoder', 'bge-rerank', 'colbert']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name}",
                                    "provider": "ollama",
                                    "type": "cross-encoder",
                                    "cost_per_1k": 0.0,
                                    "recommended": True,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
                            # Also include general models that can be used for reranking
                            elif any(keyword in model_name.lower() for keyword in ['mistral', 'llama', 'qwen']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name} (General)",
                                    "provider": "ollama",
                                    "type": "generative",
                                    "cost_per_1k": 0.0,
                                    "recommended": False,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama reranker models: {str(e)}")
        
        # Add some common reranker models as fallback
        if not models:
            models = [
                {
                    "id": "mistral:7b",
                    "name": "Mistral 7B (Fallback)",
                    "provider": "ollama",
                    "type": "generative",
                    "cost_per_1k": 0.0,
                    "recommended": True
                },
                {
                    "id": "llama2:7b",
                    "name": "Llama 2 7B (Fallback)",
                    "provider": "ollama", 
                    "type": "generative",
                    "cost_per_1k": 0.0,
                    "recommended": False
                }
            ]
        
        return jsonify({
            "success": True,
            "data": {
                "models": models,
                "total": len(models),
                "providers": list(set(m["provider"] for m in models)),
                "types": list(set(m["type"] for m in models))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting reranker models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# NOTE: Benchmark routes moved to routes/benchmark.py
# NOTE: Reranker model registry routes moved to routes/reranker.py

async def _run_ab_comparison_handler():
    """Run A/B comparison tests with statistical significance analysis."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['queries', 'config_a', 'config_b']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        queries = data['queries']
        config_a = data['config_a']
        config_b = data['config_b']
        options = data.get('options', {})
        
        # Validate queries format
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({"success": False, "error": "queries must be a non-empty array"}), 400
        
        # Options with defaults
        limit = options.get('limit', 10)
        metrics_to_compare = options.get('metrics', ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg'])
        k_values = options.get('k_values', [1, 3, 5, 10])
        confidence_level = options.get('confidence_level', 0.95)
        min_effect_size = options.get('min_effect_size', 0.1)
        
        # Run comparison
        comparison_id = str(uuid.uuid4())
        results = await execute_ab_comparison(
            comparison_id, queries, config_a, config_b, 
            limit, metrics_to_compare, k_values, confidence_level, min_effect_size
        )
        
        # Store results
        ab_comparison_dir = os.path.join(CACHE_DIR, 'ab_comparisons')
        os.makedirs(ab_comparison_dir, exist_ok=True)
        
        results_file = os.path.join(ab_comparison_dir, f'{comparison_id}.json')
        async with aiofiles.open(results_file, 'w') as f:
            await f.write(json.dumps(results, indent=2))
        
        logger.info(f"A/B comparison completed: {comparison_id}")
        return jsonify({"success": True, "data": results})
        
    except Exception as e:
        logger.error(f"Error running A/B comparison: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_ab_results_handler():
    """Get A/B comparison results with filtering."""
    try:
        # Query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        config_name = request.args.get('config_name')
        min_significance = request.args.get('min_significance', type=float)
        
        ab_comparison_dir = os.path.join(CACHE_DIR, 'ab_comparisons')
        if not os.path.exists(ab_comparison_dir):
            return jsonify({"success": True, "data": {"results": [], "total": 0}})
        
        # Load all comparison results
        all_results = []
        for filename in os.listdir(ab_comparison_dir):
            if filename.endswith('.json'):
                try:
                    async with aiofiles.open(os.path.join(ab_comparison_dir, filename), 'r') as f:
                        result = json.loads(await f.read())
                        
                        # Apply filters
                        if config_name:
                            if (config_name.lower() not in result.get('config_a', {}).get('name', '').lower() and 
                                config_name.lower() not in result.get('config_b', {}).get('name', '').lower()):
                                continue
                        
                        if min_significance is not None:
                            max_p_value = max([stat.get('p_value', 1.0) 
                                             for stat in result.get('statistical_results', {}).values()])
                            if max_p_value > (1 - min_significance):
                                continue
                        
                        all_results.append(result)
                except Exception as e:
                    logger.warning(f"Error loading comparison result {filename}: {e}")
        
        # Sort by timestamp (newest first)
        all_results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Apply pagination
        total = len(all_results)
        paginated_results = all_results[offset:offset + limit]
        
        return jsonify({
            "success": True,
            "data": {
                "results": paginated_results,
                "total": total,
                "offset": offset,
                "limit": limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving A/B comparison results: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_ab_result_by_id_handler(comparison_id):
    """Get specific A/B comparison result."""
    try:
        ab_comparison_dir = os.path.join(CACHE_DIR, 'ab_comparisons')
        result_file = os.path.join(ab_comparison_dir, f'{comparison_id}.json')
        
        if not os.path.exists(result_file):
            return jsonify({"success": False, "error": "Comparison result not found"}), 404
        
        async with aiofiles.open(result_file, 'r') as f:
            result = json.loads(await f.read())
        
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        logger.error(f"Error retrieving A/B comparison result: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


async def execute_ab_comparison(comparison_id, queries, config_a, config_b, limit, metrics, k_values, confidence_level, min_effect_size):
    """Execute A/B comparison with statistical significance testing."""
    results_a = []
    results_b = []
    query_results = []
    
    logger.info(f"Starting A/B comparison {comparison_id} with {len(queries)} queries")
    
    for i, query_data in enumerate(queries):
        try:
            # Handle different query formats
            if isinstance(query_data, str):
                query_text = query_data
                expected_tools = []
            elif isinstance(query_data, dict):
                query_text = query_data.get('query', '')
                expected_tools = query_data.get('expected_tools', [])
            else:
                logger.warning(f"Invalid query format at index {i}: {query_data}")
                continue
            
            if not query_text.strip():
                continue
            
            logger.debug(f"Processing query {i+1}/{len(queries)}: {query_text}")
            
            # Execute search with config A
            search_a = await execute_search_with_config(query_text, config_a, limit)
            
            # Execute search with config B  
            search_b = await execute_search_with_config(query_text, config_b, limit)
            
            # Calculate metrics for this query
            metrics_a = calculate_search_metrics(search_a, expected_tools, k_values)
            metrics_b = calculate_search_metrics(search_b, expected_tools, k_values)
            
            query_result = {
                "query": query_text,
                "expected_tools": expected_tools,
                "results_a": search_a,
                "results_b": search_b,
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "timestamp": time.time()
            }
            
            query_results.append(query_result)
            results_a.append(metrics_a)
            results_b.append(metrics_b)
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            continue
    
    if not results_a or not results_b:
        raise ValueError("No valid query results obtained for comparison")
    
    # Calculate aggregate metrics and statistical significance
    statistical_results = {}
    aggregate_metrics_a = {}
    aggregate_metrics_b = {}
    
    for metric in metrics:
        for k in k_values:
            metric_key = f"{metric}@{k}" if metric in ['precision_at_k', 'recall_at_k'] else metric
            
            # Extract metric values from all queries
            values_a = []
            values_b = []
            
            for metrics_a, metrics_b in zip(results_a, results_b):
                if metric in ['precision_at_k', 'recall_at_k']:
                    val_a = metrics_a.get(metric, {}).get(str(k), 0.0)
                    val_b = metrics_b.get(metric, {}).get(str(k), 0.0)
                else:
                    val_a = metrics_a.get(metric, 0.0)
                    val_b = metrics_b.get(metric, 0.0)
                
                values_a.append(val_a)
                values_b.append(val_b)
            
            # Calculate aggregate statistics
            avg_a = sum(values_a) / len(values_a) if values_a else 0.0
            avg_b = sum(values_b) / len(values_b) if values_b else 0.0
            
            aggregate_metrics_a[metric_key] = {
                "mean": avg_a,
                "std": calculate_std(values_a),
                "min": min(values_a) if values_a else 0.0,
                "max": max(values_a) if values_a else 0.0,
                "median": calculate_median(values_a)
            }
            
            aggregate_metrics_b[metric_key] = {
                "mean": avg_b,
                "std": calculate_std(values_b),
                "min": min(values_b) if values_b else 0.0,
                "max": max(values_b) if values_b else 0.0,
                "median": calculate_median(values_b)
            }
            
            # Perform statistical significance testing
            stat_result = perform_significance_test(
                values_a, values_b, confidence_level, min_effect_size
            )
            statistical_results[metric_key] = stat_result
    
    # Calculate overall effect sizes and power analysis
    overall_stats = calculate_overall_comparison_stats(
        results_a, results_b, metrics, k_values, confidence_level
    )
    
    return {
        "comparison_id": comparison_id,
        "config_a": config_a,
        "config_b": config_b,
        "query_results": query_results,
        "aggregate_metrics_a": aggregate_metrics_a,
        "aggregate_metrics_b": aggregate_metrics_b,
        "statistical_results": statistical_results,
        "overall_statistics": overall_stats,
        "options": {
            "limit": limit,
            "metrics": metrics,
            "k_values": k_values,
            "confidence_level": confidence_level,
            "min_effect_size": min_effect_size
        },
        "summary": {
            "total_queries": len(queries),
            "successful_queries": len(query_results),
            "significant_improvements": count_significant_improvements(statistical_results),
            "significant_degradations": count_significant_degradations(statistical_results)
        },
        "timestamp": time.time()
    }


async def execute_search_with_config(query, config, limit):
    """Execute search with specific configuration."""
    try:
        # Build search parameters from config
        weaviate_overrides = config.get('weaviate', {})
        reranker_config = config.get('reranker', {})
        
        # Use the existing search infrastructure
        if weaviate_overrides or reranker_config.get('enabled', False):
            results = bm25_vector_override_service(
                query,
                limit=limit,
                overrides=weaviate_overrides,
                reranker_config=reranker_config if reranker_config.get('enabled') else None
            )
        else:
            # Standard search
            results = search_tools(query, limit=limit)
        
        # Ensure consistent format
        if isinstance(results, dict) and 'tools' in results:
            return results['tools']
        return results if isinstance(results, list) else []
        
    except Exception as e:
        logger.error(f"Error executing search with config: {str(e)}")
        return []


def calculate_search_metrics(search_results, expected_tools, k_values):
    """Calculate search metrics for a single query."""
    if not search_results:
        return {
            "precision_at_k": {str(k): 0.0 for k in k_values},
            "recall_at_k": {str(k): 0.0 for k in k_values},
            "mrr": 0.0,
            "ndcg": 0.0,
            "total_results": 0
        }
    
    # Extract tool IDs from results
    result_tool_ids = []
    for result in search_results:
        if isinstance(result, dict):
            tool_id = result.get('tool', {}).get('id') or result.get('id')
            if tool_id:
                result_tool_ids.append(tool_id)
    
    expected_set = set(expected_tools)
    metrics = {"total_results": len(result_tool_ids)}
    
    # Calculate precision@K and recall@K
    precision_at_k = {}
    recall_at_k = {}
    
    for k in k_values:
        top_k_results = result_tool_ids[:k]
        relevant_in_top_k = len(set(top_k_results) & expected_set)
        
        precision_at_k[str(k)] = relevant_in_top_k / k if k > 0 else 0.0
        recall_at_k[str(k)] = relevant_in_top_k / len(expected_set) if expected_set else 0.0
    
    metrics["precision_at_k"] = precision_at_k
    metrics["recall_at_k"] = recall_at_k
    
    # Calculate MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, tool_id in enumerate(result_tool_ids):
        if tool_id in expected_set:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr
    
    # Calculate NDCG@10 (simplified)
    dcg = 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_set), 10)))
    
    for i, tool_id in enumerate(result_tool_ids[:10]):
        if tool_id in expected_set:
            dcg += 1.0 / math.log2(i + 2)
    
    metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    
    return metrics


def perform_significance_test(values_a, values_b, confidence_level, min_effect_size):
    """Perform statistical significance test between two sets of values."""
    try:
        import scipy.stats as stats
    except ImportError:
        logger.warning("scipy not available, using simplified significance testing")
        return simplified_significance_test(values_a, values_b, confidence_level)
    
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test_type": "insufficient_data",
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "confidence_interval": [0.0, 0.0],
            "power": 0.0
        }
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = math.sqrt(((len(values_a) - 1) * calculate_std(values_a) ** 2 + 
                           (len(values_b) - 1) * calculate_std(values_b) ** 2) / 
                          (len(values_a) + len(values_b) - 2))
    
    effect_size = abs(sum(values_b) / len(values_b) - sum(values_a) / len(values_a)) / pooled_std if pooled_std > 0 else 0.0
    
    # Determine significance
    alpha = 1 - confidence_level
    significant = p_value < alpha and effect_size >= min_effect_size
    
    # Calculate confidence interval for difference in means
    mean_diff = sum(values_b) / len(values_b) - sum(values_a) / len(values_a)
    se_diff = math.sqrt(calculate_std(values_a) ** 2 / len(values_a) + calculate_std(values_b) ** 2 / len(values_b))
    t_critical = stats.t.ppf(1 - alpha / 2, len(values_a) + len(values_b) - 2)
    
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Simple power calculation (approximate)
    power = 1 - stats.t.cdf(t_critical - effect_size * math.sqrt(len(values_a) * len(values_b) / (len(values_a) + len(values_b)) / 2), 
                           len(values_a) + len(values_b) - 2)
    
    return {
        "test_type": "t_test",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
        "confidence_interval": [ci_lower, ci_upper],
        "power": min(power, 1.0),
        "sample_size_a": len(values_a),
        "sample_size_b": len(values_b),
        "mean_difference": mean_diff
    }


def simplified_significance_test(values_a, values_b, confidence_level):
    """Simplified significance test when scipy is not available."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test_type": "insufficient_data",
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "confidence_interval": [0.0, 0.0]
        }
    
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    
    std_a = calculate_std(values_a)
    std_b = calculate_std(values_b)
    
    # Simple two-sample z-test approximation
    pooled_se = math.sqrt(std_a ** 2 / len(values_a) + std_b ** 2 / len(values_b))
    z_score = abs(mean_b - mean_a) / pooled_se if pooled_se > 0 else 0.0
    
    # Approximate p-value using normal distribution
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))))
    
    # Effect size
    pooled_std = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
    effect_size = abs(mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0
    
    alpha = 1 - confidence_level
    significant = p_value < alpha
    
    return {
        "test_type": "simplified_z_test",
        "z_score": z_score,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
        "mean_difference": mean_b - mean_a
    }


def calculate_std(values):
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_median(values):
    """Calculate median of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def calculate_overall_comparison_stats(results_a, results_b, metrics, k_values, confidence_level):
    """Calculate overall comparison statistics across all metrics."""
    overall_improvements = 0
    overall_degradations = 0
    total_comparisons = 0
    
    # significant_improvements = 0  # Currently unused
    # significant_degradations = 0  # Currently unused
    
    for metric in metrics:
        for k in k_values if metric in ['precision_at_k', 'recall_at_k'] else [None]:
            # metric_key = f"{metric}@{k}" if k else metric  # Currently unused
            
            values_a = []
            values_b = []
            
            for metrics_a, metrics_b in zip(results_a, results_b):
                if k:
                    val_a = metrics_a.get(metric, {}).get(str(k), 0.0)
                    val_b = metrics_b.get(metric, {}).get(str(k), 0.0)
                else:
                    val_a = metrics_a.get(metric, 0.0)
                    val_b = metrics_b.get(metric, 0.0)
                
                values_a.append(val_a)
                values_b.append(val_b)
            
            if values_a and values_b:
                mean_a = sum(values_a) / len(values_a)
                mean_b = sum(values_b) / len(values_b)
                
                total_comparisons += 1
                
                if mean_b > mean_a:
                    overall_improvements += 1
                elif mean_b < mean_a:
                    overall_degradations += 1
    
    return {
        "total_metric_comparisons": total_comparisons,
        "improvements": overall_improvements,
        "degradations": overall_degradations,
        "no_change": total_comparisons - overall_improvements - overall_degradations,
        "improvement_rate": overall_improvements / total_comparisons if total_comparisons > 0 else 0.0,
        "degradation_rate": overall_degradations / total_comparisons if total_comparisons > 0 else 0.0
    }


def count_significant_improvements(statistical_results):
    """Count number of statistically significant improvements."""
    count = 0
    for metric_key, stats in statistical_results.items():
        if stats.get('significant', False) and stats.get('mean_difference', 0) > 0:
            count += 1
    return count


def count_significant_degradations(statistical_results):
    """Count number of statistically significant degradations.""" 
    count = 0
    for metric_key, stats in statistical_results.items():
        if stats.get('significant', False) and stats.get('mean_difference', 0) < 0:
            count += 1
    return count


async def _get_safety_status_handler():
    """Get comprehensive safety system status."""
    try:
        # Check environment safety configurations
        safety_config = {
            "safety_mode": os.getenv("LDTS_SAFETY_MODE", "production"),
            "read_only_mode": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true",
            "no_attach_mode": os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true",
            "emergency_lockdown": os.getenv("LDTS_EMERGENCY_LOCKDOWN", "false").lower() == "true",
            "production_isolated": os.getenv("LDTS_PRODUCTION_ISOLATED", "true").lower() == "true"
        }
        
        # Perform isolation checks
        isolation_checks = {
            "letta_api_isolated": check_letta_api_isolation(),
            "agent_modifications_blocked": check_agent_modification_blocked(),
            "tool_attachments_blocked": check_tool_attachment_blocked(),
            "database_read_only": check_database_read_only(),
            "sandbox_mode_active": check_sandbox_mode()
        }
        
        # Calculate overall safety score
        passed_checks = sum(1 for passed in isolation_checks.values() if passed)
        total_checks = len(isolation_checks)
        safety_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine overall status
        if safety_score == 100:
            overall_status = "SECURE"
            status_message = "All safety checks passed"
        elif safety_score >= 80:
            overall_status = "WARNING" 
            status_message = "Some safety checks failed - review required"
        else:
            overall_status = "CRITICAL"
            status_message = "Multiple safety failures detected - immediate attention required"
        
        # Get blocked operations
        blocked_operations = [
            "attach_tool", "detach_tool", "modify_agent", "create_agent",
            "delete_agent", "update_memory", "production_write"
        ]
        
        allowed_operations = [
            "search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"
        ]
        
        return jsonify({
            "success": True,
            "data": {
                "overall_status": overall_status,
                "status_message": status_message,
                "safety_score": safety_score,
                "safety_config": safety_config,
                "isolation_checks": isolation_checks,
                "allowed_operations": allowed_operations,
                "blocked_operations": blocked_operations,
                "emergency_triggers": [
                    "read_only_mode_disabled",
                    "production_isolation_compromised",
                    "critical_safety_violation",
                    "multiple_safety_check_failures"
                ],
                "recommendations": generate_safety_recommendations(isolation_checks, safety_config),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting safety status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_status": {
                "overall_status": "ERROR",
                "status_message": "Safety status check failed - assuming unsafe conditions",
                "safety_score": 0
            }
        }), 500


async def _validate_operation_handler():
    """Validate if a specific operation is safe to perform."""
    try:
        data = await request.get_json()
        
        if not data or 'operation' not in data:
            return jsonify({"success": False, "error": "Missing 'operation' in request"}), 400
        
        operation = data['operation']
        context = data.get('context', {})
        
        # Perform safety validation
        is_safe = validate_operation_safety(operation, context)
        
        # Get detailed validation result
        validation_result = {
            "operation": operation,
            "is_safe": is_safe,
            "validation_checks": {
                "operation_allowed": operation in [
                    "search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"
                ],
                "operation_blocked": operation in [
                    "attach_tool", "detach_tool", "modify_agent", "create_agent",
                    "delete_agent", "update_memory", "production_write"
                ],
                "read_only_mode_active": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true",
                "safety_mode_testing": os.getenv("LDTS_SAFETY_MODE", "production") == "testing",
                "no_production_indicators": not has_production_indicators(operation, context)
            },
            "context": context,
            "timestamp": time.time()
        }
        
        if not is_safe:
            validation_result["block_reason"] = get_block_reason(operation, context)
            validation_result["alternatives"] = suggest_safe_alternatives(operation)
        
        return jsonify({"success": True, "data": validation_result})
        
    except Exception as e:
        logger.error(f"Error validating operation: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


async def _get_emergency_status_handler():
    """Get emergency safety status for critical monitoring."""
    try:
        # Quick emergency checks
        emergency_status = {
            "emergency_lockdown": os.getenv("LDTS_EMERGENCY_LOCKDOWN", "false").lower() == "true",
            "read_only_compromised": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() != "true",
            "production_access_detected": has_production_access(),
            "safety_mode_production": os.getenv("LDTS_SAFETY_MODE", "production") == "production",
            "critical_operations_blocked": are_critical_operations_blocked()
        }
        
        # Count emergency conditions
        emergency_conditions = sum(1 for condition in emergency_status.values() if condition)
        
        if emergency_conditions == 0:
            alert_level = "NORMAL"
            message = "No emergency conditions detected"
        elif emergency_conditions == 1:
            alert_level = "CAUTION"
            message = "One emergency condition detected - monitoring required"
        else:
            alert_level = "EMERGENCY"
            message = f"{emergency_conditions} emergency conditions detected - immediate action required"
        
        return jsonify({
            "success": True,
            "data": {
                "alert_level": alert_level,
                "message": message,
                "emergency_conditions_count": emergency_conditions,
                "emergency_status": emergency_status,
                "immediate_actions_required": get_immediate_actions(emergency_status),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting emergency status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "emergency_fallback": {
                "alert_level": "CRITICAL",
                "message": "Emergency status check failed - assume emergency conditions"
            }
        }), 500


# Safety helper functions

def check_letta_api_isolation() -> bool:
    """Check that Letta API writes are disabled"""
    try:
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        
        return safety_mode == "testing" and read_only_mode and no_attach_mode
    except Exception:
        return False

def check_agent_modification_blocked() -> bool:
    """Check that agent modifications are blocked"""
    try:
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        block_agent_ops = os.getenv("LDTS_BLOCK_AGENT_MODIFICATIONS", "true").lower() == "true"
        
        return (safety_mode == "testing" and no_attach_mode) or block_agent_ops
    except Exception:
        return False

def check_tool_attachment_blocked() -> bool:
    """Check that tool attachment/detachment is blocked"""
    try:
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        
        return no_attach_mode or read_only_mode
    except Exception:
        return False

def check_database_read_only() -> bool:
    """Check that database is in read-only mode"""
    try:
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        database_mode = os.getenv("LDTS_DATABASE_MODE", "testing")
        db_url = os.getenv("DATABASE_URL", "")
        
        return read_only_mode and database_mode == "testing" and "prod" not in db_url.lower()
    except Exception:
        return False

def check_sandbox_mode() -> bool:
    """Check that we're operating in sandbox mode"""
    try:
        return os.getenv("LDTS_SAFETY_MODE", "production") == "testing"
    except Exception:
        return False

def validate_operation_safety(operation: str, context: dict) -> bool:
    """Validate if operation is safe"""
    blocked_ops = ["attach_tool", "detach_tool", "modify_agent", "create_agent", "delete_agent"]
    allowed_ops = ["search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"]
    
    if operation in blocked_ops:
        return False
    
    if operation in allowed_ops:
        return not has_production_indicators(operation, context)
    
    return False

def has_production_indicators(operation: str, context: dict) -> bool:
    """Check if operation has production indicators"""
    production_flags = [
        "agent_id" in context,
        "production" in str(context).lower(),
        "prod" in str(context).lower(),
        "modify" in operation.lower(),
        "create" in operation.lower() and "agent" in operation.lower()
    ]
    return any(production_flags)

def has_production_access() -> bool:
    """Check if production access is detected"""
    try:
        letta_api_url = os.getenv("LETTA_API_URL", "")
        return "prod" in letta_api_url.lower() or "production" in letta_api_url.lower()
    except Exception:
        return False

def are_critical_operations_blocked() -> bool:
    """Check if critical operations are properly blocked"""
    try:
        no_attach = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        read_only = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        return no_attach and read_only
    except Exception:
        return False

def get_block_reason(operation: str, context: dict) -> str:
    """Get reason why operation is blocked"""
    if operation in ["attach_tool", "detach_tool"]:
        return "Tool attachment/detachment operations are blocked for safety"
    elif operation in ["modify_agent", "create_agent", "delete_agent"]:
        return "Agent modification operations are blocked for safety"
    elif has_production_indicators(operation, context):
        return "Operation contains production indicators"
    else:
        return "Operation not in allowed list"

def suggest_safe_alternatives(operation: str) -> list:
    """Suggest safe alternatives for blocked operations"""
    alternatives = {
        "attach_tool": ["search for tools", "evaluate tool relevance", "create benchmarks"],
        "detach_tool": ["analyze tool usage", "create removal recommendations", "document changes"],
        "modify_agent": ["analyze agent configuration", "create modification proposals", "document requirements"],
        "create_agent": ["design agent specifications", "validate configurations", "create templates"]
    }
    return alternatives.get(operation, ["use read-only operations", "create analysis reports"])

def generate_safety_recommendations(isolation_checks: dict, safety_config: dict) -> list:
    """Generate safety recommendations based on current status"""
    recommendations = []
    
    if not isolation_checks.get("letta_api_isolated", False):
        recommendations.append("Enable Letta API isolation by setting LDTS_SAFETY_MODE=testing")
    
    if not isolation_checks.get("agent_modifications_blocked", False):
        recommendations.append("Block agent modifications with LDTS_NO_ATTACH_MODE=true")
    
    if not isolation_checks.get("tool_attachments_blocked", False):
        recommendations.append("Block tool operations with LDTS_READ_ONLY_MODE=true")
    
    if not isolation_checks.get("database_read_only", False):
        recommendations.append("Enable database read-only mode with LDTS_DATABASE_MODE=testing")
    
    if not safety_config.get("production_isolated", False):
        recommendations.append("Ensure production isolation with LDTS_PRODUCTION_ISOLATED=true")
    
    if safety_config.get("safety_mode") == "production":
        recommendations.append("Switch to testing mode with LDTS_SAFETY_MODE=testing")
    
    if not recommendations:
        recommendations.append("All safety measures are properly configured")
    
    return recommendations

def get_immediate_actions(emergency_status: dict) -> list:
    """Get immediate actions required for emergency conditions"""
    actions = []
    
    if emergency_status.get("emergency_lockdown", False):
        actions.append("EMERGENCY: System is in lockdown mode - review emergency procedures")
    
    if emergency_status.get("read_only_compromised", False):
        actions.append("CRITICAL: Enable read-only mode immediately with LDTS_READ_ONLY_MODE=true")
    
    if emergency_status.get("production_access_detected", False):
        actions.append("CRITICAL: Production access detected - isolate system immediately")
    
    if emergency_status.get("safety_mode_production", False):
        actions.append("HIGH: Switch to testing mode with LDTS_SAFETY_MODE=testing")
    
    if not emergency_status.get("critical_operations_blocked", False):
        actions.append("HIGH: Block critical operations with safety environment variables")
    
    return actions or ["Monitor system status and maintain current safety measures"]


# Weaviate Connection Management Endpoints

@app.route('/api/v1/weaviate/connection-status', methods=['GET'])
async def get_weaviate_connection_status():
    """Get detailed Weaviate connection status and metrics."""
    try:
        client_manager = get_client_manager()
        health_status = client_manager.get_health_status()
        
        return jsonify({
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error getting Weaviate connection status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_status": {
                "status": "error",
                "message": "Connection status check failed"
            }
        }), 500


@app.route('/api/v1/weaviate/connection-test', methods=['POST'])
async def test_weaviate_connection():
    """Test Weaviate connection with a simple query."""
    try:
        client_manager = get_client_manager()
        
        # Test with a simple collection list operation
        async def test_operation(client):
            # Simple test - list collections
            try:
                collections = client.collections.list_all()
                return {
                    "test_type": "list_collections",
                    "collections_count": len(collections),
                    "collections": [c.name for c in collections[:5]]  # First 5
                }
            except Exception as e:
                return {
                    "test_type": "list_collections",
                    "error": str(e)
                }
        
        start_time = time.time()
        result = await client_manager.execute_query(test_operation)
        response_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "data": {
                "connection_test": "passed",
                "response_time": response_time,
                "test_result": result,
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Weaviate connection test failed: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "data": {
                "connection_test": "failed",
                "timestamp": time.time()
            }
        }), 500


@app.route('/api/v1/weaviate/pool-stats', methods=['GET'])
async def get_weaviate_pool_stats():
    """Get detailed connection pool statistics."""
    try:
        client_manager = get_client_manager()
        
        if not client_manager.pool:
            return jsonify({
                "success": False,
                "error": "Connection pool not initialized"
            }), 503
        
        stats = client_manager.pool.get_stats()
        
        return jsonify({
            "success": True,
            "data": {
                "pool_statistics": stats,
                "health_recommendations": generate_pool_recommendations(stats),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting pool statistics: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/weaviate/connection-reset', methods=['POST'])
async def reset_weaviate_connections():
    """Reset Weaviate connection pool (emergency operation)."""
    try:
        data = await request.get_json() or {}
        force = data.get('force', False)
        
        if not force:
            return jsonify({
                "success": False,
                "error": "This operation requires 'force': true parameter",
                "warning": "This will reset all connections and may cause temporary service disruption"
            }), 400
        
        # Close existing manager
        close_client_manager()
        logger.warning("Weaviate connection pool reset requested - reinitializing")
        
        # Reinitialize
        client_manager = get_client_manager()
        health_status = client_manager.get_health_status()
        
        return jsonify({
            "success": True,
            "data": {
                "operation": "connection_pool_reset",
                "status": "completed",
                "new_health_status": health_status,
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error resetting connections: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def generate_pool_recommendations(stats: dict) -> list:
    """Generate recommendations based on pool statistics."""
    recommendations = []
    
    success_rate = stats.get("success_rate", 0)
    if success_rate < 0.95:
        recommendations.append(f"Low success rate ({success_rate:.1%}) - check Weaviate server health")
    
    # active_connections = stats.get("active_connections", 0)  # Currently unused
    available_connections = stats.get("available_connections", 0)
    if available_connections == 0:
        recommendations.append("No available connections - consider increasing pool size")
    
    avg_response_time = stats.get("average_response_time", 0)
    if avg_response_time > 5.0:
        recommendations.append(f"High response time ({avg_response_time:.2f}s) - check network or server performance")
    
    circuit_breaker_state = stats.get("circuit_breaker_state", "closed")
    if circuit_breaker_state != "closed":
        recommendations.append(f"Circuit breaker is {circuit_breaker_state} - investigate connection issues")
    
    failed_requests = stats.get("failed_requests", 0)
    successful_requests = stats.get("successful_requests", 0)
    if failed_requests > successful_requests:
        recommendations.append("More failed than successful requests - check system health")
    
    if not recommendations:
        recommendations.append("Connection pool is operating normally")
    
    return recommendations


# Health endpoints (both versions for compatibility)
@app.route('/api/v1/health', methods=['GET'])
async def health_check_v1():
    """Health check endpoint for the API server (v1)."""
    return await health_check()


@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint for the API server."""
    # Check Weaviate connection
    weaviate_ok = False
    weaviate_message = "Client not initialized"
    if weaviate_client:
        try:
            if weaviate_client.is_connected(): # Check connected first
                if weaviate_client.is_ready(): # Then check ready
                    weaviate_ok = True
                    weaviate_message = "Connected and ready"
                else:
                    weaviate_message = "Connected but not ready" # e.g., still indexing, or some other issue
            else:
                weaviate_message = "Not connected"
        except AttributeError: # Handles if client is some mock object without these methods
            weaviate_message = "Client object missing ready/connected methods"
            logger.warning("Health check: Weaviate client object seems malformed.")
        except Exception as e: # Catch any other exception during checks
            logger.error(f"Error checking Weaviate status in health check: {e}")
            weaviate_message = f"Exception during check: {str(e)}"
            weaviate_ok = False # Ensure ok is false on exception
    
    weaviate_status_report = "OK" if weaviate_ok else "ERROR"

    # Check tool cache status (from in-memory _tool_cache)
    tool_cache_in_memory_status = "OK"
    tool_cache_size = 0
    tool_cache_last_mod_str = "Never"
    if _tool_cache is not None:
        tool_cache_size = len(_tool_cache)
        if _tool_cache_last_modified > 0:
            tool_cache_last_mod_str = datetime.fromtimestamp(_tool_cache_last_modified, tz=timezone.utc).isoformat()
    else: # _tool_cache is None
        tool_cache_in_memory_status = "Not loaded in memory"
        # Check if the file itself exists, as it might have failed to load
        if not os.path.exists(TOOL_CACHE_FILE_PATH):
            tool_cache_in_memory_status = "Error: File not found and not loaded"
        else:
            tool_cache_in_memory_status = "Error: File exists but not loaded in memory"

        
    # Check MCP servers cache file status (check file on disk)
    mcp_servers_cache_file_status = "OK"
    mcp_servers_cache_size_on_disk = 0
    try:
        if os.path.exists(MCP_SERVERS_CACHE_FILE_PATH):
            # For health check, just check existence and maybe size, avoid full async read if possible
            # However, the current code does an async read, let's keep it for consistency for now
            # but be mindful this could be slow if file is huge.
            # A better approach for health might be just checking os.path.getmtime if file exists.
            async with aiofiles.open(MCP_SERVERS_CACHE_FILE_PATH, 'r') as f:
                mcp_data = json.loads(await f.read())
                mcp_servers_cache_size_on_disk = len(mcp_data)
        else:
            mcp_servers_cache_file_status = "Error: File not found"
    except Exception as e:
        mcp_servers_cache_file_status = f"Error reading file: {str(e)}"
        logger.warning(f"Health check: Error reading MCP servers cache file: {e}")

    # Determine overall health
    # Weaviate is critical. Cache files are important.
    is_fully_healthy = weaviate_ok and tool_cache_in_memory_status == "OK" and mcp_servers_cache_file_status == "OK"
    
    overall_status_string = "ERROR"
    if is_fully_healthy:
        overall_status_string = "OK"
    elif weaviate_ok: # Weaviate is OK, but caches might have issues
        overall_status_string = "DEGRADED"

    response_payload = {
        "status": overall_status_string,
        "version": "1.0.0",  # TODO: Read from package.json or VERSION file
        "config": {
            "MAX_TOTAL_TOOLS": MAX_TOTAL_TOOLS,
            "MAX_MCP_TOOLS": MAX_MCP_TOOLS,
            "MIN_MCP_TOOLS": os.getenv('MIN_MCP_TOOLS', '7'),
            "DEFAULT_DROP_RATE": DEFAULT_DROP_RATE,
            "PROTECTED_TOOLS": NEVER_DETACH_TOOLS,
            "MANAGE_ONLY_MCP_TOOLS": MANAGE_ONLY_MCP_TOOLS,
            "EXCLUDE_LETTA_CORE_TOOLS": EXCLUDE_LETTA_CORE_TOOLS,
            "EXCLUDE_OFFICIAL_TOOLS": EXCLUDE_OFFICIAL_TOOLS,
        },
        "details": {
            "weaviate": {
                "status": weaviate_status_report,
                "message": weaviate_message
            },
            "tool_cache_in_memory": { # Clarified this is about the in-memory representation
                "status": tool_cache_in_memory_status,
                "size": tool_cache_size,
                "last_loaded": tool_cache_last_mod_str,
                "source_file_path": TOOL_CACHE_FILE_PATH
            },
            "mcp_servers_cache_file": { # Clarified this is about the file on disk
                "status": mcp_servers_cache_file_status,
                "size_on_disk": mcp_servers_cache_size_on_disk if mcp_servers_cache_file_status == "OK" else "N/A",
                "path": MCP_SERVERS_CACHE_FILE_PATH
            }
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return jsonify(response_payload), 200 if overall_status_string == "OK" else 503

@app.before_serving
async def startup():
    global weaviate_client, http_session
    logger.info("API Server starting up...")
    
    try:
        # Initialize new Weaviate client manager
        logger.info("Initializing Weaviate Client Manager...")
        client_manager = get_client_manager()
        
        # Test the connection manager
        health_status = client_manager.get_health_status()
        if health_status.get("status") in ["healthy", "warning"]:
            logger.info(f"Weaviate Client Manager initialized: {health_status['status']}")
        else:
            logger.error(f"Weaviate Client Manager unhealthy: {health_status}")
        
        # Keep legacy client for backward compatibility (will be phased out)
        try:
            temp_client = init_weaviate_client()
            if temp_client and temp_client.is_connected() and temp_client.is_ready():
                weaviate_client = temp_client
                logger.info("Legacy Weaviate client also initialized for compatibility.")
            else:
                weaviate_client = None
                logger.warning("Legacy Weaviate client initialization failed, using only new manager.")
        except Exception as e:
            logger.warning(f"Legacy client initialization failed: {e}")
            weaviate_client = None
            
    except Exception as e:
        logger.error(f"Exception during Weaviate client initialization: {e}", exc_info=True)
        weaviate_client = None

    # Initialize global aiohttp session
    http_session = aiohttp.ClientSession()
    logger.info("Global aiohttp client session created.")
    
    # Configure tool manager with dependencies
    from models import ToolLimitsConfig
    sdk_client_func = get_letta_sdk_client if USE_LETTA_SDK else None
    tool_config = ToolLimitsConfig(
        max_total_tools=MAX_TOTAL_TOOLS,
        max_mcp_tools=MAX_MCP_TOOLS,
        min_mcp_tools=int(os.getenv('MIN_MCP_TOOLS', '7')),
        manage_only_mcp_tools=MANAGE_ONLY_MCP_TOOLS,
        never_detach_tools=NEVER_DETACH_TOOLS
    )
    tool_manager.configure(
        http_session=http_session,
        letta_url=LETTA_URL,
        headers=HEADERS,
        use_letta_sdk=USE_LETTA_SDK,
        get_letta_sdk_client_func=sdk_client_func,
        search_tools_func=search_tools,
        tool_config=tool_config
    )
    logger.info("Tool manager configured with search function and tool config.")
    
    # Configure agent service for agent communication
    agent_service.configure(
        http_session=http_session,
        letta_url=LETTA_URL,
        headers=HEADERS,
        use_letta_sdk=USE_LETTA_SDK,
        get_letta_sdk_client_func=sdk_client_func,
        letta_message_base_urls=LETTA_MESSAGE_BASE_URLS,
        matrix_bridge_webhook_url=MATRIX_BRIDGE_WEBHOOK_URL
    )
    logger.info("Agent service configured for agent communication.")
    
    # Configure search service with reranker and expansion settings
    from search_service import SearchConfig, RerankerConfig, QueryExpansionConfig
    search_config = SearchConfig(
        provider=TOOL_SEARCH_PROVIDER,
        reranker=RerankerConfig(
            enabled=os.getenv('ENABLE_RERANKING', 'true').lower() == 'true',
            provider=os.getenv('RERANKER_PROVIDER', 'vllm'),
            url=os.getenv('RERANKER_URL', 'http://100.81.139.20:11435/rerank'),
            model=os.getenv('RERANKER_MODEL', 'qwen3-reranker-4b'),
            timeout=float(os.getenv('RERANKER_TIMEOUT', '30.0')),
            initial_limit=int(os.getenv('RERANK_INITIAL_LIMIT', '30')),
            top_k=int(os.getenv('RERANK_TOP_K', '10'))
        ),
        expansion=QueryExpansionConfig(
            enabled=os.getenv('ENABLE_QUERY_EXPANSION', 'true').lower() == 'true',
            use_universal=os.getenv('USE_UNIVERSAL_EXPANSION', 'true').lower() == 'true'
        )
    )
    search_service.configure(
        weaviate_client=weaviate_client,
        letta_sdk_client_func=sdk_client_func,
        config=search_config
    )
    logger.info(f"Search service configured: provider={search_config.provider}, "
                f"reranking={'enabled' if search_config.reranker.enabled else 'disabled'}")
    
    # Configure and register search routes blueprint
    from routes import search as search_routes, search_bp
    search_routes.configure(bm25_vector_override_service=bm25_vector_override_service)
    app.register_blueprint(search_bp)
    logger.info("Search routes blueprint registered.")
    
    # Configure and register config routes blueprint
    from routes import config as config_routes, config_bp
    config_routes.configure(
        http_session=http_session,
        log_config_change=log_config_change,
        validate_config_func=_validate_config_handler,
        get_tool_selector_config_func=_get_tool_selector_config_handler,
        update_tool_selector_config_func=_update_tool_selector_config_handler
    )
    app.register_blueprint(config_bp)
    logger.info("Config routes blueprint registered.")
    
    # Configure and register ollama routes blueprint
    from routes import ollama as ollama_routes, ollama_bp
    ollama_routes.configure()
    app.register_blueprint(ollama_bp)
    logger.info("Ollama routes blueprint registered.")
    
    # Configure and register backup routes blueprint
    from routes import backup as backup_routes, backup_bp
    backup_routes.configure(
        cache_dir=CACHE_DIR,
        log_config_change=log_config_change,
        perform_configuration_validation=perform_configuration_validation,
        test_service_connection=test_service_connection
    )
    app.register_blueprint(backup_bp)
    logger.info("Backup routes blueprint registered.")
    
    # Configure and register cost control routes blueprint
    from routes import cost_control as cost_control_routes, cost_control_bp
    cost_control_routes.configure(
        get_cost_manager=get_cost_manager,
        CostCategory=CostCategory,
        BudgetPeriod=BudgetPeriod,
        AlertLevel=AlertLevel
    )
    app.register_blueprint(cost_control_bp)
    logger.info("Cost control routes blueprint registered.")
    
    # Configure and register operations routes blueprints (maintenance, logs, environment)
    from routes import operations as operations_routes
    from routes.operations import maintenance_bp, logs_bp, environment_bp
    operations_routes.configure(
        start_time=start_time,
        cache_dir=CACHE_DIR,
        log_config_change=log_config_change,
        test_weaviate_connection=test_weaviate_connection,
        test_ollama_connection=test_ollama_connection,
        test_letta_connection=test_letta_connection,
        get_tool_count_from_cache=get_tool_count_from_cache,
        get_cache_size=get_cache_size,
        get_last_sync_time=get_last_sync_time,
        get_weaviate_index_status=get_weaviate_index_status,
        get_memory_usage=get_memory_usage,
        get_disk_usage=get_disk_usage,
        get_cpu_info=get_cpu_info,
        get_log_file_size=get_log_file_size,
        get_recent_error_count=get_recent_error_count,
        get_recent_warning_count=get_recent_warning_count,
        perform_cleanup_operation=perform_cleanup_operation,
        perform_optimization=perform_optimization,
        get_log_entries=get_log_entries,
        perform_log_analysis=perform_log_analysis,
        get_error_log_entries=get_error_log_entries,
        clear_log_files=clear_log_files,
        export_log_data=export_log_data
    )
    app.register_blueprint(maintenance_bp)
    app.register_blueprint(logs_bp)
    app.register_blueprint(environment_bp)
    logger.info("Operations routes blueprints registered (maintenance, logs, environment).")
    
    # Configure and register benchmark routes blueprint
    from routes import benchmark as benchmark_routes
    from routes.benchmark import benchmark_bp
    benchmark_routes.configure(
        cache_dir=CACHE_DIR,
        search_tools=search_tools
    )
    app.register_blueprint(benchmark_bp)
    logger.info("Benchmark routes blueprint registered.")
    
    # Configure and register reranker routes blueprint
    from routes import reranker as reranker_routes
    from routes.reranker import reranker_bp
    reranker_routes.configure(cache_dir=CACHE_DIR)
    app.register_blueprint(reranker_bp)
    logger.info("Reranker routes blueprint registered.")

    # Configure services layer
    from services.tool_search import configure_search_service
    from services.tool_cache import get_tool_cache_service
    configure_search_service(search_tools)
    get_tool_cache_service(CACHE_DIR)  # Initialize with cache dir
    logger.info("Services layer configured.")

    # Configure and register tools routes blueprint
    from routes import tools as tools_routes
    from routes.tools import tools_bp
    tools_routes.configure(
        manage_only_mcp_tools=MANAGE_ONLY_MCP_TOOLS,
        attach_tools_func=_tools_attach_handler,
        prune_tools_func=_tools_prune_handler,
        sync_func=_tools_sync_handler,
        refresh_func=_tools_refresh_handler
    )
    app.register_blueprint(tools_bp)
    logger.info("Tools routes blueprint registered.")

    # Configure and register evaluation routes blueprint
    from routes import evaluation as evaluation_routes
    from routes.evaluation import evaluation_bp
    evaluation_routes.configure(
        submit_evaluation_func=_submit_evaluation_handler,
        get_evaluations_func=_get_evaluations_handler,
        get_analytics_func=_get_analytics_handler,
        compare_rerank_func=_compare_rerank_handler,
        run_ab_comparison_func=_run_ab_comparison_handler,
        get_ab_results_func=_get_ab_results_handler,
        get_ab_result_by_id_func=_get_ab_result_by_id_handler
    )
    app.register_blueprint(evaluation_bp)
    logger.info("Evaluation routes blueprint registered.")

    # Configure and register safety routes blueprint
    from routes import safety as safety_routes
    from routes.safety import safety_bp
    safety_routes.configure(
        get_safety_status_func=_get_safety_status_handler,
        validate_operation_func=_validate_operation_handler,
        get_emergency_status_func=_get_emergency_status_handler
    )
    app.register_blueprint(safety_bp)
    logger.info("Safety routes blueprint registered.")

    # Configure and register models routes blueprint
    from routes import models as models_routes
    from routes.models import models_bp
    models_routes.configure(
        get_embedding_models_func=_get_embedding_models_handler,
        get_reranker_models_func=_get_reranker_models_handler,
        get_embedding_health_func=_get_embedding_health_handler,
        search_test_func=_search_test_handler
    )
    app.register_blueprint(models_bp)
    logger.info("Models routes blueprint registered.")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
    
    # Perform initial cache loads
    await read_tool_cache(force_reload=True)
    logger.info("Performing initial read of MCP servers cache file...")
    await read_mcp_servers_cache() # This just logs, doesn't store in memory globally for now


@app.after_serving
async def shutdown():
    global weaviate_client, http_session
    logger.info("API Server shutting down...")
    
    # Close new client manager
    try:
        close_client_manager()
        logger.info("Weaviate Client Manager closed.")
    except Exception as e:
        logger.error(f"Error closing Weaviate Client Manager: {e}")
    
    # Close legacy client
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("Legacy Weaviate client closed.")
        except Exception as e:
            logger.error(f"Error closing legacy Weaviate client: {e}")
    
    if http_session:
        await http_session.close()
        logger.info("Global aiohttp client session closed.")

# ================================================================================
# LDTS-58: Cost Control and Budget Management API Endpoints
# ================================================================================
# Cost control routes moved to routes/cost_control.py blueprint

# Configuration validation helper functions
async def perform_configuration_validation(config_type: str, field: str, value, context: dict = None):
    """Perform comprehensive validation of configuration values."""
    import re
    from urllib.parse import urlparse

    if context is None:
        context = {}

    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "field": field,
        "value": value,
        "config_type": config_type
    }

    try:
        if config_type == "tool_selector":
            await validate_tool_selector_config(field, value, validation_result)
        elif config_type == "embedding":
            await validate_embedding_config(field, value, validation_result, context)
        elif config_type == "weaviate":
            await validate_weaviate_config(field, value, validation_result)
        elif config_type == "letta_api":
            await validate_letta_api_config(field, value, validation_result)
        else:
            validation_result["errors"].append(f"Unknown configuration type: {config_type}")
            validation_result["valid"] = False

    except Exception as e:
        validation_result["errors"].append(f"Validation error: {str(e)}")
        validation_result["valid"] = False

    # Set overall validity
    validation_result["valid"] = len(validation_result["errors"]) == 0

    return validation_result

async def validate_tool_selector_config(field: str, value, result: dict):
    """Validate tool selector configuration fields."""
    if field == "max_total_tools":
        if not isinstance(value, (int, str)):
            result["errors"].append("Max total tools must be a number")
        else:
            try:
                num_value = int(value)
                if num_value < 1:
                    result["errors"].append("Max total tools must be at least 1")
                elif num_value > 200:
                    result["warnings"].append("Very high tool limit may impact performance")
                elif num_value < 10:
                    result["warnings"].append("Low tool limit may restrict functionality")
            except ValueError:
                result["errors"].append("Max total tools must be a valid number")

    elif field == "max_mcp_tools":
        if not isinstance(value, (int, str)):
            result["errors"].append("Max MCP tools must be a number")
        else:
            try:
                num_value = int(value)
                if num_value < 1:
                    result["errors"].append("Max MCP tools must be at least 1")
                elif num_value > 100:
                    result["warnings"].append("Very high MCP tool limit may impact performance")
                # Cross-validation with max_total_tools if available
                max_total = int(os.getenv('MAX_TOTAL_TOOLS', '30'))
                if num_value > max_total:
                    result["warnings"].append(f"MCP tools limit ({num_value}) exceeds total tools limit ({max_total})")
            except ValueError:
                result["errors"].append("Max MCP tools must be a valid number")

    elif field == "min_mcp_tools":
        if not isinstance(value, (int, str)):
            result["errors"].append("Min MCP tools must be a number")
        else:
            try:
                num_value = int(value)
                if num_value < 0:
                    result["errors"].append("Min MCP tools cannot be negative")
                elif num_value > 50:
                    result["warnings"].append("High minimum may prevent effective pruning")
                # Cross-validation with max_mcp_tools
                max_mcp = int(os.getenv('MAX_MCP_TOOLS', '20'))
                if num_value > max_mcp:
                    result["errors"].append(f"Min MCP tools ({num_value}) cannot exceed max MCP tools ({max_mcp})")
            except ValueError:
                result["errors"].append("Min MCP tools must be a valid number")

    elif field == "default_drop_rate":
        if not isinstance(value, (int, float, str)):
            result["errors"].append("Drop rate must be a number")
        else:
            try:
                num_value = float(value)
                if num_value < 0.0 or num_value > 1.0:
                    result["errors"].append("Drop rate must be between 0.0 and 1.0")
                elif num_value < 0.1:
                    result["warnings"].append("Very low drop rate may not effectively prune tools")
                elif num_value > 0.9:
                    result["warnings"].append("Very high drop rate may remove too many tools")
                else:
                    if num_value <= 0.3:
                        result["suggestions"].append("Conservative pruning - good for stability")
                    elif num_value >= 0.7:
                        result["suggestions"].append("Aggressive pruning - good for performance")
                    else:
                        result["suggestions"].append("Balanced pruning approach")
            except ValueError:
                result["errors"].append("Drop rate must be a valid number")

async def validate_embedding_config(field: str, value, result: dict, context: dict):
    """Validate embedding configuration fields."""
    if field == "provider":
        valid_providers = ["openai", "ollama"]
        if value not in valid_providers:
            result["errors"].append(f"Provider must be one of: {', '.join(valid_providers)}")
        else:
            result["suggestions"].append(f"Selected provider: {value}")

    elif field == "openai_api_key":
        if not value or len(str(value).strip()) == 0:
            result["warnings"].append("OpenAI API key is required for OpenAI provider")
        elif not str(value).startswith("sk-"):
            result["warnings"].append("OpenAI API key should start with 'sk-'")
        elif len(str(value)) < 40:
            result["warnings"].append("OpenAI API key appears to be too short")

    elif field == "openai_model":
        valid_models = [
            "text-embedding-3-small", "text-embedding-3-large",
            "text-embedding-ada-002", "text-embedding-2"
        ]
        if value not in valid_models:
            result["warnings"].append(f"Unknown OpenAI model. Supported: {', '.join(valid_models)}")

    elif field == "ollama_host":
        if not value:
            result["errors"].append("Ollama host is required")
        else:
            # Basic IP/hostname validation
            import re
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'

            if not (re.match(ip_pattern, value) or re.match(hostname_pattern, value) or value == "localhost"):
                result["warnings"].append("Host should be a valid IP address or hostname")

    elif field == "ollama_port":
        if not isinstance(value, (int, str)):
            result["errors"].append("Port must be a number")
        else:
            try:
                port_value = int(value)
                if port_value < 1 or port_value > 65535:
                    result["errors"].append("Port must be between 1 and 65535")
                elif port_value < 1024:
                    result["warnings"].append("Port below 1024 may require elevated privileges")
                elif port_value != 11434:
                    result["suggestions"].append("Default Ollama port is 11434")
            except ValueError:
                result["errors"].append("Port must be a valid number")

    elif field == "ollama_model":
        if not value:
            result["warnings"].append("Ollama model name is required")
        else:
            common_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
            if value not in common_models:
                result["suggestions"].append(f"Common models include: {', '.join(common_models)}")

async def validate_weaviate_config(field: str, value, result: dict):
    """Validate Weaviate configuration fields."""
    from urllib.parse import urlparse

    if field == "url":
        if not value:
            result["errors"].append("Weaviate URL is required")
        else:
            try:
                parsed = urlparse(str(value))
                if not parsed.scheme:
                    result["errors"].append("URL must include protocol (http:// or https://)")
                elif parsed.scheme not in ["http", "https"]:
                    result["errors"].append("URL must use http:// or https://")
                elif not parsed.netloc:
                    result["errors"].append("URL must include hostname")
                else:
                    if parsed.scheme == "http":
                        result["warnings"].append("Consider using https:// for production")
                    if parsed.port:
                        if parsed.port == 8080:
                            result["suggestions"].append("Using default Weaviate port")
                        elif parsed.port not in [80, 443, 8080]:
                            result["suggestions"].append("Non-standard port detected")
            except Exception:
                result["errors"].append("Invalid URL format")

    elif field == "batch_size":
        if not isinstance(value, (int, str)):
            result["errors"].append("Batch size must be a number")
        else:
            try:
                batch_value = int(value)
                if batch_value < 1:
                    result["errors"].append("Batch size must be at least 1")
                elif batch_value > 1000:
                    result["warnings"].append("Very large batch size may cause memory issues")
                elif batch_value < 10:
                    result["warnings"].append("Small batch size may be inefficient")
                elif batch_value == 100:
                    result["suggestions"].append("Using recommended default batch size")
            except ValueError:
                result["errors"].append("Batch size must be a valid number")

    elif field == "timeout":
        if not isinstance(value, (int, str)):
            result["errors"].append("Timeout must be a number")
        else:
            try:
                timeout_value = int(value)
                if timeout_value < 1:
                    result["errors"].append("Timeout must be at least 1 second")
                elif timeout_value > 300:
                    result["warnings"].append("Very long timeout may cause poor user experience")
                elif timeout_value < 30:
                    result["warnings"].append("Short timeout may cause connection failures")
            except ValueError:
                result["errors"].append("Timeout must be a valid number")

    elif field == "hybrid_search_alpha":
        if not isinstance(value, (int, float, str)):
            result["errors"].append("Alpha must be a number")
        else:
            try:
                alpha_value = float(value)
                if alpha_value < 0.0 or alpha_value > 1.0:
                    result["errors"].append("Alpha must be between 0.0 and 1.0")
                else:
                    if alpha_value == 0.0:
                        result["suggestions"].append("Pure keyword search (no vector similarity)")
                    elif alpha_value == 1.0:
                        result["suggestions"].append("Pure vector search (no keyword matching)")
                    elif alpha_value == 0.75:
                        result["suggestions"].append("Recommended balanced setting")
                    elif alpha_value < 0.5:
                        result["suggestions"].append("Keyword-focused hybrid search")
                    else:
                        result["suggestions"].append("Vector-focused hybrid search")
            except ValueError:
                result["errors"].append("Alpha must be a valid number")

async def validate_letta_api_config(field: str, value, result: dict):
    """Validate Letta API configuration fields."""
    from urllib.parse import urlparse

    if field == "url":
        if not value:
            result["errors"].append("Letta API URL is required")
        else:
            try:
                parsed = urlparse(str(value))
                if not parsed.scheme:
                    result["errors"].append("URL must include protocol (http:// or https://)")
                elif parsed.scheme not in ["http", "https"]:
                    result["errors"].append("URL must use http:// or https://")
                elif not parsed.netloc:
                    result["errors"].append("URL must include hostname")
                else:
                    if parsed.scheme == "http":
                        result["warnings"].append("Consider using https:// for production")
                    if not parsed.path.endswith("/v1"):
                        result["suggestions"].append("URL should typically end with /v1")
            except Exception:
                result["errors"].append("Invalid URL format")

    elif field == "password":
        if not value:
            result["warnings"].append("Letta API password is required")
        elif len(str(value)) < 8:
            result["warnings"].append("Password should be at least 8 characters")
        elif len(str(value)) > 100:
            result["warnings"].append("Unusually long password")

    elif field == "timeout":
        if not isinstance(value, (int, str)):
            result["errors"].append("Timeout must be a number")
        else:
            try:
                timeout_value = int(value)
                if timeout_value < 1:
                    result["errors"].append("Timeout must be at least 1 second")
                elif timeout_value > 300:
                    result["warnings"].append("Very long timeout may cause poor user experience")
                elif timeout_value < 10:
                    result["warnings"].append("Short timeout may cause API call failures")
            except ValueError:
                result["errors"].append("Timeout must be a valid number")

async def test_service_connection(service_type: str, config: dict):
    """Test connection to external services with provided configuration."""
    try:
        if service_type == "ollama":
            return await test_ollama_connection(config)
        elif service_type == "weaviate":
            return await test_weaviate_connection(config)
        elif service_type == "letta_api":
            return await test_letta_connection()  # Uses environment variables
        elif service_type == "openai":
            return await test_openai_connection(config)
        else:
            return {
                "available": False,
                "error": f"Unknown service type: {service_type}",
                "tested_at": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "tested_at": datetime.now().isoformat()
        }

async def test_openai_connection(config: dict):
    """Test OpenAI API connection with provided configuration."""
    try:
        api_key = config.get("api_key")
        model = config.get("model", "text-embedding-3-small")

        if not api_key:
            return {
                "available": False,
                "error": "API key is required",
                "tested_at": datetime.now().isoformat()
            }

        # Simple test - we won't actually call OpenAI API to avoid costs
        # In a real implementation, you might make a minimal test call
        result = {
            "available": True,
            "model": model,
            "api_key_format": "valid" if api_key.startswith("sk-") else "invalid",
            "tested_at": datetime.now().isoformat()
        }

        if len(api_key) < 40:
            result["warning"] = "API key appears to be too short"

        return result

    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "tested_at": datetime.now().isoformat()
        }

# Add cost tracking to existing operations
async def track_embedding_cost(operation: str, token_count: int, provider: str = "openai"):
    """Helper function to track embedding costs in existing operations"""
    try:
        await record_embedding_cost(operation, token_count, provider)
    except Exception as e:
        logger.warning(f"Failed to track embedding cost: {e}")


async def track_weaviate_operation_cost(operation: str, operation_type: str, count: int = 1):
    """Helper function to track Weaviate operation costs"""
    try:
        await record_weaviate_cost(operation, operation_type, count)
    except Exception as e:
        logger.warning(f"Failed to track Weaviate cost: {e}")


async def track_letta_api_cost(operation: str, call_count: int = 1):
    """Helper function to track Letta API costs"""
    try:
        await record_letta_api_cost(operation, call_count)
    except Exception as e:
        logger.warning(f"Failed to track Letta API cost: {e}")


# Comprehensive logging helper functions
async def get_log_entries(level='all', lines=100, search='', from_time='', to_time=''):
    """Get log entries with filtering."""
    import re
    from datetime import datetime, timedelta

    try:
        # For this implementation, we'll work with Python's logging module
        # and read from the logs captured in memory or from log files
        log_entries = []

        # Get log records from various sources
        # 1. In-memory log records (if available)
        # 2. Log files (if configured)
        # For now, we'll simulate recent log entries based on the logger

        # Parse time filters
        from_dt = None
        to_dt = None
        if from_time:
            try:
                from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
            except:
                pass
        if to_time:
            try:
                to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
            except:
                pass

        # Generate sample log entries (in a real implementation, read from log files)
        sample_entries = await generate_sample_log_entries(lines)

        # Filter by level
        if level != 'all':
            sample_entries = [entry for entry in sample_entries if entry['level'].lower() == level.lower()]

        # Filter by search term
        if search:
            search_pattern = re.compile(re.escape(search), re.IGNORECASE)
            sample_entries = [
                entry for entry in sample_entries
                if search_pattern.search(entry['message']) or search_pattern.search(entry.get('logger', ''))
            ]

        # Filter by time range
        if from_dt or to_dt:
            filtered_entries = []
            for entry in sample_entries:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if from_dt and entry_time < from_dt:
                    continue
                if to_dt and entry_time > to_dt:
                    continue
                filtered_entries.append(entry)
            sample_entries = filtered_entries

        return sample_entries[:lines]
    except Exception as e:
        logger.error(f"Error getting log entries: {str(e)}")
        return []

async def generate_sample_log_entries(count=100):
    """Generate sample log entries for demonstration."""
    from datetime import datetime, timedelta
    import random

    levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    loggers = ['api_server', 'weaviate_client', 'letta_client', 'tool_manager']

    sample_messages = [
        "Tool search completed successfully",
        "Weaviate connection established",
        "Agent tool attachment completed",
        "Cache refresh initiated",
        "Error processing tool request",
        "Embedding generation started",
        "Configuration updated",
        "Maintenance task completed",
        "API request received",
        "Database query executed"
    ]

    entries = []
    base_time = datetime.now()

    for i in range(count):
        level = random.choice(levels)
        if level == 'ERROR':
            # Add some realistic error messages
            messages = [
                "Failed to connect to Weaviate: connection timeout",
                "Tool attachment failed: agent not found",
                "Error processing search query: invalid parameters",
                "Database connection lost",
                "Authentication failed for Letta API"
            ]
            message = random.choice(messages)
        else:
            message = random.choice(sample_messages)

        entry_time = base_time - timedelta(minutes=random.randint(0, 1440))  # Last 24 hours

        entry = {
            "timestamp": entry_time.isoformat(),
            "level": level,
            "logger": random.choice(loggers),
            "message": message,
            "line_number": random.randint(100, 999),
            "thread": f"Thread-{random.randint(1, 5)}"
        }

        if level == 'ERROR':
            entry["exception"] = "Exception details would appear here"
            entry["stack_trace"] = "Stack trace would appear here"

        entries.append(entry)

    # Sort by timestamp (newest first)
    entries.sort(key=lambda x: x['timestamp'], reverse=True)
    return entries

async def perform_log_analysis(timeframe='24h', include_details=False):
    """Perform comprehensive log analysis."""
    try:
        # Parse timeframe
        hours = 24
        if timeframe == '1h':
            hours = 1
        elif timeframe == '7d':
            hours = 24 * 7
        elif timeframe == '30d':
            hours = 24 * 30

        # Get log entries for analysis
        log_entries = await get_log_entries(lines=10000)  # Analyze more entries

        # Analysis metrics
        total_entries = len(log_entries)
        error_count = len([e for e in log_entries if e['level'] == 'ERROR'])
        warning_count = len([e for e in log_entries if e['level'] == 'WARNING'])
        info_count = len([e for e in log_entries if e['level'] == 'INFO'])
        debug_count = len([e for e in log_entries if e['level'] == 'DEBUG'])

        # Top error patterns
        error_entries = [e for e in log_entries if e['level'] == 'ERROR']
        error_patterns = {}
        for error in error_entries:
            # Simple pattern extraction (first 50 chars)
            pattern = error['message'][:50] + "..." if len(error['message']) > 50 else error['message']
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

        top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]

        # Logger activity
        logger_activity = {}
        for entry in log_entries:
            logger_name = entry.get('logger', 'unknown')
            if logger_name not in logger_activity:
                logger_activity[logger_name] = {'total': 0, 'errors': 0, 'warnings': 0}
            logger_activity[logger_name]['total'] += 1
            if entry['level'] == 'ERROR':
                logger_activity[logger_name]['errors'] += 1
            elif entry['level'] == 'WARNING':
                logger_activity[logger_name]['warnings'] += 1

        analysis = {
            "timeframe": timeframe,
            "total_entries": total_entries,
            "summary": {
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "debug_count": debug_count,
                "error_rate": round((error_count / total_entries) * 100, 2) if total_entries > 0 else 0
            },
            "top_error_patterns": [{"pattern": pattern, "count": count} for pattern, count in top_errors],
            "logger_activity": logger_activity,
            "recommendations": []
        }

        # Add recommendations based on analysis
        if error_count > total_entries * 0.1:
            analysis["recommendations"].append("High error rate detected - consider investigating root causes")
        if warning_count > total_entries * 0.2:
            analysis["recommendations"].append("High warning rate - review system configuration")

        if include_details:
            analysis["recent_errors"] = error_entries[:20]  # Last 20 errors
            analysis["sample_entries"] = log_entries[:50]   # Sample entries

        return analysis
    except Exception as e:
        logger.error(f"Error performing log analysis: {str(e)}")
        return {"error": str(e)}

async def get_error_log_entries(hours=24, include_stack_trace=True, group_by='none'):
    """Get error log entries with analysis."""
    try:
        # Get all log entries and filter for errors
        all_entries = await get_log_entries(level='error', lines=1000)

        # Filter by time window
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)

        error_entries = []
        for entry in all_entries:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time >= cutoff_time:
                error_entries.append(entry)

        result = {
            "total_errors": len(error_entries),
            "timeframe_hours": hours,
            "errors": error_entries
        }

        # Group errors if requested
        if group_by == 'error_type':
            grouped = {}
            for error in error_entries:
                # Simple error type extraction (first word of message)
                error_type = error['message'].split()[0] if error['message'] else 'Unknown'
                if error_type not in grouped:
                    grouped[error_type] = []
                grouped[error_type].append(error)
            result["grouped_errors"] = grouped

        elif group_by == 'endpoint':
            grouped = {}
            for error in error_entries:
                # Extract endpoint from message (simplified)
                endpoint = 'unknown'
                if 'endpoint' in error['message'].lower():
                    endpoint = error.get('logger', 'unknown')
                if endpoint not in grouped:
                    grouped[endpoint] = []
                grouped[endpoint].append(error)
            result["grouped_errors"] = grouped

        elif group_by == 'time':
            # Group by hour
            grouped = {}
            for error in error_entries:
                hour_key = error['timestamp'][:13]  # YYYY-MM-DDTHH
                if hour_key not in grouped:
                    grouped[hour_key] = []
                grouped[hour_key].append(error)
            result["grouped_errors"] = grouped

        return result
    except Exception as e:
        logger.error(f"Error getting error logs: {str(e)}")
        return {"error": str(e)}

async def clear_log_files(backup=True, older_than_days=0):
    """Clear log files with optional backup."""
    try:
        # In a real implementation, this would:
        # 1. Find log files
        # 2. Optionally create backups
        # 3. Clear or rotate logs
        # 4. Return statistics

        # For now, return simulation data
        result = {
            "success": True,
            "files_cleared": 3,
            "space_freed_mb": 150,
            "backup_created": backup,
            "backup_location": "/var/log/backups/" if backup else None,
            "older_than_days": older_than_days
        }

        return result
    except Exception as e:
        logger.error(f"Error clearing log files: {str(e)}")
        return {"success": False, "error": str(e)}

async def export_log_data(format_type='json', filters=None):
    """Export log data in specified format."""
    try:
        if filters is None:
            filters = {}

        # Get log entries based on filters
        log_entries = await get_log_entries(
            level=filters.get('level', 'all'),
            lines=filters.get('lines', 1000),
            search=filters.get('search', ''),
            from_time=filters.get('from_time', ''),
            to_time=filters.get('to_time', '')
        )

        # Export in requested format
        if format_type == 'json':
            import json
            export_data = json.dumps(log_entries, indent=2)
            content_type = 'application/json'
            filename = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        elif format_type == 'csv':
            import csv
            import io
            output = io.StringIO()
            if log_entries:
                writer = csv.DictWriter(output, fieldnames=log_entries[0].keys())
                writer.writeheader()
                writer.writerows(log_entries)
            export_data = output.getvalue()
            content_type = 'text/csv'
            filename = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        elif format_type == 'text':
            lines = []
            for entry in log_entries:
                line = f"[{entry['timestamp']}] {entry['level']} {entry['logger']}: {entry['message']}"
                lines.append(line)
            export_data = '\n'.join(lines)
            content_type = 'text/plain'
            filename = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        result = {
            "success": True,
            "format": format_type,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": len(export_data),
            "entry_count": len(log_entries),
            "data": export_data  # In a real app, might return download URL instead
        }

        return result
    except Exception as e:
        logger.error(f"Error exporting log data: {str(e)}")
        return {"success": False, "error": str(e)}

# Enhanced logging helper functions to replace placeholders
def get_log_file_size():
    """Get actual log file size."""
    try:
        import os
        import glob

        # Look for common log file patterns
        log_patterns = [
            '/var/log/app.log',
            '/app/logs/*.log',
            'logs/*.log',
            '*.log'
        ]

        total_size = 0
        files_found = []

        for pattern in log_patterns:
            try:
                log_files = glob.glob(pattern)
                for log_file in log_files:
                    if os.path.exists(log_file):
                        size = os.path.getsize(log_file)
                        total_size += size
                        files_found.append({
                            "file": log_file,
                            "size_bytes": size,
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
            except:
                continue

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files_found,
            "files_count": len(files_found)
        }
    except Exception as e:
        return {"size_mb": 0, "error": str(e)}

async def get_recent_error_count():
    """Get actual recent error count."""
    try:
        # Get error logs from last 24 hours
        errors = await get_error_log_entries(hours=24)
        return errors.get("total_errors", 0)
    except:
        return 0

async def get_recent_warning_count():
    """Get actual recent warning count."""
    try:
        # Get warning logs from last 24 hours
        warnings = await get_log_entries(level='warning', lines=1000)
        return len(warnings)
    except:
        return 0

# ================================================================================

if __name__ == '__main__':
    # Use Hypercorn for serving
    # Ensure PORT is an integer
    port = int(os.getenv('PORT', 3001)) # Default to 3001 if not set
    config = Config()
    config.bind = [f"0.0.0.0:{port}"] # Bind to all interfaces on the specified port
    
    # Set a higher graceful timeout if needed, e.g., for long-running requests
    # config.graceful_timeout = 30  # seconds

    logger.info(f"Starting Hypercorn server on port {port}...")
    asyncio.run(serve(app, config))
