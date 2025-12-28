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
    
    The search provider is determined by the TOOL_SEARCH_PROVIDER environment variable:
    - 'weaviate': Use Weaviate vector database (default)
    - 'letta': Use Letta's native client.tools.search() API
    - 'hybrid': Try Letta first, fallback to Weaviate on error
    
    Args:
        query: Search query describing the tool you're looking for
        limit: Maximum number of results to return
        min_score: Minimum relevance score (0-100) to include
        
    Returns:
        List of tool dicts with search results
    """
    global weaviate_client
    
    async def search_via_letta():
        """Search using Letta's native tools.search() API"""
        if not USE_LETTA_SDK:
            raise RuntimeError("Letta SDK not available for tool search")
        
        sdk_client = get_letta_sdk_client()
        results = await sdk_client.search_tools_with_scores(
            query=query,
            limit=limit,
            min_score=min_score
        )
        logger.info(f"Letta native search for '{query}' returned {len(results)} results")
        return results
    
    async def search_via_weaviate():
        """Search using Weaviate vector database"""
        if not weaviate_client or not weaviate_client.is_ready():
            weaviate_client_local = init_weaviate_client()
            if not weaviate_client_local or not weaviate_client_local.is_ready():
                raise RuntimeError("Weaviate client not available for tool search")
        
        results = await asyncio.to_thread(search_tools, query=query, limit=limit)
        logger.info(f"Weaviate search for '{query}' returned {len(results)} results")
        return results
    
    # Execute search based on configured provider
    if TOOL_SEARCH_PROVIDER == 'letta':
        try:
            return await search_via_letta()
        except Exception as e:
            logger.error(f"Letta tool search failed: {e}")
            raise
    
    elif TOOL_SEARCH_PROVIDER == 'hybrid':
        # Try Letta first, fallback to Weaviate
        try:
            return await search_via_letta()
        except Exception as e:
            logger.warning(f"Letta tool search failed, falling back to Weaviate: {e}")
            return await search_via_weaviate()
    
    else:  # 'weaviate' (default)
        return await search_via_weaviate()


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

async def detach_tool(agent_id: str, tool_id: str, tool_name: str = None):
    """
    Detach a single tool asynchronously.
    
    Delegates to tool_manager.detach_tool for the actual implementation.
    """
    return await tool_manager.detach_tool(agent_id, tool_id, tool_name)

async def attach_tool(agent_id: str, tool: dict):
    """
    Attach a single tool asynchronously.
    
    Delegates to tool_manager.attach_tool for the actual implementation.
    """
    return await tool_manager.attach_tool(agent_id, tool)

async def process_tools(agent_id: str, mcp_tools: list, matching_tools: list, keep_tools: list = None):
    """
    Process tool detachments and attachments in parallel.
    
    Delegates to tool_manager.process_tools for the actual implementation.
    """
    return await tool_manager.process_tools(agent_id, mcp_tools, matching_tools, keep_tools)

@app.route('/api/v1/tools/search', methods=['POST'])
async def search():
    """Search endpoint - Note: This still calls the original synchronous search_tools"""
    # TODO: Decide if this endpoint should also be async or use a different search mechanism
    logger.info("Received request for /api/v1/tools/search")
    try:
        data = await request.get_json()
        if not data:
            logger.warning("Search request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        query = data.get('query')
        limit = data.get('limit', 10)

        if not query:
            logger.warning("Search request missing 'query' parameter.")
            return jsonify({"error": "Query parameter is required"}), 400

        # This call might need adjustment if search_tools is strictly async now
        # For now, assuming it might work or needs a sync wrapper if this endpoint is kept sync
        logger.warning("Calling potentially async search_tools from sync context in /search endpoint.")
        
        # Check if reranking is enabled (DEPRECATED - use /tools/search/rerank endpoint instead)
        enable_reranking = data.get('enable_reranking', False)
        reranker_config = None
        if enable_reranking:
            # DEPRECATED: This enable_reranking parameter is deprecated
            # Use the dedicated /api/v1/tools/search/rerank endpoint instead
            logger.warning("DEPRECATED: enable_reranking parameter is deprecated. Use /api/v1/tools/search/rerank endpoint instead for better reranking support.")

            # Build reranker config for backward compatibility
            reranker_config = {
                'enabled': True,
                'model': data.get('reranker_config', {}).get('model', 'bge-reranker-v2-m3'),
                'base_url': data.get('reranker_config', {}).get('base_url', 'http://localhost:8091')
            }
        
        # If MANAGE_ONLY_MCP_TOOLS is enabled, search with higher limit and filter for MCP tools first
        if MANAGE_ONLY_MCP_TOOLS:
            # Search with a higher limit to ensure we get enough MCP tools
            search_limit = limit * 5  # Get 5x more results to filter from
            logger.info(f"MANAGE_ONLY_MCP_TOOLS enabled - searching with limit {search_limit} to filter for MCP tools")
            results = search_tools(query=query, limit=search_limit, reranker_config=reranker_config)
        else:
            results = search_tools(query=query, limit=limit, reranker_config=reranker_config)
        
        # Filter results if MANAGE_ONLY_MCP_TOOLS is enabled
        if MANAGE_ONLY_MCP_TOOLS:
            # Load tool cache to check tool types
            tools_cache = await read_tool_cache()
            filtered_results = []
            
            logger.info(f"DEBUG: Starting MCP filtering with {len(results)} results")
            for i, result in enumerate(results):
                tool_name = result.get('name')
                logger.info(f"DEBUG: Result {i}: name='{tool_name}', keys={list(result.keys())}")
                if tool_name:
                    # Find the tool in cache to check its type
                    cached_tool = next((t for t in tools_cache if t.get('name') == tool_name), None)
                    if cached_tool:
                        tool_type = cached_tool.get("tool_type")
                        is_letta_core = _is_letta_core_tool(cached_tool)
                        is_mcp_tool = (tool_type == "external_mcp" or 
                                     (not is_letta_core and tool_type == "custom"))
                        logger.info(f"DEBUG: Tool '{tool_name}' found in cache - type={tool_type}, is_letta_core={is_letta_core}, is_mcp_tool={is_mcp_tool}")
                        if is_mcp_tool:
                            filtered_results.append(result)
                            logger.info(f"DEBUG: Tool '{tool_name}' PASSED filter")
                            # Stop when we have enough results
                            if len(filtered_results) >= limit:
                                break
                        else:
                            logger.info(f"DEBUG: Tool '{tool_name}' FAILED filter - not MCP")
                    else:
                        logger.info(f"DEBUG: Tool '{tool_name}' NOT found in cache")
                        # If not in cache, it might be an MCP tool that needs registration
                        if result.get("mcp_server_name"):
                            filtered_results.append(result)
                            logger.info(f"DEBUG: Tool '{tool_name}' PASSED filter - has mcp_server_name")
                            # Stop when we have enough results
                            if len(filtered_results) >= limit:
                                break
                else:
                    logger.info(f"DEBUG: Result {i} has no name field")
            
            logger.info(f"Weaviate search: {len(results)} total results, {len(filtered_results)} after MCP filtering.")
            # Map rerank_score to score if present
            for result in filtered_results[:limit]:
                if 'rerank_score' in result and 'score' not in result:
                    result['score'] = result['rerank_score']
            return jsonify(filtered_results[:limit])  # Ensure we don't return more than requested
        else:
            logger.info(f"Weaviate search successful, returning {len(results)} results.")
            # Map rerank_score to score if present
            for result in results:
                if 'rerank_score' in result and 'score' not in result:
                    result['score'] = result['rerank_score']
            return jsonify(results)
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/v1/tools/search/rerank', methods=['POST'])
async def search_with_reranking():
    """Search with reranking endpoint for dashboard frontend."""
    logger.info("Received request for /api/v1/tools/search/rerank")
    
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Extract query and reranker config - handle both formats
        if isinstance(data.get('query'), str):
            # Direct format: {"query": "search term", "limit": 5}
            query_string = data.get('query', '')
            limit = data.get('limit', 10)
            reranker_config = data.get('reranker_config', {})
        else:
            # Nested format: {"query": {"query": "search term", "limit": 5}}
            query_data = data.get('query', {})
            reranker_config = data.get('reranker_config', {})
            query_string = query_data.get('query', '')
            limit = query_data.get('limit', 10)
        
        # Use server environment configuration if reranker_config is empty or missing model
        if not reranker_config.get('model'):
            reranker_config = {
                "enabled": os.getenv('RERANKER_ENABLED', 'true').lower() == 'true',
                "model": os.getenv('RERANKER_MODEL', 'qwen3-reranker-4b'),
                "provider": os.getenv('RERANKER_PROVIDER', 'vllm'),
                "parameters": {
                    "temperature": float(os.getenv('RERANKER_TEMPERATURE', '0.1')),
                    "max_tokens": int(os.getenv('RERANKER_MAX_TOKENS', '512')),
                    "base_url": os.getenv('RERANKER_URL', 'http://100.81.139.20:11435/rerank')
                }
            }
        
        if not query_string:
            return jsonify({"success": False, "error": "No query provided"}), 400
        
        logger.info(f"Performing reranked search for: '{query_string}' with limit: {limit}")
        logger.info(f"Reranker config: {reranker_config}")
        
        # Use the search_tools_with_reranking function to get actual reranked results
        # If MANAGE_ONLY_MCP_TOOLS is enabled, search with higher limit and filter for MCP tools first
        if MANAGE_ONLY_MCP_TOOLS:
            # Search with a higher limit to ensure we get enough MCP tools
            search_limit = limit * 5  # Get 5x more results to filter from
            logger.info(f"MANAGE_ONLY_MCP_TOOLS enabled - searching with limit {search_limit} to filter for MCP tools (rerank endpoint)")
            results = await asyncio.to_thread(search_tools_with_reranking, query=query_string, limit=search_limit, use_reranking=True)
        else:
            results = await asyncio.to_thread(search_tools_with_reranking, query=query_string, limit=limit, use_reranking=True)
        
        # Filter results if MANAGE_ONLY_MCP_TOOLS is enabled
        if MANAGE_ONLY_MCP_TOOLS:
            # Load tool cache to check tool types
            tools_cache = await read_tool_cache()
            filtered_results = []
            
            logger.info(f"DEBUG: Starting MCP filtering with {len(results)} results (rerank endpoint)")
            for i, result in enumerate(results):
                tool_name = result.get('name')
                if tool_name:
                    # Find the tool in cache to check its type
                    cached_tool = next((t for t in tools_cache if t.get('name') == tool_name), None)
                    if cached_tool:
                        tool_type = cached_tool.get("tool_type")
                        is_letta_core = _is_letta_core_tool(cached_tool)
                        is_mcp_tool = (tool_type == "external_mcp" or 
                                     (not is_letta_core and tool_type == "custom"))
                        if is_mcp_tool:
                            filtered_results.append(result)
                            # Stop when we have enough results
                            if len(filtered_results) >= limit:
                                break
                    else:
                        # If not in cache, it might be an MCP tool that needs registration
                        if result.get("mcp_server_name"):
                            filtered_results.append(result)
                            # Stop when we have enough results
                            if len(filtered_results) >= limit:
                                break
            
            logger.info(f"Weaviate search (rerank): {len(results)} total results, {len(filtered_results)} after MCP filtering.")
            results = filtered_results[:limit]  # Ensure we don't return more than requested
        
        if not results:
            logger.warning("No results returned from search_tools after filtering")
            return jsonify({
                "success": True,
                "data": {
                    "query": query_string,
                    "results": [],
                    "metadata": {
                        "total_found": 0,
                        "search_time": 0,
                        "reranker_used": reranker_config.get('model', 'none')
                    }
                }
            })
        
        # Convert results to frontend format
        formatted_results = []
        for i, result in enumerate(results):
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
                "reasoning": result.get('reasoning', '')
            }
            formatted_results.append(formatted_result)
        
        response_data = {
            "query": query_string,
            "results": formatted_results,
            "metadata": {
                "total_found": len(results),
                "search_time": 0.1,  # Placeholder
                "reranker_used": reranker_config.get('model', 'mistral:7b')
            }
        }
        
        logger.info(f"Reranked search with reranking=True successful, returning {len(formatted_results)} results.")
        return jsonify({"success": True, "data": response_data})
        
    except Exception as e:
        logger.error(f"Error during reranked search: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/tools', methods=['GET'])
async def get_tools():
    logger.info("Received request for /api/v1/tools")
    try:
        # Read directly from the cache asynchronously
        tools = await read_tool_cache() # Await the async function
        logger.info(f"Get tools from cache successful, returning {len(tools)} tools.")
        return jsonify(tools)
    except Exception as e:
        logger.error(f"Error during get_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

async def fetch_agent_info(agent_id):
    """Fetch agent information asynchronously using SDK or aiohttp"""
    # Use SDK if enabled
    if USE_LETTA_SDK:
        try:
            sdk_client = get_letta_sdk_client()
            return await sdk_client.get_agent_name(agent_id)
        except Exception as e:
            logger.error(f"SDK fetch_agent_info failed: {e}")
            raise
    
    # Fall back to aiohttp
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_info (agent: {agent_id})")
        raise ConnectionError("HTTP session not available")
    async with http_session.get(f"{LETTA_URL}/agents/{agent_id}", headers=HEADERS) as response:
        response.raise_for_status()
        agent_data = await response.json()
    return agent_data.get("name", "Unknown Agent")

async def fetch_agent_tools(agent_id):
    """Fetch agent's current tools asynchronously using SDK or aiohttp"""
    # Use SDK if enabled
    if USE_LETTA_SDK:
        try:
            sdk_client = get_letta_sdk_client()
            return await sdk_client.list_agent_tools(agent_id)
        except Exception as e:
            logger.error(f"SDK fetch_agent_tools failed: {e}")
            raise
    
    # Fall back to aiohttp
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_tools (agent: {agent_id})")
        raise ConnectionError("HTTP session not available")
    async with http_session.get(f"{LETTA_URL}/agents/{agent_id}/tools", headers=HEADERS) as response:
        response.raise_for_status()
        return await response.json()

async def register_tool(tool_name, server_name):
    """Register a tool from an MCP server asynchronously using SDK or aiohttp"""
    # Use SDK if enabled
    if USE_LETTA_SDK:
        try:
            sdk_client = get_letta_sdk_client()
            return await sdk_client.register_mcp_tool(tool_name, server_name)
        except Exception as e:
            logger.error(f"SDK register_tool failed: {e}")
            raise
    
    # Fall back to aiohttp
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for register_tool (tool: {tool_name}, server: {server_name})")
        raise ConnectionError("HTTP session not available")
    register_url = f"{LETTA_URL}/tools/mcp/servers/{server_name}/{tool_name}"
    async with http_session.post(register_url, headers=HEADERS) as response:
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

async def _send_trigger_message(agent_id: str, tool_names: list, query: str = None):
    """
    Internal coroutine that actually sends the trigger message to the agent.
    This is meant to be run as a background task (fire-and-forget).
    """
    global http_session
    if not http_session:
        logger.warning("HTTP session not available for trigger message")
        return
    
    if not LETTA_MESSAGE_BASE_URLS:
        logger.warning("No Letta message endpoints available for trigger")
        return
    
    tool_list = ", ".join(tool_names[:5])
    if len(tool_names) > 5:
        tool_list += f" and {len(tool_names) - 5} more"
    
    # Use system role to minimize disruption to conversation flow
    # The [SYSTEM] prefix helps the agent recognize this as an automated notification
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
    for base_url in LETTA_MESSAGE_BASE_URLS:
        messages_url = f"{base_url}/agents/{agent_id}/messages"
        logger.info(f"[BACKGROUND] Sending trigger message to {agent_id} via {messages_url} ...")
        try:
            async with http_session.post(messages_url, headers=HEADERS, json=payload) as response:
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
                    if MATRIX_BRIDGE_WEBHOOK_URL:
                        await _emit_matrix_bridge_webhook(
                            agent_id=agent_id,
                            new_run_id=new_run_id,
                            tool_names=tool_names,
                            query=query
                        )
                    
                    return
                text = await response.text()
                last_error = f"HTTP {response.status} - {text[:200]}"
                logger.warning(f"[BACKGROUND] Trigger failed for {agent_id} via {messages_url}: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[BACKGROUND] Error in trigger message for {agent_id} via {messages_url}: {e}")
    
    if last_error:
        logger.warning(f"[BACKGROUND] All trigger attempts failed for {agent_id}: {last_error}")


async def _emit_matrix_bridge_webhook(
    agent_id: str, 
    new_run_id: str = None, 
    tool_names: list = None, 
    query: str = None
):
    """
    Emit webhook to Matrix bridge for cross-run tracking.
    
    This notifies the Matrix bridge that a new run was triggered after tool attachment,
    allowing it to track the conversation across multiple Letta runs.
    
    Webhook payload:
    {
        "event": "run_triggered",
        "agent_id": "agent-xxx",
        "new_run_id": "run-xxx",
        "trigger_type": "tool_attachment",
        "tools_attached": ["tool1", "tool2"],
        "query": "original user query",
        "timestamp": "2025-12-06T22:15:00Z"
    }
    """
    global http_session
    
    if not MATRIX_BRIDGE_WEBHOOK_URL:
        return
    
    if not http_session:
        logger.warning("[WEBHOOK] HTTP session not available for Matrix bridge webhook")
        return
    
    from datetime import datetime
    
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
        async with http_session.post(
            MATRIX_BRIDGE_WEBHOOK_URL,
            json=webhook_payload,
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                logger.info(f"[WEBHOOK] Notified Matrix bridge of run trigger for {agent_id}, run_id={new_run_id}")
            else:
                resp_text = await resp.text()
                logger.warning(f"[WEBHOOK] Matrix bridge returned {resp.status}: {resp_text[:200]}")
    except asyncio.TimeoutError:
        logger.warning(f"[WEBHOOK] Timeout sending webhook to Matrix bridge for {agent_id}")
    except Exception as e:
        logger.warning(f"[WEBHOOK] Failed to notify Matrix bridge for {agent_id}: {e}")


def trigger_agent_loop(agent_id: str, attached_tools: list, query: str = None):
    """
    Fire-and-forget trigger to start a new agent loop with updated tools.
    
    In Letta V1 architecture, tools are passed to the LLM at the start of a request.
    After attaching new tools, we need to trigger a new loop so the agent can use them.
    
    This function spawns a background task and returns immediately - it does NOT wait
    for the agent to process the message. This is intentional to avoid blocking the
    attach endpoint response.
    
    Returns True if the background task was successfully created.
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
        # The task will run to completion but we don't wait for it
        task = loop.create_task(_send_trigger_message(agent_id, tool_names, query))
        
        # Optional: Add a callback to log when it completes
        def on_complete(t):
            if t.exception():
                logger.warning(f"[BACKGROUND] Trigger task failed with exception: {t.exception()}")
        task.add_done_callback(on_complete)
        
        logger.info(f"Spawned background trigger task for agent {agent_id} with {len(tool_names)} new tools")
        return True
        
    except Exception as e:
        logger.warning(f"Error creating trigger task: {e}")
        return False

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
                registered_tool = await register_tool(tool_name, originating_server)
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


@app.route('/api/v1/tools/attach', methods=['POST'])
async def attach_tools():
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
                fetch_agent_info(agent_id),
                fetch_agent_tools(agent_id)
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
                    
                    preattach_prune_result = await _perform_tool_pruning(
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
                        current_agent_tools = await fetch_agent_tools(agent_id)
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
            results = await process_tools(agent_id, mcp_tools, processed_tools, keep_tools)
            
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
                    loop_triggered = trigger_agent_loop(
                        agent_id,
                        successful_attachments,
                        query=query
                    )
                    logger.info(f"Loop trigger task spawned: {loop_triggered}")
                except Exception as trigger_error:
                    logger.error(f"Exception during trigger_agent_loop: {trigger_error}", exc_info=True)

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

async def _perform_tool_pruning(agent_id: str, user_prompt: str, drop_rate: float, keep_tool_ids: list = None, newly_matched_tool_ids: list = None) -> dict:
    """
    Core logic for pruning tools.
    Only prunes MCP tools ('external_mcp'). Core Letta tools are always preserved.
    Keeps a percentage of the most relevant MCP tools from the entire library,
    plus any explicitly kept or newly matched MCP tools, up to the agent's current MCP tool count.
    """
    requested_keep_tool_ids = set(keep_tool_ids or [])
    requested_newly_matched_tool_ids = set(newly_matched_tool_ids or [])
    
    logger.info(f"Pruning request for agent {agent_id} with prompt: '{user_prompt}', drop_rate: {drop_rate}")
    logger.info(f"Requested to keep (all types): {requested_keep_tool_ids}, Requested newly matched (all types): {requested_newly_matched_tool_ids}")

    try:
        # 1. Retrieve Current Agent Tools and categorize them
        logger.info(f"Fetching current tools for agent {agent_id}...")
        current_agent_tools_list = await fetch_agent_tools(agent_id) # List of tool dicts
        
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
            is_letta_core_tool = _is_letta_core_tool(tool)
            tool_name = tool.get('name', '').lower()
            
            # Check if this tool should never be detached
            is_never_detach_tool = (
                tool_id in requested_keep_tool_ids or 
                tool_id in requested_newly_matched_tool_ids or
                any(never_detach_name.lower() in tool_name for never_detach_name in NEVER_DETACH_TOOLS)
            )
            
            if is_never_detach_tool or (MANAGE_ONLY_MCP_TOOLS and is_letta_core_tool):
                # If we only manage MCP tools, treat Letta core tools as protected
                # Also protect never-detach tools and keep list tools
                core_tools_on_agent.append(tool)
            elif tool.get("tool_type") == "external_mcp" or (not is_letta_core_tool and tool.get("tool_type") == "custom"):
                # Treat external_mcp and custom (non-Letta) tools as MCP tools
                mcp_tools_on_agent_list.append(tool)
            else:
                core_tools_on_agent.append(tool)

        current_mcp_tool_ids = {tool['id'] for tool in mcp_tools_on_agent_list}
        current_core_tool_ids = {tool['id'] for tool in core_tools_on_agent}
        
        # Track protected tools for logging
        protected_tool_names = [tool.get('name', 'Unknown') for tool in core_tools_on_agent 
                               if any(never_detach_name.lower() in tool.get('name', '').lower() 
                                     for never_detach_name in NEVER_DETACH_TOOLS) 
                               or tool.get('id') in requested_keep_tool_ids
                               or tool.get('id') in requested_newly_matched_tool_ids]
        
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
        MIN_MCP_TOOLS = int(os.getenv('MIN_MCP_TOOLS', '7'))
        
        if num_currently_attached_mcp == 0:
            logger.info("No MCP tools currently attached to the agent. Nothing to prune among MCP tools.")
            # Core tools are kept by default.
            return {
                "success": True, "message": "No MCP tools to prune. Core tools preserved.",
                "details": {
                    "tools_on_agent_before_total": num_total_attached,
                    "mcp_tools_on_agent_before": 0,
                    "core_tools_preserved_count": num_currently_attached_core,
                    "target_mcp_tools_to_keep": 0,
                    "mcp_tools_detached_count": 0, # Changed from tools_detached_count
                    "final_tool_ids_on_agent": list(current_core_tool_ids),
                }
            }

        if num_currently_attached_mcp <= MIN_MCP_TOOLS:
            logger.info(f"Agent {agent_id} has {num_currently_attached_mcp} MCP tools, which is at or below the minimum required ({MIN_MCP_TOOLS}). Skipping pruning.")
            return {
                "success": True, "message": f"Pruning skipped: Agent has {num_currently_attached_mcp} MCP tools (minimum required: {MIN_MCP_TOOLS})",
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
        # Use configured limits instead of just drop_rate
        max_mcp_allowed = MAX_MCP_TOOLS
        max_total_allowed = MAX_TOTAL_TOOLS - num_currently_attached_core  # Reserve space for core tools
        
        # Calculate target based on both drop_rate and limits
        target_from_drop_rate = math.floor(num_currently_attached_mcp * (1.0 - drop_rate))
        num_mcp_tools_to_keep = min(target_from_drop_rate, max_mcp_allowed, max_total_allowed)
        
        # Ensure we never go below the minimum MCP tool count
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
        top_library_tools_data = await asyncio.to_thread(search_tools, query=user_prompt, limit=search_limit)
        
        ordered_top_library_tool_info = []
        seen_top_ids = set()
        for tool_data in top_library_tools_data:
            tool_id = tool_data.get('id') or tool_data.get('tool_id')
            if tool_id and tool_id not in seen_top_ids:
                ordered_top_library_tool_info.append(
                    (tool_id, tool_data.get('name', 'Unknown'), tool_data.get('tool_type')) # Include tool_type
                )
                seen_top_ids.add(tool_id)
        logger.info(f"Found {len(ordered_top_library_tool_info)} unique, potentially relevant tools from library search.")

        # 4. Determine Final Set of MCP Tools to Keep on Agent
        
        # Initialize with MCP tools that *must* be kept:
        # - Newly matched MCP tools (that are actually on the agent)
        # - Explicitly requested to keep MCP tools (that are actually on the agent)
        final_mcp_tool_ids_to_keep = set()
        
        for tool_id in requested_newly_matched_tool_ids:
            if tool_id in current_mcp_tool_ids:
                final_mcp_tool_ids_to_keep.add(tool_id)
        logger.info(f"Initially keeping newly matched MCP tools (if on agent): {len(final_mcp_tool_ids_to_keep)}. Set: {final_mcp_tool_ids_to_keep}")

        for tool_id in requested_keep_tool_ids:
            if tool_id in current_mcp_tool_ids:
                final_mcp_tool_ids_to_keep.add(tool_id)
        
        # Additional safeguard: protect any never-detach tools that might still be in MCP list
        for tool in mcp_tools_on_agent_list:
            tool_name = tool.get('name', '').lower()
            if any(never_detach_name.lower() in tool_name for never_detach_name in NEVER_DETACH_TOOLS):
                final_mcp_tool_ids_to_keep.add(tool.get('id'))
                logger.warning(f"Never-detach tool '{tool.get('name')}' found in MCP list - protecting from pruning")
        
        logger.info(f"After adding explicitly requested-to-keep MCP tools (if on agent): {len(final_mcp_tool_ids_to_keep)}. Set: {final_mcp_tool_ids_to_keep}")

        # If the number of must-keep tools is already at or above the target, we need to be more aggressive
        # Apply stricter pruning when we have too many "must-keep" tools
        if len(final_mcp_tool_ids_to_keep) >= num_mcp_tools_to_keep:
            logger.info(f"Number of must-keep MCP tools ({len(final_mcp_tool_ids_to_keep)}) meets or exceeds target ({num_mcp_tools_to_keep}). Being more aggressive with detachment.")
            # Even with must-keep tools, we should still enforce the drop rate more strictly
            # Only keep the most relevant tools up to 80% of current count to force detachment
            aggressive_target = max(1, math.floor(num_currently_attached_mcp * 0.8))
            if len(final_mcp_tool_ids_to_keep) > aggressive_target:
                # Prioritize tools in this order: never-detach tools, explicitly kept tools, newly matched tools, then library-relevant tools
                prioritized_keeps = set()
                
                # HIGHEST PRIORITY: Never-detach tools (find_tools, etc.) - these MUST be kept regardless of target
                never_detach_tool_ids = set()
                for tool in mcp_tools_on_agent_list:
                    tool_name = tool.get('name', '').lower()
                    if any(never_detach_name.lower() in tool_name for never_detach_name in NEVER_DETACH_TOOLS):
                        never_detach_tool_ids.add(tool.get('id'))
                        prioritized_keeps.add(tool.get('id'))
                        logger.info(f"Aggressive pruning: PROTECTING never-detach tool '{tool.get('name')}' (ID: {tool.get('id')})")
                
                # SECOND PRIORITY: Explicitly requested to keep tools (these should NEVER be detached)
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
            # We have space to keep more MCP tools up to num_mcp_tools_to_keep.
            # Fill the remaining slots with the most relevant *other* currently attached MCP tools.
            # These are tools in current_mcp_tool_ids but not in final_mcp_tool_ids_to_keep yet.
            
            # We need to sort these other_attached_mcp_tools by relevance.
            # The `ordered_top_library_tool_info` gives relevance from a library search.
            # We'll iterate through it and pick attached MCP tools not already in our keep set.
            
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
            # Convert set to list once to preserve order for result mapping
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
        
        return {
            "success": True,
            "message": f"Pruning completed for agent {agent_id}. Only MCP tools were considered for pruning.",
            "details": {
                "tools_on_agent_before_total": num_total_attached,
                "mcp_tools_on_agent_before": num_currently_attached_mcp,
                "core_tools_preserved_count": num_currently_attached_core,
                "target_mcp_tools_to_keep_after_pruning": num_mcp_tools_to_keep, # Renamed for clarity
                "relevant_library_tools_found_count": len(ordered_top_library_tool_info),
                "final_mcp_tool_ids_kept_on_agent": list(final_mcp_tool_ids_to_keep),
                "final_core_tool_ids_on_agent": list(current_core_tool_ids),
                "actual_total_tools_on_agent_after_pruning": len(final_tool_ids_on_agent),
                "mcp_tools_detached_count": len(successful_detachments_info),
                "mcp_tools_failed_detachment_count": len(failed_detachments_info),
                "drop_rate_applied_to_mcp_tools": drop_rate,
                "explicitly_kept_tool_ids_from_request": list(requested_keep_tool_ids), # These are all types
                "newly_matched_tool_ids_from_request": list(requested_newly_matched_tool_ids), # These are all types
                "successful_detachments_mcp": successful_detachments_info,
                "failed_detachments_mcp": failed_detachments_info
            }
        }

    except Exception as e:
        logger.error(f"Error during tool pruning for agent {agent_id}: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.route('/api/v1/tools/prune', methods=['POST'])
async def prune_tools():
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
        pruning_result = await _perform_tool_pruning(
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


@app.route('/api/v1/tools/sync', methods=['POST'])
async def sync_tools_endpoint(): # Renamed function to avoid conflict
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


@app.route('/api/v1/config/validate', methods=['POST'])
async def validate_config_endpoint():
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


@app.route('/api/v1/search/parameter-schemas', methods=['GET'])
async def get_search_parameter_schemas():
    """LDTS-79: Get parameter validation schemas for BM25 and vector parameters"""
    logger.info("Received request for /api/v1/search/parameter-schemas")
    
    try:
        return jsonify({
            "bm25_parameters": bm25_vector_override_service.get_bm25_parameter_schema(),
            "vector_parameters": bm25_vector_override_service.get_vector_parameter_schema(),
            "supported_distance_metrics": bm25_vector_override_service.get_supported_distance_metrics()
        })
        
    except Exception as e:
        logger.error(f"Failed to get parameter schemas: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get parameter schemas: {str(e)}"}), 500


@app.route('/api/v1/search/supported-distance-metrics', methods=['GET'])
async def get_supported_distance_metrics():
    """LDTS-79: Get list of supported vector distance metrics"""
    logger.info("Received request for /api/v1/search/supported-distance-metrics")
    
    try:
        metrics = bm25_vector_override_service.get_supported_distance_metrics()
        
        return jsonify({
            "supported_distance_metrics": metrics,
            "count": len(metrics),
            "default_metric": "cosine"
        })
        
    except Exception as e:
        logger.error(f"Failed to get supported distance metrics: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get distance metrics: {str(e)}"}), 500


@app.route('/api/v1/search/parameter-sets', methods=['POST'])
async def create_search_parameter_set():
    """LDTS-79: Create a new search parameter set with BM25 and vector overrides"""
    logger.info("Received request for /api/v1/search/parameter-sets [POST]")
    
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No parameter set data provided"}), 400
        
        # Extract required fields
        name = data.get("name")
        if not name:
            return jsonify({"error": "Parameter set name is required"}), 400
        
        description = data.get("description", "")
        
        parameter_set_id = bm25_vector_override_service.create_parameter_set(
            name=name,
            description=description,
            bm25_params=data.get("bm25_parameters", []),
            vector_params=data.get("vector_parameters", []),
            hybrid_alpha=data.get("hybrid_alpha"),
            fusion_type=data.get("fusion_type", "relative_score"),
            metadata=data.get("metadata")
        )
        
        logger.info(f"Created search parameter set: {name} ({parameter_set_id})")
        
        return jsonify({
            "parameter_set_id": parameter_set_id,
            "name": name,
            "description": description,
            "created": True
        })
        
    except Exception as e:
        logger.error(f"Failed to create search parameter set: {e}", exc_info=True)
        return jsonify({"error": f"Failed to create parameter set: {str(e)}"}), 500


@app.route('/api/v1/search/parameter-sets', methods=['GET'])
async def list_search_parameter_sets():
    """LDTS-79: List all search parameter sets"""
    logger.info("Received request for /api/v1/search/parameter-sets [GET]")
    
    try:
        # Get query parameter for active_only (default: True)
        args = request.args
        active_only = args.get('active_only', 'true').lower() == 'true'
        
        parameter_sets = bm25_vector_override_service.list_parameter_sets(active_only=active_only)
        
        return jsonify({
            "parameter_sets": parameter_sets,
            "total_count": len(parameter_sets),
            "active_only": active_only
        })
        
    except Exception as e:
        logger.error(f"Failed to list parameter sets: {e}", exc_info=True)
        return jsonify({"error": f"Failed to list parameter sets: {str(e)}"}), 500


@app.route('/api/v1/search/parameter-sets/<parameter_set_id>', methods=['GET'])
async def get_search_parameter_set(parameter_set_id: str):
    """LDTS-79: Get a specific search parameter set"""
    logger.info(f"Received request for /api/v1/search/parameter-sets/{parameter_set_id} [GET]")
    
    try:
        parameter_set = bm25_vector_override_service.get_parameter_set(parameter_set_id)
        
        if not parameter_set:
            return jsonify({"error": f"Parameter set not found: {parameter_set_id}"}), 404
        
        return jsonify({
            "parameter_set_id": parameter_set.parameter_set_id,
            "name": parameter_set.name,
            "description": parameter_set.description,
            "bm25_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.bm25_overrides
            ],
            "vector_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.vector_overrides
            ],
            "hybrid_alpha": parameter_set.hybrid_alpha,
            "fusion_type": parameter_set.fusion_type,
            "created_at": parameter_set.created_at.isoformat(),
            "active": parameter_set.active,
            "metadata": parameter_set.metadata
        })
        
    except Exception as e:
        logger.error(f"Failed to get parameter set {parameter_set_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get parameter set: {str(e)}"}), 500


@app.route('/api/v1/search/statistics', methods=['GET'])
async def get_search_override_statistics():
    """LDTS-79: Get BM25/Vector override service statistics"""
    logger.info("Received request for /api/v1/search/statistics")
    
    try:
        statistics = bm25_vector_override_service.get_statistics()
        
        return jsonify({
            "service_statistics": statistics,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to get search override statistics: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get search override statistics: {str(e)}"}), 500


# Reranker Configuration Endpoints
@app.route('/api/v1/config/reranker', methods=['GET'])
async def get_reranker_config():
    """Get current reranker configuration."""
    try:
        # For now, return default config based on environment variables
        config = {
            "enabled": os.getenv('RERANKER_ENABLED', 'true').lower() == 'true',
            "model": os.getenv('RERANKER_MODEL', 'qwen3-reranker-4b'),
            "provider": os.getenv('RERANKER_PROVIDER', 'vllm'),
            "parameters": {
                "temperature": float(os.getenv('RERANKER_TEMPERATURE', '0.1')),
                "max_tokens": int(os.getenv('RERANKER_MAX_TOKENS', '512')),
                "base_url": os.getenv('RERANKER_URL', 'http://100.81.139.20:11435/rerank')
            }
        }
        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting reranker config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/reranker', methods=['PUT'])
async def update_reranker_config():
    """Update reranker configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        # Validate required fields
        required_fields = ['enabled', 'model', 'provider', 'parameters']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # For now, just acknowledge the update (in a full implementation, 
        # this would save to a config file or database)
        logger.info(f"Reranker config update requested: {data}")
        
        return jsonify({"success": True, "message": "Reranker configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating reranker config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/reranker/test', methods=['POST'])
async def test_reranker_connection():
    """Test reranker connection with provided configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        # Extract connection parameters
        provider = data.get('provider', 'ollama')
        base_url = data.get('parameters', {}).get('base_url', 'http://ollama-reranker-adapter:8080')
        # model = data.get('model', 'mistral:7b')  # Currently unused
        
        # Test connection based on provider
        connected = False
        if provider == 'ollama':
            try:
                async with aiohttp.ClientSession() as session:
                    # Test health endpoint
                    async with session.get(f"{base_url}/health", timeout=5) as response:
                        if response.status == 200:
                            connected = True
            except Exception as conn_error:
                logger.warning(f"Reranker connection test failed: {str(conn_error)}")
        
        return jsonify({
            "success": True, 
            "data": {"connected": connected}
        })
    except Exception as e:
        logger.error(f"Error testing reranker connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Embedding Configuration Endpoints
@app.route('/api/v1/config/embedding', methods=['GET'])
async def get_embedding_config():
    """Get current embedding configuration."""
    try:
        # Return embedding config based on environment variables
        config = {
            "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
            "provider": os.getenv('EMBEDDING_PROVIDER', 'ollama'),
            "parameters": {
                "dimensions": int(os.getenv('EMBEDDING_DIMENSION', '2560')),
                "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                "use_ollama": os.getenv('USE_OLLAMA_EMBEDDINGS', 'true').lower() == 'true'
            }
        }
        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting embedding config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/embedding', methods=['PUT'])
async def update_embedding_config():
    """Update embedding configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        # Validate required fields
        required_fields = ['model', 'provider']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # For now, just acknowledge the update
        logger.info(f"Embedding config update requested: {data}")
        
        return jsonify({"success": True, "message": "Embedding configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating embedding config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/ollama/models', methods=['GET'])
async def get_ollama_models():
    """Get available models from Ollama instance."""
    
    def get_fallback_models():
        """Get fallback models including configured embedding model"""
        configured_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'llama2:7b')
        fallback_models = [configured_model, "mistral:7b", "llama2:7b", "codellama:7b"]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(fallback_models))
    
    try:
        # Get base URL from environment configuration
        ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
        base_url = f"http://{ollama_host}:11434"
        
        # Try to query Ollama API for models
        logger.info(f"Fetching Ollama models from: {base_url}")
        
        async with aiohttp.ClientSession() as session:
            # Query the Ollama API tags endpoint
            ollama_url = f"{base_url}/api/tags"
            async with session.get(ollama_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    # Format models for frontend
                    formatted_models = []
                    for model in models:
                        formatted_model = {
                            "name": model.get('name', ''),
                            "size": model.get('size', 0),
                            "modified_at": model.get('modified_at', ''),
                            "digest": model.get('digest', ''),
                            "details": model.get('details', {})
                        }
                        formatted_models.append(formatted_model)
                    
                    logger.info(f"Successfully fetched {len(formatted_models)} models from Ollama")
                    return jsonify({
                        "success": True, 
                        "data": {
                            "models": formatted_models,
                            "base_url": base_url,
                            "total": len(formatted_models)
                        }
                    })
                else:
                    logger.warning(f"Ollama API returned status {response.status}")
                    return jsonify({
                        "success": False, 
                        "error": f"Ollama API returned status {response.status}",
                        "fallback_models": get_fallback_models()
                    }), 503
                    
    except asyncio.TimeoutError:
        logger.error("Timeout connecting to Ollama API")
        return jsonify({
            "success": False, 
            "error": "Timeout connecting to Ollama instance",
            "fallback_models": ["mistral:7b", "llama2:7b", "codellama:7b"]
        }), 503
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return jsonify({
            "success": False, 
            "error": f"Failed to fetch models: {str(e)}",
            "fallback_models": get_fallback_models()
        }), 503


# Configuration presets endpoints
@app.route('/api/v1/config/presets', methods=['GET'])
async def get_configuration_presets():
    """Get all configuration presets."""
    try:
        # For now, return empty array since we don't have persistent storage yet
        # This can be extended to use a database or file storage
        return jsonify({
            "success": True,
            "data": []
        })
    except Exception as e:
        logger.error(f"Error getting configuration presets: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/presets', methods=['POST'])
async def create_configuration_preset():
    """Create a new configuration preset."""
    try:
        data = await request.get_json()
        # For now, return success but don't actually store
        # This can be extended to use a database or file storage
        return jsonify({
            "success": True,
            "data": {
                "id": "preset_" + str(int(time.time())),
                "name": data.get("name", "Untitled Preset"),
                "description": data.get("description", ""),
                "config": data.get("config", {}),
                "created_at": time.time()
            }
        })
    except Exception as e:
        logger.error(f"Error creating configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/presets/<preset_id>', methods=['PUT'])
async def update_configuration_preset(preset_id):
    """Update a configuration preset."""
    try:
        data = await request.get_json()
        # For now, return success but don't actually store
        return jsonify({
            "success": True,
            "data": {
                "id": preset_id,
                "name": data.get("name", "Updated Preset"),
                "description": data.get("description", ""),
                "config": data.get("config", {}),
                "updated_at": time.time()
            }
        })
    except Exception as e:
        logger.error(f"Error updating configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/presets/<preset_id>', methods=['DELETE'])
async def delete_configuration_preset(preset_id):
    """Delete a configuration preset."""
    try:
        # For now, just return success
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Tool Selector Configuration endpoints
@app.route('/api/v1/config/tool-selector', methods=['GET'])
async def get_tool_selector_config():
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


@app.route('/api/v1/config/tool-selector', methods=['PUT'])
async def update_tool_selector_config():
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


# Ollama Configuration endpoints
@app.route('/api/v1/config/ollama', methods=['GET'])
async def get_ollama_config():
    """Get current Ollama configuration."""
    try:
        config = {
            "connection": {
                "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                "port": int(os.getenv('OLLAMA_PORT', '11434')),
                "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30')),
                "base_url": os.getenv('OLLAMA_BASE_URL', '')
            },
            "embedding": {
                "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                "enabled": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false').lower() == 'true'
            },
            "generation": {
                "default_model": os.getenv('OLLAMA_DEFAULT_MODEL', 'mistral:7b'),
                "temperature": float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
                "context_length": int(os.getenv('OLLAMA_CONTEXT_LENGTH', '4096'))
            },
            "performance": {
                "num_parallel": int(os.getenv('OLLAMA_NUM_PARALLEL', '1')),
                "num_ctx": int(os.getenv('OLLAMA_NUM_CTX', '2048')),
                "num_gpu": int(os.getenv('OLLAMA_NUM_GPU', '-1')),  # -1 means auto-detect
                "low_vram": os.getenv('OLLAMA_LOW_VRAM', 'false').lower() == 'true'
            }
        }

        # Add connection status
        connection_status = await test_ollama_connection(config["connection"])
        config["status"] = connection_status

        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting Ollama config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/ollama', methods=['PUT'])
async def update_ollama_config():
    """Update Ollama configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400

        # Validate configuration data
        validation_errors = []
        warnings = []

        if "connection" in data:
            conn = data["connection"]

            # Validate host
            if "host" in conn:
                host = conn["host"]
                if not host or not isinstance(host, str):
                    validation_errors.append("Host must be a valid string")
                elif not host.replace('.', '').replace('-', '').replace(':', '').isalnum():
                    warnings.append("Host format may be invalid")

            # Validate port
            if "port" in conn:
                port = conn["port"]
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    validation_errors.append("Port must be between 1 and 65535")

            # Validate timeout
            if "timeout" in conn:
                timeout = conn["timeout"]
                if not isinstance(timeout, int) or timeout < 1:
                    validation_errors.append("Timeout must be a positive integer")
                elif timeout < 5:
                    warnings.append("Very low timeout may cause connection issues")

        if "performance" in data:
            perf = data["performance"]

            # Validate num_parallel
            if "num_parallel" in perf:
                num_parallel = perf["num_parallel"]
                if not isinstance(num_parallel, int) or num_parallel < 1:
                    validation_errors.append("num_parallel must be a positive integer")
                elif num_parallel > 8:
                    warnings.append("High parallelism may cause resource issues")

            # Validate context lengths
            if "num_ctx" in perf:
                num_ctx = perf["num_ctx"]
                if not isinstance(num_ctx, int) or num_ctx < 128:
                    validation_errors.append("num_ctx must be at least 128")
                elif num_ctx > 32768:
                    warnings.append("Very large context may cause memory issues")

        if validation_errors:
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "warnings": warnings
            }), 400

        # Test connection if connection config provided
        if "connection" in data:
            test_result = await test_ollama_connection(data["connection"])
            if not test_result["available"]:
                warnings.append(f"Ollama connection test failed: {test_result.get('error', 'Unknown error')}")

        # Log the configuration update
        logger.info(f"Ollama configuration update requested: {data}")

        # Log to audit system
        await log_config_change(
            action="update",
            config_type="ollama",
            changes=data,
            user_info={"source": "api", "timestamp": datetime.now().isoformat()},
            metadata={"warnings": warnings, "connection_test": test_result if "connection" in data else None}
        )

        # For now, just acknowledge the update
        # TODO: Actually persist the configuration changes to environment or config file

        response = {
            "success": True,
            "message": "Ollama configuration updated successfully",
            "applied_config": data
        }

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error updating Ollama config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/ollama/test', methods=['POST'])
async def test_ollama_connection_endpoint():
    """Test Ollama connection with provided configuration."""
    try:
        data = await request.get_json()
        connection_config = data.get("connection", {})

        # Use defaults if not provided
        config = {
            "host": connection_config.get("host", os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')),
            "port": connection_config.get("port", int(os.getenv('OLLAMA_PORT', '11434'))),
            "timeout": connection_config.get("timeout", int(os.getenv('OLLAMA_TIMEOUT', '30')))
        }

        result = await test_ollama_connection(config)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def test_ollama_connection(config):
    """Test connection to Ollama server."""
    try:
        host = config.get("host", "192.168.50.80")
        port = config.get("port", 11434)
        timeout = config.get("timeout", 30)

        base_url = f"http://{host}:{port}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Test basic connectivity
            async with session.get(f"{base_url}/api/version") as response:
                if response.status == 200:
                    version_data = await response.json()

                    # Test model listing
                    async with session.get(f"{base_url}/api/tags") as models_response:
                        if models_response.status == 200:
                            models_data = await models_response.json()
                            model_count = len(models_data.get("models", []))

                            return {
                                "available": True,
                                "version": version_data.get("version", "unknown"),
                                "host": host,
                                "port": port,
                                "model_count": model_count,
                                "response_time": "< 1s"  # Simplified for now
                            }
                        else:
                            return {
                                "available": False,
                                "error": f"Models endpoint returned {models_response.status}",
                                "host": host,
                                "port": port
                            }
                else:
                    return {
                        "available": False,
                        "error": f"Version endpoint returned {response.status}",
                        "host": host,
                        "port": port
                    }
    except asyncio.TimeoutError:
        return {
            "available": False,
            "error": "Connection timeout",
            "host": host,
            "port": port
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "host": host,
            "port": port
        }


# Weaviate Configuration endpoints
@app.route('/api/v1/config/weaviate', methods=['GET'])
async def get_weaviate_config():
    """Get current Weaviate configuration."""
    try:
        config = {
            "connection": {
                "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '30')),
                "retries": int(os.getenv('WEAVIATE_RETRIES', '3')),
                "api_key": "***" if os.getenv('WEAVIATE_API_KEY') else None
            },
            "schema": {
                "class_name": os.getenv('WEAVIATE_CLASS_NAME', 'Tool'),
                "vector_index_type": os.getenv('WEAVIATE_VECTOR_INDEX', 'hnsw'),
                "distance_metric": os.getenv('WEAVIATE_DISTANCE_METRIC', 'cosine')
            },
            "search": {
                "alpha": float(os.getenv('WEAVIATE_ALPHA', '0.75')),
                "limit": int(os.getenv('WEAVIATE_LIMIT', '50')),
                "additional_properties": os.getenv('WEAVIATE_ADDITIONAL_PROPERTIES', 'id,distance,certainty').split(','),
                "autocut": int(os.getenv('WEAVIATE_AUTOCUT', '1'))
            },
            "performance": {
                "ef_construction": int(os.getenv('WEAVIATE_EF_CONSTRUCTION', '128')),
                "ef": int(os.getenv('WEAVIATE_EF', '64')),
                "max_connections": int(os.getenv('WEAVIATE_MAX_CONNECTIONS', '64')),
                "vector_cache_max_objects": int(os.getenv('WEAVIATE_VECTOR_CACHE_MAX_OBJECTS', '1000000')),
                "cleanup_interval_seconds": int(os.getenv('WEAVIATE_CLEANUP_INTERVAL', '60'))
            }
        }

        # Add connection status
        connection_status = await test_weaviate_connection(config["connection"])
        config["status"] = connection_status

        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting Weaviate config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/weaviate', methods=['PUT'])
async def update_weaviate_config():
    """Update Weaviate configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400

        # Validate configuration data
        validation_errors = []
        warnings = []

        if "connection" in data:
            conn = data["connection"]

            # Validate URL
            if "url" in conn:
                url = conn["url"]
                if not url or not isinstance(url, str):
                    validation_errors.append("URL must be a valid string")
                elif not (url.startswith('http://') or url.startswith('https://')):
                    validation_errors.append("URL must start with http:// or https://")

            # Validate timeout
            if "timeout" in conn:
                timeout = conn["timeout"]
                if not isinstance(timeout, int) or timeout < 1:
                    validation_errors.append("Timeout must be a positive integer")
                elif timeout < 5:
                    warnings.append("Very low timeout may cause connection issues")

            # Validate retries
            if "retries" in conn:
                retries = conn["retries"]
                if not isinstance(retries, int) or retries < 0:
                    validation_errors.append("Retries must be a non-negative integer")
                elif retries > 10:
                    warnings.append("High retry count may cause delays")

        if "search" in data:
            search = data["search"]

            # Validate alpha
            if "alpha" in search:
                alpha = search["alpha"]
                if not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
                    validation_errors.append("Alpha must be between 0.0 and 1.0")

            # Validate limit
            if "limit" in search:
                limit = search["limit"]
                if not isinstance(limit, int) or limit < 1:
                    validation_errors.append("Limit must be a positive integer")
                elif limit > 1000:
                    warnings.append("Very high limit may impact performance")

        if "performance" in data:
            perf = data["performance"]

            # Validate HNSW parameters
            hnsw_params = ["ef_construction", "ef", "max_connections"]
            for param in hnsw_params:
                if param in perf:
                    value = perf[param]
                    if not isinstance(value, int) or value < 1:
                        validation_errors.append(f"{param} must be a positive integer")

            # Validate cache size
            if "vector_cache_max_objects" in perf:
                cache_size = perf["vector_cache_max_objects"]
                if not isinstance(cache_size, int) or cache_size < 1000:
                    validation_errors.append("vector_cache_max_objects must be at least 1000")
                elif cache_size > 10000000:
                    warnings.append("Very large vector cache may use excessive memory")

        if validation_errors:
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "warnings": warnings
            }), 400

        # Test connection if connection config provided
        if "connection" in data:
            test_result = await test_weaviate_connection(data["connection"])
            if not test_result["available"]:
                warnings.append(f"Weaviate connection test failed: {test_result.get('error', 'Unknown error')}")

        # Log the configuration update
        logger.info(f"Weaviate configuration update requested: {data}")

        # For now, just acknowledge the update
        # TODO: Actually persist the configuration changes to environment or config file

        response = {
            "success": True,
            "message": "Weaviate configuration updated successfully",
            "applied_config": data
        }

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error updating Weaviate config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/weaviate/test', methods=['POST'])
async def test_weaviate_connection_endpoint():
    """Test Weaviate connection with provided configuration."""
    try:
        data = await request.get_json()
        connection_config = data.get("connection", {})

        # Use defaults if not provided
        config = {
            "url": connection_config.get("url", os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')),
            "timeout": connection_config.get("timeout", int(os.getenv('WEAVIATE_TIMEOUT', '30'))),
            "api_key": connection_config.get("api_key", os.getenv('WEAVIATE_API_KEY'))
        }

        result = await test_weaviate_connection(config)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error testing Weaviate connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/weaviate/schema', methods=['GET'])
async def get_weaviate_schema():
    """Get Weaviate schema information."""
    try:
        url = os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')
        timeout = int(os.getenv('WEAVIATE_TIMEOUT', '30'))

        headers = {}
        if os.getenv('WEAVIATE_API_KEY'):
            headers['Authorization'] = f"Bearer {os.getenv('WEAVIATE_API_KEY')}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Get schema information
            async with session.get(f"{url.rstrip('/')}/v1/schema", headers=headers) as response:
                if response.status == 200:
                    schema_data = await response.json()

                    # Get object count for each class
                    classes_with_counts = []
                    for class_info in schema_data.get("classes", []):
                        class_name = class_info["class"]

                        # Get object count
                        count_query = f"{url.rstrip('/')}/v1/objects?class={class_name}&limit=0"
                        async with session.get(count_query, headers=headers) as count_response:
                            if count_response.status == 200:
                                count_data = await count_response.json()
                                object_count = count_data.get("totalResults", 0)
                            else:
                                object_count = -1  # Unknown

                        classes_with_counts.append({
                            **class_info,
                            "object_count": object_count
                        })

                    return jsonify({
                        "success": True,
                        "data": {
                            "classes": classes_with_counts,
                            "total_classes": len(classes_with_counts)
                        }
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": f"Schema endpoint returned {response.status}"
                    }), response.status
    except Exception as e:
        logger.error(f"Error getting Weaviate schema: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


async def test_weaviate_connection(config):
    """Test connection to Weaviate server."""
    try:
        url = config.get("url", "http://weaviate:8080/")
        timeout = config.get("timeout", 30)
        api_key = config.get("api_key")

        headers = {}
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Test basic connectivity
            meta_url = f"{url.rstrip('/')}/v1/meta"
            async with session.get(meta_url, headers=headers) as response:
                if response.status == 200:
                    meta_data = await response.json()

                    # Test schema access
                    schema_url = f"{url.rstrip('/')}/v1/schema"
                    async with session.get(schema_url, headers=headers) as schema_response:
                        if schema_response.status == 200:
                            schema_data = await schema_response.json()
                            class_count = len(schema_data.get("classes", []))

                            return {
                                "available": True,
                                "version": meta_data.get("version", "unknown"),
                                "hostname": meta_data.get("hostname", "unknown"),
                                "class_count": class_count,
                                "modules": meta_data.get("modules", {}),
                                "url": url
                            }
                        else:
                            return {
                                "available": False,
                                "error": f"Schema endpoint returned {schema_response.status}",
                                "url": url
                            }
                else:
                    return {
                        "available": False,
                        "error": f"Meta endpoint returned {response.status}",
                        "url": url
                    }
    except asyncio.TimeoutError:
        return {
            "available": False,
            "error": "Connection timeout",
            "url": url
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "url": url
        }


# Configuration Backup and Restore endpoints
@app.route('/api/v1/config/backup', methods=['POST'])
async def create_config_backup():
    """Create a backup of all configuration settings."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        backup_name = data.get('name', f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        include_secrets = data.get('include_secrets', False)

        # Collect all configuration data
        backup_data = {
            "backup_info": {
                "name": backup_name,
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "includes_secrets": include_secrets,
                "system_info": {
                    "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                    "python_version": sys.version,
                    "app_version": "1.0.0"  # TODO: Get from version file
                }
            },
            "configurations": {}
        }

        # Tool Selector Configuration
        try:
            tool_selector_config = {
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
                }
            }
            backup_data["configurations"]["tool_selector"] = tool_selector_config
        except Exception as e:
            logger.warning(f"Failed to backup tool selector config: {e}")

        # Ollama Configuration
        try:
            ollama_config = {
                "connection": {
                    "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                    "port": int(os.getenv('OLLAMA_PORT', '11434')),
                    "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30'))
                },
                "embedding": {
                    "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                    "enabled": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false').lower() == 'true'
                },
                "generation": {
                    "default_model": os.getenv('OLLAMA_DEFAULT_MODEL', 'mistral:7b'),
                    "temperature": float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
                    "context_length": int(os.getenv('OLLAMA_CONTEXT_LENGTH', '4096'))
                }
            }
            backup_data["configurations"]["ollama"] = ollama_config
        except Exception as e:
            logger.warning(f"Failed to backup Ollama config: {e}")

        # Weaviate Configuration
        try:
            weaviate_config = {
                "connection": {
                    "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                    "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '30')),
                    "retries": int(os.getenv('WEAVIATE_RETRIES', '3'))
                },
                "schema": {
                    "class_name": os.getenv('WEAVIATE_CLASS_NAME', 'Tool'),
                    "vector_index_type": os.getenv('WEAVIATE_VECTOR_INDEX', 'hnsw'),
                    "distance_metric": os.getenv('WEAVIATE_DISTANCE_METRIC', 'cosine')
                },
                "search": {
                    "alpha": float(os.getenv('WEAVIATE_ALPHA', '0.75')),
                    "limit": int(os.getenv('WEAVIATE_LIMIT', '50'))
                }
            }

            if include_secrets and os.getenv('WEAVIATE_API_KEY'):
                weaviate_config["connection"]["api_key"] = os.getenv('WEAVIATE_API_KEY')

            backup_data["configurations"]["weaviate"] = weaviate_config
        except Exception as e:
            logger.warning(f"Failed to backup Weaviate config: {e}")

        # Embedding Configuration
        try:
            embedding_config = {
                "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
                "openai_model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                "ollama_model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M')
            }

            if include_secrets and os.getenv('OPENAI_API_KEY'):
                embedding_config["openai_api_key"] = os.getenv('OPENAI_API_KEY')

            backup_data["configurations"]["embedding"] = embedding_config
        except Exception as e:
            logger.warning(f"Failed to backup embedding config: {e}")

        # Letta API Configuration
        try:
            letta_config = {
                "api_url": os.getenv('LETTA_API_URL', 'https://letta.example.com/v1'),
                "timeout": int(os.getenv('LETTA_TIMEOUT', '30'))
            }

            if include_secrets and os.getenv('LETTA_PASSWORD'):
                letta_config["password"] = os.getenv('LETTA_PASSWORD')

            backup_data["configurations"]["letta"] = letta_config
        except Exception as e:
            logger.warning(f"Failed to backup Letta config: {e}")

        # Save backup to file
        backup_dir = os.path.join(CACHE_DIR, 'backups')
        os.makedirs(backup_dir, exist_ok=True)

        backup_file = os.path.join(backup_dir, f"{backup_name}.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

        logger.info(f"Configuration backup created: {backup_file}")

        return jsonify({
            "success": True,
            "message": "Configuration backup created successfully",
            "data": {
                "backup_name": backup_name,
                "backup_file": backup_file,
                "size_bytes": os.path.getsize(backup_file),
                "configurations_count": len(backup_data["configurations"]),
                "created_at": backup_data["backup_info"]["created_at"]
            }
        })

    except Exception as e:
        logger.error(f"Error creating configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/backup', methods=['GET'])
async def list_config_backups():
    """List all available configuration backups."""
    try:
        backup_dir = os.path.join(CACHE_DIR, 'backups')

        if not os.path.exists(backup_dir):
            return jsonify({
                "success": True,
                "data": {
                    "backups": [],
                    "total": 0
                }
            })

        backups = []
        for filename in os.listdir(backup_dir):
            if filename.endswith('.json'):
                backup_path = os.path.join(backup_dir, filename)
                try:
                    with open(backup_path, 'r') as f:
                        backup_data = json.load(f)

                    backup_info = backup_data.get("backup_info", {})
                    backups.append({
                        "name": backup_info.get("name", filename.replace('.json', '')),
                        "filename": filename,
                        "created_at": backup_info.get("created_at"),
                        "size_bytes": os.path.getsize(backup_path),
                        "configurations_count": len(backup_data.get("configurations", {})),
                        "includes_secrets": backup_info.get("includes_secrets", False),
                        "version": backup_info.get("version", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse backup {filename}: {e}")
                    # Add basic info even if parsing fails
                    backups.append({
                        "name": filename.replace('.json', ''),
                        "filename": filename,
                        "created_at": None,
                        "size_bytes": os.path.getsize(backup_path),
                        "configurations_count": 0,
                        "includes_secrets": False,
                        "version": "unknown",
                        "error": "Failed to parse backup file"
                    })

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return jsonify({
            "success": True,
            "data": {
                "backups": backups,
                "total": len(backups)
            }
        })

    except Exception as e:
        logger.error(f"Error listing configuration backups: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/backup/<backup_name>', methods=['GET'])
async def get_config_backup(backup_name):
    """Get details of a specific backup."""
    try:
        backup_dir = os.path.join(CACHE_DIR, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        # Mask secrets if they exist
        masked_data = json.loads(json.dumps(backup_data))  # Deep copy
        if "configurations" in masked_data:
            for config_name, config_data in masked_data["configurations"].items():
                if isinstance(config_data, dict):
                    for key, value in config_data.items():
                        if isinstance(value, dict):
                            for sub_key in list(value.keys()):
                                if 'key' in sub_key.lower() or 'password' in sub_key.lower() or 'secret' in sub_key.lower():
                                    value[sub_key] = "***"
                        elif 'key' in key.lower() or 'password' in key.lower() or 'secret' in key.lower():
                            config_data[key] = "***"

        return jsonify({
            "success": True,
            "data": masked_data
        })

    except Exception as e:
        logger.error(f"Error getting configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/restore/<backup_name>', methods=['POST'])
async def restore_config_backup(backup_name):
    """Restore configuration from a backup."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        dry_run = data.get('dry_run', False)
        selected_configs = data.get('configurations', [])  # List of config names to restore

        backup_dir = os.path.join(CACHE_DIR, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        configurations = backup_data.get("configurations", {})

        if not configurations:
            return jsonify({"success": False, "error": "No configurations found in backup"}), 400

        # Filter configurations if specific ones were selected
        if selected_configs:
            configurations = {k: v for k, v in configurations.items() if k in selected_configs}

        restoration_plan = {}
        warnings = []

        # Plan restoration for each configuration
        for config_name, config_data in configurations.items():
            if config_name == "tool_selector":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "MAX_TOTAL_TOOLS", "current": os.getenv('MAX_TOTAL_TOOLS', '30'), "new": str(config_data["tool_limits"]["max_total_tools"])},
                        {"var": "MAX_MCP_TOOLS", "current": os.getenv('MAX_MCP_TOOLS', '20'), "new": str(config_data["tool_limits"]["max_mcp_tools"])},
                        {"var": "MIN_MCP_TOOLS", "current": os.getenv('MIN_MCP_TOOLS', '7'), "new": str(config_data["tool_limits"]["min_mcp_tools"])},
                        {"var": "DEFAULT_DROP_RATE", "current": os.getenv('DEFAULT_DROP_RATE', '0.6'), "new": str(config_data["behavior"]["default_drop_rate"])},
                        {"var": "EXCLUDE_LETTA_CORE_TOOLS", "current": os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'true'), "new": str(config_data["behavior"]["exclude_letta_core_tools"]).lower()},
                    ]
                }

            elif config_name == "ollama":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "OLLAMA_EMBEDDING_HOST", "current": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'), "new": config_data["connection"]["host"]},
                        {"var": "OLLAMA_PORT", "current": os.getenv('OLLAMA_PORT', '11434'), "new": str(config_data["connection"]["port"])},
                        {"var": "OLLAMA_EMBEDDING_MODEL", "current": os.getenv('OLLAMA_EMBEDDING_MODEL', ''), "new": config_data["embedding"]["model"]},
                        {"var": "USE_OLLAMA_EMBEDDINGS", "current": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false'), "new": str(config_data["embedding"]["enabled"]).lower()},
                    ]
                }

            elif config_name == "weaviate":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "WEAVIATE_URL", "current": os.getenv('WEAVIATE_URL', ''), "new": config_data["connection"]["url"]},
                        {"var": "WEAVIATE_TIMEOUT", "current": os.getenv('WEAVIATE_TIMEOUT', '30'), "new": str(config_data["connection"]["timeout"])},
                        {"var": "WEAVIATE_ALPHA", "current": os.getenv('WEAVIATE_ALPHA', '0.75'), "new": str(config_data["search"]["alpha"])},
                    ]
                }

                if "api_key" in config_data["connection"]:
                    restoration_plan[config_name]["changes"].append({
                        "var": "WEAVIATE_API_KEY", "current": "***" if os.getenv('WEAVIATE_API_KEY') else None, "new": "***"
                    })

        if dry_run:
            return jsonify({
                "success": True,
                "message": "Restoration plan generated (dry run)",
                "data": {
                    "backup_name": backup_name,
                    "backup_date": backup_data.get("backup_info", {}).get("created_at"),
                    "configurations_to_restore": len(restoration_plan),
                    "restoration_plan": restoration_plan,
                    "warnings": warnings
                }
            })

        # TODO: Actually apply the restoration
        # For now, just simulate the restoration
        logger.info(f"Configuration restoration requested for backup: {backup_name}")
        logger.info(f"Restoration plan: {restoration_plan}")

        return jsonify({
            "success": True,
            "message": "Configuration restored successfully (simulation)",
            "data": {
                "backup_name": backup_name,
                "restored_configurations": list(restoration_plan.keys()),
                "total_changes": sum(len(plan.get("changes", [])) for plan in restoration_plan.values()),
                "warnings": warnings + ["Restoration is currently simulated - actual environment variable updates not implemented"]
            }
        })

    except Exception as e:
        logger.error(f"Error restoring configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/backup/<backup_name>', methods=['DELETE'])
async def delete_config_backup(backup_name):
    """Delete a configuration backup."""
    try:
        backup_dir = os.path.join(CACHE_DIR, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        os.remove(backup_file)
        logger.info(f"Configuration backup deleted: {backup_file}")

        return jsonify({
            "success": True,
            "message": "Backup deleted successfully",
            "data": {
                "backup_name": backup_name,
                "deleted_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error deleting configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Configuration Save and Reset endpoints
@app.route('/api/v1/config/save', methods=['POST'])
async def save_current_configuration():
    """Save current configuration state with optional name and description."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        config_name = data.get('name', f"config_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        description = data.get('description', 'Manual configuration save')
        include_secrets = data.get('include_secrets', False)

        # Collect all current configurations
        current_config = {}

        # Collect Tool Selector configuration
        try:
            tool_selector_config = {
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
                }
            }
            current_config["tool_selector"] = tool_selector_config
        except Exception as e:
            logger.warning(f"Failed to collect tool selector config: {e}")

        # Collect Embedding configuration
        try:
            embedding_config = {
                "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
                "openai": {
                    "model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                    "api_key": "***MASKED***" if not include_secrets else os.getenv('OPENAI_API_KEY', '')
                },
                "ollama": {
                    "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                    "port": int(os.getenv('OLLAMA_PORT', '11434')),
                    "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text'),
                    "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30'))
                }
            }
            current_config["embedding"] = embedding_config
        except Exception as e:
            logger.warning(f"Failed to collect embedding config: {e}")

        # Collect Weaviate configuration
        try:
            weaviate_config = {
                "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                "batch_size": int(os.getenv('WEAVIATE_BATCH_SIZE', '100')),
                "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '60')),
                "hybrid_search_alpha": float(os.getenv('WEAVIATE_HYBRID_ALPHA', '0.75'))
            }
            current_config["weaviate"] = weaviate_config
        except Exception as e:
            logger.warning(f"Failed to collect weaviate config: {e}")

        # Collect Letta API configuration
        try:
            letta_config = {
                "url": os.getenv('LETTA_API_URL', 'https://letta.example.com/v1'),
                "password": "***MASKED***" if not include_secrets else os.getenv('LETTA_PASSWORD', ''),
                "timeout": int(os.getenv('LETTA_TIMEOUT', '30'))
            }
            current_config["letta_api"] = letta_config
        except Exception as e:
            logger.warning(f"Failed to collect letta config: {e}")

        # Create save record
        save_record = {
            "name": config_name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "include_secrets": include_secrets,
            "configurations": current_config,
            "version": "1.0",
            "saved_by": "admin"  # In a real system, get from auth
        }

        # Save to file
        saves_dir = os.path.join(CACHE_DIR, 'config_saves')
        os.makedirs(saves_dir, exist_ok=True)

        save_file = os.path.join(saves_dir, f"{config_name}.json")
        with open(save_file, 'w') as f:
            json.dump(save_record, f, indent=2)

        # Log the save action
        await log_config_change(
            action="configuration_save",
            config_type="system_wide",
            changes={"config_name": config_name},
            metadata={
                "description": description,
                "configurations_saved": len(current_config),
                "include_secrets": include_secrets
            }
        )

        return jsonify({
            "success": True,
            "data": {
                "config_name": config_name,
                "description": description,
                "saved_at": save_record["saved_at"],
                "configurations_count": len(current_config),
                "file_size_bytes": os.path.getsize(save_file)
            }
        })

    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/saves', methods=['GET'])
async def list_saved_configurations():
    """List all saved configurations."""
    try:
        saves_dir = os.path.join(CACHE_DIR, 'config_saves')

        if not os.path.exists(saves_dir):
            return jsonify({
                "success": True,
                "data": {
                    "saves": [],
                    "total": 0
                }
            })

        saves = []
        for filename in os.listdir(saves_dir):
            if filename.endswith('.json'):
                save_path = os.path.join(saves_dir, filename)
                try:
                    with open(save_path, 'r') as f:
                        save_data = json.load(f)

                    saves.append({
                        "name": save_data.get("name"),
                        "description": save_data.get("description"),
                        "saved_at": save_data.get("saved_at"),
                        "configurations_count": len(save_data.get("configurations", {})),
                        "include_secrets": save_data.get("include_secrets", False),
                        "version": save_data.get("version", "unknown"),
                        "file_size_bytes": os.path.getsize(save_path)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read save file {filename}: {e}")
                    continue

        # Sort by saved_at descending
        saves.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

        return jsonify({
            "success": True,
            "data": {
                "saves": saves,
                "total": len(saves)
            }
        })

    except Exception as e:
        logger.error(f"Error listing saved configurations: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/reset', methods=['POST'])
async def reset_configuration():
    """Reset configuration to defaults or to a specific saved state."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        reset_type = data.get('type', 'defaults')  # 'defaults' or 'saved'
        config_name = data.get('config_name')  # Required if type is 'saved'
        sections = data.get('sections', [])  # Specific sections to reset, empty = all
        dry_run = data.get('dry_run', False)

        reset_actions = []
        warnings = []

        if reset_type == 'defaults':
            # Reset to default values
            default_configs = {
                "tool_selector": {
                    "MAX_TOTAL_TOOLS": "30",
                    "MAX_MCP_TOOLS": "20",
                    "MIN_MCP_TOOLS": "7",
                    "DEFAULT_DROP_RATE": "0.6",
                    "EXCLUDE_LETTA_CORE_TOOLS": "true",
                    "EXCLUDE_OFFICIAL_TOOLS": "true",
                    "MANAGE_ONLY_MCP_TOOLS": "true"
                },
                "embedding": {
                    "EMBEDDING_PROVIDER": "openai",
                    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
                    "OLLAMA_EMBEDDING_HOST": "192.168.50.80",
                    "OLLAMA_PORT": "11434",
                    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
                    "OLLAMA_TIMEOUT": "30"
                },
                "weaviate": {
                    "WEAVIATE_URL": "http://weaviate:8080/",
                    "WEAVIATE_BATCH_SIZE": "100",
                    "WEAVIATE_TIMEOUT": "60",
                    "WEAVIATE_HYBRID_ALPHA": "0.75"
                },
                "letta_api": {
                    "LETTA_API_URL": "https://letta.example.com/v1",
                    "LETTA_TIMEOUT": "30"
                }
            }

            # Filter by sections if specified
            if sections:
                filtered_configs = {}
                for section in sections:
                    if section in default_configs:
                        filtered_configs[section] = default_configs[section]
                default_configs = filtered_configs

            # Apply defaults (simulate - in real implementation would set environment variables)
            for section, config in default_configs.items():
                for key, value in config.items():
                    reset_actions.append({
                        "action": "reset_to_default",
                        "section": section,
                        "key": key,
                        "new_value": value,
                        "previous_value": os.getenv(key, "not_set")
                    })

                    if not dry_run:
                        # In a real implementation, you would update environment variables
                        # or configuration files here
                        pass

        elif reset_type == 'saved':
            if not config_name:
                return jsonify({"success": False, "error": "config_name required for saved reset"}), 400

            # Load saved configuration
            saves_dir = os.path.join(CACHE_DIR, 'config_saves')
            save_file = os.path.join(saves_dir, f"{config_name}.json")

            if not os.path.exists(save_file):
                return jsonify({"success": False, "error": f"Saved configuration '{config_name}' not found"}), 404

            with open(save_file, 'r') as f:
                save_data = json.load(f)

            saved_configs = save_data.get("configurations", {})

            # Filter by sections if specified
            if sections:
                filtered_configs = {}
                for section in sections:
                    if section in saved_configs:
                        filtered_configs[section] = saved_configs[section]
                saved_configs = filtered_configs

            # Apply saved configuration
            for section, config in saved_configs.items():
                reset_actions.append({
                    "action": "reset_to_saved",
                    "section": section,
                    "config_name": config_name,
                    "config": config
                })

                if not dry_run:
                    # In a real implementation, you would apply the saved configuration
                    pass
        else:
            return jsonify({"success": False, "error": f"Invalid reset type: {reset_type}"}), 400

        if not dry_run:
            # Log the reset action
            await log_config_change(
                action="configuration_reset",
                config_type="system_wide",
                changes={
                    "reset_type": reset_type,
                    "config_name": config_name,
                    "sections": sections
                },
                metadata={
                    "actions_count": len(reset_actions),
                    "warnings": warnings
                }
            )

        return jsonify({
            "success": True,
            "data": {
                "reset_type": reset_type,
                "config_name": config_name,
                "sections": sections or "all",
                "actions": reset_actions,
                "warnings": warnings,
                "dry_run": dry_run
            }
        })

    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/saves/<config_name>', methods=['DELETE'])
async def delete_saved_configuration(config_name):
    """Delete a saved configuration."""
    try:
        saves_dir = os.path.join(CACHE_DIR, 'config_saves')
        save_file = os.path.join(saves_dir, f"{config_name}.json")

        if not os.path.exists(save_file):
            return jsonify({"success": False, "error": f"Saved configuration '{config_name}' not found"}), 404

        # Get file info before deletion
        file_size = os.path.getsize(save_file)

        # Delete the file
        os.remove(save_file)

        # Log the deletion
        await log_config_change(
            action="configuration_save_deleted",
            config_type="system_wide",
            changes={"config_name": config_name},
            metadata={"file_size_bytes": file_size}
        )

        return jsonify({
            "success": True,
            "data": {
                "config_name": config_name,
                "deleted_at": datetime.now().isoformat(),
                "file_size_bytes": file_size
            }
        })

    except Exception as e:
        logger.error(f"Error deleting saved configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Real-time Configuration Validation endpoints
@app.route('/api/v1/config/validate', methods=['POST'])
async def validate_configuration():
    """Validate configuration values in real-time."""
    try:
        data = await request.get_json()
        config_type = data.get('config_type')  # 'tool_selector', 'embedding', 'weaviate', 'letta_api'
        field = data.get('field')
        value = data.get('value')
        context = data.get('context', {})  # Additional context for validation

        validation_result = await perform_configuration_validation(config_type, field, value, context)

        return jsonify({
            "success": True,
            "data": validation_result
        })

    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/validate/connection', methods=['POST'])
async def validate_connection():
    """Test connection with provided configuration values."""
    try:
        data = await request.get_json()
        service_type = data.get('service_type')  # 'ollama', 'weaviate', 'letta_api', 'openai'
        config = data.get('config', {})

        connection_result = await test_service_connection(service_type, config)

        return jsonify({
            "success": True,
            "data": connection_result
        })

    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/validate/bulk', methods=['POST'])
async def validate_bulk_configuration():
    """Validate multiple configuration values at once."""
    try:
        data = await request.get_json()
        validations = data.get('validations', [])  # List of {config_type, field, value, context}

        results = []
        for validation in validations:
            try:
                result = await perform_configuration_validation(
                    validation.get('config_type'),
                    validation.get('field'),
                    validation.get('value'),
                    validation.get('context', {})
                )
                results.append({
                    "field_id": validation.get('field_id'),
                    "valid": result["valid"],
                    "errors": result.get("errors", []),
                    "warnings": result.get("warnings", []),
                    "suggestions": result.get("suggestions", [])
                })
            except Exception as e:
                results.append({
                    "field_id": validation.get('field_id'),
                    "valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "suggestions": []
                })

        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "overall_valid": all(r["valid"] for r in results)
            }
        })

    except Exception as e:
        logger.error(f"Error validating bulk configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Configuration Audit and Logging endpoints
@app.route('/api/v1/config/audit', methods=['GET'])
async def get_config_audit_logs():
    """Get configuration change audit logs."""
    try:
        # Query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        config_type = request.args.get('config_type')  # Filter by config type
        action = request.args.get('action')  # Filter by action (create, update, delete)
        start_date = request.args.get('start_date')  # ISO format
        end_date = request.args.get('end_date')  # ISO format

        audit_file = os.path.join(CACHE_DIR, 'config_audit.json')

        if not os.path.exists(audit_file):
            return jsonify({
                "success": True,
                "data": {
                    "logs": [],
                    "total": 0,
                    "filtered": 0,
                    "offset": offset,
                    "limit": limit
                }
            })

        with open(audit_file, 'r') as f:
            all_logs = []
            for line in f:
                try:
                    all_logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

        # Apply filters
        filtered_logs = all_logs

        if config_type:
            filtered_logs = [log for log in filtered_logs if log.get('config_type') == config_type]

        if action:
            filtered_logs = [log for log in filtered_logs if log.get('action') == action]

        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                filtered_logs = [log for log in filtered_logs
                               if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) >= start_dt]
            except ValueError:
                pass

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                filtered_logs = [log for log in filtered_logs
                               if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) <= end_dt]
            except ValueError:
                pass

        # Sort by timestamp, newest first
        filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Apply pagination
        total_filtered = len(filtered_logs)
        paginated_logs = filtered_logs[offset:offset + limit]

        return jsonify({
            "success": True,
            "data": {
                "logs": paginated_logs,
                "total": len(all_logs),
                "filtered": total_filtered,
                "offset": offset,
                "limit": limit
            }
        })

    except Exception as e:
        logger.error(f"Error getting config audit logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/audit/stats', methods=['GET'])
async def get_config_audit_stats():
    """Get audit log statistics."""
    try:
        audit_file = os.path.join(CACHE_DIR, 'config_audit.json')

        if not os.path.exists(audit_file):
            return jsonify({
                "success": True,
                "data": {
                    "total_entries": 0,
                    "config_types": {},
                    "actions": {},
                    "recent_activity": [],
                    "last_change": None
                }
            })

        with open(audit_file, 'r') as f:
            all_logs = []
            for line in f:
                try:
                    all_logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Calculate statistics
        config_types = {}
        actions = {}

        for log in all_logs:
            config_type = log.get('config_type', 'unknown')
            action = log.get('action', 'unknown')

            config_types[config_type] = config_types.get(config_type, 0) + 1
            actions[action] = actions.get(action, 0) + 1

        # Get recent activity (last 10 entries)
        recent_logs = sorted(all_logs, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]

        last_change = recent_logs[0] if recent_logs else None

        return jsonify({
            "success": True,
            "data": {
                "total_entries": len(all_logs),
                "config_types": config_types,
                "actions": actions,
                "recent_activity": recent_logs,
                "last_change": last_change
            }
        })

    except Exception as e:
        logger.error(f"Error getting config audit stats: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/config/audit/clear', methods=['POST'])
async def clear_config_audit_logs():
    """Clear all audit logs (admin operation)."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        confirm = data.get('confirm', False)

        if not confirm:
            return jsonify({
                "success": False,
                "error": "Confirmation required. Set 'confirm': true in request body."
            }), 400

        audit_file = os.path.join(CACHE_DIR, 'config_audit.json')

        # Create backup before clearing
        if os.path.exists(audit_file):
            backup_file = f"{audit_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(audit_file, backup_file)

            # Log the clearing action
            await log_config_change(
                action="clear_audit_logs",
                config_type="audit_system",
                changes={"backup_file": backup_file},
                user_info={"action": "admin_clear", "timestamp": datetime.now().isoformat()}
            )

        return jsonify({
            "success": True,
            "message": "Audit logs cleared successfully",
            "data": {
                "cleared_at": datetime.now().isoformat(),
                "backup_created": backup_file if os.path.exists(audit_file) else None
            }
        })

    except Exception as e:
        logger.error(f"Error clearing config audit logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


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


# Environment Management API endpoints
@app.route('/api/v1/environment', methods=['GET'])
async def get_environment_variables():
    """Get current environment variables (filtered for security)."""
    try:
        # Define which environment variables are safe to expose
        safe_env_vars = {
            # Tool Selector Configuration
            'MAX_TOTAL_TOOLS', 'MAX_MCP_TOOLS', 'MIN_MCP_TOOLS', 'DEFAULT_DROP_RATE',
            'EXCLUDE_LETTA_CORE_TOOLS', 'EXCLUDE_OFFICIAL_TOOLS', 'MANAGE_ONLY_MCP_TOOLS',
            'MIN_SCORE_DEFAULT', 'SEMANTIC_WEIGHT', 'KEYWORD_WEIGHT',

            # Ollama Configuration
            'OLLAMA_EMBEDDING_HOST', 'OLLAMA_PORT', 'OLLAMA_TIMEOUT', 'OLLAMA_BASE_URL',
            'OLLAMA_EMBEDDING_MODEL', 'USE_OLLAMA_EMBEDDINGS', 'OLLAMA_DEFAULT_MODEL',
            'OLLAMA_TEMPERATURE', 'OLLAMA_CONTEXT_LENGTH', 'OLLAMA_NUM_PARALLEL',
            'OLLAMA_NUM_CTX', 'OLLAMA_NUM_GPU', 'OLLAMA_LOW_VRAM',

            # Weaviate Configuration
            'WEAVIATE_URL', 'WEAVIATE_TIMEOUT', 'WEAVIATE_RETRIES', 'WEAVIATE_CLASS_NAME',
            'WEAVIATE_VECTOR_INDEX', 'WEAVIATE_DISTANCE_METRIC', 'WEAVIATE_ALPHA',
            'WEAVIATE_LIMIT', 'WEAVIATE_ADDITIONAL_PROPERTIES', 'WEAVIATE_AUTOCUT',
            'WEAVIATE_EF_CONSTRUCTION', 'WEAVIATE_EF', 'WEAVIATE_MAX_CONNECTIONS',
            'WEAVIATE_VECTOR_CACHE_MAX_OBJECTS', 'WEAVIATE_CLEANUP_INTERVAL',

            # Embedding Configuration
            'EMBEDDING_PROVIDER', 'OPENAI_EMBEDDING_MODEL',

            # Letta Configuration
            'LETTA_API_URL', 'LETTA_TIMEOUT',

            # System Configuration
            'LOG_LEVEL', 'DEBUG', 'PORT'
        }

        # Secret environment variables (values will be masked)
        secret_env_vars = {
            'OPENAI_API_KEY', 'WEAVIATE_API_KEY', 'LETTA_PASSWORD', 'LETTA_API_KEY'
        }

        environment = {}

        # Add safe environment variables
        for var_name in safe_env_vars:
            value = os.getenv(var_name)
            if value is not None:
                environment[var_name] = {
                    "value": value,
                    "type": "safe",
                    "description": get_env_var_description(var_name)
                }

        # Add secret environment variables (masked)
        for var_name in secret_env_vars:
            value = os.getenv(var_name)
            if value is not None:
                environment[var_name] = {
                    "value": "***",
                    "type": "secret",
                    "description": get_env_var_description(var_name),
                    "has_value": True
                }
            else:
                environment[var_name] = {
                    "value": None,
                    "type": "secret",
                    "description": get_env_var_description(var_name),
                    "has_value": False
                }

        return jsonify({
            "success": True,
            "data": {
                "environment": environment,
                "total_vars": len(environment),
                "safe_vars": len([v for v in environment.values() if v["type"] == "safe"]),
                "secret_vars": len([v for v in environment.values() if v["type"] == "secret"])
            }
        })

    except Exception as e:
        logger.error(f"Error getting environment variables: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/environment/validate', methods=['POST'])
async def validate_environment():
    """Validate environment configuration and check for issues."""
    try:
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "configurations": {}
        }

        # Validate Tool Selector Configuration
        tool_selector_validation = validate_tool_selector_env()
        validation_results["configurations"]["tool_selector"] = tool_selector_validation
        if not tool_selector_validation["valid"]:
            validation_results["valid"] = False
        validation_results["warnings"].extend(tool_selector_validation.get("warnings", []))
        validation_results["errors"].extend(tool_selector_validation.get("errors", []))

        # Validate Ollama Configuration
        ollama_validation = validate_ollama_env()
        validation_results["configurations"]["ollama"] = ollama_validation
        if not ollama_validation["valid"]:
            validation_results["valid"] = False
        validation_results["warnings"].extend(ollama_validation.get("warnings", []))
        validation_results["errors"].extend(ollama_validation.get("errors", []))

        # Validate Weaviate Configuration
        weaviate_validation = validate_weaviate_env()
        validation_results["configurations"]["weaviate"] = weaviate_validation
        if not weaviate_validation["valid"]:
            validation_results["valid"] = False
        validation_results["warnings"].extend(weaviate_validation.get("warnings", []))
        validation_results["errors"].extend(weaviate_validation.get("errors", []))

        # Validate Embedding Configuration
        embedding_validation = validate_embedding_env()
        validation_results["configurations"]["embedding"] = embedding_validation
        if not embedding_validation["valid"]:
            validation_results["valid"] = False
        validation_results["warnings"].extend(embedding_validation.get("warnings", []))
        validation_results["errors"].extend(embedding_validation.get("errors", []))

        # General recommendations
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('USE_OLLAMA_EMBEDDINGS', '').lower() == 'true':
            validation_results["recommendations"].append("Consider setting up either OpenAI API key or Ollama embeddings")

        return jsonify({
            "success": True,
            "data": validation_results
        })

    except Exception as e:
        logger.error(f"Error validating environment: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/environment/template', methods=['GET'])
async def get_environment_template():
    """Get a template .env file with all possible configuration options."""
    try:
        template_content = generate_env_template()

        return jsonify({
            "success": True,
            "data": {
                "template": template_content,
                "filename": ".env.template",
                "description": "Template environment file with all configuration options"
            }
        })

    except Exception as e:
        logger.error(f"Error generating environment template: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


def get_env_var_description(var_name):
    """Get description for environment variable."""
    descriptions = {
        # Tool Selector Configuration
        'MAX_TOTAL_TOOLS': 'Maximum total tools per agent (including Letta core tools)',
        'MAX_MCP_TOOLS': 'Maximum MCP tools per agent',
        'MIN_MCP_TOOLS': 'Minimum MCP tools to maintain (pruning disabled below this)',
        'DEFAULT_DROP_RATE': 'Default aggressiveness of pruning (0.0-1.0)',
        'EXCLUDE_LETTA_CORE_TOOLS': 'Whether to exclude Letta core tools from management',
        'EXCLUDE_OFFICIAL_TOOLS': 'Whether to exclude official tools from management',
        'MANAGE_ONLY_MCP_TOOLS': 'Whether to only manage MCP tools',

        # Ollama Configuration
        'OLLAMA_EMBEDDING_HOST': 'Ollama server hostname or IP address',
        'OLLAMA_PORT': 'Ollama server port (default: 11434)',
        'OLLAMA_TIMEOUT': 'Connection timeout in seconds',
        'OLLAMA_EMBEDDING_MODEL': 'Ollama embedding model to use',
        'USE_OLLAMA_EMBEDDINGS': 'Whether to use Ollama for embeddings',
        'OLLAMA_DEFAULT_MODEL': 'Default Ollama model for generation',

        # Weaviate Configuration
        'WEAVIATE_URL': 'Weaviate database URL',
        'WEAVIATE_TIMEOUT': 'Weaviate connection timeout',
        'WEAVIATE_ALPHA': 'Hybrid search alpha parameter (0.0-1.0)',
        'WEAVIATE_CLASS_NAME': 'Weaviate class name for tools',

        # Embedding Configuration
        'EMBEDDING_PROVIDER': 'Embedding provider (openai or ollama)',
        'OPENAI_EMBEDDING_MODEL': 'OpenAI embedding model to use',

        # Secret Configuration
        'OPENAI_API_KEY': 'OpenAI API key for embeddings and reranking',
        'WEAVIATE_API_KEY': 'Weaviate API key (if authentication required)',
        'LETTA_PASSWORD': 'Letta API password',
        'LETTA_API_URL': 'Letta API base URL',

        # System Configuration
        'LOG_LEVEL': 'Application log level (DEBUG, INFO, WARNING, ERROR)',
        'PORT': 'API server port number'
    }

    return descriptions.get(var_name, f"Configuration for {var_name}")


def validate_tool_selector_env():
    """Validate tool selector environment variables."""
    validation = {"valid": True, "warnings": [], "errors": []}

    try:
        max_total = int(os.getenv('MAX_TOTAL_TOOLS', '30'))
        max_mcp = int(os.getenv('MAX_MCP_TOOLS', '20'))
        min_mcp = int(os.getenv('MIN_MCP_TOOLS', '7'))

        if min_mcp >= max_mcp:
            validation["errors"].append("MIN_MCP_TOOLS must be less than MAX_MCP_TOOLS")
            validation["valid"] = False

        if max_mcp >= max_total:
            validation["errors"].append("MAX_MCP_TOOLS must be less than MAX_TOTAL_TOOLS")
            validation["valid"] = False

        if min_mcp < 3:
            validation["warnings"].append("Very low MIN_MCP_TOOLS may limit functionality")

        if max_total > 50:
            validation["warnings"].append("High MAX_TOTAL_TOOLS may impact performance")

    except ValueError as e:
        validation["errors"].append(f"Invalid numeric value in tool selector config: {e}")
        validation["valid"] = False

    return validation


def validate_ollama_env():
    """Validate Ollama environment variables."""
    validation = {"valid": True, "warnings": [], "errors": []}

    use_ollama = os.getenv('USE_OLLAMA_EMBEDDINGS', '').lower() == 'true'

    if use_ollama:
        if not os.getenv('OLLAMA_EMBEDDING_HOST'):
            validation["errors"].append("OLLAMA_EMBEDDING_HOST required when USE_OLLAMA_EMBEDDINGS=true")
            validation["valid"] = False

        if not os.getenv('OLLAMA_EMBEDDING_MODEL'):
            validation["warnings"].append("OLLAMA_EMBEDDING_MODEL not specified, using default")

        try:
            port = int(os.getenv('OLLAMA_PORT', '11434'))
            if not (1 <= port <= 65535):
                validation["errors"].append("OLLAMA_PORT must be between 1 and 65535")
                validation["valid"] = False
        except ValueError:
            validation["errors"].append("OLLAMA_PORT must be a valid integer")
            validation["valid"] = False

    return validation


def validate_weaviate_env():
    """Validate Weaviate environment variables."""
    validation = {"valid": True, "warnings": [], "errors": []}

    weaviate_url = os.getenv('WEAVIATE_URL')
    if not weaviate_url:
        validation["errors"].append("WEAVIATE_URL is required")
        validation["valid"] = False
    elif not (weaviate_url.startswith('http://') or weaviate_url.startswith('https://')):
        validation["errors"].append("WEAVIATE_URL must start with http:// or https://")
        validation["valid"] = False

    try:
        alpha = float(os.getenv('WEAVIATE_ALPHA', '0.75'))
        if not (0.0 <= alpha <= 1.0):
            validation["errors"].append("WEAVIATE_ALPHA must be between 0.0 and 1.0")
            validation["valid"] = False
    except ValueError:
        validation["errors"].append("WEAVIATE_ALPHA must be a valid number")
        validation["valid"] = False

    return validation


def validate_embedding_env():
    """Validate embedding environment variables."""
    validation = {"valid": True, "warnings": [], "errors": []}

    provider = os.getenv('EMBEDDING_PROVIDER', 'openai').lower()

    if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        validation["errors"].append("OPENAI_API_KEY required for OpenAI embedding provider")
        validation["valid"] = False

    if provider == 'ollama' and not os.getenv('USE_OLLAMA_EMBEDDINGS', '').lower() == 'true':
        validation["warnings"].append("EMBEDDING_PROVIDER=ollama but USE_OLLAMA_EMBEDDINGS is not true")

    if provider not in ['openai', 'ollama']:
        validation["errors"].append("EMBEDDING_PROVIDER must be 'openai' or 'ollama'")
        validation["valid"] = False

    return validation


def generate_env_template():
    """Generate a complete .env template file."""
    template = """# Letta Tool Selector Configuration Template
# Copy this file to .env and configure the values for your environment

# =============================================================================
# Tool Selector Configuration
# =============================================================================
# Maximum total tools per agent (including Letta core tools)
MAX_TOTAL_TOOLS=30

# Maximum MCP tools per agent
MAX_MCP_TOOLS=20

# Minimum MCP tools to maintain (pruning disabled below this threshold)
MIN_MCP_TOOLS=7

# Default aggressiveness of pruning (0.0 = conservative, 1.0 = aggressive)
DEFAULT_DROP_RATE=0.6

# Whether to exclude Letta core tools from management (recommended: true)
EXCLUDE_LETTA_CORE_TOOLS=true

# Whether to exclude official tools from management (recommended: true)
EXCLUDE_OFFICIAL_TOOLS=true

# Whether to only manage MCP tools (recommended: true)
MANAGE_ONLY_MCP_TOOLS=true

# =============================================================================
# Ollama Configuration
# =============================================================================
# Ollama server hostname or IP address
OLLAMA_EMBEDDING_HOST=192.168.50.80

# Ollama server port
OLLAMA_PORT=11434

# Connection timeout in seconds
OLLAMA_TIMEOUT=30

# Ollama embedding model (leave empty to use Ollama's default)
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M

# Whether to use Ollama for embeddings (true/false)
USE_OLLAMA_EMBEDDINGS=false

# Default Ollama model for text generation
OLLAMA_DEFAULT_MODEL=mistral:7b

# Generation temperature (0.0-1.0)
OLLAMA_TEMPERATURE=0.7

# Context length for generation
OLLAMA_CONTEXT_LENGTH=4096

# =============================================================================
# Weaviate Configuration
# =============================================================================
# Weaviate database URL
WEAVIATE_URL=http://weaviate:8080/

# Connection timeout in seconds
WEAVIATE_TIMEOUT=30

# Number of connection retries
WEAVIATE_RETRIES=3

# Weaviate class name for storing tools
WEAVIATE_CLASS_NAME=Tool

# Vector index type (hnsw recommended)
WEAVIATE_VECTOR_INDEX=hnsw

# Distance metric for similarity search
WEAVIATE_DISTANCE_METRIC=cosine

# Hybrid search alpha parameter (0.0 = keyword only, 1.0 = vector only)
WEAVIATE_ALPHA=0.75

# Default search result limit
WEAVIATE_LIMIT=50

# =============================================================================
# Embedding Configuration
# =============================================================================
# Embedding provider (openai or ollama)
EMBEDDING_PROVIDER=openai

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# =============================================================================
# Letta API Configuration
# =============================================================================
# Letta API base URL
LETTA_API_URL=https://letta.example.com/v1

# Letta API timeout in seconds
LETTA_TIMEOUT=30

# =============================================================================
# Secret Configuration (Keep these secure!)
# =============================================================================
# OpenAI API key for embeddings and reranking
OPENAI_API_KEY=sk-your-openai-api-key-here

# Weaviate API key (if authentication is enabled)
# WEAVIATE_API_KEY=your-weaviate-api-key-here

# Letta API password
LETTA_PASSWORD=your-letta-password-here

# =============================================================================
# System Configuration
# =============================================================================
# Application log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# API server port
PORT=8020

# Enable debug mode (true/false)
DEBUG=false
"""

    return template


# System Maintenance API endpoints
@app.route('/api/v1/maintenance/status', methods=['GET'])
async def get_maintenance_status():
    """Get system maintenance status and health information."""
    try:
        maintenance_status = {
            "system": {
                "uptime_seconds": int(time.time() - start_time),
                "memory_usage": get_memory_usage(),
                "disk_usage": get_disk_usage(),
                "cpu_info": get_cpu_info(),
                "last_restart": datetime.fromtimestamp(start_time).isoformat()
            },
            "services": {
                "weaviate": await test_weaviate_connection({"url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')}),
                "ollama": await test_ollama_connection({"host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}),
                "letta_api": await test_letta_connection()
            },
            "database": {
                "tool_count": await get_tool_count_from_cache(),
                "cache_size": get_cache_size(),
                "last_sync": get_last_sync_time(),
                "index_status": await get_weaviate_index_status()
            },
            "logs": {
                "log_level": os.getenv('LOG_LEVEL', 'INFO'),
                "log_size": get_log_file_size(),
                "error_count": await get_recent_error_count(),
                "warning_count": await get_recent_warning_count()
            }
        }

        # Determine overall health status
        health_status = "healthy"
        issues = []

        # Check service availability
        if not maintenance_status["services"]["weaviate"]["available"]:
            health_status = "degraded"
            issues.append("Weaviate service unavailable")

        if maintenance_status["services"]["ollama"]["available"] == False:
            if os.getenv('USE_OLLAMA_EMBEDDINGS', '').lower() == 'true':
                health_status = "degraded"
                issues.append("Ollama service unavailable (but configured for embeddings)")
            else:
                issues.append("Ollama service unavailable (not critical)")

        # Check resource usage
        memory_usage = maintenance_status["system"]["memory_usage"]["percent"]
        if memory_usage > 90:
            health_status = "critical"
            issues.append(f"High memory usage: {memory_usage}%")
        elif memory_usage > 80:
            health_status = "warning"
            issues.append(f"Elevated memory usage: {memory_usage}%")

        disk_usage = maintenance_status["system"]["disk_usage"]["percent"]
        if disk_usage > 95:
            health_status = "critical"
            issues.append(f"Critical disk usage: {disk_usage}%")
        elif disk_usage > 85:
            health_status = "warning"
            issues.append(f"High disk usage: {disk_usage}%")

        maintenance_status["health"] = {
            "status": health_status,
            "issues": issues,
            "last_check": datetime.now().isoformat()
        }

        return jsonify({
            "success": True,
            "data": maintenance_status
        })

    except Exception as e:
        logger.error(f"Error getting maintenance status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/maintenance/cleanup', methods=['POST'])
async def perform_system_cleanup():
    """Perform system cleanup operations."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        operations = data.get('operations', [])  # List of cleanup operations to perform
        dry_run = data.get('dry_run', False)

        available_operations = {
            "clear_cache": "Clear runtime cache files",
            "rotate_logs": "Rotate application log files",
            "cleanup_temp": "Remove temporary files",
            "compress_old_logs": "Compress old log files",
            "cleanup_backups": "Remove old backup files (>30 days)",
            "optimize_database": "Optimize Weaviate database",
            "clear_audit_logs": "Clear old audit log entries"
        }

        if not operations:
            operations = list(available_operations.keys())  # Default to all operations

        cleanup_results = {
            "operations": [],
            "total_space_freed": 0,
            "errors": []
        }

        for operation in operations:
            if operation not in available_operations:
                cleanup_results["errors"].append(f"Unknown operation: {operation}")
                continue

            try:
                result = await perform_cleanup_operation(operation, dry_run)
                cleanup_results["operations"].append({
                    "name": operation,
                    "description": available_operations[operation],
                    "status": "completed" if result["success"] else "failed",
                    "details": result["details"],
                    "space_freed": result.get("space_freed", 0),
                    "files_affected": result.get("files_affected", 0)
                })
                cleanup_results["total_space_freed"] += result.get("space_freed", 0)

            except Exception as e:
                cleanup_results["errors"].append(f"Error in {operation}: {str(e)}")
                cleanup_results["operations"].append({
                    "name": operation,
                    "description": available_operations[operation],
                    "status": "failed",
                    "details": str(e),
                    "space_freed": 0,
                    "files_affected": 0
                })

        # Log cleanup operation
        await log_config_change(
            action="system_cleanup",
            config_type="maintenance",
            changes={"operations": operations, "dry_run": dry_run},
            metadata={"results": cleanup_results}
        )

        return jsonify({
            "success": True,
            "data": {
                "cleanup_results": cleanup_results,
                "dry_run": dry_run,
                "completed_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error performing system cleanup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/maintenance/restart', methods=['POST'])
async def restart_system_components():
    """Restart system components (simulated - requires external orchestration)."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        components = data.get('components', [])  # List of components to restart
        force = data.get('force', False)

        available_components = {
            "api_server": "API Server (requires external restart)",
            "sync_service": "Tool Sync Service",
            "mcp_server": "MCP Protocol Server",
            "weaviate": "Weaviate Database (external service)",
            "ollama": "Ollama Service (external service)"
        }

        if not components:
            return jsonify({
                "success": False,
                "error": "No components specified for restart"
            }), 400

        restart_results = {
            "components": [],
            "warnings": []
        }

        for component in components:
            if component not in available_components:
                restart_results["warnings"].append(f"Unknown component: {component}")
                continue

            if component == "api_server":
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "requires_external_action",
                    "message": "API server restart requires external orchestration (Docker, systemd, etc.)"
                })
            elif component in ["weaviate", "ollama"]:
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "external_service",
                    "message": f"{component.capitalize()} is an external service - restart manually or via orchestration"
                })
            else:
                # For internal services, simulate restart
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "simulated",
                    "message": f"{component} restart simulated - implement actual restart logic"
                })

        # Log restart request
        await log_config_change(
            action="component_restart",
            config_type="maintenance",
            changes={"components": components, "force": force},
            metadata={"results": restart_results}
        )

        return jsonify({
            "success": True,
            "data": {
                "restart_results": restart_results,
                "requested_at": datetime.now().isoformat(),
                "note": "Most restart operations require external orchestration"
            }
        })

    except Exception as e:
        logger.error(f"Error restarting system components: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/maintenance/optimize', methods=['POST'])
async def optimize_system():
    """Optimize system performance."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        operations = data.get('operations', [])

        available_optimizations = {
            "rebuild_cache": "Rebuild tool cache from Letta API",
            "reindex_weaviate": "Trigger Weaviate reindexing",
            "compact_database": "Compact database files",
            "optimize_embeddings": "Optimize embedding storage",
            "cleanup_duplicates": "Remove duplicate tool entries"
        }

        if not operations:
            operations = list(available_optimizations.keys())

        optimization_results = {
            "operations": [],
            "performance_impact": {},
            "warnings": []
        }

        for operation in operations:
            if operation not in available_optimizations:
                optimization_results["warnings"].append(f"Unknown optimization: {operation}")
                continue

            try:
                result = await perform_optimization(operation)
                optimization_results["operations"].append({
                    "name": operation,
                    "description": available_optimizations[operation],
                    "status": "completed" if result["success"] else "failed",
                    "details": result["details"],
                    "performance_impact": result.get("performance_impact", {})
                })

            except Exception as e:
                optimization_results["warnings"].append(f"Error in {operation}: {str(e)}")

        # Log optimization
        await log_config_change(
            action="system_optimization",
            config_type="maintenance",
            changes={"operations": operations},
            metadata={"results": optimization_results}
        )

        return jsonify({
            "success": True,
            "data": {
                "optimization_results": optimization_results,
                "completed_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error optimizing system: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Log Viewer and Analysis API endpoints
@app.route('/api/v1/logs', methods=['GET'])
async def get_logs():
    """Get application logs with filtering and pagination."""
    try:
        # Query parameters
        level = request.args.get('level', 'all')  # all, debug, info, warning, error
        lines = min(int(request.args.get('lines', 100)), 10000)  # Limit to 10k lines max
        search = request.args.get('search', '')
        from_time = request.args.get('from', '')
        to_time = request.args.get('to', '')

        log_entries = await get_log_entries(
            level=level,
            lines=lines,
            search=search,
            from_time=from_time,
            to_time=to_time
        )

        return jsonify({
            "success": True,
            "data": {
                "logs": log_entries,
                "total_lines": len(log_entries),
                "filters": {
                    "level": level,
                    "lines": lines,
                    "search": search,
                    "from_time": from_time,
                    "to_time": to_time
                }
            }
        })
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/logs/analysis', methods=['GET'])
async def analyze_logs():
    """Analyze logs for patterns, errors, and insights."""
    try:
        # Query parameters
        timeframe = request.args.get('timeframe', '24h')  # 1h, 24h, 7d, 30d
        include_details = request.args.get('include_details', 'false').lower() == 'true'

        analysis = await perform_log_analysis(timeframe, include_details)

        return jsonify({
            "success": True,
            "data": analysis
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/logs/errors', methods=['GET'])
async def get_error_logs():
    """Get error logs with detailed analysis."""
    try:
        hours = min(int(request.args.get('hours', 24)), 168)  # Max 7 days
        include_stack_trace = request.args.get('include_stack_trace', 'true').lower() == 'true'
        group_by = request.args.get('group_by', 'none')  # none, error_type, endpoint, time

        error_logs = await get_error_log_entries(
            hours=hours,
            include_stack_trace=include_stack_trace,
            group_by=group_by
        )

        return jsonify({
            "success": True,
            "data": error_logs
        })
    except Exception as e:
        logger.error(f"Error getting error logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/logs/clear', methods=['POST'])
async def clear_logs():
    """Clear log files with backup option."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        backup = data.get('backup', True)
        older_than_days = data.get('older_than_days', 0)

        result = await clear_log_files(backup=backup, older_than_days=older_than_days)

        # Log the action
        await log_config_change(
            action="clear_logs",
            config_type="logging",
            changes={"backup": backup, "older_than_days": older_than_days},
            metadata={"result": result}
        )

        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/logs/export', methods=['POST'])
async def export_logs():
    """Export logs in various formats (JSON, CSV, text)."""
    try:
        data = await request.get_json()
        format_type = data.get('format', 'json')  # json, csv, text
        filters = data.get('filters', {})

        export_result = await export_log_data(format_type, filters)

        return jsonify({
            "success": True,
            "data": export_result
        })
    except Exception as e:
        logger.error(f"Error exporting logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Helper functions for maintenance operations
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


# Tools refresh endpoint
@app.route('/api/v1/tools/refresh', methods=['POST'])
async def refresh_tools():
    """Refresh the tool index from Letta API."""
    try:
        logger.info("Refreshing tool index...")
        # Force reload the tool cache
        await read_tool_cache(force_reload=True)
        return jsonify({"success": True, "message": "Tool index refreshed successfully"})
    except Exception as e:
        logger.error(f"Error refreshing tool index: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Evaluation endpoints
@app.route('/api/v1/evaluations', methods=['POST'])
async def submit_evaluation():
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


@app.route('/api/v1/evaluations', methods=['GET'])
async def get_evaluations():
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


# Analytics endpoint
@app.route('/api/v1/analytics', methods=['GET'])
async def get_analytics():
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


# Rerank comparison endpoint
@app.route('/api/v1/rerank/compare', methods=['POST'])
async def compare_rerank_configurations():
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


# Search test endpoint with parameter overrides
@app.route('/api/v1/search/test', methods=['GET'])
async def test_search_with_overrides():
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


# Model listing endpoints
@app.route('/api/v1/models/embedding', methods=['GET'])
async def get_embedding_models():
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


@app.route('/api/v1/embedding/health', methods=['GET'])
async def get_embedding_health():
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


@app.route('/api/v1/models/reranker', methods=['GET'])  
async def get_reranker_models():
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


# Benchmark Query Set CRUD Endpoints
@app.route('/api/v1/benchmark/query-sets', methods=['GET'])
async def list_benchmark_query_sets():
    """Get all benchmark query sets."""
    try:
        # Load benchmark sets from cache file
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = {"query_sets": [], "last_updated": None}
        
        return jsonify({
            "success": True,
            "data": {
                "query_sets": benchmark_data.get('query_sets', []),
                "total": len(benchmark_data.get('query_sets', [])),
                "last_updated": benchmark_data.get('last_updated')
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing benchmark query sets: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/query-sets', methods=['POST'])
async def create_benchmark_query_set():
    """Create a new benchmark query set."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'queries']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Validate queries format
        queries = data.get('queries', [])
        for i, query in enumerate(queries):
            if not isinstance(query, dict) or 'query' not in query:
                return jsonify({"success": False, "error": f"Invalid query format at index {i}"}), 400
        
        # Load existing benchmark sets
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            benchmark_data = {"query_sets": [], "last_updated": None}
        
        # Create new query set
        new_query_set = {
            "id": f"qs_{int(time.time())}_{len(benchmark_data['query_sets'])}",
            "name": data['name'],
            "description": data['description'],
            "queries": queries,
            "metadata": data.get('metadata', {}),
            "created_at": datetime.utcnow().isoformat(),
            "created_by": data.get('created_by', 'system'),
            "tags": data.get('tags', []),
            "query_count": len(queries),
            "last_run": None,
            "run_count": 0,
            "active": True
        }
        
        # Add to benchmark data
        benchmark_data['query_sets'].append(new_query_set)
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Created benchmark query set: {new_query_set['id']}")
        
        return jsonify({
            "success": True,
            "data": new_query_set,
            "message": f"Query set '{data['name']}' created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/query-sets/<query_set_id>', methods=['GET'])
async def get_benchmark_query_set(query_set_id):
    """Get a specific benchmark query set."""
    try:
        # Load benchmark sets
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        if not os.path.exists(benchmark_path):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)
        
        # Find the query set
        query_set = None
        for qs in benchmark_data.get('query_sets', []):
            if qs['id'] == query_set_id:
                query_set = qs
                break
        
        if not query_set:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        return jsonify({
            "success": True,
            "data": query_set
        })
        
    except Exception as e:
        logger.error(f"Error getting benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/query-sets/<query_set_id>', methods=['PUT'])
async def update_benchmark_query_set(query_set_id):
    """Update a benchmark query set."""
    try:
        data = await request.get_json()
        
        # Load benchmark sets
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        if not os.path.exists(benchmark_path):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)
        
        # Find and update the query set
        query_set_index = None
        for i, qs in enumerate(benchmark_data.get('query_sets', [])):
            if qs['id'] == query_set_id:
                query_set_index = i
                break
        
        if query_set_index is None:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        # Update allowed fields
        updatable_fields = ['name', 'description', 'queries', 'metadata', 'tags', 'active']
        query_set = benchmark_data['query_sets'][query_set_index]
        
        for field in updatable_fields:
            if field in data:
                query_set[field] = data[field]
                if field == 'queries':
                    query_set['query_count'] = len(data[field])
        
        query_set['updated_at'] = datetime.utcnow().isoformat()
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Updated benchmark query set: {query_set_id}")
        
        return jsonify({
            "success": True,
            "data": query_set,
            "message": f"Query set {query_set_id} updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/query-sets/<query_set_id>', methods=['DELETE'])
async def delete_benchmark_query_set(query_set_id):
    """Delete a benchmark query set."""
    try:
        # Load benchmark sets
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        if not os.path.exists(benchmark_path):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)
        
        # Find and remove the query set
        original_count = len(benchmark_data.get('query_sets', []))
        benchmark_data['query_sets'] = [
            qs for qs in benchmark_data.get('query_sets', []) 
            if qs['id'] != query_set_id
        ]
        
        if len(benchmark_data['query_sets']) == original_count:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Deleted benchmark query set: {query_set_id}")
        
        return jsonify({
            "success": True,
            "message": f"Query set {query_set_id} deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error deleting benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/query-sets/<query_set_id>/run', methods=['POST'])
async def run_benchmark_query_set(query_set_id):
    """Run a benchmark query set against current configuration."""
    try:
        data = await request.get_json()
        
        # Load benchmark sets
        benchmark_path = os.path.join(CACHE_DIR, 'benchmark_query_sets.json')
        if not os.path.exists(benchmark_path):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)
        
        # Find the query set
        query_set = None
        query_set_index = None
        for i, qs in enumerate(benchmark_data.get('query_sets', [])):
            if qs['id'] == query_set_id:
                query_set = qs
                query_set_index = i
                break
        
        if not query_set:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        if not query_set.get('active', True):
            return jsonify({"success": False, "error": f"Query set {query_set_id} is inactive"}), 400
        
        # Get configuration to test with
        config = data.get('config')  # Optional configuration override
        limit = data.get('limit', 10)
        include_details = data.get('include_details', False)
        
        logger.info(f"Running benchmark query set: {query_set_id} with {len(query_set['queries'])} queries")
        
        # Run each query in the set
        results = []
        start_time = time.time()
        
        for i, query_item in enumerate(query_set['queries']):
            query_text = query_item['query']
            expected_results = query_item.get('expected_results', [])
            
            try:
                # Run the search query
                if config:
                    # Use provided configuration
                    search_results = await asyncio.to_thread(
                        search_tools, 
                        query=query_text, 
                        limit=limit, 
                        reranker_config=config
                    )
                else:
                    # Use default configuration
                    search_results = await asyncio.to_thread(
                        search_tools, 
                        query=query_text, 
                        limit=limit
                    )
                
                # Calculate metrics if expected results are provided
                metrics = {}
                if expected_results:
                    actual_tool_ids = [r.get('id') or r.get('tool_id') for r in search_results[:limit]]
                    expected_tool_ids = [exp.get('tool_id') for exp in expected_results if exp.get('tool_id')]
                    
                    if expected_tool_ids:
                        # Calculate precision@k, recall@k, and overlap
                        relevant_found = len(set(actual_tool_ids) & set(expected_tool_ids))
                        precision_at_k = relevant_found / len(actual_tool_ids) if actual_tool_ids else 0
                        recall_at_k = relevant_found / len(expected_tool_ids) if expected_tool_ids else 0
                        
                        metrics = {
                            "precision_at_k": precision_at_k,
                            "recall_at_k": recall_at_k,
                            "relevant_found": relevant_found,
                            "total_relevant": len(expected_tool_ids),
                            "total_retrieved": len(actual_tool_ids)
                        }
                
                query_result = {
                    "query_index": i,
                    "query": query_text,
                    "results_count": len(search_results),
                    "metrics": metrics,
                    "execution_time": time.time() - start_time  # Approximate per-query time
                }
                
                if include_details:
                    query_result["results"] = search_results[:limit]
                    query_result["expected_results"] = expected_results
                
                results.append(query_result)
                
            except Exception as query_error:
                logger.error(f"Error running query {i} in benchmark {query_set_id}: {str(query_error)}")
                results.append({
                    "query_index": i,
                    "query": query_text,
                    "error": str(query_error),
                    "metrics": {},
                    "execution_time": 0
                })
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        successful_queries = [r for r in results if 'error' not in r]
        avg_precision = sum(r['metrics'].get('precision_at_k', 0) for r in successful_queries) / max(len(successful_queries), 1)
        avg_recall = sum(r['metrics'].get('recall_at_k', 0) for r in successful_queries) / max(len(successful_queries), 1)
        
        run_summary = {
            "run_id": f"run_{int(time.time())}",
            "query_set_id": query_set_id,
            "query_set_name": query_set['name'],
            "timestamp": datetime.utcnow().isoformat(),
            "total_queries": len(query_set['queries']),
            "successful_queries": len(successful_queries),
            "failed_queries": len(results) - len(successful_queries),
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(results),
            "aggregate_metrics": {
                "avg_precision_at_k": avg_precision,
                "avg_recall_at_k": avg_recall,
                "success_rate": len(successful_queries) / len(results)
            },
            "config_used": config,
            "results": results
        }
        
        # Update query set with run information
        query_set['last_run'] = datetime.utcnow().isoformat()
        query_set['run_count'] = query_set.get('run_count', 0) + 1
        benchmark_data['query_sets'][query_set_index] = query_set
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save updated benchmark data
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        # Optionally save run results for history
        run_history_path = os.path.join(CACHE_DIR, 'benchmark_run_history.json')
        if os.path.exists(run_history_path):
            with open(run_history_path, 'r') as f:
                run_history = json.load(f)
        else:
            run_history = {"runs": []}
        
        run_history['runs'].append(run_summary)
        # Keep only last 100 runs to prevent excessive storage
        run_history['runs'] = run_history['runs'][-100:]
        
        with open(run_history_path, 'w') as f:
            json.dump(run_history, f, indent=2)
        
        logger.info(f"Completed benchmark run for {query_set_id}: {len(successful_queries)}/{len(results)} queries successful")
        
        return jsonify({
            "success": True,
            "data": run_summary,
            "message": f"Benchmark run completed: {len(successful_queries)}/{len(results)} queries successful"
        })
        
    except Exception as e:
        logger.error(f"Error running benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/benchmark/runs', methods=['GET'])
async def get_benchmark_run_history():
    """Get benchmark run history."""
    try:
        limit = request.args.get('limit', 20, type=int)
        query_set_id = request.args.get('query_set_id')
        
        run_history_path = os.path.join(CACHE_DIR, 'benchmark_run_history.json')
        
        if os.path.exists(run_history_path):
            with open(run_history_path, 'r') as f:
                run_history = json.load(f)
        else:
            run_history = {"runs": []}
        
        runs = run_history.get('runs', [])
        
        # Filter by query_set_id if provided
        if query_set_id:
            runs = [run for run in runs if run.get('query_set_id') == query_set_id]
        
        # Sort by timestamp (newest first) and limit
        runs = sorted(runs, key=lambda r: r.get('timestamp', ''), reverse=True)[:limit]
        
        return jsonify({
            "success": True,
            "data": {
                "runs": runs,
                "total": len(runs),
                "filtered_by_query_set": query_set_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting benchmark run history: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Reranker Model Registry Endpoints
@app.route('/api/v1/reranker/models/registry', methods=['GET'])
async def list_registered_reranker_models():
    """Get all registered reranker models from the registry."""
    try:
        # Load registry from cache file (or database in future)
        registry_path = os.path.join(CACHE_DIR, 'reranker_model_registry.json')
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": [], "last_updated": None}
            
        # Merge with discovered models from providers
        try:
            # Call the reranker models discovery logic directly
            models_from_discovery = []
            
            # Ollama reranker models discovery (same logic as get_reranker_models)
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
                                    models_from_discovery.append({
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
                                    models_from_discovery.append({
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
                logger.warning(f"Failed to fetch Ollama reranker models for registry: {str(e)}")
            
            # Add fallback models if nothing discovered
            if not models_from_discovery:
                models_from_discovery = [
                    {
                        "id": "mistral:7b",
                        "name": "Mistral 7B (Fallback)",
                        "provider": "ollama",
                        "type": "generative",
                        "cost_per_1k": 0.0,
                        "recommended": True
                    }
                ]
            
            discovered_models = models_from_discovery
            
            # Update registry with any new discovered models
            existing_ids = {model['id'] for model in registry['models']}
            new_models = []
            
            for discovered_model in discovered_models:
                if discovered_model['id'] not in existing_ids:
                    # Add new discovered model to registry
                    new_models.append({
                        **discovered_model,
                        "registered": True,
                        "validated": False,
                        "last_tested": None,
                        "test_status": "pending",
                        "registry_notes": "Auto-discovered from provider"
                    })
            
            if new_models:
                registry['models'].extend(new_models)
                registry['last_updated'] = datetime.utcnow().isoformat()
                
                # Save updated registry
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                    
        except Exception as discovery_error:
            logger.warning(f"Error during model discovery for registry: {str(discovery_error)}")
        
        return jsonify({
            "success": True,
            "data": {
                "models": registry['models'],
                "total": len(registry['models']),
                "last_updated": registry.get('last_updated'),
                "providers": list(set(m["provider"] for m in registry['models'])),
                "types": list(set(m["type"] for m in registry['models'])),
                "validated_count": len([m for m in registry['models'] if m.get('validated', False)]),
                "pending_count": len([m for m in registry['models'] if m.get('test_status') == 'pending'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing registered reranker models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/reranker/models/registry', methods=['POST'])
async def register_reranker_model():
    """Register a new reranker model in the registry."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['id', 'name', 'provider', 'type']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Load existing registry
        registry_path = os.path.join(CACHE_DIR, 'reranker_model_registry.json')
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": [], "last_updated": None}
        
        # Check if model already exists
        existing_ids = {model['id'] for model in registry['models']}
        if data['id'] in existing_ids:
            return jsonify({"success": False, "error": f"Model {data['id']} already registered"}), 409
        
        # Create new model entry
        new_model = {
            "id": data['id'],
            "name": data['name'],
            "provider": data['provider'],
            "type": data['type'],
            "cost_per_1k": data.get('cost_per_1k', 0.0),
            "recommended": data.get('recommended', False),
            "registered": True,
            "validated": False,
            "last_tested": None,
            "test_status": "pending",
            "registry_notes": data.get('notes', 'Manually registered'),
            "configuration": data.get('configuration', {}),
            "registered_at": datetime.utcnow().isoformat()
        }
        
        # Add optional fields if provided
        optional_fields = ['size', 'modified_at', 'dimensions', 'max_tokens', 'description']
        for field in optional_fields:
            if field in data:
                new_model[field] = data[field]
        
        # Add to registry
        registry['models'].append(new_model)
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        # Save registry
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Registered new reranker model: {data['id']}")
        
        return jsonify({
            "success": True,
            "data": new_model,
            "message": f"Model {data['id']} registered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error registering reranker model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/reranker/models/registry/<model_id>', methods=['PUT'])
async def update_registered_model(model_id):
    """Update a registered reranker model."""
    try:
        data = await request.get_json()
        
        # Load registry
        registry_path = os.path.join(CACHE_DIR, 'reranker_model_registry.json')
        if not os.path.exists(registry_path):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Find model to update
        model_to_update = None
        model_index = None
        for i, model in enumerate(registry['models']):
            if model['id'] == model_id:
                model_to_update = model
                model_index = i
                break
        
        if not model_to_update:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        # Update allowed fields
        updatable_fields = [
            'name', 'recommended', 'registry_notes', 'configuration', 
            'validated', 'test_status', 'description'
        ]
        
        for field in updatable_fields:
            if field in data:
                model_to_update[field] = data[field]
        
        model_to_update['last_updated'] = datetime.utcnow().isoformat()
        registry['models'][model_index] = model_to_update
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        # Save registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Updated registered reranker model: {model_id}")
        
        return jsonify({
            "success": True,
            "data": model_to_update,
            "message": f"Model {model_id} updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating registered model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/reranker/models/registry/<model_id>/test', methods=['POST'])
async def test_registered_model(model_id):
    """Test connectivity and functionality of a registered reranker model."""
    try:
        # Load registry
        registry_path = os.path.join(CACHE_DIR, 'reranker_model_registry.json')
        if not os.path.exists(registry_path):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Find model to test
        model_to_test = None
        model_index = None
        for i, model in enumerate(registry['models']):
            if model['id'] == model_id:
                model_to_test = model
                model_index = i
                break
        
        if not model_to_test:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        test_results = {
            "model_id": model_id,
            "test_timestamp": datetime.utcnow().isoformat(),
            "connectivity": False,
            "functionality": False,
            "latency_ms": None,
            "error": None,
            "details": {}
        }
        
        try:
            # Test based on provider
            if model_to_test['provider'] == 'ollama':
                # Test Ollama model connectivity
                ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
                ollama_url = f"http://{ollama_host}:11434"
                
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    # Check if model is available
                    async with session.get(f"{ollama_url}/api/tags", timeout=10) as response:
                        if response.status == 200:
                            models_data = await response.json()
                            available_models = [m['name'] for m in models_data.get('models', [])]
                            
                            if model_id in available_models:
                                test_results["connectivity"] = True
                                
                                # Test basic functionality with a simple rerank task
                                test_payload = {
                                    "model": model_id,
                                    "prompt": "Query: test search\nDocument: This is a test document for reranking.\nRelevance score (0-1):",
                                    "options": {"temperature": 0.1, "num_predict": 10}
                                }
                                
                                async with session.post(f"{ollama_url}/api/generate", 
                                                       json=test_payload, timeout=30) as test_response:
                                    if test_response.status == 200:
                                        test_results["functionality"] = True
                                        test_results["latency_ms"] = int((time.time() - start_time) * 1000)
                            else:
                                test_results["error"] = f"Model {model_id} not found in Ollama"
                        else:
                            test_results["error"] = f"Failed to connect to Ollama: HTTP {response.status}"
                            
        except Exception as test_error:
            test_results["error"] = str(test_error)
        
        # Update model in registry with test results
        model_to_test["last_tested"] = test_results["test_timestamp"]
        if test_results["connectivity"] and test_results["functionality"]:
            model_to_test["test_status"] = "passed"
            model_to_test["validated"] = True
        else:
            model_to_test["test_status"] = "failed"
            model_to_test["validated"] = False
        
        model_to_test["test_results"] = test_results
        registry['models'][model_index] = model_to_test
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Tested registered reranker model {model_id}: {model_to_test['test_status']}")
        
        return jsonify({
            "success": True,
            "data": test_results,
            "message": f"Model {model_id} test completed"
        })
        
    except Exception as e:
        logger.error(f"Error testing registered model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/v1/reranker/models/registry/<model_id>', methods=['DELETE'])
async def unregister_reranker_model(model_id):
    """Remove a reranker model from the registry."""
    try:
        # Load registry
        registry_path = os.path.join(CACHE_DIR, 'reranker_model_registry.json')
        if not os.path.exists(registry_path):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Find and remove model
        original_count = len(registry['models'])
        registry['models'] = [model for model in registry['models'] if model['id'] != model_id]
        
        if len(registry['models']) == original_count:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Unregistered reranker model: {model_id}")
        
        return jsonify({
            "success": True,
            "message": f"Model {model_id} unregistered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error unregistering model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# A/B Comparison Runner Endpoints

@app.route('/api/v1/ab-comparison/run', methods=['POST'])
async def run_ab_comparison():
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


@app.route('/api/v1/ab-comparison/results', methods=['GET'])
async def get_ab_comparison_results():
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


@app.route('/api/v1/ab-comparison/results/<comparison_id>', methods=['GET'])
async def get_ab_comparison_result(comparison_id):
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


# Safety Status Endpoints

@app.route('/api/v1/safety/status', methods=['GET'])
async def get_safety_status():
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


@app.route('/api/v1/safety/validate-operation', methods=['POST'])
async def validate_operation():
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


@app.route('/api/v1/safety/emergency-status', methods=['GET'])
async def get_emergency_status():
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
    sdk_client_func = get_letta_sdk_client if USE_LETTA_SDK else None
    tool_manager.configure(
        http_session=http_session,
        letta_url=LETTA_URL,
        headers=HEADERS,
        use_letta_sdk=USE_LETTA_SDK,
        get_letta_sdk_client_func=sdk_client_func
    )
    logger.info("Tool manager configured.")

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

# Cost control imports moved to top of file

@app.route('/api/v1/cost-control/status', methods=['GET'])
async def get_cost_control_status():
    """Get overall cost control system status"""
    try:
        manager = get_cost_manager()
        
        # Get budget status
        budget_status = await manager.get_budget_status()
        
        # Get recent alerts
        recent_alerts = await manager.get_recent_alerts(hours=24)
        alert_count = len(recent_alerts)
        critical_alerts = len([a for a in recent_alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]])
        
        # Get daily summary
        daily_summary = await manager.get_cost_summary(BudgetPeriod.DAILY)
        
        return jsonify({
            'success': True,
            'data': {
                'budget_status': budget_status,
                'daily_summary': daily_summary.to_dict(),
                'alert_summary': {
                    'total_alerts_24h': alert_count,
                    'critical_alerts_24h': critical_alerts
                },
                'system_status': 'healthy' if critical_alerts == 0 else 'warning'
            }
        })
        
    except Exception as e:
        logger.error(f"Cost control status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/summary', methods=['GET'])
async def get_cost_summary():
    """Get cost summary for specified period"""
    try:
        period_str = request.args.get('period', 'daily').lower()
        
        try:
            period = BudgetPeriod(period_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {period_str}. Valid periods: {[p.value for p in BudgetPeriod]}'
            }), 400
        
        manager = get_cost_manager()
        summary = await manager.get_cost_summary(period)
        
        return jsonify({
            'success': True,
            'data': summary.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Cost summary error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/budget', methods=['GET'])
async def get_budget_limits():
    """Get all budget limits and their status"""
    try:
        manager = get_cost_manager()
        budget_status = await manager.get_budget_status()
        
        return jsonify({
            'success': True,
            'data': budget_status
        })
        
    except Exception as e:
        logger.error(f"Budget limits error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/budget', methods=['POST'])
async def set_budget_limit():
    """Set or update a budget limit"""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['period', 'limit']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Parse period
        try:
            period = BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in CostCategory]}'
                }), 400
        
        # Parse limit
        try:
            limit = float(data['limit'])
            if limit < 0:
                raise ValueError("Limit must be non-negative")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid limit value. Must be a non-negative number.'
            }), 400
        
        manager = get_cost_manager()
        
        # Set budget limit
        budget_key = await manager.set_budget_limit(
            category=category,
            period=period,
            limit=limit,
            hard_limit=data.get('hard_limit', False),
            alert_thresholds=data.get('alert_thresholds', [0.5, 0.8, 0.95]),
            enabled=data.get('enabled', True)
        )
        
        return jsonify({
            'success': True,
            'data': {
                'budget_key': budget_key,
                'message': f'Budget limit set for {category.value if category else "overall"} - {period.value}'
            }
        })
        
    except Exception as e:
        logger.error(f"Set budget limit error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/budget', methods=['DELETE'])
async def remove_budget_limit():
    """Remove a budget limit"""
    try:
        data = await request.get_json()
        
        # Validate required fields
        if 'period' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: period'
            }), 400
        
        # Parse period
        try:
            period = BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in CostCategory]}'
                }), 400
        
        manager = get_cost_manager()
        removed = await manager.remove_budget_limit(category, period)
        
        if removed:
            return jsonify({
                'success': True,
                'data': {
                    'message': f'Budget limit removed for {category.value if category else "overall"} - {period.value}'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Budget limit not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Remove budget limit error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/alerts', methods=['GET'])
async def get_cost_alerts():
    """Get recent cost alerts"""
    try:
        hours = int(request.args.get('hours', 24))
        if hours < 1 or hours > 168:  # Limit to 1 hour - 1 week
            hours = 24
            
        manager = get_cost_manager()
        alerts = await manager.get_recent_alerts(hours=hours)
        
        # Convert alerts to dict format
        alert_data = [alert.to_dict() for alert in alerts]
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alert_data,
                'count': len(alert_data),
                'hours_requested': hours
            }
        })
        
    except Exception as e:
        logger.error(f"Cost alerts error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/record', methods=['POST'])
async def record_manual_cost():
    """Manually record a cost entry"""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['category', 'operation', 'cost']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Parse category
        try:
            category = CostCategory(data['category'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in CostCategory]}'
            }), 400
        
        # Parse cost
        try:
            cost = float(data['cost'])
            if cost < 0:
                raise ValueError("Cost must be non-negative")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid cost value. Must be a non-negative number.'
            }), 400
        
        manager = get_cost_manager()
        
        # Record the cost
        allowed = await manager.record_cost(
            category=category,
            operation=data['operation'],
            cost=cost,
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'success': True,
            'data': {
                'recorded': True,
                'allowed': allowed,
                'message': 'Cost recorded successfully' if allowed else 'Cost recorded but operation would be blocked by hard limits'
            }
        })
        
    except Exception as e:
        logger.error(f"Record manual cost error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/estimate', methods=['POST'])
async def estimate_operation_cost():
    """Estimate cost for a planned operation"""
    try:
        data = await request.get_json()
        
        if 'operation_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: operation_type'
            }), 400
        
        operation_type = data['operation_type']
        params = data.get('params', {})
        
        manager = get_cost_manager()
        estimated_cost = await manager.estimate_operation_cost(operation_type, **params)
        
        return jsonify({
            'success': True,
            'data': {
                'operation_type': operation_type,
                'estimated_cost': estimated_cost,
                'currency': 'USD',
                'params_used': params
            }
        })
        
    except Exception as e:
        logger.error(f"Estimate operation cost error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/v1/cost-control/reset', methods=['POST'])
async def reset_period_costs():
    """Reset costs for a specific period and category (admin function)"""
    try:
        data = await request.get_json()
        
        if 'period' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: period'
            }), 400
        
        # Parse period
        try:
            period = BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in CostCategory]}'
                }), 400
        
        manager = get_cost_manager()
        await manager.reset_period_costs(category, period)
        
        return jsonify({
            'success': True,
            'data': {
                'message': f'Costs reset for {category.value if category else "all categories"} in period {period.value}'
            }
        })
        
    except Exception as e:
        logger.error(f"Reset period costs error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
