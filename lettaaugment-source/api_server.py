from quart import Quart, request, jsonify
# Restore search_tools import, remove get_all_tools as cache is used for listing
from weaviate_tool_search_with_reranking import search_tools, init_client as init_weaviate_client
from weaviate_client_manager import get_client_manager, close_client_manager, weaviate_client
from weaviate_tool_search import get_embedding_for_text, get_tool_embedding_by_id # Keep some functions from original
from upload_tools_to_weaviate import upload_tools
import os
import requests
import asyncio
import aiohttp
import aiofiles # Import aiofiles
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
import json # Added json import
import time # Need time for cache timeout check
import math # For cosine similarity and math.floor
import uuid # For generating comparison IDs
from concurrent.futures import ThreadPoolExecutor
from hypercorn.config import Config
from hypercorn.asyncio import serve
from simple_config_validation import validate_configuration
from bm25_vector_overrides import bm25_vector_override_service
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Quart(__name__)
# Load .env file - try container path first, then current directory
if os.path.exists('/app/.env'):
    load_dotenv('/app/.env')
else:
    load_dotenv()

LETTA_URL = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1').replace('http://', 'https://')
if not LETTA_URL.endswith('/v1'):
    LETTA_URL = LETTA_URL.rstrip('/') + '/v1'

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
NEVER_DETACH_TOOLS = [name.strip() for name in os.getenv('NEVER_DETACH_TOOLS', 'find_tools').split(',') if name.strip()]

# Default minimum score threshold for tool attachment (0-100)
DEFAULT_MIN_SCORE = float(os.getenv('DEFAULT_MIN_SCORE', '35.0'))

logger.info(f"Tool management configuration:")
logger.info(f"  MAX_TOTAL_TOOLS: {MAX_TOTAL_TOOLS}")
logger.info(f"  MAX_MCP_TOOLS: {MAX_MCP_TOOLS}")
logger.info(f"  DEFAULT_MIN_SCORE: {DEFAULT_MIN_SCORE}")
logger.info(f"  EXCLUDE_LETTA_CORE_TOOLS: {EXCLUDE_LETTA_CORE_TOOLS}")
logger.info(f"  EXCLUDE_OFFICIAL_TOOLS: {EXCLUDE_OFFICIAL_TOOLS}")
logger.info(f"  MANAGE_ONLY_MCP_TOOLS: {MANAGE_ONLY_MCP_TOOLS}")
logger.info(f"  NEVER_DETACH_TOOLS: {NEVER_DETACH_TOOLS}")

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

async def detach_tool(agent_id: str, tool_id: str):
    """Detach a single tool asynchronously using the global session"""
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for detach_tool (agent: {agent_id}, tool: {tool_id})")
        return {"success": False, "tool_id": tool_id, "error": "HTTP session not available"}
    try:
        detach_url = f"{LETTA_URL}/agents/{agent_id}/tools/detach/{tool_id}"

        # Add timeout to prevent hanging requests
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout

        async with http_session.patch(detach_url, headers=HEADERS, timeout=timeout) as response:
            try:
                response_data = await response.json()
            except aiohttp.ContentTypeError: # More specific exception
                # Handle case where response is not JSON
                response_text = await response.text()
                logger.warning(f"Non-JSON response from detach endpoint: {response_text}")
                response_data = {"text": response_text}

            if response.status == 200:
                # logger.info(f"Successfully detached tool {tool_id}")
                return {"success": True, "tool_id": tool_id}
            elif response.status == 404:
                # Tool might already be detached or doesn't exist
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

async def attach_tool(agent_id: str, tool: dict):
    """Attach a single tool asynchronously using the global session"""
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for attach_tool (agent: {agent_id})")
        return {"success": False, "tool_id": tool.get('tool_id') or tool.get('id'), "name": tool.get('name', 'Unknown'), "error": "HTTP session not available"}
    try:
        tool_name = tool.get('name', 'Unknown')
        tool_id = tool.get('tool_id') or tool.get('id')
        if not tool_id:
            logger.error(f"No tool ID found for tool {tool_name}")
            return {"success": False, "tool_id": None, "name": tool_name, "error": "No tool ID available"}

        # logger.info(f"Attempting to attach tool {tool_name} ({tool_id}) to agent {agent_id}")
        attach_url = f"{LETTA_URL}/agents/{agent_id}/tools/attach/{tool_id}"
        async with http_session.patch(attach_url, headers=HEADERS) as response:
            if response.status == 200:
                return {
                    "success": True,
                    "tool_id": tool_id,
                    "name": tool.get("name"),
                    # Calculate score based on distance if available from search_tools result
                    "match_score": 100 * (1 - tool.get("distance", 0)) if "distance" in tool else 100
                }
            else:
                logger.error(f"Failed to attach tool {tool_id}: HTTP {response.status}")
                return {"success": False, "tool_id": tool_id, "name": tool.get("name")}
    except Exception as e:
        logger.error(f"Error attaching tool {tool_id}: {str(e)}")
        return {"success": False, "tool_id": tool_id, "name": tool.get("name")}

async def process_tools(agent_id: str, mcp_tools: list, matching_tools: list, keep_tools: list = None):
    """Process tool detachments and attachments in parallel using the global session"""
    keep_tools = keep_tools or []
    logger.info(f"Processing tools for agent {agent_id}")
    logger.info(f"Current unique MCP tools: {len(mcp_tools)}")
    logger.info(f"Tools to attach: {len(matching_tools)}")
    logger.info(f"Tools to keep: {len(keep_tools)}")

    # Get MIN_MCP_TOOLS from environment
    MIN_MCP_TOOLS = int(os.getenv('MIN_MCP_TOOLS', '7'))
    
    # Create a set of tool IDs to keep (including the ones we're about to attach)
    keep_tool_ids = set()
    # Add explicitly kept tools
    for tool_id in keep_tools:
        if tool_id:
            keep_tool_ids.add(tool_id)
    # Add new tools being attached
    for tool in matching_tools:
        tool_id = tool.get("id") or tool.get("tool_id")
        if tool_id:
            keep_tool_ids.add(tool_id)

    logger.info(f"Tool IDs to keep: {keep_tool_ids}")

    # Use the global http_session, assuming it's initialized
    if not http_session:
        logger.error(f"HTTP session not initialized for process_tools (agent: {agent_id})")
        # Decide how to handle this - raise error, return failure?
        # For now, log and return an error structure
        return {
            "detached_tools": [],
            "failed_detachments": [t.get("tool_id") or t.get("id") for t in mcp_tools],
            "successful_attachments": [],
            "failed_attachments": matching_tools, # Mark all as failed if session is down
            "error": "HTTP session not available"
        }

    # First, detach all existing MCP tools that aren't in the keep list
    tools_to_detach = []

    # Get all current MCP tool IDs
    current_mcp_tool_ids = set()
    for tool in mcp_tools:
        tool_id = tool.get("tool_id") or tool.get("id")
        if tool_id:
            current_mcp_tool_ids.add(tool_id)

    # Calculate how many MCP tools we'll have after detachments and attachments
    # Current MCP tools - tools to detach + new tools to attach
    potential_detach_count = 0
    for tool in mcp_tools:
        tool_id = tool.get("tool_id") or tool.get("id")
        if tool_id and tool_id not in keep_tool_ids:
            potential_detach_count += 1
    
    # How many MCP tools will remain after all operations
    new_tools_count = len([t for t in matching_tools if (t.get("id") or t.get("tool_id")) not in current_mcp_tool_ids])
    remaining_mcp_tools = len(mcp_tools) - potential_detach_count + new_tools_count
    
    logger.info(f"MIN_MCP_TOOLS check: current={len(mcp_tools)}, potential_detach={potential_detach_count}, new_tools={new_tools_count}, remaining={remaining_mcp_tools}, min_required={MIN_MCP_TOOLS}")
    
    # If we would go below MIN_MCP_TOOLS, limit detachments
    if remaining_mcp_tools < MIN_MCP_TOOLS:
        # Calculate how many we can actually detach
        max_detach_allowed = max(0, len(mcp_tools) + new_tools_count - MIN_MCP_TOOLS)
        logger.warning(f"Limiting detachments to preserve MIN_MCP_TOOLS={MIN_MCP_TOOLS}. Max allowed detachments: {max_detach_allowed}")
        
        # Build list of tools to detach, but limit to max_detach_allowed
        detach_count = 0
        for tool in mcp_tools:
            tool_id = tool.get("tool_id") or tool.get("id")
            tool_name = tool.get("name", "Unknown")
            
            # If tool ID is valid and not in the keep list
            if tool_id and tool_id not in keep_tool_ids:
                if detach_count < max_detach_allowed:
                    tools_to_detach.append({
                        "id": tool_id,
                        "tool_id": tool_id,
                        "name": tool_name
                    })
                    detach_count += 1
                else:
                    logger.info(f"Preserving tool {tool_name} ({tool_id}) to maintain MIN_MCP_TOOLS")
    else:
        # Find tools to detach (current tools that aren't in keep_tool_ids)
        for tool in mcp_tools:
            tool_id = tool.get("tool_id") or tool.get("id")
            tool_name = tool.get("name", "Unknown")

            # If tool ID is valid and not in the keep list
            if tool_id and tool_id not in keep_tool_ids:
                # logger.info(f"Will detach tool: {tool_name} ({tool_id})")
                tools_to_detach.append({
                    "id": tool_id,
                    "tool_id": tool_id,
                    "name": tool_name
                })

    logger.info(f"Tools to detach: {len(tools_to_detach)}")
    if tools_to_detach:
        # logger.info(f"Will detach: {', '.join([t.get('name', 'Unknown') + ' (' + t.get('id', 'Unknown ID') + ')' for t in tools_to_detach])}")
        pass  # Add pass to maintain block structure after commenting out log

    # Run detachments in parallel
    detach_tasks = []
    for tool in tools_to_detach:
        tool_id = tool.get("tool_id") or tool.get("id")
        if tool_id:
            # Note: The retry logic within detach_tool itself is removed for simplicity here.
            # If retries are crucial, detach_tool should handle them internally,
            # or a more complex parallel retry mechanism would be needed.
            detach_tasks.append(detach_tool(agent_id, tool_id)) # Pass only necessary args

    if detach_tasks:
        logger.info(f"Executing {len(detach_tasks)} detach operations in parallel...")
        detach_results = await asyncio.gather(*detach_tasks, return_exceptions=True)
        # Handle potential exceptions returned by gather
        processed_detach_results = []
        for i, result in enumerate(detach_results):
            tool_id_for_error = tools_to_detach[i].get("tool_id") or tools_to_detach[i].get("id")
            if isinstance(result, Exception):
                logger.error(f"Exception during parallel detach for tool ID {tool_id_for_error}: {result}")
                processed_detach_results.append({"success": False, "tool_id": tool_id_for_error, "error": str(result)})
            else:
                processed_detach_results.append(result)
        detach_results = processed_detach_results # Use the processed list
    else:
        detach_results = []
        logger.info("No detach tasks to execute.")


    # Process detachment results
    detached = [r["tool_id"] for r in detach_results if r and r.get("success")] # Add check for None result
    failed_detach = [r["tool_id"] for r in detach_results if r and not r.get("success")] # Add check for None result

    # Run all attachments in parallel
    attach_tasks = [attach_tool(agent_id, tool) # Pass only necessary args
                   for tool in matching_tools]
    attach_results = await asyncio.gather(*attach_tasks, return_exceptions=True) # Handle exceptions here too

    # Process attachment results (including exceptions)
    successful_attachments = []
    failed_attachments = []
    for i, result in enumerate(attach_results):
        tool_info = matching_tools[i] # Get corresponding tool info
        tool_id_for_error = tool_info.get("tool_id") or tool_info.get("id")
        tool_name_for_error = tool_info.get("name", "Unknown")

        if isinstance(result, Exception):
            logger.error(f"Exception during parallel attach for tool {tool_name_for_error} ({tool_id_for_error}): {result}")
            failed_attachments.append({"success": False, "tool_id": tool_id_for_error, "name": tool_name_for_error, "error": str(result)})
        elif isinstance(result, dict) and result.get("success"):
             successful_attachments.append(result)
        else: # It's a dict but success is False or structure is unexpected
            logger.warning(f"Failed attach result for tool {tool_name_for_error} ({tool_id_for_error}): {result}")
            # Ensure it has a standard structure even if attach_tool failed internally
            failed_attachments.append({
                "success": False,
                "tool_id": tool_id_for_error,
                "name": tool_name_for_error,
                "error": result.get("error", "Unknown attachment failure") if isinstance(result, dict) else "Unexpected result type"
            })


    # Return processed results (successful_attachments and failed_attachments populated by the loop above)
    return {
        "detached_tools": detached,
        "failed_detachments": failed_detach,
        "successful_attachments": successful_attachments, # Use lists populated in the loop
        "failed_attachments": failed_attachments  # Use lists populated in the loop
    }

@app.route('/api/v1/tools/search', methods=['POST'])
async def search():
    """Search endpoint - Note: This still calls the original synchronous search_tools"""
    # TODO: Decide if this endpoint should also be async or use a different search mechanism
    logger.info(f"Received request for /api/v1/tools/search")
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
        results = search_tools(query=query, limit=limit) # Await the async version
        
        # Filter results if MANAGE_ONLY_MCP_TOOLS is enabled
        if MANAGE_ONLY_MCP_TOOLS:
            # Load tool cache to check tool types
            tools_cache = await read_tool_cache()
            filtered_results = []
            
            for result in results:
                tool_name = result.get('name')
                if tool_name:
                    # Find the tool in cache to check its type
                    cached_tool = next((t for t in tools_cache if t.get('name') == tool_name), None)
                    if cached_tool:
                        # Check if it's an MCP tool
                        is_mcp_tool = (cached_tool.get("tool_type") == "external_mcp" or 
                                     (not _is_letta_core_tool(cached_tool) and cached_tool.get("tool_type") == "custom"))
                        if is_mcp_tool:
                            filtered_results.append(result)
                        else:
                            logger.debug(f"Filtering out non-MCP tool '{tool_name}' from search results")
                    else:
                        # If not in cache, it might be an MCP tool that needs registration
                        if result.get("mcp_server_name"):
                            filtered_results.append(result)
            
            logger.info(f"Weaviate search: {len(results)} total results, {len(filtered_results)} after MCP filtering.")
            return jsonify(filtered_results)
        else:
            logger.info(f"Weaviate search successful, returning {len(results)} results.")
            return jsonify(results)
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/v1/tools/search/rerank', methods=['POST'])
async def search_with_reranking():
    """Search with reranking endpoint for dashboard frontend."""
    logger.info(f"Received request for /api/v1/tools/search/rerank")
    
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Extract query and reranker config
        query_data = data.get('query', {})
        reranker_config = data.get('reranker_config', {})
        
        # Extract query string
        query_string = query_data.get('query', '')
        if not query_string:
            return jsonify({"success": False, "error": "No query provided"}), 400
            
        limit = query_data.get('limit', 10)
        
        logger.info(f"Performing reranked search for: '{query_string}' with limit: {limit}")
        logger.info(f"Reranker config: {reranker_config}")
        
        # Use the same search_tools function but with reranking enabled
        results = await asyncio.to_thread(search_tools, query=query_string, limit=limit)
        
        if not results:
            logger.warning("No results returned from search_tools")
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
        
        logger.info(f"Reranked search successful, returning {len(formatted_results)} results.")
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
    """Fetch agent information asynchronously using the global session"""
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_info (agent: {agent_id})")
        raise ConnectionError("HTTP session not available") # Or return default?
    async with http_session.get(f"{LETTA_URL}/agents/{agent_id}", headers=HEADERS) as response:
        response.raise_for_status()
        agent_data = await response.json()
    return agent_data.get("name", "Unknown Agent")

async def fetch_agent_tools(agent_id):
    """Fetch agent's current tools asynchronously using the global session"""
    global http_session
    if not http_session:
        logger.error(f"HTTP session not initialized for fetch_agent_tools (agent: {agent_id})")
        raise ConnectionError("HTTP session not available")
    async with http_session.get(f"{LETTA_URL}/agents/{agent_id}/tools", headers=HEADERS) as response:
        response.raise_for_status()
        return await response.json()

async def register_tool(tool_name, server_name):
    """Register a tool from an MCP server asynchronously using the global session"""
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

            # 5. Perform detachments and attachments
            results = await process_tools(agent_id, mcp_tools, processed_tools, keep_tools)
            
            # 6. Optionally, trigger pruning after successful attachments if a query was provided
            if query and results.get("successful_attachments"):
                successful_attachment_ids = [t['tool_id'] for t in results["successful_attachments"]]
                
                # Check if we have enough MCP tools to warrant pruning
                MIN_MCP_TOOLS = int(os.getenv('MIN_MCP_TOOLS', '7'))
                try:
                    current_agent_tools = await fetch_agent_tools(agent_id)
                    mcp_tools_count = sum(1 for tool in current_agent_tools 
                                         if tool.get("tool_type") == "external_mcp" or 
                                         (not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"))
                    
                    if mcp_tools_count <= MIN_MCP_TOOLS:
                        logger.info(f"Skipping auto-pruning: Agent has {mcp_tools_count} MCP tools (minimum required: {MIN_MCP_TOOLS})")
                    else:
                        logger.info(f"Calling tool pruning after successful attachment of {len(successful_attachment_ids)} tools for agent {agent_id}")
                        pruning_result = await _perform_tool_pruning(
                            agent_id=agent_id,
                            user_prompt=query, # Use the same query for pruning context
                            drop_rate=DEFAULT_DROP_RATE, # Use configurable drop rate from environment
                            keep_tool_ids=keep_tools, # Preserve tools explicitly asked to be kept
                            newly_matched_tool_ids=successful_attachment_ids # Preserve newly attached tools
                        )
                        if pruning_result.get("success"):
                            logger.info(f"Tool pruning completed successfully: {pruning_result.get('details', {}).get('mcp_tools_detached_count', 0)} tools pruned")
                        else:
                            logger.warning(f"Tool pruning failed: {pruning_result.get('error', 'Unknown error')}")
                        
                except Exception as prune_error:
                    logger.error(f"Error during tool pruning after attachment: {prune_error}")
                    # Continue execution - don't fail the whole attach operation due to pruning issues
            else:
                logger.info("Skipping tool pruning - no successful attachments or no query provided")

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
                    "target_agent": agent_id
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
    """
    # Check tool_type for obvious Letta tools
    letta_tool_types = [
        'letta_core', 'letta_voice_sleeptime_core', 'letta_sleeptime_core', 
        'letta_memory_core', 'letta_files_core', 'letta_builtin', 'letta_multi_agent_core'
    ]
    
    tool_type = tool.get('tool_type', '')
    if tool_type in letta_tool_types:
        return True
    
    # Check by tool name for additional core tools and important management tools
    core_tool_names = [
        'send_message', 'conversation_search', 'archival_memory_insert', 
        'archival_memory_search', 'core_memory_append', 'core_memory_replace', 
        'pause_heartbeats', 'find_attach_tools'  # Include tool management functions
    ]
    
    tool_name = tool.get('name', '')
    if tool_name in core_tool_names:
        return True
    
    return False

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
            
        logger.info(f"Target MCP tools calculation:")
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
            detach_tasks = [detach_tool(agent_id, tool_id) for tool_id in mcp_tools_to_detach_ids]
            logger.info(f"Executing {len(detach_tasks)} detach operations for MCP tools in parallel...")
            detach_results = await asyncio.gather(*detach_tasks, return_exceptions=True)

            id_to_name_map = {tool['id']: tool.get('name', 'Unknown') for tool in mcp_tools_on_agent_list}

            for i, result in enumerate(detach_results):
                tool_id_detached = list(mcp_tools_to_detach_ids)[i] 
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
            "model": os.getenv('RERANKER_MODEL', 'mistral:7b'),
            "provider": os.getenv('RERANKER_PROVIDER', 'ollama'),
            "parameters": {
                "temperature": float(os.getenv('RERANKER_TEMPERATURE', '0.1')),
                "max_tokens": int(os.getenv('RERANKER_MAX_TOKENS', '512')),
                "base_url": os.getenv('OLLAMA_RERANKER_BASE_URL', 'http://ollama-reranker-adapter:8080')
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
        model = data.get('model', 'mistral:7b')
        
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
                "score": result.get('score', 0),
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
            results = await search_tools_service(query, limit=limit)
        
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
    
    significant_improvements = 0
    significant_degradations = 0
    
    for metric in metrics:
        for k in k_values if metric in ['precision_at_k', 'recall_at_k'] else [None]:
            metric_key = f"{metric}@{k}" if k else metric
            
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
    
    active_connections = stats.get("active_connections", 0)
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
            logger.warning(f"Health check: Weaviate client object seems malformed.")
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

from cost_control_manager import (
    get_cost_manager, CostCategory, BudgetPeriod, AlertLevel,
    record_embedding_cost, record_weaviate_cost, record_letta_api_cost
)

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
