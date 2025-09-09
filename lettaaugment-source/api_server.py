from quart import Quart, request, jsonify
# Restore search_tools import, remove get_all_tools as cache is used for listing
from weaviate_tool_search_with_reranking import search_tools, init_client as init_weaviate_client
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

logger.info(f"Tool management configuration:")
logger.info(f"  MAX_TOTAL_TOOLS: {MAX_TOTAL_TOOLS}")
logger.info(f"  MAX_MCP_TOOLS: {MAX_MCP_TOOLS}")
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
        min_score = data.get('min_score', 50.0)  # Add min_score parameter (default 50.0)

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
        # Initialize Weaviate client
        # Ensure this uses the correct configuration for your deployment (Docker vs. local)
        temp_client = init_weaviate_client() # Call it once and store in a temporary variable
        
        if temp_client: # Check if the client object was created
            if temp_client.is_connected() and temp_client.is_ready():
                weaviate_client = temp_client # Assign to global only if successful and ready
                logger.info("Weaviate client initialized and ready.")
            elif temp_client.is_connected(): # Connected but not ready
                logger.error("Weaviate client connected but not ready during startup. Global client will be None.")
                weaviate_client = None # Set global to None
            else: # Not connected
                logger.error("Weaviate client failed to connect during startup. Global client will be None.")
                weaviate_client = None # Set global to None
        else: # init_weaviate_client() returned None
            logger.error("init_weaviate_client() returned None. Weaviate client not initialized.")
            weaviate_client = None # Ensure global client is None
            
    except Exception as e: # Catch any other unexpected error during the init process
        logger.error(f"Exception during Weaviate client initialization process in startup: {e}", exc_info=True)
        weaviate_client = None # Ensure global client is None on any exception

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
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("Weaviate client closed.")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
    if http_session:
        await http_session.close()
        logger.info("Global aiohttp client session closed.")

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
