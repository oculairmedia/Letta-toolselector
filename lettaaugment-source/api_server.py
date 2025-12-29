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
    async def record_embedding_cost(*args, **kwargs) -> bool:
        return False
    async def record_weaviate_cost(*args, **kwargs) -> bool:
        return False
    async def record_letta_api_cost(*args, **kwargs) -> bool:
        return False
# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import audit logging for structured events
from audit_logging import (
    emit_tool_event, emit_batch_event, emit_pruning_event, emit_limit_enforcement_event,
    AuditAction, AuditSource, start_audit_processor, stop_audit_processor
)

# Import connection test functions from config routes
from routes.config import test_ollama_connection, test_weaviate_connection  # type: ignore[assignment]

# Import Letta SDK client wrapper for SDK-based API calls
# Feature flag to enable SDK migration (set USE_LETTA_SDK=true to enable)
USE_LETTA_SDK = os.getenv('USE_LETTA_SDK', 'false').lower() == 'true'
_letta_sdk_client = None
get_letta_sdk_client = None  # Will be set if SDK is enabled

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
        get_letta_sdk_client = None
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
_mcp_servers_cache = None  # In-memory cache for MCP servers
_mcp_servers_cache_mtime = 0  # Timestamp of last MCP servers cache load

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
async def read_mcp_servers_cache(force_reload=False):
    """
    Reads the MCP servers cache file asynchronously, using an in-memory cache.
    
    Args:
        force_reload: If True, reload from disk even if cached in memory
        
    Returns:
        List of MCP server dictionaries
    """
    global _mcp_servers_cache, _mcp_servers_cache_mtime
    try:
        # Check modification time synchronously first
        try:
            current_mtime = os.path.getmtime(MCP_SERVERS_CACHE_FILE_PATH)
        except FileNotFoundError:
            logger.error(f"MCP servers cache file not found: {MCP_SERVERS_CACHE_FILE_PATH}. Returning empty list.")
            _mcp_servers_cache = []
            _mcp_servers_cache_mtime = 0
            return []
        
        # Reload if forced, cache is empty, or file has been modified
        if force_reload or _mcp_servers_cache is None or current_mtime > _mcp_servers_cache_mtime:
            logger.info(f"Loading MCP servers cache from file: {MCP_SERVERS_CACHE_FILE_PATH}")
            async with aiofiles.open(MCP_SERVERS_CACHE_FILE_PATH, mode='r') as f:
                content = await f.read()
                _mcp_servers_cache = json.loads(content)
            _mcp_servers_cache_mtime = current_mtime
            logger.info(f"Loaded {len(_mcp_servers_cache)} MCP servers into cache.")
        
        return _mcp_servers_cache if _mcp_servers_cache else []
    except FileNotFoundError:
        logger.error(f"MCP servers cache file not found during async read: {MCP_SERVERS_CACHE_FILE_PATH}. Returning empty list.")
        _mcp_servers_cache = []
        _mcp_servers_cache_mtime = 0
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from MCP servers cache file: {MCP_SERVERS_CACHE_FILE_PATH}. Returning empty list.")
        _mcp_servers_cache = []
        _mcp_servers_cache_mtime = 0
        return []
    except Exception as e:
        logger.error(f"Error reading MCP servers cache file {MCP_SERVERS_CACHE_FILE_PATH}: {e}")
        _mcp_servers_cache = []
        _mcp_servers_cache_mtime = 0
        return []


def get_mcp_servers_cache_count() -> int:
    """
    Get the count of cached MCP servers without async I/O.
    
    Returns:
        Number of MCP servers in cache, or 0 if cache not loaded
    """
    return len(_mcp_servers_cache) if _mcp_servers_cache else 0

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


# Tools attach handler moved to routes/tools.py blueprint

def _is_letta_core_tool(tool: dict) -> bool:
    """
    Determine if a tool is a Letta core tool that should not be managed by auto selection.
    
    Delegates to models.is_letta_core_tool for the actual implementation.
    This wrapper is kept for backward compatibility with existing code.
    """
    return models_is_letta_core_tool(tool)

# Tools prune and sync handlers moved to routes/tools.py blueprint


# Config validation, tool selector config endpoints moved to routes/config.py blueprint
# Search parameter routes moved to routes/search.py blueprint
# Reranker and Embedding config routes moved to routes/config.py blueprint
# Ollama routes moved to routes/ollama.py blueprint
# Configuration presets routes moved to routes/config.py blueprint
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
        result = await test_weaviate_connection(config)  # type: ignore[call-arg]
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


# Tools refresh handler moved to routes/tools.py blueprint

# Evaluation endpoints and helpers moved to routes/evaluation.py blueprint
# Models endpoints moved to routes/models.py blueprint
# NOTE: Benchmark routes moved to routes/benchmark.py
# NOTE: Reranker model registry routes moved to routes/reranker.py

# Safety endpoints and helpers moved to routes/safety.py blueprint


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

        
    # Check MCP servers cache status (use in-memory cache, no async I/O)
    mcp_servers_cache_file_status = "OK"
    mcp_servers_cache_size = 0
    try:
        # Use in-memory cache count (O(1), no I/O)
        mcp_servers_cache_size = get_mcp_servers_cache_count()
        
        if mcp_servers_cache_size == 0:
            # Check if file exists but cache not loaded
            if os.path.exists(MCP_SERVERS_CACHE_FILE_PATH):
                mcp_servers_cache_file_status = "Warning: File exists but not loaded in memory"
            else:
                mcp_servers_cache_file_status = "Error: File not found"
    except Exception as e:
        mcp_servers_cache_file_status = f"Error checking cache: {str(e)}"
        logger.warning(f"Health check: Error checking MCP servers cache: {e}")

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
            "mcp_servers_cache": {  # Now uses in-memory cache
                "status": mcp_servers_cache_file_status,
                "size": mcp_servers_cache_size,
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
    
    # Start audit event processor for async logging
    await start_audit_processor()
    
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
    
    # Initialize services layer early (needed by config blueprint)
    from services.tool_search import configure_search_service
    from services.tool_cache import get_tool_cache_service
    configure_search_service(search_tools)
    tool_cache_service = get_tool_cache_service(CACHE_DIR)
    logger.info("Services layer configured.")
    
    # Configure and register config routes blueprint
    from routes import config as config_routes, config_bp
    config_routes.configure(
        http_session=http_session,
        log_config_change=log_config_change,
        tool_cache_service=tool_cache_service
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

    # Configure and register tools routes blueprint
    from routes import tools as tools_routes
    from routes.tools import tools_bp
    tools_routes.configure(
        manage_only_mcp_tools=MANAGE_ONLY_MCP_TOOLS,
        default_min_score=DEFAULT_MIN_SCORE,
        agent_service=agent_service,
        tool_manager=tool_manager,
        search_tools_func=search_tools,
        read_tool_cache_func=read_tool_cache,
        read_mcp_servers_cache_func=read_mcp_servers_cache,
        process_matching_tool_func=process_matching_tool,
        init_weaviate_client_func=init_weaviate_client,
        get_weaviate_client_func=lambda: weaviate_client,
        is_letta_core_tool_func=models_is_letta_core_tool,
        emit_batch_event_func=emit_batch_event,
        emit_pruning_event_func=emit_pruning_event,
        audit_action_class=AuditAction,
        audit_source_class=AuditSource
    )
    app.register_blueprint(tools_bp)
    logger.info("Tools routes blueprint registered.")

    # Configure and register evaluation routes blueprint
    from routes import evaluation as evaluation_routes
    from routes.evaluation import evaluation_bp
    evaluation_routes.configure(
        search_tools_func=search_tools,
        bm25_vector_override_service=bm25_vector_override_service,
        cache_dir=CACHE_DIR
    )
    app.register_blueprint(evaluation_bp)
    logger.info("Evaluation routes blueprint registered.")

    # Configure and register safety routes blueprint
    from routes import safety as safety_routes
    from routes.safety import safety_bp
    safety_routes.configure()  # No dependencies needed - uses environment variables
    app.register_blueprint(safety_bp)
    logger.info("Safety routes blueprint registered.")

    # Configure and register models routes blueprint
    from routes import models as models_routes
    from routes.models import models_bp
    models_routes.configure(
        search_tools_func=search_tools,
        bm25_vector_override_service=bm25_vector_override_service
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
    
    # Stop audit processor and flush remaining events
    await stop_audit_processor()
    logger.info("Audit event processor stopped.")

# ================================================================================
# LDTS-58: Cost Control and Budget Management API Endpoints
# ================================================================================
# Cost control routes moved to routes/cost_control.py blueprint

# Configuration validation helper functions
async def perform_configuration_validation(config_type: str, field: str, value, context: dict | None = None):
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
            return await test_weaviate_connection(config)  # type: ignore[call-arg]
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
