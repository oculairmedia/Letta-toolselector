"""
Tools Blueprint

Provides tool management endpoints for searching, attaching, detaching, and pruning tools.

Routes:
- POST /api/v1/tools/search - Search for tools
- POST /api/v1/tools/search/rerank - Search with reranking
- GET /api/v1/tools - List all tools
- POST /api/v1/tools/attach - Attach tools to an agent
- POST /api/v1/tools/prune - Prune excess tools from an agent
- POST /api/v1/tools/sync - Sync tool cache
- POST /api/v1/tools/refresh - Refresh tool cache
"""

from quart import Blueprint, request, jsonify
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# Create the blueprint
tools_bp = Blueprint('tools', __name__)

# Module state - injected via configure()
_search_func: Optional[Callable] = None
_search_with_rerank_func: Optional[Callable] = None
_list_tools_func: Optional[Callable] = None
_attach_tools_func: Optional[Callable] = None
_prune_tools_func: Optional[Callable] = None
_sync_func: Optional[Callable] = None
_refresh_func: Optional[Callable] = None


def configure(
    search_func: Optional[Callable] = None,
    search_with_rerank_func: Optional[Callable] = None,
    list_tools_func: Optional[Callable] = None,
    attach_tools_func: Optional[Callable] = None,
    prune_tools_func: Optional[Callable] = None,
    sync_func: Optional[Callable] = None,
    refresh_func: Optional[Callable] = None
):
    """
    Configure the tools blueprint with required dependencies.
    
    Args:
        search_func: Function for /tools/search
        search_with_rerank_func: Function for /tools/search/rerank
        list_tools_func: Function for GET /tools
        attach_tools_func: Function for /tools/attach
        prune_tools_func: Function for /tools/prune
        sync_func: Function for /tools/sync
        refresh_func: Function for /tools/refresh
    """
    global _search_func, _search_with_rerank_func, _list_tools_func
    global _attach_tools_func, _prune_tools_func, _sync_func, _refresh_func
    
    _search_func = search_func
    _search_with_rerank_func = search_with_rerank_func
    _list_tools_func = list_tools_func
    _attach_tools_func = attach_tools_func
    _prune_tools_func = prune_tools_func
    _sync_func = sync_func
    _refresh_func = refresh_func
    
    logger.info("Tools blueprint configured")


@tools_bp.route('/api/v1/tools/search', methods=['POST'])
async def search():
    """Search for tools matching a query."""
    if not _search_func:
        return jsonify({"error": "Search not configured"}), 503
    
    return await _search_func()


@tools_bp.route('/api/v1/tools/search/rerank', methods=['POST'])
async def search_with_reranking():
    """Search with reranking for better relevance."""
    if not _search_with_rerank_func:
        return jsonify({"error": "Reranking search not configured"}), 503
    
    return await _search_with_rerank_func()


@tools_bp.route('/api/v1/tools', methods=['GET'])
async def list_tools():
    """List all available tools."""
    if not _list_tools_func:
        return jsonify({"error": "List tools not configured"}), 503
    
    return await _list_tools_func()


@tools_bp.route('/api/v1/tools/attach', methods=['POST'])
async def attach_tools():
    """Attach tools to an agent."""
    if not _attach_tools_func:
        return jsonify({"error": "Attach tools not configured"}), 503
    
    return await _attach_tools_func()


@tools_bp.route('/api/v1/tools/prune', methods=['POST'])
async def prune_tools():
    """Prune excess tools from an agent."""
    if not _prune_tools_func:
        return jsonify({"error": "Prune tools not configured"}), 503
    
    return await _prune_tools_func()


@tools_bp.route('/api/v1/tools/sync', methods=['POST'])
async def sync_tools():
    """Sync tool cache with upstream sources."""
    if not _sync_func:
        return jsonify({"error": "Sync not configured"}), 503
    
    return await _sync_func()


@tools_bp.route('/api/v1/tools/refresh', methods=['POST'])
async def refresh_tools():
    """Refresh tool cache."""
    if not _refresh_func:
        return jsonify({"error": "Refresh not configured"}), 503
    
    return await _refresh_func()
