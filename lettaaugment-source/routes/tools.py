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

import os
from quart import Blueprint, request, jsonify
import logging
from typing import Optional, Callable, List, Dict, Any

from services.tool_cache import ToolCacheService, get_tool_cache_service
from services.tool_search import ToolSearchService

logger = logging.getLogger(__name__)

# Create the blueprint
tools_bp = Blueprint('tools', __name__)

# Configuration flags - set via configure()
_manage_only_mcp_tools: bool = True

# Heavy operation handlers - still delegated to api_server for now
# These have too many dependencies to move in one step
_attach_tools_func: Optional[Callable] = None
_prune_tools_func: Optional[Callable] = None
_sync_func: Optional[Callable] = None
_refresh_func: Optional[Callable] = None


def configure(
    manage_only_mcp_tools: bool = True,
    attach_tools_func: Optional[Callable] = None,
    prune_tools_func: Optional[Callable] = None,
    sync_func: Optional[Callable] = None,
    refresh_func: Optional[Callable] = None
):
    """
    Configure the tools blueprint.
    
    Args:
        manage_only_mcp_tools: Whether to filter for MCP tools only
        attach_tools_func: Function for /tools/attach (delegated)
        prune_tools_func: Function for /tools/prune (delegated)
        sync_func: Function for /tools/sync (delegated)
        refresh_func: Function for /tools/refresh (delegated)
    """
    global _manage_only_mcp_tools
    global _attach_tools_func, _prune_tools_func, _sync_func, _refresh_func
    
    _manage_only_mcp_tools = manage_only_mcp_tools
    _attach_tools_func = attach_tools_func
    _prune_tools_func = prune_tools_func
    _sync_func = sync_func
    _refresh_func = refresh_func
    
    logger.info(f"Tools blueprint configured (manage_only_mcp={manage_only_mcp_tools})")


# =============================================================================
# Search Endpoints - Full implementation
# =============================================================================

@tools_bp.route('/api/v1/tools/search', methods=['POST'])
async def search():
    """Search for tools matching a query."""
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

        # Check for deprecated reranking parameter
        enable_reranking = data.get('enable_reranking', False)
        reranker_config = None
        if enable_reranking:
            logger.warning("DEPRECATED: enable_reranking parameter is deprecated. "
                         "Use /api/v1/tools/search/rerank endpoint instead.")
            reranker_config = {
                'enabled': True,
                'model': data.get('reranker_config', {}).get('model', 'bge-reranker-v2-m3'),
                'base_url': data.get('reranker_config', {}).get('base_url', 'http://localhost:8091')
            }
        
        # Search with higher limit if filtering for MCP tools
        if _manage_only_mcp_tools:
            search_limit = limit * 5
            logger.info(f"MANAGE_ONLY_MCP_TOOLS enabled - searching with limit {search_limit}")
        else:
            search_limit = limit
            
        results = ToolSearchService.search(query=query, limit=search_limit, reranker_config=reranker_config)
        
        # Filter for MCP tools if enabled
        if _manage_only_mcp_tools:
            cache_service = get_tool_cache_service()
            tools_cache = await cache_service.read_tool_cache()
            filtered_results = _filter_mcp_results(results, tools_cache, limit)
            
            logger.info(f"Search: {len(results)} total, {len(filtered_results)} after MCP filtering")
            _normalize_scores(filtered_results)
            return jsonify(filtered_results[:limit])
        else:
            logger.info(f"Search successful, returning {len(results)} results")
            _normalize_scores(results)
            return jsonify(results)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route('/api/v1/tools/search/rerank', methods=['POST'])
async def search_with_reranking():
    """Search with reranking for better relevance."""
    logger.info("Received request for /api/v1/tools/search/rerank")
    
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
            
        # Build reranker config
        reranker_config = {
            'enabled': True,
            'model': data.get('reranker_model', 'bge-reranker-v2-m3'),
            'base_url': data.get('reranker_base_url', os.getenv('RERANKER_BASE_URL', 'http://localhost:8091')),
            'top_k': data.get('reranker_top_k', limit)
        }
        
        logger.info(f"Performing reranked search for: '{query}' with config: {reranker_config}")
        
        # Search with higher limit if filtering for MCP tools
        if _manage_only_mcp_tools:
            search_limit = limit * 5
            logger.info(f"MANAGE_ONLY_MCP_TOOLS enabled - searching with limit {search_limit}")
        else:
            search_limit = limit
            
        results = ToolSearchService.search(query=query, limit=search_limit, reranker_config=reranker_config)
        
        if not results:
            logger.warning("No results returned from search_tools")
            return jsonify([])
        
        # Filter for MCP tools if enabled
        if _manage_only_mcp_tools:
            cache_service = get_tool_cache_service()
            tools_cache = await cache_service.read_tool_cache()
            filtered_results = _filter_mcp_results(results, tools_cache, limit)
            
            logger.info(f"Reranked search: {len(results)} total, {len(filtered_results)} after MCP filtering")
            formatted = _format_search_results(filtered_results[:limit])
            return jsonify(formatted)
        else:
            logger.info(f"Reranked search successful, returning {len(results)} results")
            formatted = _format_search_results(results[:limit])
            return jsonify(formatted)
            
    except Exception as e:
        logger.error(f"Error during reranked search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route('/api/v1/tools', methods=['GET'])
async def list_tools():
    """List all available tools from cache."""
    logger.info("Received request for /api/v1/tools")
    
    try:
        cache_service = get_tool_cache_service()
        tools = await cache_service.read_tool_cache()
        logger.info(f"Returning {len(tools)} tools from cache")
        return jsonify(tools)
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# Delegated Endpoints - Still use handlers from api_server
# =============================================================================

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


# =============================================================================
# Helper Functions
# =============================================================================

def _filter_mcp_results(results: List[Dict], tools_cache: List[Dict], limit: int) -> List[Dict]:
    """
    Filter search results to only include MCP tools.
    
    Args:
        results: Raw search results
        tools_cache: Cached tools for type checking
        limit: Maximum results to return
        
    Returns:
        Filtered list of MCP tools
    """
    cache_service = get_tool_cache_service()
    filtered = []
    
    for result in results:
        tool_name = result.get('name')
        if not tool_name:
            continue
            
        # Find tool in cache
        cached_tool = next((t for t in tools_cache if t.get('name') == tool_name), None)
        
        if cached_tool:
            tool_type = cached_tool.get("tool_type", "")
            is_letta_core = cache_service.is_letta_core_tool(cached_tool)
            is_mcp_tool = (tool_type == "external_mcp" or 
                         (not is_letta_core and tool_type == "custom"))
            
            if is_mcp_tool:
                filtered.append(result)
                if len(filtered) >= limit:
                    break
        else:
            # Not in cache - include if has mcp_server_name
            if result.get("mcp_server_name"):
                filtered.append(result)
                if len(filtered) >= limit:
                    break
    
    return filtered


def _normalize_scores(results: List[Dict]) -> None:
    """Map rerank_score to score field if present."""
    for result in results:
        if 'rerank_score' in result and 'score' not in result:
            result['score'] = result['rerank_score']


def _format_search_results(results: List[Dict]) -> List[Dict]:
    """Format search results for API response."""
    formatted = []
    for i, result in enumerate(results):
        formatted_result = {
            "name": result.get('name', ''),
            "description": result.get('description', ''),
            "source": result.get('source', 'unknown'),
            "score": result.get('rerank_score', result.get('score', 0)),
            "rank": i + 1,
            "tool_type": result.get('tool_type', ''),
            "mcp_server_name": result.get('mcp_server_name'),
            "tags": result.get('tags', []),
            "json_schema": result.get('json_schema')
        }
        formatted.append(formatted_result)
    return formatted
