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
import asyncio
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
_default_min_score: float = 0.0

# Dependencies injected via configure()
_agent_service: Optional[Any] = None
_tool_manager: Optional[Any] = None
_search_tools_func: Optional[Callable] = None
_read_tool_cache_func: Optional[Callable] = None
_read_mcp_servers_cache_func: Optional[Callable] = None
_process_matching_tool_func: Optional[Callable] = None
_init_weaviate_client_func: Optional[Callable] = None
_get_weaviate_client_func: Optional[Callable] = None
_is_letta_core_tool_func: Optional[Callable] = None

# Audit functions
_emit_batch_event_func: Optional[Callable] = None
_emit_pruning_event_func: Optional[Callable] = None
_audit_action_class: Optional[Any] = None
_audit_source_class: Optional[Any] = None


def configure(
    manage_only_mcp_tools: bool = True,
    default_min_score: float = 0.0,
    agent_service: Optional[Any] = None,
    tool_manager: Optional[Any] = None,
    search_tools_func: Optional[Callable] = None,
    read_tool_cache_func: Optional[Callable] = None,
    read_mcp_servers_cache_func: Optional[Callable] = None,
    process_matching_tool_func: Optional[Callable] = None,
    init_weaviate_client_func: Optional[Callable] = None,
    get_weaviate_client_func: Optional[Callable] = None,
    is_letta_core_tool_func: Optional[Callable] = None,
    emit_batch_event_func: Optional[Callable] = None,
    emit_pruning_event_func: Optional[Callable] = None,
    audit_action_class: Optional[Any] = None,
    audit_source_class: Optional[Any] = None
):
    """
    Configure the tools blueprint with required dependencies.
    
    Args:
        manage_only_mcp_tools: Whether to filter for MCP tools only
        default_min_score: Default minimum score threshold for tool matching
        agent_service: AgentService instance for agent operations
        tool_manager: ToolManager instance for tool operations
        search_tools_func: Function to search tools
        read_tool_cache_func: Function to read tool cache
        read_mcp_servers_cache_func: Function to read MCP servers cache
        process_matching_tool_func: Function to process matching tools
        init_weaviate_client_func: Function to initialize Weaviate client
        get_weaviate_client_func: Function to get current Weaviate client
        is_letta_core_tool_func: Function to check if tool is Letta core
        emit_batch_event_func: Function to emit batch audit events
        emit_pruning_event_func: Function to emit pruning audit events
        audit_action_class: AuditAction enum class
        audit_source_class: AuditSource enum class
    """
    global _manage_only_mcp_tools, _default_min_score
    global _agent_service, _tool_manager, _search_tools_func
    global _read_tool_cache_func, _read_mcp_servers_cache_func
    global _process_matching_tool_func, _init_weaviate_client_func
    global _get_weaviate_client_func, _is_letta_core_tool_func
    global _emit_batch_event_func, _emit_pruning_event_func
    global _audit_action_class, _audit_source_class
    
    _manage_only_mcp_tools = manage_only_mcp_tools
    _default_min_score = default_min_score
    _agent_service = agent_service
    _tool_manager = tool_manager
    _search_tools_func = search_tools_func
    _read_tool_cache_func = read_tool_cache_func
    _read_mcp_servers_cache_func = read_mcp_servers_cache_func
    _process_matching_tool_func = process_matching_tool_func
    _init_weaviate_client_func = init_weaviate_client_func
    _get_weaviate_client_func = get_weaviate_client_func
    _is_letta_core_tool_func = is_letta_core_tool_func
    _emit_batch_event_func = emit_batch_event_func
    _emit_pruning_event_func = emit_pruning_event_func
    _audit_action_class = audit_action_class
    _audit_source_class = audit_source_class
    
    logger.info(f"Tools blueprint configured (manage_only_mcp={manage_only_mcp_tools})")


# =============================================================================
# Helper Functions
# =============================================================================

def _is_letta_core_tool(tool: dict) -> bool:
    """Check if a tool is a Letta core tool."""
    if _is_letta_core_tool_func:
        return _is_letta_core_tool_func(tool)
    # Fallback implementation
    cache_service = get_tool_cache_service()
    return cache_service.is_letta_core_tool(tool)


def _filter_mcp_results(results: List[Dict], tools_cache: List[Dict], limit: int) -> List[Dict]:
    """
    Filter search results to only include MCP tools.
    
    Uses O(1) dictionary lookup instead of O(n) linear search for better performance
    when filtering large result sets against large tool caches.
    
    Args:
        results: Raw search results
        tools_cache: Cached tools for type checking (unused, kept for API compatibility)
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
            
        # O(1) lookup by name instead of O(n) linear search
        cached_tool = cache_service.get_tool_by_name(tool_name)
        
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


# =============================================================================
# Search Endpoints
# =============================================================================

@tools_bp.route('/api/v1/tools/search', methods=['POST'])
async def search():
    """Search for tools matching a query."""
    logger.debug("Received request for /api/v1/tools/search")
    
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
            logger.debug("MANAGE_ONLY_MCP_TOOLS enabled - searching with limit %d", search_limit)
        else:
            search_limit = limit
            
        results = ToolSearchService.search(query=query, limit=search_limit, reranker_config=reranker_config)
        
        # Filter for MCP tools if enabled
        if _manage_only_mcp_tools:
            cache_service = get_tool_cache_service()
            tools_cache = await cache_service.read_tool_cache()
            filtered_results = _filter_mcp_results(results, tools_cache, limit)
            
            logger.debug("Search: %d total, %d after MCP filtering", len(results), len(filtered_results))
            _normalize_scores(filtered_results)
            return jsonify(filtered_results[:limit])
        else:
            logger.debug("Search successful, returning %d results", len(results))
            _normalize_scores(results)
            return jsonify(results)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route('/api/v1/tools/search/rerank', methods=['POST'])
async def search_with_reranking():
    """Search with reranking for better relevance."""
    logger.debug("Received request for /api/v1/tools/search/rerank")
    
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
        
        logger.debug("Performing reranked search for: '%s' with config: %s", query, reranker_config)
        
        # Search with higher limit if filtering for MCP tools
        if _manage_only_mcp_tools:
            search_limit = limit * 5
            logger.debug("MANAGE_ONLY_MCP_TOOLS enabled - searching with limit %d", search_limit)
        else:
            search_limit = limit
            
        results = ToolSearchService.search(query=query, limit=search_limit, reranker_config=reranker_config)
        
        if not results:
            logger.debug("No results returned from search_tools")
            return jsonify([])
        
        # Filter for MCP tools if enabled
        if _manage_only_mcp_tools:
            cache_service = get_tool_cache_service()
            tools_cache = await cache_service.read_tool_cache()
            filtered_results = _filter_mcp_results(results, tools_cache, limit)
            
            logger.debug("Reranked search: %d total, %d after MCP filtering", len(results), len(filtered_results))
            formatted = _format_search_results(filtered_results[:limit])
            return jsonify(formatted)
        else:
            logger.debug("Reranked search successful, returning %d results", len(results))
            formatted = _format_search_results(results[:limit])
            return jsonify(formatted)
            
    except Exception as e:
        logger.error(f"Error during reranked search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route('/api/v1/tools', methods=['GET'])
async def list_tools():
    """List all available tools from cache."""
    logger.debug("Received request for /api/v1/tools")
    
    try:
        cache_service = get_tool_cache_service()
        tools = await cache_service.read_tool_cache()
        logger.debug("Returning %d tools from cache", len(tools))
        return jsonify(tools)
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# Tool Attachment Endpoint
# =============================================================================

@tools_bp.route('/api/v1/tools/attach', methods=['POST'])
async def attach_tools():
    """Handle tool attachment requests with parallel processing using cache."""
    logger.debug("Received request for %s", request.path)
    
    # Check required dependencies
    if not _agent_service or not _tool_manager:
        return jsonify({"error": "Tool attachment not configured - missing agent_service or tool_manager"}), 503
    if not _search_tools_func:
        return jsonify({"error": "Tool attachment not configured - missing search_tools function"}), 503
    
    try:
        data = await request.get_json()
        if not data:
            logger.warning("Attach request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        query = data.get('query', '')
        limit = data.get('limit', 10)
        agent_id = data.get('agent_id')
        keep_tools = data.get('keep_tools', [])
        min_score = data.get('min_score', _default_min_score)
        skip_loop_trigger = data.get('skip_loop_trigger', False)
        
        logger.debug("Attach request payload: skip_loop_trigger=%s, keys=%s", skip_loop_trigger, list(data.keys()))

        if not agent_id:
            logger.warning("Attach request missing 'agent_id'.")
            return jsonify({"error": "agent_id is required"}), 400

        try:
            # 1. Fetch agent-specific info (name and current tools) directly from Letta
            agent_name, current_agent_tools = await asyncio.gather(
                _agent_service.fetch_agent_info(agent_id),
                _tool_manager.fetch_agent_tools(agent_id)
            )

            # 2. Identify unique MCP tools currently on the agent
            mcp_tools = []
            seen_tool_ids = set()
            logger.debug("Getting current tools directly from agent %s (%s)...", agent_name, agent_id)
            logger.debug("Total tools on agent: %d", len(current_agent_tools))
            
            # Precompute MCP status once per tool (avoid redundant _is_letta_core_tool calls)
            mcp_status: dict[str, bool] = {}
            for tool in current_agent_tools:
                tool_id = tool.get("id") or tool.get("tool_id")
                if tool_id:
                    is_mcp = (tool.get("tool_type") == "external_mcp" or 
                              (not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"))
                    mcp_status[tool_id] = is_mcp
            
            mcp_count = sum(1 for is_mcp in mcp_status.values() if is_mcp)
            logger.debug("Found %d total MCP tools, checking for duplicates...", mcp_count)

            for tool in current_agent_tools:
                tool_id = tool.get("id") or tool.get("tool_id")
                if tool_id and mcp_status.get(tool_id) and tool_id not in seen_tool_ids:
                    seen_tool_ids.add(tool_id)
                    tool_copy = tool.copy()
                    tool_copy["id"] = tool_id
                    tool_copy["tool_id"] = tool_id
                    mcp_tools.append(tool_copy)

            # 3. Search for matching tools - ensure Weaviate client is ready
            weaviate_client = _get_weaviate_client_func() if _get_weaviate_client_func else None
            
            if not weaviate_client or not weaviate_client.is_ready():
                logger.warning("Weaviate client not ready or not initialized at /attach endpoint. Attempting re-initialization...")
                if _init_weaviate_client_func:
                    weaviate_client = _init_weaviate_client_func()
                    if not weaviate_client or not weaviate_client.is_ready():
                        logger.error("Failed to re-initialize Weaviate client for /attach. Cannot perform search.")
                        return jsonify({"error": "Weaviate client not available after re-attempt"}), 500
                    logger.debug("Weaviate client successfully re-initialized for /attach endpoint.")
                else:
                    return jsonify({"error": "Weaviate client not available and no init function configured"}), 500
            
            logger.debug("Running Weaviate search for query '%s' directly...", query)
            # Call the synchronous search_tools function in a separate thread
            matching_tools_from_search = await asyncio.to_thread(
                _search_tools_func,
                query=query,
                limit=limit
            )
            
            logger.debug("Found %d matching tools from Weaviate search.", len(matching_tools_from_search))
            
            # 3.5. Filter tools by min_score threshold
            filtered_tools = []
            for tool in matching_tools_from_search:
                tool_score = tool.get('rerank_score')
                if tool_score is None:
                    tool_score = tool.get('score', 0)
                
                tool_score_percent = tool_score * 100
                
                if tool_score_percent >= min_score:
                    filtered_tools.append(tool)
                    logger.debug(f"Tool '{tool.get('name')}' passed filter with score {tool_score_percent:.1f}% >= {min_score}%")
                else:
                    logger.debug(f"Tool '{tool.get('name')}' filtered out with score {tool_score_percent:.1f}% < {min_score}%")
            
            logger.debug("Score filtering: %d of %d tools passed min_score threshold of %s%%", len(filtered_tools), len(matching_tools_from_search), min_score)

            # 4. Process matching tools (check cache, register if needed)
            letta_tools_cache = await _read_tool_cache_func() if _read_tool_cache_func else []
            mcp_servers = await _read_mcp_servers_cache_func() if _read_mcp_servers_cache_func else []

            if _process_matching_tool_func:
                process_tasks = [_process_matching_tool_func(tool, letta_tools_cache, mcp_servers) for tool in filtered_tools]
                processed_tools_results = await asyncio.gather(*process_tasks, return_exceptions=True)
                
                processed_tools = []
                for i, res in enumerate(processed_tools_results):
                    if isinstance(res, Exception):
                        logger.error(f"Error processing tool candidate {matching_tools_from_search[i].get('name', 'Unknown')}: {res}")
                    elif res:
                        processed_tools.append(res)
                
                logger.debug("Successfully processed/registered %d tools for attachment consideration.", len(processed_tools))
            else:
                processed_tools = filtered_tools
                logger.warning("No process_matching_tool_func configured, using filtered tools directly")

            # 5. Pre-attach pruning: Check if we need to make room before attaching new tools
            MAX_TOTAL_TOOLS = int(os.getenv('MAX_TOTAL_TOOLS', '30'))
            MAX_MCP_TOOLS = int(os.getenv('MAX_MCP_TOOLS', '20'))
            MIN_MCP_TOOLS = int(os.getenv('MIN_MCP_TOOLS', '7'))
            
            total_current_tools = len(current_agent_tools)
            mcp_current_count = len(mcp_tools)
            core_current_count = total_current_tools - mcp_current_count
            
            new_tool_ids = set()
            for tool in processed_tools:
                tool_id = tool.get("id") or tool.get("tool_id")
                if tool_id and tool_id not in seen_tool_ids:
                    new_tool_ids.add(tool_id)
            
            new_tools_count = len(new_tool_ids)
            logger.info(f"Pre-attach analysis: current_total={total_current_tools}, current_mcp={mcp_current_count}, core={core_current_count}, new_tools={new_tools_count}")
            logger.info(f"Limits: MAX_TOTAL={MAX_TOTAL_TOOLS}, MAX_MCP={MAX_MCP_TOOLS}, MIN_MCP={MIN_MCP_TOOLS}")
            
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
                
                min_removals_for_mcp = max(0, projected_mcp - MAX_MCP_TOOLS)
                min_removals_for_total = max(0, projected_total - MAX_TOTAL_TOOLS)
                min_removals_needed = max(min_removals_for_mcp, min_removals_for_total)
                
                max_removals_allowed = max(0, mcp_current_count - MIN_MCP_TOOLS)
                tools_to_remove = min(min_removals_needed, max_removals_allowed)
                
                logger.info(f"Pre-attach pruning: need to remove {min_removals_needed} tools (min_for_mcp={min_removals_for_mcp}, min_for_total={min_removals_for_total})")
                logger.info(f"Pre-attach pruning: can remove up to {max_removals_allowed} tools (respecting MIN_MCP_TOOLS={MIN_MCP_TOOLS})")
                logger.info(f"Pre-attach pruning: will remove {tools_to_remove} tools")
                
                if tools_to_remove > 0:
                    effective_drop_rate = min(0.9, tools_to_remove / max(1, mcp_current_count))
                    
                    logger.info(f"Pre-attach pruning: using drop_rate={effective_drop_rate:.2f} to remove ~{tools_to_remove} tools")
                    
                    preattach_prune_result = await _tool_manager.perform_tool_pruning(
                        agent_id=agent_id,
                        user_prompt=query,
                        drop_rate=effective_drop_rate,
                        keep_tool_ids=keep_tools,
                        newly_matched_tool_ids=[]
                    )
                    
                    if preattach_prune_result.get("success"):
                        removed_count = preattach_prune_result.get("details", {}).get("mcp_tools_detached_count", 0)
                        logger.info(f"Pre-attach pruning completed: removed {removed_count} tools to make room")
                        
                        # Use remaining_tools from pruning result to avoid redundant API call
                        current_agent_tools = preattach_prune_result.get("remaining_tools", [])
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
            results = await _tool_manager.process_tools(agent_id, mcp_tools, processed_tools, keep_tools)
            
            # 6.5. Emit audit events for attachments and detachments
            if _emit_batch_event_func and _audit_action_class and _audit_source_class:
                try:
                    import uuid
                    correlation_id = str(uuid.uuid4())
                    
                    if results.get("successful_attachments"):
                        _emit_batch_event_func(
                            action=_audit_action_class.ATTACH,
                            agent_id=agent_id,
                            tools=results["successful_attachments"],
                            source=_audit_source_class.API_ATTACH,
                            reason=f"Query match: {query[:100] if query else 'no query'}" if query else "Requested tool attachment",
                            correlation_id=correlation_id,
                            success_count=len(results["successful_attachments"]),
                            failure_count=0
                        )
                    
                    if results.get("failed_attachments"):
                        _emit_batch_event_func(
                            action=_audit_action_class.ATTACH,
                            agent_id=agent_id,
                            tools=[{"tool_id": t.get("tool_id") or t.get("id"), 
                                   "name": t.get("name", "unknown"), 
                                   "success": False} 
                                  for t in results["failed_attachments"]],
                            source=_audit_source_class.API_ATTACH,
                            reason="Attachment failed",
                            correlation_id=correlation_id,
                            success_count=0,
                            failure_count=len(results["failed_attachments"])
                        )
                    
                    if results.get("detached_tools"):
                        _emit_batch_event_func(
                            action=_audit_action_class.DETACH,
                            agent_id=agent_id,
                            tools=[{"tool_id": tool_id, "name": "unknown", "success": True} 
                                  for tool_id in results["detached_tools"]],
                            source=_audit_source_class.API_ATTACH,
                            reason="Making room for new tools",
                            correlation_id=correlation_id,
                            success_count=len(results["detached_tools"]),
                            failure_count=0
                        )
                    
                except Exception as audit_error:
                    logger.warning(f"Failed to emit audit events: {audit_error}")
            
            # 7. Trigger a new agent loop so newly attached tools are available
            loop_triggered = False
            successful_attachments = results.get("successful_attachments", [])
            logger.info(f"Checking if loop trigger needed: {len(successful_attachments)} successful attachments, skip_loop_trigger={skip_loop_trigger}")
            if successful_attachments and not skip_loop_trigger:
                logger.info(f"Triggering agent loop for {agent_id} with query: {query}")
                try:
                    loop_triggered = _agent_service.trigger_agent_loop(
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
                    "processed_count": len(matching_tools_from_search),
                    "passed_filter_count": len(filtered_tools),
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
            logger.error(f"Error during tool management: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error during attach_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# Tool Pruning Endpoint
# =============================================================================

@tools_bp.route('/api/v1/tools/prune', methods=['POST'])
async def prune_tools():
    """Prune tools attached to an agent based on their relevance to a user's prompt."""
    logger.debug("Received request for /api/v1/tools/prune")
    
    if not _tool_manager:
        return jsonify({"error": "Tool pruning not configured - missing tool_manager"}), 503
    
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

        if drop_rate is None or not isinstance(drop_rate, (int, float)) or not (0 <= drop_rate <= 1):
            logger.warning(f"Prune request has invalid 'drop_rate': {drop_rate}. Must be between 0 and 1.")
            return jsonify({"error": "drop_rate must be a number between 0 and 1"}), 400

        # Call the core pruning logic
        pruning_result = await _tool_manager.perform_tool_pruning(
            agent_id=agent_id,
            user_prompt=user_prompt,
            drop_rate=drop_rate,
            keep_tool_ids=keep_tool_ids,
            newly_matched_tool_ids=newly_matched_tool_ids
        )

        # Emit audit events for pruning operation
        if _emit_batch_event_func and _emit_pruning_event_func and _audit_action_class and _audit_source_class:
            try:
                import uuid
                correlation_id = str(uuid.uuid4())
                
                if pruning_result.get("success"):
                    details = pruning_result.get("details", {})
                    
                    _emit_pruning_event_func(
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
                    
                    if details.get("successful_detachments_mcp"):
                        _emit_batch_event_func(
                            action=_audit_action_class.DETACH,
                            agent_id=agent_id,
                            tools=details.get("successful_detachments_mcp", []),
                            source=_audit_source_class.API_PRUNE,
                            reason=f"Pruning with drop_rate={drop_rate}",
                            correlation_id=correlation_id,
                            success_count=len(details.get("successful_detachments_mcp", [])),
                            failure_count=len(details.get("failed_detachments_mcp", []))
                        )
                    
                    if details.get("failed_detachments_mcp"):
                        _emit_batch_event_func(
                            action=_audit_action_class.DETACH,
                            agent_id=agent_id,
                            tools=[{"tool_id": t.get("tool_id"), 
                                   "name": t.get("name", "unknown"), 
                                   "success": False} 
                                  for t in details.get("failed_detachments_mcp", [])],
                            source=_audit_source_class.API_PRUNE,
                            reason="Detachment failed during pruning",
                            correlation_id=correlation_id,
                            success_count=0,
                            failure_count=len(details.get("failed_detachments_mcp", []))
                        )
            
            except Exception as audit_error:
                logger.warning(f"Failed to emit audit events for pruning: {audit_error}")

        if pruning_result.get("success"):
            return jsonify(pruning_result)
        else:
            return jsonify(pruning_result), 500

    except Exception as e:
        logger.error(f"Error during prune_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# Tool Sync/Refresh Endpoints
# =============================================================================

@tools_bp.route('/api/v1/tools/sync', methods=['POST'])
async def sync_tools():
    """Endpoint to manually trigger the sync process (for testing/debugging)."""
    logger.debug("Received request for /api/v1/tools/sync")
    try:
        from sync_service import sync_tools as do_sync_tools
        await do_sync_tools()
        logger.info("Manual sync process completed successfully.")
        return jsonify({"message": "Sync process completed successfully."})
    except ImportError:
        logger.error("Could not import sync_tools from sync_service.")
        return jsonify({"error": "Sync service function not found."}), 500
    except Exception as e:
        logger.error(f"Error during manual sync: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error during sync: {str(e)}"}), 500


@tools_bp.route('/api/v1/tools/refresh', methods=['POST'])
async def refresh_tools():
    """Refresh the tool index from Letta API."""
    logger.debug("Received request for /api/v1/tools/refresh")
    try:
        if _read_tool_cache_func:
            await _read_tool_cache_func(force_reload=True)
        else:
            cache_service = get_tool_cache_service()
            await cache_service.read_tool_cache(force_reload=True)
        
        logger.info("Tool index refresh completed successfully.")
        return jsonify({"success": True, "message": "Tool index refreshed successfully"})
    except Exception as e:
        logger.error(f"Error refreshing tool index: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
