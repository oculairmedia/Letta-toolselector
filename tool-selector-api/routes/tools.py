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
- GET /api/v1/tools/lookup - Direct tool lookup by name/ID with fuzzy matching
- POST /api/v1/tools/direct-attach - Direct attach tools by name/ID
- POST /api/v1/tools/direct-detach - Direct detach tools by name/ID
- GET /api/v1/tools/agent/<agent_id> - List agent tools with categorization
- GET /api/v1/tools/inspect/<tool_name_or_id> - Inspect tool metadata and schema
"""

import os
import asyncio
import time
import difflib
from quart import Blueprint, request, jsonify
import logging
from typing import Optional, Callable, List, Dict, Any

# Import metrics (optional - graceful degradation if not available)
try:
    from metrics import record_search, record_attach, record_prune

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    record_search = record_attach = record_prune = None

from services.tool_cache import ToolCacheService, get_tool_cache_service
from services.tool_search import ToolSearchService
from models import ToolLimitsConfig

logger = logging.getLogger(__name__)

# Create the blueprint
tools_bp = Blueprint("tools", __name__)

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

# Pin service (injected via configure)
_pin_service: Optional[Any] = None
_tool_config: Optional[Any] = None


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
    audit_source_class: Optional[Any] = None,
    pin_service: Optional[Any] = None,
    tool_config: Optional[Any] = None,
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
    global _pin_service, _tool_config

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
    _pin_service = pin_service
    _tool_config = tool_config

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
        tool_name = result.get("name")
        if not tool_name:
            continue

        # O(1) lookup by name instead of O(n) linear search
        cached_tool = cache_service.get_tool_by_name(tool_name)

        if cached_tool:
            tool_type = cached_tool.get("tool_type", "")
            is_letta_core = cache_service.is_letta_core_tool(cached_tool)
            is_mcp_tool = tool_type == "external_mcp" or (not is_letta_core and tool_type == "custom")

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
        if "rerank_score" in result and "score" not in result:
            result["score"] = result["rerank_score"]


def _format_search_results(results: List[Dict]) -> List[Dict]:
    """Format search results for API response."""
    formatted = []
    for i, result in enumerate(results):
        formatted_result = {
            "name": result.get("name", ""),
            "description": result.get("description", ""),
            "source": result.get("source", "unknown"),
            "score": result.get("rerank_score", result.get("score", 0)),
            "rank": i + 1,
            "tool_type": result.get("tool_type", ""),
            "mcp_server_name": result.get("mcp_server_name"),
            "tags": result.get("tags", []),
            "json_schema": result.get("json_schema"),
        }
        formatted.append(formatted_result)
    return formatted


# =============================================================================
# Search Endpoints
# =============================================================================


@tools_bp.route("/api/v1/tools/search", methods=["POST"])
async def search():
    """Search for tools matching a query."""
    logger.debug("Received request for /api/v1/tools/search")
    start_time = time.time()

    try:
        data = await request.get_json()
        if not data:
            logger.warning("Search request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        query = data.get("query")
        limit = data.get("limit", 10)

        if not query:
            logger.warning("Search request missing 'query' parameter.")
            return jsonify({"error": "Query parameter is required"}), 400

        # Check for deprecated reranking parameter
        enable_reranking = data.get("enable_reranking", False)
        reranker_config = None
        if enable_reranking:
            logger.warning(
                "DEPRECATED: enable_reranking parameter is deprecated. "
                "Use /api/v1/tools/search/rerank endpoint instead."
            )
            reranker_config = {
                "enabled": True,
                "model": data.get("reranker_config", {}).get("model", "bge-reranker-v2-m3"),
                "base_url": data.get("reranker_config", {}).get("base_url", "http://localhost:8091"),
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
            final_results = filtered_results[:limit]
        else:
            logger.debug("Search successful, returning %d results", len(results))
            _normalize_scores(results)
            final_results = results

        # Track metrics
        if METRICS_ENABLED and record_search:
            duration = time.time() - start_time
            search_type = "reranked" if enable_reranking else "hybrid"
            record_search(search_type=search_type, duration=duration, result_count=len(final_results))

        return jsonify(final_results)

    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/search/rerank", methods=["POST"])
async def search_with_reranking():
    """Search with reranking for better relevance."""
    logger.debug("Received request for /api/v1/tools/search/rerank")

    try:
        data = await request.get_json()
        if not data:
            logger.warning("Search request received with no JSON body.")
            return jsonify({"error": "Request body must be JSON"}), 400

        query = data.get("query")
        limit = data.get("limit", 10)

        if not query:
            logger.warning("Search request missing 'query' parameter.")
            return jsonify({"error": "Query parameter is required"}), 400

        # Build reranker config
        reranker_config = {
            "enabled": True,
            "model": data.get("reranker_model", "bge-reranker-v2-m3"),
            "base_url": data.get("reranker_base_url", os.getenv("RERANKER_BASE_URL", "http://localhost:8091")),
            "top_k": data.get("reranker_top_k", limit),
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


@tools_bp.route("/api/v1/tools", methods=["GET"])
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


@tools_bp.route("/api/v1/tools/attach", methods=["POST"])
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

        query = data.get("query", "")
        limit = data.get("limit", 10)
        agent_id = data.get("agent_id")
        keep_tools = data.get("keep_tools", [])
        min_score = data.get("min_score", _default_min_score)
        skip_loop_trigger = data.get("skip_loop_trigger", False)

        logger.debug("Attach request payload: skip_loop_trigger=%s, keys=%s", skip_loop_trigger, list(data.keys()))

        if not agent_id:
            logger.warning("Attach request missing 'agent_id'.")
            return jsonify({"error": "agent_id is required"}), 400

        try:
            # 1. Fetch agent-specific info (name and current tools) directly from Letta
            agent_name, current_agent_tools = await asyncio.gather(
                _agent_service.fetch_agent_info(agent_id), _tool_manager.fetch_agent_tools(agent_id)
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
                    is_mcp = tool.get("tool_type") == "external_mcp" or (
                        not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"
                    )
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
                logger.warning(
                    "Weaviate client not ready or not initialized at /attach endpoint. Attempting re-initialization..."
                )
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
            matching_tools_from_search = await asyncio.to_thread(_search_tools_func, query=query, limit=limit)

            logger.debug("Found %d matching tools from Weaviate search.", len(matching_tools_from_search))

            # 3.5. Filter tools by min_score threshold
            filtered_tools = []
            for tool in matching_tools_from_search:
                tool_score = tool.get("rerank_score")
                if tool_score is None:
                    tool_score = tool.get("score", 0)

                tool_score_percent = tool_score * 100

                if tool_score_percent >= min_score:
                    filtered_tools.append(tool)
                    logger.debug(
                        f"Tool '{tool.get('name')}' passed filter with score {tool_score_percent:.1f}% >= {min_score}%"
                    )
                else:
                    logger.debug(
                        f"Tool '{tool.get('name')}' filtered out with score {tool_score_percent:.1f}% < {min_score}%"
                    )

            logger.debug(
                "Score filtering: %d of %d tools passed min_score threshold of %s%%",
                len(filtered_tools),
                len(matching_tools_from_search),
                min_score,
            )

            # 4. Process matching tools (check cache, register if needed)
            letta_tools_cache = await _read_tool_cache_func() if _read_tool_cache_func else []
            mcp_servers = await _read_mcp_servers_cache_func() if _read_mcp_servers_cache_func else []

            if _process_matching_tool_func:
                process_tasks = [
                    _process_matching_tool_func(tool, letta_tools_cache, mcp_servers) for tool in filtered_tools
                ]
                processed_tools_results = await asyncio.gather(*process_tasks, return_exceptions=True)

                processed_tools = []
                for i, res in enumerate(processed_tools_results):
                    if isinstance(res, Exception):
                        logger.error(
                            f"Error processing tool candidate {matching_tools_from_search[i].get('name', 'Unknown')}: {res}"
                        )
                    elif res:
                        processed_tools.append(res)

                logger.debug(
                    "Successfully processed/registered %d tools for attachment consideration.", len(processed_tools)
                )
            else:
                processed_tools = filtered_tools
                logger.warning("No process_matching_tool_func configured, using filtered tools directly")

            # 5. Pre-attach pruning: Check if we need to make room before attaching new tools
            MAX_TOTAL_TOOLS = int(os.getenv("MAX_TOTAL_TOOLS", "30"))
            MAX_MCP_TOOLS = int(os.getenv("MAX_MCP_TOOLS", "20"))
            MIN_MCP_TOOLS = int(os.getenv("MIN_MCP_TOOLS", "7"))

            total_current_tools = len(current_agent_tools)
            mcp_current_count = len(mcp_tools)
            core_current_count = total_current_tools - mcp_current_count

            new_tool_ids = set()
            for tool in processed_tools:
                tool_id = tool.get("id") or tool.get("tool_id")
                if tool_id and tool_id not in seen_tool_ids:
                    new_tool_ids.add(tool_id)

            new_tools_count = len(new_tool_ids)
            logger.info(
                f"Pre-attach analysis: current_total={total_current_tools}, current_mcp={mcp_current_count}, core={core_current_count}, new_tools={new_tools_count}"
            )
            logger.info(f"Limits: MAX_TOTAL={MAX_TOTAL_TOOLS}, MAX_MCP={MAX_MCP_TOOLS}, MIN_MCP={MIN_MCP_TOOLS}")

            projected_total = total_current_tools + new_tools_count
            projected_mcp = mcp_current_count + new_tools_count

            logger.info(f"Projected after attach: total={projected_total}, mcp={projected_mcp}")

            # Determine if we need pre-attach pruning
            needs_preattach_pruning = False
            if projected_total > MAX_TOTAL_TOOLS:
                logger.warning(
                    f"Pre-attach check: projected total ({projected_total}) exceeds MAX_TOTAL_TOOLS ({MAX_TOTAL_TOOLS})"
                )
                needs_preattach_pruning = True
            elif projected_mcp > MAX_MCP_TOOLS:
                logger.warning(
                    f"Pre-attach check: projected MCP count ({projected_mcp}) exceeds MAX_MCP_TOOLS ({MAX_MCP_TOOLS})"
                )
                needs_preattach_pruning = True

            # Perform pre-attach pruning if needed
            if needs_preattach_pruning and query:
                logger.info("Executing pre-attach pruning to make room for new tools...")

                min_removals_for_mcp = max(0, projected_mcp - MAX_MCP_TOOLS)
                min_removals_for_total = max(0, projected_total - MAX_TOTAL_TOOLS)
                min_removals_needed = max(min_removals_for_mcp, min_removals_for_total)

                max_removals_allowed = max(0, mcp_current_count - MIN_MCP_TOOLS)
                tools_to_remove = min(min_removals_needed, max_removals_allowed)

                logger.info(
                    f"Pre-attach pruning: need to remove {min_removals_needed} tools (min_for_mcp={min_removals_for_mcp}, min_for_total={min_removals_for_total})"
                )
                logger.info(
                    f"Pre-attach pruning: can remove up to {max_removals_allowed} tools (respecting MIN_MCP_TOOLS={MIN_MCP_TOOLS})"
                )
                logger.info(f"Pre-attach pruning: will remove {tools_to_remove} tools")

                if tools_to_remove > 0:
                    effective_drop_rate = min(0.9, tools_to_remove / max(1, mcp_current_count))

                    logger.info(
                        f"Pre-attach pruning: using drop_rate={effective_drop_rate:.2f} to remove ~{tools_to_remove} tools"
                    )

                    preattach_prune_result = await _tool_manager.perform_tool_pruning(
                        agent_id=agent_id,
                        user_prompt=query,
                        drop_rate=effective_drop_rate,
                        keep_tool_ids=keep_tools,
                        newly_matched_tool_ids=[],
                    )

                    if preattach_prune_result.get("success"):
                        removed_count = preattach_prune_result.get("details", {}).get("mcp_tools_detached_count", 0)
                        logger.info(f"Pre-attach pruning completed: removed {removed_count} tools to make room")

                        # Use remaining_tools from pruning result to avoid redundant API call
                        current_agent_tools = preattach_prune_result.get("remaining_tools", [])
                        mcp_tools = []
                        seen_tool_ids = set()

                        for tool in current_agent_tools:
                            is_mcp_tool = tool.get("tool_type") == "external_mcp" or (
                                not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"
                            )

                            if is_mcp_tool:
                                tool_id = tool.get("id") or tool.get("tool_id")
                                if tool_id and tool_id not in seen_tool_ids:
                                    seen_tool_ids.add(tool_id)
                                    tool_copy = tool.copy()
                                    tool_copy["id"] = tool_id
                                    tool_copy["tool_id"] = tool_id
                                    mcp_tools.append(tool_copy)

                        logger.info(
                            f"After pre-attach pruning: total_tools={len(current_agent_tools)}, mcp_tools={len(mcp_tools)}"
                        )
                    else:
                        logger.warning(
                            f"Pre-attach pruning failed: {preattach_prune_result.get('error', 'Unknown error')}"
                        )
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
                            reason=f"Query match: {query[:100] if query else 'no query'}"
                            if query
                            else "Requested tool attachment",
                            correlation_id=correlation_id,
                            success_count=len(results["successful_attachments"]),
                            failure_count=0,
                        )

                    if results.get("failed_attachments"):
                        _emit_batch_event_func(
                            action=_audit_action_class.ATTACH,
                            agent_id=agent_id,
                            tools=[
                                {
                                    "tool_id": t.get("tool_id") or t.get("id"),
                                    "name": t.get("name", "unknown"),
                                    "success": False,
                                }
                                for t in results["failed_attachments"]
                            ],
                            source=_audit_source_class.API_ATTACH,
                            reason="Attachment failed",
                            correlation_id=correlation_id,
                            success_count=0,
                            failure_count=len(results["failed_attachments"]),
                        )

                    if results.get("detached_tools"):
                        _emit_batch_event_func(
                            action=_audit_action_class.DETACH,
                            agent_id=agent_id,
                            tools=[
                                {"tool_id": tool_id, "name": "unknown", "success": True}
                                for tool_id in results["detached_tools"]
                            ],
                            source=_audit_source_class.API_ATTACH,
                            reason="Making room for new tools",
                            correlation_id=correlation_id,
                            success_count=len(results["detached_tools"]),
                            failure_count=0,
                        )

                except Exception as audit_error:
                    logger.warning(f"Failed to emit audit events: {audit_error}")

            # 7. Fetch pinned tools for the response (before loop trigger to avoid latency)
            pinned_tools_info = []
            if _pin_service and agent_id:
                try:
                    pinned_ids = await _pin_service.get_pinned_tools(agent_id)
                    if pinned_ids:
                        # Build name map from in-memory mcp_tools + successful attachments
                        # This avoids a redundant HTTP call to Letta API
                        id_to_name = {}
                        for t in mcp_tools:
                            tid = t.get("id") or t.get("tool_id")
                            if tid:
                                id_to_name[tid] = t.get("name", "")
                        for t in results.get("successful_attachments", []):
                            tid = t.get("tool_id") or t.get("id")
                            if tid:
                                id_to_name[tid] = t.get("name", "")
                        pinned_tools_info = [
                            {"tool_id": tid, "name": id_to_name.get(tid, "unknown")} for tid in pinned_ids
                        ]
                except Exception as pin_err:
                    logger.warning(f"Could not fetch pinned tools for response: {pin_err}")

            # 8. Trigger a new agent loop so newly attached tools are available
            loop_triggered = False
            successful_attachments = results.get("successful_attachments", [])
            logger.info(
                f"Checking if loop trigger needed: {len(successful_attachments)} successful attachments, skip_loop_trigger={skip_loop_trigger}"
            )
            if successful_attachments and not skip_loop_trigger:
                logger.info(f"Triggering agent loop for {agent_id} with query: {query}")
                try:
                    loop_triggered = _agent_service.trigger_agent_loop(agent_id, successful_attachments, query=query)
                    logger.info(f"Loop trigger task spawned: {loop_triggered}")
                except Exception as trigger_error:
                    logger.error(f"Exception during agent_service.trigger_agent_loop: {trigger_error}", exc_info=True)
            return jsonify(
                {
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
                        "preserved_tools": results.get("protected_tool_ids", keep_tools),
                        "target_agent": agent_id,
                        "loop_triggered": loop_triggered,
                        "pinned_tools": pinned_tools_info,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error during tool management: {str(e)}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error during attach_tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# Tool Pruning Endpoint
# =============================================================================


@tools_bp.route("/api/v1/tools/prune", methods=["POST"])
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
        agent_id = data.get("agent_id")
        user_prompt = data.get("user_prompt")
        drop_rate = data.get("drop_rate")

        # Extract optional parameters
        keep_tool_ids = data.get("keep_tool_ids", [])
        newly_matched_tool_ids = data.get("newly_matched_tool_ids", [])

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
            newly_matched_tool_ids=newly_matched_tool_ids,
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
                        tools_protected=details.get("explicitly_kept_tool_ids_from_request", [])
                        + details.get("newly_matched_tool_ids_from_request", []),
                        drop_rate=drop_rate,
                        correlation_id=correlation_id,
                        metadata={
                            "mcp_tools_before": details.get("mcp_tools_on_agent_before", 0),
                            "target_mcp_tools": details.get("target_mcp_tools_to_keep_after_pruning", 0),
                            "user_prompt_snippet": user_prompt[:100] if user_prompt else "no prompt",
                            "failed_detachments": len(details.get("failed_detachments_mcp", [])),
                        },
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
                            failure_count=len(details.get("failed_detachments_mcp", [])),
                        )

                    if details.get("failed_detachments_mcp"):
                        _emit_batch_event_func(
                            action=_audit_action_class.DETACH,
                            agent_id=agent_id,
                            tools=[
                                {"tool_id": t.get("tool_id"), "name": t.get("name", "unknown"), "success": False}
                                for t in details.get("failed_detachments_mcp", [])
                            ],
                            source=_audit_source_class.API_PRUNE,
                            reason="Detachment failed during pruning",
                            correlation_id=correlation_id,
                            success_count=0,
                            failure_count=len(details.get("failed_detachments_mcp", [])),
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


@tools_bp.route("/api/v1/tools/sync", methods=["POST"])
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


@tools_bp.route("/api/v1/tools/refresh", methods=["POST"])
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


# =============================================================================
# Fuzzy Matching Helper
# =============================================================================


def fuzzy_match_tools(query: str, tool_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find fuzzy matches for a tool name query.

    Uses a weighted combination of:
    1. Prefix matching (highest priority)
    2. Substring matching
    3. difflib close matches (Levenshtein-like distance)

    Operates on tool_names list — O(n) but cache is small (~200-500 tools).
    Uses only stdlib (difflib.get_close_matches), no new dependencies.

    Args:
        query: Tool name query string
        tool_names: List of known tool names to match against
        limit: Maximum number of suggestions to return (default 5)

    Returns:
        List of dicts with 'name' and 'score' keys, sorted by score descending
    """
    if not query or not tool_names:
        return []

    query_lower = query.lower()
    scored: Dict[str, float] = {}

    # 1. Prefix match (highest weight — user typed the beginning)
    for name in tool_names:
        name_lower = name.lower()
        if name_lower.startswith(query_lower):
            # Shorter names that match = closer match
            score = 0.95 - (len(name) - len(query)) * 0.005
            scored[name] = max(scored.get(name, 0), score)

    # 2. Substring match (query appears somewhere in the name)
    for name in tool_names:
        name_lower = name.lower()
        if query_lower in name_lower and name not in scored:
            pos = name_lower.index(query_lower)
            coverage = len(query) / len(name)
            score = 0.70 + coverage * 0.15 - pos * 0.003
            scored[name] = max(scored.get(name, 0), score)

    # 3. difflib close matches (handles typos / transpositions)
    close = difflib.get_close_matches(query_lower, [n.lower() for n in tool_names], n=limit * 2, cutoff=0.4)
    name_lookup = {n.lower(): n for n in tool_names}
    for match in close:
        original_name = name_lookup.get(match, match)
        if original_name not in scored:
            ratio = difflib.SequenceMatcher(None, query_lower, match).ratio()
            scored[original_name] = max(scored.get(original_name, 0), ratio * 0.80)

    results = [
        {"name": name, "score": round(score, 2)}
        for name, score in sorted(scored.items(), key=lambda x: x[1], reverse=True)
    ]
    return results[:limit]


def _get_suggestions(query: str) -> List[Dict[str, Any]]:
    """Get fuzzy suggestions for a tool name query using the tool cache."""
    cache_service = get_tool_cache_service()
    all_names = [t.get("name", "") for t in cache_service.get_cached_tools() if t.get("name")]
    return fuzzy_match_tools(query, all_names)


def _build_tool_response(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Build a standardized tool response dict from a cache entry."""
    return {
        "id": tool.get("id") or tool.get("tool_id"),
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "tool_type": tool.get("tool_type", ""),
        "source_type": tool.get("source_type", ""),
        "json_schema": tool.get("json_schema"),
        "tags": tool.get("tags", []),
        "module": tool.get("module", ""),
        "server_name": tool.get("mcp_server_name", ""),
    }


async def _ensure_cache_loaded():
    """Ensure the tool cache is loaded (handles cold start)."""
    cache_service = get_tool_cache_service()
    tools = cache_service.get_cached_tools()
    if not tools:
        logger.info("Tool cache empty, triggering load...")
        await cache_service.read_tool_cache(force_reload=True)


# =============================================================================
# Direct Tool Management Endpoints
# =============================================================================


@tools_bp.route("/api/v1/tools/lookup", methods=["GET"])
async def lookup_tool():
    """
    Look up a tool by exact name, ID, or fuzzy match.

    Query params:
        tool_name (str, optional): Tool name to match
        tool_id (str, optional): Exact tool ID
        fuzzy (bool, optional, default false): Enable fuzzy/prefix matching
    """
    logger.debug("Received request for /api/v1/tools/lookup")

    try:
        tool_name = request.args.get("tool_name")
        tool_id = request.args.get("tool_id")
        fuzzy = request.args.get("fuzzy", "false").lower() == "true"

        if not tool_name and not tool_id:
            return jsonify({"error": "At least one of tool_name or tool_id is required"}), 400

        await _ensure_cache_loaded()
        cache_service = get_tool_cache_service()

        # Exact match by ID
        if tool_id:
            tool = cache_service.get_tool_by_id(tool_id)
            if tool:
                return jsonify({"success": True, "tool": _build_tool_response(tool)})
            # ID not found
            return jsonify(
                {
                    "success": False,
                    "error": "Tool not found",
                    "tool_not_found": True,
                    "searched_by": "id",
                    "value": tool_id,
                    "suggestions": [],
                }
            ), 404

        # Exact match by name
        tool = cache_service.get_tool_by_name(tool_name) if tool_name else None
        if tool:
            return jsonify({"success": True, "tool": _build_tool_response(tool)})

        # Not found — compute suggestions
        suggestions = _get_suggestions(tool_name) if tool_name else []

        if fuzzy and suggestions:
            # Fuzzy mode: return suggestions as the successful result
            return jsonify({"success": True, "exact_match": None, "suggestions": suggestions})

        # 404 with suggestions (PM requirement: ALWAYS include suggestions)
        return jsonify(
            {
                "success": False,
                "error": "Tool not found",
                "tool_not_found": True,
                "searched_by": "name",
                "value": tool_name,
                "suggestions": suggestions,
            }
        ), 404

    except Exception as e:
        logger.error(f"Error during tool lookup: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/direct-attach", methods=["POST"])
async def direct_attach_tools():
    """
    Attach specific tools to an agent by name or ID.

    Bypasses semantic search entirely. Supports single and bulk operations.
    DOES NOT trigger auto-pruning.
    """
    logger.debug("Received request for /api/v1/tools/direct-attach")

    if not _tool_manager:
        return jsonify({"error": "Tool attachment not configured - missing tool_manager"}), 503

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        agent_id = data.get("agent_id")
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400

        tools_input = data.get("tools", [])
        pin = data.get("pin", False)

        # Normalize single tool to list
        if isinstance(tools_input, dict):
            tools_input = [tools_input]

        if not tools_input:
            return jsonify({"error": "tools array is required and cannot be empty"}), 400

        await _ensure_cache_loaded()
        cache_service = get_tool_cache_service()

        # Fetch current agent tools for "already attached" detection
        current_tools = await _tool_manager.fetch_agent_tools(agent_id)
        current_tool_ids = {(t.get("id") or t.get("tool_id")) for t in current_tools if t.get("id") or t.get("tool_id")}

        attached = []
        failed = []
        pinned_names = []

        async def _resolve_and_attach(item: Dict[str, Any]):
            """Resolve a tool by name/ID and attach it."""
            name = item.get("name")
            tid = item.get("tool_id") or item.get("id")

            # Resolve tool from cache
            tool = None
            if tid:
                tool = cache_service.get_tool_by_id(tid)
            if not tool and name:
                tool = cache_service.get_tool_by_name(name)

            if not tool:
                query = name or tid or "unknown"
                # TODO: Also provide suggestions for ID-based lookups (e.g. typo'd tool IDs). Low priority. (PM feedback)
                suggestions = _get_suggestions(query) if name else []
                return {
                    "name": name or tid,
                    "error": "Tool not found in cache",
                    "tool_not_found": True,
                    "suggestions": suggestions,
                }

            resolved_id = tool.get("id") or tool.get("tool_id")
            resolved_name = tool.get("name", "")

            # Already attached?
            if resolved_id in current_tool_ids:
                return {"tool_id": resolved_id, "name": resolved_name, "status": "already_attached"}

            # Attach
            result = await _tool_manager.attach_tool(agent_id, tool)
            if result.get("success"):
                return {"tool_id": resolved_id, "name": resolved_name, "status": "attached"}
            else:
                return {
                    "tool_id": resolved_id,
                    "name": resolved_name,
                    "error": result.get("error", "Attachment failed"),
                }

        # Process all tools in parallel
        tasks = [_resolve_and_attach(item) for item in tools_input]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append(
                    {
                        "name": tools_input[i].get("name") or tools_input[i].get("tool_id", "unknown"),
                        "error": str(result),
                    }
                )
            elif isinstance(result, dict) and "error" in result:
                failed.append(result)
            elif isinstance(result, dict):
                attached.append(result)

        # Pin tools if requested
        if pin and _pin_service and attached:
            ids_to_pin = [
                r["tool_id"]
                for r in attached
                if r.get("status") in ("attached", "already_attached") and r.get("tool_id")
            ]
            if ids_to_pin:
                await _pin_service.pin_tools(agent_id, ids_to_pin)
                pinned_names = [r["name"] for r in attached if r.get("tool_id") in ids_to_pin]

        total_attached = sum(1 for r in attached if r.get("status") == "attached")

        return jsonify(
            {
                "success": True,
                "message": f"Attached {total_attached} tools",
                "details": {"attached": attached, "failed": failed, "pinned": pinned_names},
            }
        )

    except Exception as e:
        logger.error(f"Error during direct attach: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/direct-detach", methods=["POST"])
async def direct_detach_tools():
    """
    Detach specific tools from an agent by name or ID.

    Respects protected (NEVER_DETACH_TOOLS) and pinned tools.
    Supports single and bulk operations.
    """
    logger.debug("Received request for /api/v1/tools/direct-detach")

    if not _tool_manager:
        return jsonify({"error": "Tool detachment not configured - missing tool_manager"}), 503

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        agent_id = data.get("agent_id")
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400

        tools_input = data.get("tools", [])
        force_unpin = data.get("unpin", False)

        # Normalize single tool to list
        if isinstance(tools_input, dict):
            tools_input = [tools_input]

        if not tools_input:
            return jsonify({"error": "tools array is required and cannot be empty"}), 400

        await _ensure_cache_loaded()
        cache_service = get_tool_cache_service()

        # Fetch current agent tools
        current_tools = await _tool_manager.fetch_agent_tools(agent_id)
        current_tool_map = {}
        for t in current_tools:
            tid = t.get("id") or t.get("tool_id")
            if tid:
                current_tool_map[tid] = t
                name = t.get("name", "")
                if name:
                    current_tool_map[name] = t

        # Get pinned tools for this agent
        pinned_ids = set()
        if _pin_service:
            pinned_ids = set(await _pin_service.get_pinned_tools(agent_id))

        # Get protected tool checker
        # ToolLimitsConfig imported at module level
        config = _tool_config if _tool_config else ToolLimitsConfig.from_env()

        detached = []
        refused = []
        failed = []

        async def _resolve_and_detach(item: Dict[str, Any]):
            """Resolve a tool by name/ID and detach it."""
            name = item.get("name")
            tid = item.get("tool_id") or item.get("id")

            # Resolve from cache first (for metadata)
            tool = None
            if tid:
                tool = cache_service.get_tool_by_id(tid)
            if not tool and name:
                tool = cache_service.get_tool_by_name(name)

            if not tool:
                # Try resolving from current agent tools directly
                lookup_key = name or tid
                if lookup_key and lookup_key in current_tool_map:
                    tool = current_tool_map[lookup_key]

            if not tool:
                query = name or tid or "unknown"
                suggestions = _get_suggestions(query) if name else []
                return "failed", {
                    "name": name or tid,
                    "error": "Tool not found in cache",
                    "tool_not_found": True,
                    "suggestions": suggestions,
                }

            resolved_id = tool.get("id") or tool.get("tool_id")
            resolved_name = tool.get("name", "")

            # Check if attached to agent
            if resolved_id not in current_tool_map:
                return "failed", {"name": resolved_name, "error": "Tool not attached to agent"}

            # Check protected tools (global)
            if config.should_protect_tool(resolved_name):
                return "refused", {"name": resolved_name, "reason": "Protected tool - cannot detach"}

            # Check pinned tools (per-agent)
            if resolved_id in pinned_ids and not force_unpin:
                return "refused", {"name": resolved_name, "reason": "Pinned tool - use unpin=true to force detach"}

            # Unpin if force_unpin is set
            if resolved_id in pinned_ids and force_unpin and _pin_service:
                await _pin_service.unpin_tools(agent_id, [resolved_id])

            # Detach
            result = await _tool_manager.detach_tool(agent_id, resolved_id, resolved_name)
            if result.get("success"):
                return "detached", {"tool_id": resolved_id, "name": resolved_name, "status": "detached"}
            else:
                return "failed", {
                    "tool_id": resolved_id,
                    "name": resolved_name,
                    "error": result.get("error", "Detachment failed"),
                }

        # Process all tools in parallel
        tasks = [_resolve_and_detach(item) for item in tools_input]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append(
                    {
                        "name": tools_input[i].get("name") or tools_input[i].get("tool_id", "unknown"),
                        "error": str(result),
                    }
                )
            elif isinstance(result, tuple):
                category, detail = result
                if category == "detached":
                    detached.append(detail)
                elif category == "refused":
                    refused.append(detail)
                else:
                    failed.append(detail)

        total_detached = len(detached)

        return jsonify(
            {
                "success": True,
                "message": f"Detached {total_detached} tools",
                "details": {"detached": detached, "refused": refused, "failed": failed},
            }
        )

    except Exception as e:
        logger.error(f"Error during direct detach: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/agent/<agent_id>", methods=["GET"])
async def list_agent_tools(agent_id: str):
    """
    List all tools currently attached to an agent, categorized by type.

    Query params:
        filter (str, optional): 'all' (default), 'mcp', 'core'
        include_schema (bool, optional, default false): Include JSON schemas
    """
    logger.debug("Received request for /api/v1/tools/agent/%s", agent_id)

    if not _tool_manager:
        return jsonify({"error": "Tool listing not configured - missing tool_manager"}), 503

    try:
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400

        tool_filter = request.args.get("filter", "all")
        include_schema = request.args.get("include_schema", "false").lower() == "true"

        # Fetch current agent tools
        agent_tools = await _tool_manager.fetch_agent_tools(agent_id)

        # Get pin and protection info
        pinned_ids = set()
        if _pin_service:
            pinned_ids = set(await _pin_service.get_pinned_tools(agent_id))

        # ToolLimitsConfig imported at module level
        config = _tool_config if _tool_config else ToolLimitsConfig.from_env()

        core_count = 0
        mcp_count = 0
        pinned_count = 0
        protected_count = 0
        tools_list = []

        for tool in agent_tools:
            tool_id = tool.get("id") or tool.get("tool_id")
            tool_name = tool.get("name", "")
            is_core = _is_letta_core_tool(tool)
            is_mcp = tool.get("tool_type") == "external_mcp" or (not is_core and tool.get("tool_type") == "custom")
            is_pinned = tool_id in pinned_ids
            is_protected = config.should_protect_tool(tool_name)

            if is_core:
                core_count += 1
            else:
                mcp_count += 1
            if is_pinned:
                pinned_count += 1
            if is_protected:
                protected_count += 1

            # Apply filter
            if tool_filter == "mcp" and not is_mcp:
                continue
            if tool_filter == "core" and not is_core:
                continue

            tool_entry = {
                "tool_id": tool_id,
                "name": tool_name,
                "tool_type": tool.get("tool_type", ""),
                "source_type": tool.get("source_type", ""),
                "description": tool.get("description", ""),
                "is_pinned": is_pinned,
                "is_protected": is_protected,
            }

            if include_schema:
                tool_entry["json_schema"] = tool.get("json_schema")

            tools_list.append(tool_entry)

        return jsonify(
            {
                "success": True,
                "agent_id": agent_id,
                "tool_count": len(tools_list),
                "summary": {
                    "core_tools": core_count,
                    "mcp_tools": mcp_count,
                    "pinned_tools": pinned_count,
                    "protected_tools": protected_count,
                },
                "tools": tools_list,
            }
        )

    except Exception as e:
        logger.error(f"Error listing agent tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/pins/<agent_id>", methods=["GET"])
async def get_agent_pins(agent_id: str):
    """
    Get pinned tools for an agent.

    Returns the list of pinned tool IDs with resolved names.
    Agents can call this to discover which of their tools are pinned.
    """
    logger.debug("Received request for /api/v1/tools/pins/%s", agent_id)

    if not _pin_service:
        return jsonify({"error": "Pin service not configured"}), 503

    try:
        if not agent_id:
            return jsonify({"error": "agent_id is required"}), 400

        pinned_ids = await _pin_service.get_pinned_tools(agent_id)

        # Resolve tool names if tool_manager is available
        pinned_tools = []
        if _tool_manager and pinned_ids:
            try:
                agent_tools = await _tool_manager.fetch_agent_tools(agent_id)
                id_to_name = {(t.get("id") or t.get("tool_id")): t.get("name", "") for t in agent_tools}
                for tid in pinned_ids:
                    pinned_tools.append(
                        {
                            "tool_id": tid,
                            "name": id_to_name.get(tid, "unknown"),
                        }
                    )
            except Exception as resolve_err:
                logger.warning(f"Could not resolve pin names: {resolve_err}")
                pinned_tools = [{"tool_id": tid, "name": "unknown"} for tid in pinned_ids]
        else:
            pinned_tools = [{"tool_id": tid, "name": "unknown"} for tid in pinned_ids]

        return jsonify(
            {
                "success": True,
                "agent_id": agent_id,
                "pinned_count": len(pinned_tools),
                "pinned_tools": pinned_tools,
            }
        )

    except Exception as e:
        logger.error(f"Error getting pinned tools: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@tools_bp.route("/api/v1/tools/inspect/<path:tool_name_or_id>", methods=["GET"])
async def inspect_tool(tool_name_or_id: str):
    """
    Inspect full metadata and schema for a tool before attaching.

    The path parameter accepts either a tool name or tool ID.
    Resolution order: try ID first (UUIDs are unambiguous), then name.
    """
    logger.debug("Received request for /api/v1/tools/inspect/%s", tool_name_or_id)

    try:
        await _ensure_cache_loaded()
        cache_service = get_tool_cache_service()

        # Try ID first (UUIDs are unambiguous)
        tool = cache_service.get_tool_by_id(tool_name_or_id)
        if not tool:
            tool = cache_service.get_tool_by_name(tool_name_or_id)

        if not tool:
            suggestions = _get_suggestions(tool_name_or_id)
            return jsonify(
                {
                    "success": False,
                    "error": "Tool not found",
                    "tool_not_found": True,
                    "searched_by": "name_or_id",
                    "value": tool_name_or_id,
                    "suggestions": suggestions,
                }
            ), 404

        tool_name = tool.get("name", "")
        server_name = tool.get("mcp_server_name", "")
        json_schema = tool.get("json_schema", {})

        # Build parameters_summary from json_schema
        params_summary = _build_params_summary(json_schema)

        # Find related tools (same MCP server prefix)
        related = _find_related_tools(tool_name, server_name, cache_service)

        # Check protection status
        # ToolLimitsConfig imported at module level
        config = _tool_config if _tool_config else ToolLimitsConfig.from_env()
        is_protected = config.should_protect_tool(tool_name)

        return jsonify(
            {
                "success": True,
                "tool": {
                    "id": tool.get("id") or tool.get("tool_id"),
                    "name": tool_name,
                    "description": tool.get("description", ""),
                    "tool_type": tool.get("tool_type", ""),
                    "source_type": tool.get("source_type", ""),
                    "server_name": server_name,
                    "json_schema": json_schema,
                    "parameters_summary": params_summary,
                    "tags": tool.get("tags", []),
                    "related_tools": related,
                    "is_protected": is_protected,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error inspecting tool: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def _build_params_summary(json_schema: Dict[str, Any]) -> str:
    """
    Build a human-readable one-liner summary of tool parameters from JSON schema.

    Example output: "shelf_id (int, required), name (string, required), html (string, optional)"
    """
    if not json_schema:
        return ""

    properties = json_schema.get("properties", {})
    required_fields = set(json_schema.get("required", []))

    if not properties:
        return ""

    parts = []
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "any")
        is_required = param_name in required_fields
        req_label = "required" if is_required else "optional"
        parts.append(f"{param_name} ({param_type}, {req_label})")

    return ", ".join(parts)


def _find_related_tools(tool_name: str, server_name: str, cache_service: ToolCacheService, limit: int = 5) -> List[str]:
    """
    Find related tools from the same MCP server.

    Uses server_name prefix matching. Returns up to `limit` tool names,
    excluding the tool itself.
    """
    if not server_name:
        return []

    related = []
    for t in cache_service.get_cached_tools():
        t_name = t.get("name", "")
        t_server = t.get("mcp_server_name", "")
        if t_server == server_name and t_name != tool_name:
            related.append(t_name)
            if len(related) >= limit:
                break

    return related
