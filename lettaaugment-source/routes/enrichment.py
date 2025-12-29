"""
Enrichment Routes

API endpoints for semantic enrichment operations:
- Profile MCP servers
- Enrich individual tools
- Run full enrichment sync
- Get enrichment statistics
"""

import logging
import subprocess
import sys
from typing import Dict, Any, List, Optional, Callable
from quart import Blueprint, jsonify, request
from datetime import datetime

logger = logging.getLogger(__name__)

enrichment_bp = Blueprint('enrichment', __name__, url_prefix='/api/v1/enrichment')

# Configurable dependencies
_get_tools_by_server_func: Optional[Callable[[], Dict[str, List[Dict[str, Any]]]]] = None
_get_all_tools_func: Optional[Callable[[], List[Dict[str, Any]]]] = None


def configure(
    get_tools_by_server_func: Optional[Callable] = None,
    get_all_tools_func: Optional[Callable] = None
) -> None:
    """Configure enrichment routes with dependencies."""
    global _get_tools_by_server_func, _get_all_tools_func
    
    if get_tools_by_server_func:
        _get_tools_by_server_func = get_tools_by_server_func
    if get_all_tools_func:
        _get_all_tools_func = get_all_tools_func


@enrichment_bp.route('/stats', methods=['GET'])
async def get_enrichment_stats():
    """
    Get enrichment cache statistics.
    
    Returns:
        JSON with cache stats: profiles_cached, tools_cached, etc.
    """
    try:
        from semantic_enrichment import EnrichmentCache
        cache = EnrichmentCache()
        stats = cache.get_stats()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Failed to get enrichment stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/health', methods=['GET'])
async def enrichment_health():
    """
    Check enrichment system health including Anthropic API.
    
    Returns:
        JSON with health status for cache and API
    """
    try:
        from semantic_enrichment import AnthropicClient, EnrichmentCache
        
        # Check cache
        cache = EnrichmentCache()
        cache_stats = cache.get_stats()
        
        # Check API (optional - can be slow)
        check_api = request.args.get('check_api', 'false').lower() == 'true'
        api_status = None
        
        if check_api:
            try:
                client = AnthropicClient()
                api_status = client.health_check()
            except Exception as e:
                api_status = {"healthy": False, "error": str(e)}
        
        return jsonify({
            "success": True,
            "cache": {
                "healthy": True,
                **cache_stats
            },
            "api": api_status
        })
    except Exception as e:
        logger.error(f"Enrichment health check failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/server/<server_name>/profile', methods=['POST'])
async def profile_server(server_name: str):
    """
    Profile or re-profile an MCP server.
    
    Args:
        server_name: Name of the MCP server to profile
        
    Query params:
        force: If true, regenerate even if cached
        
    Returns:
        JSON with server profile
    """
    try:
        from semantic_enrichment import MCPServerProfiler
        
        force = request.args.get('force', 'false').lower() == 'true'
        
        # Get tools for this server
        if not _get_tools_by_server_func:
            return jsonify({
                "success": False,
                "error": "get_tools_by_server function not configured"
            }), 503
        
        servers_tools = _get_tools_by_server_func()
        tools = servers_tools.get(server_name, [])
        
        if not tools:
            return jsonify({
                "success": False,
                "error": f"No tools found for server: {server_name}"
            }), 404
        
        # Profile the server
        profiler = MCPServerProfiler()
        result = profiler.profile_server(server_name, tools, force=force)
        
        if result.success:
            return jsonify({
                "success": True,
                "profile": result.profile.model_dump(mode='json') if result.profile else None,
                "duration_ms": result.duration_ms,
                "tokens_used": result.tokens_used,
                "tools_analyzed": result.tools_analyzed
            })
        else:
            return jsonify({
                "success": False,
                "error": result.error
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to profile server {server_name}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/server/<server_name>/status', methods=['GET'])
async def get_server_profile_status(server_name: str):
    """
    Get profile status for an MCP server.
    
    Returns:
        JSON with profile info and whether it needs updating
    """
    try:
        from semantic_enrichment import EnrichmentCache
        
        cache = EnrichmentCache()
        profile = cache.get_profile(server_name)
        
        if profile:
            # Check if update needed
            needs_update = False
            if _get_tools_by_server_func:
                servers_tools = _get_tools_by_server_func()
                tools = servers_tools.get(server_name, [])
                needs_update = cache.needs_profile_update(server_name, tools)
            
            return jsonify({
                "success": True,
                "exists": True,
                "profile": {
                    "server_name": profile.server_name,
                    "domain": profile.domain,
                    "tool_count": profile.tool_count,
                    "last_updated": profile.last_updated.isoformat(),
                    "profile_hash": profile.profile_hash
                },
                "needs_update": needs_update
            })
        else:
            return jsonify({
                "success": True,
                "exists": False,
                "needs_update": True
            })
            
    except Exception as e:
        logger.error(f"Failed to get profile status for {server_name}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/tool/<tool_id>/enrich', methods=['POST'])
async def enrich_tool(tool_id: str):
    """
    Enrich a single tool.
    
    Query params:
        force: If true, re-enrich even if cached
        
    Returns:
        JSON with enrichment result
    """
    try:
        from semantic_enrichment import ToolEnricher, EnrichmentCache
        
        force = request.args.get('force', 'false').lower() == 'true'
        
        # Get tool data
        if not _get_all_tools_func:
            return jsonify({
                "success": False,
                "error": "get_all_tools function not configured"
            }), 503
        
        all_tools = _get_all_tools_func()
        tool = next((t for t in all_tools if t.get('id') == tool_id or t.get('tool_id') == tool_id), None)
        
        if not tool:
            return jsonify({
                "success": False,
                "error": f"Tool not found: {tool_id}"
            }), 404
        
        # Get server profile for context
        server_name = tool.get('mcp_server_name', '')
        server_profile = None
        if server_name:
            cache = EnrichmentCache()
            server_profile = cache.get_profile(server_name)
        
        # Enrich the tool
        enricher = ToolEnricher()
        result = enricher.enrich_tool(tool, server_profile, force=force)
        
        return jsonify({
            "success": result.success,
            "tool_id": result.tool_id,
            "tool_name": result.tool_name,
            "enhanced_description": result.enhanced_description,
            "action_entities": result.action_entities,
            "semantic_keywords": result.semantic_keywords,
            "use_cases": result.use_cases,
            "duration_ms": result.duration_ms,
            "tokens_used": result.tokens_used,
            "error": result.error
        })
        
    except Exception as e:
        logger.error(f"Failed to enrich tool {tool_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/sync', methods=['POST'])
async def run_enrichment_sync():
    """
    Run full enrichment sync.
    
    This profiles all servers and enriches all tools that need updating.
    Only processes tools that have changed since last enrichment.
    
    Query params:
        force: If true, re-enrich everything
        dry_run: If true, show what would be done without making changes
        
    Returns:
        JSON with sync statistics
    """
    try:
        from semantic_enrichment import MCPServerProfiler, ToolEnricher, EnrichmentCache
        from semantic_enrichment.models import EnrichmentStats
        
        force = request.args.get('force', 'false').lower() == 'true'
        dry_run = request.args.get('dry_run', 'false').lower() == 'true'
        
        if not _get_tools_by_server_func:
            return jsonify({
                "success": False,
                "error": "get_tools_by_server function not configured"
            }), 503
        
        # Get all tools grouped by server
        servers_tools = _get_tools_by_server_func()
        
        stats = EnrichmentStats(
            total_servers=len(servers_tools),
            started_at=datetime.utcnow()
        )
        
        profiler = MCPServerProfiler()
        enricher = ToolEnricher()
        cache = EnrichmentCache()
        
        # Profile servers
        logger.info(f"Profiling {len(servers_tools)} servers (force={force}, dry_run={dry_run})")
        
        for server_name, tools in servers_tools.items():
            stats.total_tools += len(tools)
            
            # Check if profile needs update
            if not force and not cache.needs_profile_update(server_name, tools):
                stats.servers_cached += 1
                continue
            
            if dry_run:
                logger.info(f"[DRY RUN] Would profile server: {server_name}")
                stats.servers_profiled += 1
                continue
            
            # Profile the server
            result = profiler.profile_server(server_name, tools, force=force)
            if result.success:
                stats.servers_profiled += 1
                stats.total_tokens_used += result.tokens_used
            else:
                stats.servers_failed += 1
        
        # Enrich tools
        logger.info(f"Enriching tools (total={stats.total_tools})")
        
        for server_name, tools in servers_tools.items():
            server_profile = cache.get_profile(server_name)
            
            for tool in tools:
                if not force and not cache.needs_tool_enrichment(tool):
                    stats.tools_cached += 1
                    continue
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would enrich tool: {tool.get('name')}")
                    stats.tools_enriched += 1
                    continue
                
                result = enricher.enrich_tool(tool, server_profile, force=force)
                if result.success:
                    stats.tools_enriched += 1
                    stats.total_tokens_used += result.tokens_used
                else:
                    stats.tools_failed += 1
        
        stats.completed_at = datetime.utcnow()
        stats.total_duration_ms = (stats.completed_at - stats.started_at).total_seconds() * 1000
        
        return jsonify({
            "success": True,
            "dry_run": dry_run,
            "stats": {
                "total_servers": stats.total_servers,
                "servers_profiled": stats.servers_profiled,
                "servers_cached": stats.servers_cached,
                "servers_failed": stats.servers_failed,
                "total_tools": stats.total_tools,
                "tools_enriched": stats.tools_enriched,
                "tools_cached": stats.tools_cached,
                "tools_failed": stats.tools_failed,
                "total_duration_ms": stats.total_duration_ms,
                "total_tokens_used": stats.total_tokens_used
            }
        })
        
    except Exception as e:
        logger.error(f"Enrichment sync failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/weaviate/sync', methods=['POST'])
async def sync_enrichment_to_weaviate():
    """
    Sync enriched data to Weaviate.
    
    Updates the Weaviate Tool collection with enriched fields.
    
    Query params:
        dry_run: If true, show what would be done without making changes
        
    Returns:
        JSON with sync statistics
    """
    try:
        dry_run = request.args.get('dry_run', 'false').lower() == 'true'
        
        # Run the sync script
        script_path = "sync_enrichment_to_weaviate.py"
        cmd = [sys.executable, script_path]
        if dry_run:
            cmd.append("--dry-run")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        success = result.returncode == 0
        
        return jsonify({
            "success": success,
            "dry_run": dry_run,
            "output": result.stdout,
            "errors": result.stderr if not success else None
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Sync timed out after 5 minutes"
        }), 504
    except Exception as e:
        logger.error(f"Failed to sync to Weaviate: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@enrichment_bp.route('/cache/clear', methods=['POST'])
async def clear_enrichment_cache():
    """
    Clear all enrichment caches.
    
    Warning: This will require re-enriching all tools.
    
    Returns:
        JSON with confirmation
    """
    try:
        from semantic_enrichment import EnrichmentCache
        
        cache = EnrichmentCache()
        cache.clear()
        
        return jsonify({
            "success": True,
            "message": "Enrichment cache cleared"
        })
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
