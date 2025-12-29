#!/usr/bin/env python3
"""
Run Semantic Enrichment Pipeline

Profiles all MCP servers and enriches their tools using Claude Sonnet.
Uses hash-based caching to only re-enrich changed content.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from semantic_enrichment.models import EnrichmentStats
from semantic_enrichment.anthropic_client import AnthropicClient
from semantic_enrichment.enrichment_cache import EnrichmentCache
from semantic_enrichment.server_profiler import MCPServerProfiler
from semantic_enrichment.tool_enricher import ToolEnricher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tools_by_server(cache_file: str) -> dict:
    """Load tools grouped by MCP server from cache file."""
    with open(cache_file) as f:
        tools = json.load(f)
    
    # Group by mcp_server_name
    servers = {}
    for tool in tools:
        server = tool.get('mcp_server_name')
        if server:
            if server not in servers:
                servers[server] = []
            servers[server].append(tool)
    
    return servers


def run_enrichment(
    tool_cache_path: str,
    anthropic_base_url: str,
    anthropic_api_key: str,
    force: bool = False,
    servers_only: bool = False,
    server_filter: str = None,
    dry_run: bool = False
):
    """
    Run the enrichment pipeline.
    
    Args:
        tool_cache_path: Path to tool_cache.json
        anthropic_base_url: Anthropic API base URL
        anthropic_api_key: Anthropic API key
        force: Force re-enrichment of all content
        servers_only: Only profile servers, don't enrich tools
        server_filter: Only process this server (for testing)
        dry_run: Show what would be done without calling API
    """
    stats = EnrichmentStats()
    stats.started_at = datetime.utcnow()
    
    # Load tools
    logger.info(f"Loading tools from {tool_cache_path}")
    servers_tools = load_tools_by_server(tool_cache_path)
    stats.total_servers = len(servers_tools)
    stats.total_tools = sum(len(tools) for tools in servers_tools.values())
    
    logger.info(f"Found {stats.total_servers} MCP servers with {stats.total_tools} tools")
    
    # Filter if requested
    if server_filter:
        if server_filter in servers_tools:
            servers_tools = {server_filter: servers_tools[server_filter]}
            logger.info(f"Filtering to server: {server_filter}")
        else:
            logger.error(f"Server '{server_filter}' not found")
            return stats
    
    if dry_run:
        logger.info("DRY RUN - would process these servers:")
        for server, tools in servers_tools.items():
            logger.info(f"  {server}: {len(tools)} tools")
        return stats
    
    # Initialize clients
    logger.info(f"Initializing Anthropic client: {anthropic_base_url}")
    client = AnthropicClient(
        base_url=anthropic_base_url,
        api_key=anthropic_api_key,
        model="claude-sonnet-4-20250514",
        timeout=120.0
    )
    
    # Health check
    health = client.health_check()
    if not health.get("healthy"):
        logger.error(f"Anthropic API unhealthy: {health.get('error')}")
        return stats
    logger.info(f"Anthropic API healthy: {health.get('latency_ms', 0):.0f}ms latency")
    
    # Initialize cache and profiler
    cache = EnrichmentCache()
    profiler = MCPServerProfiler(client=client, cache=cache)
    enricher = ToolEnricher(client=client, cache=cache)
    
    # Phase 1: Profile all servers
    logger.info("=" * 60)
    logger.info("PHASE 1: Profiling MCP Servers")
    logger.info("=" * 60)
    
    server_profiles = {}
    for server_name, tools in servers_tools.items():
        result = profiler.profile_server(server_name, tools, force=force)
        stats.total_tokens_used += result.tokens_used
        
        if result.success:
            server_profiles[server_name] = result.profile
            if result.tokens_used > 0:
                stats.servers_profiled += 1
            else:
                stats.servers_cached += 1
        else:
            stats.servers_failed += 1
            logger.error(f"Failed to profile {server_name}: {result.error}")
    
    logger.info(
        f"Server profiling complete: {stats.servers_profiled} profiled, "
        f"{stats.servers_cached} cached, {stats.servers_failed} failed"
    )
    
    if servers_only:
        logger.info("Stopping after server profiling (--servers-only)")
        stats.completed_at = datetime.utcnow()
        return stats
    
    # Phase 2: Enrich all tools
    logger.info("=" * 60)
    logger.info("PHASE 2: Enriching Tools")
    logger.info("=" * 60)
    
    for server_name, tools in servers_tools.items():
        profile = server_profiles.get(server_name)
        logger.info(f"Enriching {len(tools)} tools from {server_name}")
        
        results = enricher.enrich_tools(tools, server_profile=profile, force=force)
        
        for result in results:
            stats.total_tokens_used += result.tokens_used
            if result.success:
                if result.tokens_used > 0:
                    stats.tools_enriched += 1
                else:
                    stats.tools_cached += 1
            else:
                stats.tools_failed += 1
    
    stats.completed_at = datetime.utcnow()
    stats.total_duration_ms = (stats.completed_at - stats.started_at).total_seconds() * 1000
    
    # Summary
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Servers: {stats.servers_profiled} profiled, {stats.servers_cached} cached, {stats.servers_failed} failed")
    logger.info(f"Tools: {stats.tools_enriched} enriched, {stats.tools_cached} cached, {stats.tools_failed} failed")
    logger.info(f"Total tokens: {stats.total_tokens_used}")
    logger.info(f"Duration: {stats.total_duration_ms / 1000:.1f}s")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run semantic enrichment pipeline")
    parser.add_argument(
        "--tool-cache", 
        default="/app/runtime_cache/tool_cache.json",
        help="Path to tool cache JSON file"
    )
    parser.add_argument(
        "--anthropic-url",
        default=os.getenv("ANTHROPIC_BASE_URL", "http://192.168.50.90:8082"),
        help="Anthropic API base URL"
    )
    parser.add_argument(
        "--anthropic-key",
        default=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-enrichment of all content"
    )
    parser.add_argument(
        "--servers-only",
        action="store_true", 
        help="Only profile servers, don't enrich individual tools"
    )
    parser.add_argument(
        "--server",
        help="Only process this specific server (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without calling API"
    )
    
    args = parser.parse_args()
    
    # Check tool cache exists
    if not os.path.exists(args.tool_cache):
        # Try alternative path
        alt_path = "/opt/stacks/lettatoolsselector/lettaaugment-source/cache/tool_cache.json"
        if os.path.exists(alt_path):
            args.tool_cache = alt_path
        else:
            logger.error(f"Tool cache not found: {args.tool_cache}")
            sys.exit(1)
    
    stats = run_enrichment(
        tool_cache_path=args.tool_cache,
        anthropic_base_url=args.anthropic_url,
        anthropic_api_key=args.anthropic_key,
        force=args.force,
        servers_only=args.servers_only,
        server_filter=args.server,
        dry_run=args.dry_run
    )
    
    # Exit with error if any failures
    if stats.servers_failed > 0 or stats.tools_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
