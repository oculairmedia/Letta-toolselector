"""
MCP Server Profiler

Generates semantic profiles for MCP servers by analyzing their tools.
Uses Claude Sonnet to understand the server's domain and capabilities.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import MCPServerProfile, ServerProfileResult
from .anthropic_client import AnthropicClient, get_anthropic_client
from .enrichment_cache import EnrichmentCache, get_enrichment_cache

logger = logging.getLogger(__name__)


class MCPServerProfiler:
    """
    Profiles MCP servers to generate semantic context for tool enrichment.
    
    Analyzes all tools from a server to understand:
    - Primary domain (e.g., "project management")
    - Capabilities (e.g., "issue tracking", "sprint planning")
    - Entity types (e.g., "issue", "project", "milestone")
    - Common actions (e.g., "create", "update", "query")
    """
    
    SYSTEM_PROMPT = """You are an expert at understanding software tools and APIs.
Your task is to analyze a set of tools from an MCP (Model Context Protocol) server
and generate a semantic profile that helps with tool search and discovery.

Focus on:
1. The primary domain this server serves
2. Key capabilities it provides
3. Entity types it works with
4. Common actions it supports
5. Natural language terms users might search for"""

    PROFILE_PROMPT = """Analyze these tools from the "{server_name}" MCP server and generate a semantic profile.

## Tools ({tool_count} total):
{tools_summary}

## Generate a profile with:
1. **domain**: Primary domain in 2-5 words (e.g., "project management and issue tracking")
2. **primary_capabilities**: 3-6 key capabilities as an array
3. **entity_types**: Main entity types this server works with (nouns)
4. **action_verbs**: Common actions supported (verbs)
5. **integration_context**: 1-2 sentences on how this fits in workflows
6. **semantic_keywords**: 10-20 related search terms users might use

Respond with JSON only:
{{
  "domain": "...",
  "primary_capabilities": ["...", "..."],
  "entity_types": ["...", "..."],
  "action_verbs": ["...", "..."],
  "integration_context": "...",
  "semantic_keywords": ["...", "..."]
}}"""

    def __init__(
        self,
        client: Optional[AnthropicClient] = None,
        cache: Optional[EnrichmentCache] = None
    ):
        """
        Initialize the profiler.
        
        Args:
            client: AnthropicClient instance (default: singleton)
            cache: EnrichmentCache instance (default: singleton)
        """
        self.client = client or get_anthropic_client()
        self.cache = cache or get_enrichment_cache()
    
    def _format_tools_summary(self, tools: List[Dict[str, Any]], max_tools: int = 30) -> str:
        """Format tool list for the prompt."""
        lines = []
        for i, tool in enumerate(tools[:max_tools]):
            name = tool.get('name', 'unknown')
            desc = tool.get('description', '')[:150]
            lines.append(f"- **{name}**: {desc}")
        
        if len(tools) > max_tools:
            lines.append(f"- ... and {len(tools) - max_tools} more tools")
        
        return "\n".join(lines)
    
    def profile_server(
        self,
        server_name: str,
        tools: List[Dict[str, Any]],
        force: bool = False
    ) -> ServerProfileResult:
        """
        Generate a semantic profile for an MCP server.
        
        Args:
            server_name: Name of the MCP server
            tools: List of tool dicts from this server
            force: If True, regenerate even if cached
            
        Returns:
            ServerProfileResult with profile or error
        """
        start_time = time.time()
        
        # Check cache
        if not force and not self.cache.needs_profile_update(server_name, tools):
            cached_profile = self.cache.get_profile(server_name)
            if cached_profile:
                logger.info(f"Using cached profile for {server_name}")
                return ServerProfileResult(
                    success=True,
                    server_name=server_name,
                    profile=cached_profile,
                    duration_ms=0,
                    tokens_used=0,
                    tools_analyzed=len(tools)
                )
        
        logger.info(f"Profiling MCP server: {server_name} ({len(tools)} tools)")
        
        try:
            # Generate prompt
            tools_summary = self._format_tools_summary(tools)
            prompt = self.PROFILE_PROMPT.format(
                server_name=server_name,
                tool_count=len(tools),
                tools_summary=tools_summary
            )
            
            # Call LLM
            result = self.client.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.2
            )
            
            # Extract metadata
            meta = result.pop("_meta", {})
            tokens_used = meta.get("tokens_used", 0)
            
            # Build profile
            profile = MCPServerProfile(
                server_name=server_name,
                domain=result.get("domain", ""),
                primary_capabilities=result.get("primary_capabilities", []),
                entity_types=result.get("entity_types", []),
                action_verbs=result.get("action_verbs", []),
                integration_context=result.get("integration_context", ""),
                semantic_keywords=result.get("semantic_keywords", []),
                profile_hash=self.cache.compute_server_hash(tools),
                last_updated=datetime.utcnow(),
                tool_count=len(tools)
            )
            
            # Cache it
            self.cache.set_profile(profile)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Profiled {server_name}: domain='{profile.domain}', "
                f"{len(profile.primary_capabilities)} capabilities, "
                f"{tokens_used} tokens, {duration_ms:.0f}ms"
            )
            
            return ServerProfileResult(
                success=True,
                server_name=server_name,
                profile=profile,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
                tools_analyzed=len(tools)
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to profile {server_name}: {e}")
            return ServerProfileResult(
                success=False,
                server_name=server_name,
                error=str(e),
                duration_ms=duration_ms,
                tokens_used=0,
                tools_analyzed=len(tools)
            )
    
    def profile_all_servers(
        self,
        servers_tools: Dict[str, List[Dict[str, Any]]],
        force: bool = False
    ) -> Dict[str, ServerProfileResult]:
        """
        Profile multiple MCP servers.
        
        Args:
            servers_tools: Dict mapping server_name -> list of tools
            force: If True, regenerate all profiles
            
        Returns:
            Dict mapping server_name -> ServerProfileResult
        """
        results = {}
        total_tokens = 0
        
        for server_name, tools in servers_tools.items():
            result = self.profile_server(server_name, tools, force=force)
            results[server_name] = result
            total_tokens += result.tokens_used
        
        # Summary
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(
            f"Profiled {success_count}/{len(servers_tools)} servers, "
            f"{total_tokens} total tokens"
        )
        
        return results


# Convenience function
def profile_mcp_server(
    server_name: str,
    tools: List[Dict[str, Any]],
    force: bool = False
) -> ServerProfileResult:
    """Profile a single MCP server."""
    profiler = MCPServerProfiler()
    return profiler.profile_server(server_name, tools, force=force)
