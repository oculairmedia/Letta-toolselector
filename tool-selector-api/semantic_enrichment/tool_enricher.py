"""
Tool Enricher

Generates semantically rich descriptions for individual tools
using MCP server context for better search accuracy.
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import MCPServerProfile, EnrichedTool, EnrichmentResult
from .anthropic_client import AnthropicClient, get_anthropic_client
from .enrichment_cache import EnrichmentCache, get_enrichment_cache

logger = logging.getLogger(__name__)


class ToolEnricher:
    """
    Enriches tool descriptions with semantic keywords and use cases.
    
    Uses MCP server profile context to generate more relevant descriptions
    that improve search accuracy.
    """
    
    SYSTEM_PROMPT = """You are a technical writer creating tool descriptions for an AI agent's semantic search system.

Your enhanced descriptions must:
1. Be clear and comprehensive (200-300 words)
2. Include natural language terms users actually search for
3. List specific use cases and scenarios
4. Extract action-entity pairs (e.g., "create issue", "delete file")
5. Use the server's domain context for accuracy

IMPORTANT: Output valid JSON only. No markdown, no explanations."""

    ENRICHMENT_PROMPT = """Enhance this tool description for semantic search.

## MCP Server Context
- **Server**: {server_name}
- **Domain**: {server_domain}
- **Capabilities**: {server_capabilities}
- **Entity Types**: {entity_types}

## Tool to Enhance
- **Name**: {tool_name}
- **Current Description**: {original_description}
- **Parameters**: 
{parameters_json}

## Generate JSON with:
1. **enhanced_description**: 200-300 word rich description including:
   - What the tool does and why users need it
   - Common scenarios and workflows
   - Related terminology and synonyms
   - Integration with other tools/systems
   
2. **action_entities**: Array of action-entity pairs this tool supports
   Examples: ["create issue", "update project", "query tasks"]
   
3. **semantic_keywords**: 10-15 search terms users might use
   Include synonyms, related concepts, and domain terms
   
4. **use_cases**: 3-5 natural language scenarios
   Examples: ["When a developer needs to track a bug...", "..."]

Respond with JSON only:
{{
  "enhanced_description": "...",
  "action_entities": ["...", "..."],
  "semantic_keywords": ["...", "..."],
  "use_cases": ["...", "..."]
}}"""

    def __init__(
        self,
        client: Optional[AnthropicClient] = None,
        cache: Optional[EnrichmentCache] = None
    ):
        """
        Initialize the enricher.
        
        Args:
            client: AnthropicClient instance (default: singleton)
            cache: EnrichmentCache instance (default: singleton)
        """
        self.client = client or get_anthropic_client()
        self.cache = cache or get_enrichment_cache()
    
    def _format_parameters(self, tool: Dict[str, Any], max_length: int = 1000) -> str:
        """Format tool parameters for the prompt."""
        json_schema = tool.get('json_schema', {})
        if isinstance(json_schema, str):
            try:
                json_schema = json.loads(json_schema)
            except json.JSONDecodeError:
                json_schema = {}
        
        if not json_schema:
            return "No parameters defined"
        
        params = json_schema.get('properties', {})
        if not params:
            return "No parameters defined"
        
        lines = []
        for name, spec in list(params.items())[:10]:  # Limit to 10 params
            param_type = spec.get('type', 'any')
            desc = spec.get('description', '')[:100]
            enum = spec.get('enum', [])
            
            line = f"  - {name} ({param_type})"
            if desc:
                line += f": {desc}"
            if enum:
                line += f" [options: {', '.join(str(e) for e in enum[:5])}]"
            lines.append(line)
        
        result = "\n".join(lines)
        if len(params) > 10:
            result += f"\n  ... and {len(params) - 10} more parameters"
        
        return result[:max_length]
    
    def enrich_tool(
        self,
        tool: Dict[str, Any],
        server_profile: Optional[MCPServerProfile] = None,
        force: bool = False
    ) -> EnrichmentResult:
        """
        Enrich a single tool with semantic information.
        
        Args:
            tool: Tool dict with name, description, json_schema
            server_profile: Optional MCP server profile for context
            force: If True, re-enrich even if cached
            
        Returns:
            EnrichmentResult with enriched data or error
        """
        tool_id = tool.get('id') or tool.get('tool_id', '')
        tool_name = tool.get('name', 'unknown')
        start_time = time.time()
        
        # Check cache
        if not force and not self.cache.needs_tool_enrichment(tool):
            cached = self.cache.get_enriched_tool(tool_id)
            if cached:
                logger.debug(f"Using cached enrichment for {tool_name}")
                return EnrichmentResult(
                    success=True,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    enhanced_description=cached.enhanced_description,
                    action_entities=cached.action_entities,
                    semantic_keywords=cached.semantic_keywords,
                    use_cases=cached.use_cases,
                    duration_ms=0,
                    tokens_used=0
                )
        
        logger.info(f"Enriching tool: {tool_name}")
        
        try:
            # Build prompt with server context
            server_name = tool.get('mcp_server_name', 'unknown')
            server_domain = ""
            server_capabilities = ""
            entity_types = ""
            
            if server_profile:
                server_domain = server_profile.domain
                server_capabilities = ", ".join(server_profile.primary_capabilities)
                entity_types = ", ".join(server_profile.entity_types)
            
            prompt = self.ENRICHMENT_PROMPT.format(
                server_name=server_name,
                server_domain=server_domain or "general purpose",
                server_capabilities=server_capabilities or "various operations",
                entity_types=entity_types or "various entities",
                tool_name=tool_name,
                original_description=tool.get('description', '')[:500],
                parameters_json=self._format_parameters(tool)
            )
            
            # Call LLM
            result = self.client.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3
            )
            
            # Extract metadata
            meta = result.pop("_meta", {})
            tokens_used = meta.get("tokens_used", 0)
            duration_ms = (time.time() - start_time) * 1000
            
            # Build enriched tool
            enriched = EnrichedTool(
                tool_id=tool_id,
                name=tool_name,
                original_description=tool.get('description', ''),
                mcp_server_name=server_name,
                enhanced_description=result.get("enhanced_description", ""),
                action_entities=result.get("action_entities", []),
                semantic_keywords=result.get("semantic_keywords", []),
                use_cases=result.get("use_cases", []),
                server_domain=server_domain,
                enrichment_hash=self.cache.compute_tool_hash(tool),
                last_enriched=datetime.utcnow(),
                enrichment_model=self.client.model
            )
            
            # Cache it
            self.cache.set_enriched_tool(enriched)
            
            logger.info(
                f"Enriched {tool_name}: {len(enriched.action_entities)} actions, "
                f"{len(enriched.semantic_keywords)} keywords, {tokens_used} tokens"
            )
            
            return EnrichmentResult(
                success=True,
                tool_id=tool_id,
                tool_name=tool_name,
                enhanced_description=enriched.enhanced_description,
                action_entities=enriched.action_entities,
                semantic_keywords=enriched.semantic_keywords,
                use_cases=enriched.use_cases,
                duration_ms=duration_ms,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to enrich {tool_name}: {e}")
            return EnrichmentResult(
                success=False,
                tool_id=tool_id,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                tokens_used=0
            )
    
    def enrich_tools(
        self,
        tools: List[Dict[str, Any]],
        server_profile: Optional[MCPServerProfile] = None,
        force: bool = False
    ) -> List[EnrichmentResult]:
        """
        Enrich multiple tools.
        
        Args:
            tools: List of tool dicts
            server_profile: Optional server profile for context
            force: If True, re-enrich all tools
            
        Returns:
            List of EnrichmentResult objects
        """
        results = []
        total_tokens = 0
        
        # Filter to tools needing enrichment
        if not force:
            tools_to_process = self.cache.get_tools_needing_enrichment(tools)
            logger.info(
                f"Enriching {len(tools_to_process)}/{len(tools)} tools "
                f"({len(tools) - len(tools_to_process)} cached)"
            )
        else:
            tools_to_process = tools
        
        for tool in tools_to_process:
            result = self.enrich_tool(tool, server_profile, force=force)
            results.append(result)
            total_tokens += result.tokens_used
        
        # Summary
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Enriched {success_count}/{len(tools_to_process)} tools, "
            f"{total_tokens} total tokens"
        )
        
        return results


# Convenience function
def enrich_tool(
    tool: Dict[str, Any],
    server_profile: Optional[MCPServerProfile] = None,
    force: bool = False
) -> EnrichmentResult:
    """Enrich a single tool."""
    enricher = ToolEnricher()
    return enricher.enrich_tool(tool, server_profile, force=force)
