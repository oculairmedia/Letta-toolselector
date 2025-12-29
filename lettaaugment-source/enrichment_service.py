"""
Enrichment Service

Automated semantic enrichment for MCP tools. Runs on a schedule to:
1. Detect tools needing enrichment (new or changed)
2. Generate server profiles for MCP servers
3. Enrich tools with action-entities, keywords, use cases
4. Sync enriched data to Weaviate

Can run standalone or be triggered by sync_service.py after new tools are detected.
"""

import os
import sys
import json
import time
import asyncio
import logging
import schedule
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_enrichment.enrichment_cache import EnrichmentCache, get_enrichment_cache
from semantic_enrichment.server_profiler import ServerProfiler
from semantic_enrichment.tool_enricher import ToolEnricher
from semantic_enrichment.anthropic_client import AnthropicClient
from semantic_enrichment.models import EnrichedTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
CACHE_DIR = os.getenv("CACHE_DIR", "/app/runtime_cache")
TOOL_CACHE_FILE = os.path.join(CACHE_DIR, "tool_cache.json")
ENRICHMENT_CACHE_DIR = os.getenv("ENRICHMENT_CACHE_DIR", "/app/enrichment_cache")


class EnrichmentService:
    """
    Service that manages semantic enrichment for tools.
    
    Integrates with sync_service.py to enrich new/changed tools automatically.
    """
    
    def __init__(
        self,
        anthropic_url: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the enrichment service.
        
        Args:
            anthropic_url: Anthropic API URL (or OpenAI-compatible proxy)
            anthropic_key: API key (can be dummy for proxy)
            cache_dir: Directory for enrichment cache
        """
        self.anthropic_url = anthropic_url or os.getenv(
            "ANTHROPIC_BASE_URL",
            "http://192.168.50.90:8082/v1"
        )
        self.anthropic_key = anthropic_key or os.getenv(
            "ANTHROPIC_API_KEY",
            "dummy"
        )
        
        # Initialize cache
        self.cache = EnrichmentCache(cache_dir or ENRICHMENT_CACHE_DIR)
        
        # Initialize clients (lazy)
        self._anthropic_client: Optional[AnthropicClient] = None
        self._server_profiler: Optional[ServerProfiler] = None
        self._tool_enricher: Optional[ToolEnricher] = None
        
        # Stats
        self.last_run: Optional[datetime] = None
        self.tools_enriched_count = 0
        self.profiles_generated_count = 0
        
        logger.info(
            f"EnrichmentService initialized: "
            f"anthropic_url={self.anthropic_url}, "
            f"cache_dir={self.cache.cache_dir}"
        )
    
    @property
    def anthropic_client(self) -> AnthropicClient:
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            self._anthropic_client = AnthropicClient(
                base_url=self.anthropic_url,
                api_key=self.anthropic_key
            )
        return self._anthropic_client
    
    @property
    def server_profiler(self) -> ServerProfiler:
        """Lazy-load server profiler."""
        if self._server_profiler is None:
            self._server_profiler = ServerProfiler(
                client=self.anthropic_client,
                cache=self.cache
            )
        return self._server_profiler
    
    @property
    def tool_enricher(self) -> ToolEnricher:
        """Lazy-load tool enricher."""
        if self._tool_enricher is None:
            self._tool_enricher = ToolEnricher(
                client=self.anthropic_client,
                cache=self.cache
            )
        return self._tool_enricher
    
    def load_tool_cache(self) -> List[Dict[str, Any]]:
        """Load tools from the sync service's cache file."""
        if not os.path.exists(TOOL_CACHE_FILE):
            logger.warning(f"Tool cache file not found: {TOOL_CACHE_FILE}")
            return []
        
        try:
            with open(TOOL_CACHE_FILE) as f:
                tools = json.load(f)
                logger.info(f"Loaded {len(tools)} tools from cache")
                return tools
        except Exception as e:
            logger.error(f"Failed to load tool cache: {e}")
            return []
    
    def group_tools_by_server(
        self,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group tools by their MCP server name."""
        servers: Dict[str, List[Dict[str, Any]]] = {}
        
        for tool in tools:
            server_name = tool.get("mcp_server_name")
            if server_name:
                if server_name not in servers:
                    servers[server_name] = []
                servers[server_name].append(tool)
        
        logger.info(f"Grouped {len(tools)} tools into {len(servers)} servers")
        return servers
    
    async def run_enrichment(
        self,
        force_all: bool = False,
        max_tools: int = 50,
        max_servers: int = 10
    ) -> Dict[str, Any]:
        """
        Run the enrichment pipeline.
        
        Args:
            force_all: If True, re-enrich all tools regardless of cache
            max_tools: Maximum number of tools to enrich per run (rate limiting)
            max_servers: Maximum number of servers to profile per run
            
        Returns:
            Dict with enrichment stats
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting enrichment run (force_all={force_all})")
        
        stats = {
            "started_at": start_time.isoformat(),
            "tools_processed": 0,
            "tools_enriched": 0,
            "servers_profiled": 0,
            "errors": [],
            "duration_seconds": 0
        }
        
        try:
            # Load tools
            tools = self.load_tool_cache()
            if not tools:
                logger.warning("No tools to enrich")
                return stats
            
            # Group by server
            servers = self.group_tools_by_server(tools)
            
            # Step 1: Profile servers that need updating
            servers_to_profile = []
            for server_name, server_tools in servers.items():
                if force_all or self.cache.needs_profile_update(server_name, server_tools):
                    servers_to_profile.append((server_name, server_tools))
            
            servers_to_profile = servers_to_profile[:max_servers]
            logger.info(f"Will profile {len(servers_to_profile)} servers")
            
            for server_name, server_tools in servers_to_profile:
                try:
                    logger.info(f"Profiling server: {server_name} ({len(server_tools)} tools)")
                    profile = await self.server_profiler.profile_server(
                        server_name=server_name,
                        tools=server_tools
                    )
                    self.cache.set_profile(profile)
                    stats["servers_profiled"] += 1
                    logger.info(f"Profiled {server_name}: domain={profile.domain}")
                except Exception as e:
                    logger.error(f"Failed to profile server {server_name}: {e}")
                    stats["errors"].append(f"Profile {server_name}: {str(e)}")
            
            # Step 2: Enrich tools that need updating
            mcp_tools = [t for t in tools if t.get("mcp_server_name")]
            
            if force_all:
                tools_to_enrich = mcp_tools[:max_tools]
            else:
                tools_to_enrich = self.cache.get_tools_needing_enrichment(mcp_tools)[:max_tools]
            
            logger.info(f"Will enrich {len(tools_to_enrich)} tools")
            
            for tool in tools_to_enrich:
                try:
                    tool_name = tool.get("name", "unknown")
                    server_name = tool.get("mcp_server_name")
                    
                    # Get server profile
                    profile = self.cache.get_profile(server_name) if server_name else None
                    
                    logger.info(f"Enriching tool: {tool_name}")
                    enriched = await self.tool_enricher.enrich_tool(
                        tool=tool,
                        server_profile=profile
                    )
                    self.cache.set_enriched_tool(enriched)
                    stats["tools_enriched"] += 1
                    stats["tools_processed"] += 1
                    
                    # Rate limiting - small delay between API calls
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Failed to enrich tool {tool.get('name')}: {e}")
                    stats["errors"].append(f"Enrich {tool.get('name')}: {str(e)}")
                    stats["tools_processed"] += 1
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            stats["duration_seconds"] = duration
            
            # Update instance stats
            self.last_run = start_time
            self.tools_enriched_count += stats["tools_enriched"]
            self.profiles_generated_count += stats["servers_profiled"]
            
            logger.info(
                f"Enrichment complete: "
                f"{stats['tools_enriched']} tools enriched, "
                f"{stats['servers_profiled']} servers profiled, "
                f"{len(stats['errors'])} errors, "
                f"{duration:.1f}s"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Enrichment run failed: {e}", exc_info=True)
            stats["errors"].append(f"Fatal: {str(e)}")
            return stats
    
    async def sync_to_weaviate(self) -> Dict[str, Any]:
        """
        Sync enriched data to Weaviate.
        
        Returns:
            Dict with sync stats
        """
        logger.info("Syncing enrichment data to Weaviate")
        
        # Import here to avoid circular imports
        try:
            import weaviate
            from weaviate.classes.init import Auth, AdditionalConfig, Timeout
        except ImportError:
            logger.error("Weaviate client not installed")
            return {"error": "Weaviate client not installed"}
        
        stats = {
            "tools_updated": 0,
            "errors": []
        }
        
        client = None
        try:
            # Connect to Weaviate
            client = weaviate.connect_to_local(
                host=os.getenv("WEAVIATE_HTTP_HOST", "weaviate"),
                port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")},
                skip_init_checks=True
            )
            
            collection = client.collections.get("Tool")
            
            # Get all enriched tools from cache
            enriched_tools = list(self.cache._tools.values())
            logger.info(f"Syncing {len(enriched_tools)} enriched tools to Weaviate")
            
            for enriched in enriched_tools:
                try:
                    # Find tool in Weaviate by tool_id
                    import weaviate.classes.query as wq
                    result = collection.query.fetch_objects(
                        filters=wq.Filter.by_property("tool_id").equal(enriched.tool_id),
                        limit=1
                    )
                    
                    if result.objects:
                        obj = result.objects[0]
                        # Update with enrichment data
                        collection.data.update(
                            uuid=obj.uuid,
                            properties={
                                "action_entities": enriched.action_entities,
                                "semantic_keywords": enriched.semantic_keywords,
                                "use_cases": enriched.use_cases,
                                "server_domain": enriched.server_domain or ""
                            }
                        )
                        stats["tools_updated"] += 1
                    else:
                        logger.debug(f"Tool {enriched.tool_name} not found in Weaviate")
                        
                except Exception as e:
                    logger.error(f"Failed to sync tool {enriched.tool_name}: {e}")
                    stats["errors"].append(f"{enriched.tool_name}: {str(e)}")
            
            logger.info(f"Weaviate sync complete: {stats['tools_updated']} tools updated")
            return stats
            
        except Exception as e:
            logger.error(f"Weaviate sync failed: {e}", exc_info=True)
            stats["errors"].append(f"Fatal: {str(e)}")
            return stats
        finally:
            if client and client.is_connected():
                client.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get enrichment service status."""
        cache_stats = self.cache.get_stats()
        return {
            "service": "enrichment",
            "status": "running",
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "total_tools_enriched": self.tools_enriched_count,
            "total_profiles_generated": self.profiles_generated_count,
            "cache": cache_stats,
            "config": {
                "anthropic_url": self.anthropic_url,
                "cache_dir": str(self.cache.cache_dir)
            }
        }


# Singleton instance for use by sync_service
_service_instance: Optional[EnrichmentService] = None


def get_enrichment_service() -> EnrichmentService:
    """Get or create the singleton EnrichmentService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EnrichmentService()
    return _service_instance


async def run_enrichment_cycle(force_all: bool = False):
    """
    Run a full enrichment cycle: enrich + sync to Weaviate.
    
    Can be called by sync_service.py after detecting new tools.
    """
    service = get_enrichment_service()
    
    # Run enrichment
    enrich_stats = await service.run_enrichment(force_all=force_all)
    
    # Sync to Weaviate if we enriched anything
    if enrich_stats.get("tools_enriched", 0) > 0:
        sync_stats = await service.sync_to_weaviate()
        enrich_stats["weaviate_sync"] = sync_stats
    
    return enrich_stats


def run_enrichment_job():
    """Synchronous wrapper for scheduled enrichment."""
    logger.info("Scheduled enrichment job triggered")
    try:
        asyncio.run(run_enrichment_cycle())
    except Exception as e:
        logger.error(f"Enrichment job failed: {e}", exc_info=True)


def main():
    """
    Main entry point for standalone enrichment service.
    
    Runs enrichment on a schedule (default: hourly).
    """
    load_dotenv()
    
    # Get enrichment interval from environment (default: 1 hour)
    enrichment_interval = int(os.getenv("ENRICHMENT_INTERVAL", "3600"))
    
    logger.info(f"Starting enrichment service (interval: {enrichment_interval}s)")
    
    # Schedule enrichment
    schedule.every(enrichment_interval).seconds.do(run_enrichment_job)
    
    # Run initial enrichment
    logger.info("Running initial enrichment...")
    run_enrichment_job()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
