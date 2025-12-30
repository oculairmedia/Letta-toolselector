"""
Enrichment Cache

Hash-based caching for MCP server profiles and tool enrichments.
Detects changes to avoid unnecessary re-enrichment.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from .models import MCPServerProfile, EnrichedTool

logger = logging.getLogger(__name__)


class EnrichmentCache:
    """
    Cache for semantic enrichment data with change detection.
    
    Stores:
    - MCP server profiles with tool list hashes
    - Enriched tool descriptions with content hashes
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache files (default: /app/enrichment_cache)
        """
        self.cache_dir = Path(cache_dir or os.getenv(
            "ENRICHMENT_CACHE_DIR",
            "/app/enrichment_cache"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles_file = self.cache_dir / "server_profiles.json"
        self.tools_file = self.cache_dir / "enriched_tools.json"
        
        # In-memory caches
        self._profiles: Dict[str, MCPServerProfile] = {}
        self._tools: Dict[str, EnrichedTool] = {}
        
        # Load from disk
        self._load_caches()
        
        logger.info(
            f"EnrichmentCache initialized: {len(self._profiles)} profiles, "
            f"{len(self._tools)} tools cached"
        )
    
    def _load_caches(self) -> None:
        """Load caches from disk."""
        # Load profiles
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file) as f:
                    data = json.load(f)
                    for name, profile_data in data.items():
                        self._profiles[name] = MCPServerProfile(**profile_data)
            except Exception as e:
                logger.warning(f"Failed to load profiles cache: {e}")
        
        # Load tools
        if self.tools_file.exists():
            try:
                with open(self.tools_file) as f:
                    data = json.load(f)
                    for tool_id, tool_data in data.items():
                        self._tools[tool_id] = EnrichedTool(**tool_data)
            except Exception as e:
                logger.warning(f"Failed to load tools cache: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles cache to disk."""
        try:
            data = {
                name: profile.model_dump(mode='json')
                for name, profile in self._profiles.items()
            }
            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save profiles cache: {e}")
    
    def _save_tools(self) -> None:
        """Save tools cache to disk."""
        try:
            data = {
                tool_id: tool.model_dump(mode='json')
                for tool_id, tool in self._tools.items()
            }
            with open(self.tools_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save tools cache: {e}")
    
    # =========================================================================
    # Hash Computation
    # =========================================================================
    
    @staticmethod
    def compute_server_hash(tools: List[Dict[str, Any]]) -> str:
        """
        Compute a hash for a server's tool list.
        
        Args:
            tools: List of tool dicts from the server
            
        Returns:
            SHA256 hash of tool names + descriptions + schemas
        """
        content = json.dumps(
            [
                (t.get('name', ''), t.get('description', ''), t.get('json_schema', ''))
                for t in sorted(tools, key=lambda x: x.get('name', ''))
            ],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @staticmethod
    def compute_tool_hash(tool: Dict[str, Any]) -> str:
        """
        Compute a hash for a tool's content.
        
        Args:
            tool: Tool dict
            
        Returns:
            SHA256 hash of tool name + description + schema
        """
        content = json.dumps({
            'name': tool.get('name', ''),
            'description': tool.get('description', ''),
            'json_schema': tool.get('json_schema', ''),
            'mcp_server_name': tool.get('mcp_server_name', '')
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    # =========================================================================
    # Server Profile Cache
    # =========================================================================
    
    def get_profile(self, server_name: str) -> Optional[MCPServerProfile]:
        """Get a cached server profile."""
        return self._profiles.get(server_name)
    
    def set_profile(self, profile: MCPServerProfile) -> None:
        """Cache a server profile."""
        self._profiles[profile.server_name] = profile
        self._save_profiles()
    
    def needs_profile_update(
        self,
        server_name: str,
        current_tools: List[Dict[str, Any]],
        max_age_days: int = 7
    ) -> bool:
        """
        Check if a server profile needs updating.
        
        Args:
            server_name: MCP server name
            current_tools: Current tool list from server
            max_age_days: Force refresh after this many days
            
        Returns:
            True if profile should be regenerated
        """
        profile = self._profiles.get(server_name)
        
        if not profile:
            logger.debug(f"Profile for {server_name} not in cache")
            return True
        
        # Check hash
        current_hash = self.compute_server_hash(current_tools)
        if profile.profile_hash != current_hash:
            logger.info(
                f"Profile for {server_name} has changed: "
                f"{profile.profile_hash[:8]} -> {current_hash[:8]}"
            )
            return True
        
        # Check age
        age = datetime.utcnow() - profile.last_updated
        if age > timedelta(days=max_age_days):
            logger.info(f"Profile for {server_name} is {age.days} days old, refreshing")
            return True
        
        return False
    
    # =========================================================================
    # Tool Enrichment Cache
    # =========================================================================
    
    def get_enriched_tool(self, tool_id: str) -> Optional[EnrichedTool]:
        """Get a cached enriched tool."""
        return self._tools.get(tool_id)
    
    def set_enriched_tool(self, tool: EnrichedTool) -> None:
        """Cache an enriched tool."""
        self._tools[tool.tool_id] = tool
        self._save_tools()
    
    def needs_tool_enrichment(
        self,
        tool: Dict[str, Any],
        max_age_days: int = 30
    ) -> bool:
        """
        Check if a tool needs enrichment.
        
        Args:
            tool: Tool dict
            max_age_days: Force refresh after this many days
            
        Returns:
            True if tool should be re-enriched
        """
        tool_id = tool.get('id') or tool.get('tool_id')
        if not tool_id:
            return True
        
        cached = self._tools.get(tool_id)
        if not cached:
            return True
        
        # Check hash
        current_hash = self.compute_tool_hash(tool)
        if cached.enrichment_hash != current_hash:
            logger.debug(f"Tool {tool.get('name')} has changed, re-enriching")
            return True
        
        # Check age
        age = datetime.utcnow() - cached.last_enriched
        if age > timedelta(days=max_age_days):
            logger.debug(f"Tool {tool.get('name')} enrichment is {age.days} days old")
            return True
        
        return False
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def get_tools_needing_enrichment(
        self,
        tools: List[Dict[str, Any]],
        max_age_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Filter tools to only those needing enrichment.
        
        Args:
            tools: List of tool dicts
            max_age_days: Max age before refresh
            
        Returns:
            List of tools that need enrichment
        """
        return [t for t in tools if self.needs_tool_enrichment(t, max_age_days)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "profiles_cached": len(self._profiles),
            "tools_cached": len(self._tools),
            "cache_dir": str(self.cache_dir),
            "profiles_file_exists": self.profiles_file.exists(),
            "tools_file_exists": self.tools_file.exists()
        }
    
    def clear(self) -> None:
        """Clear all caches."""
        self._profiles.clear()
        self._tools.clear()
        if self.profiles_file.exists():
            self.profiles_file.unlink()
        if self.tools_file.exists():
            self.tools_file.unlink()
        logger.info("Enrichment cache cleared")


# Singleton instance
_cache_instance: Optional[EnrichmentCache] = None


def get_enrichment_cache() -> EnrichmentCache:
    """Get or create the singleton EnrichmentCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EnrichmentCache()
    return _cache_instance
