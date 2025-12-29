"""
Tool Cache Service

Provides caching functionality for tools data, shared across blueprints.
"""

import os
import json
import logging
import aiofiles
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Module state
_tool_cache: List[Dict[str, Any]] = []
_tool_cache_by_name: Dict[str, Dict[str, Any]] = {}  # O(1) lookup by name
_tool_cache_by_id: Dict[str, Dict[str, Any]] = {}    # O(1) lookup by id
_cache_last_loaded: Optional[datetime] = None
_cache_dir: Optional[str] = None


def _rebuild_cache_indexes():
    """Rebuild the name and id indexes from the tool cache list."""
    global _tool_cache_by_name, _tool_cache_by_id
    _tool_cache_by_name = {t.get('name'): t for t in _tool_cache if t.get('name')}
    _tool_cache_by_id = {t.get('id', t.get('tool_id')): t for t in _tool_cache if t.get('id') or t.get('tool_id')}


class ToolCacheService:
    """Service for managing tool cache operations."""
    
    def __init__(self, cache_dir: str = None):
        """Initialize the cache service with optional cache directory."""
        global _cache_dir
        if cache_dir:
            _cache_dir = cache_dir
        elif not _cache_dir:
            _cache_dir = os.getenv('CACHE_DIR', '/app/cache')
    
    @property
    def cache_file(self) -> str:
        """Get the path to the tools cache file."""
        return os.path.join(_cache_dir or '/app/cache', 'all_tools_cache.json')
    
    async def read_tool_cache(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Read the tool cache from disk.
        
        Args:
            force_reload: If True, reload from disk even if cached in memory
            
        Returns:
            List of tool dictionaries
        """
        global _tool_cache, _cache_last_loaded
        
        # Return cached data if available and not forcing reload
        if _tool_cache and not force_reload:
            return _tool_cache
        
        try:
            if os.path.exists(self.cache_file):
                async with aiofiles.open(self.cache_file, 'r') as f:
                    content = await f.read()
                    _tool_cache = json.loads(content) if content else []
                    _rebuild_cache_indexes()  # Build O(1) lookup indexes
                    _cache_last_loaded = datetime.now(timezone.utc)
                    logger.info(f"Loaded {len(_tool_cache)} tools from cache (indexed by name and id)")
            else:
                logger.warning(f"Tool cache file not found: {self.cache_file}")
                _tool_cache = []
                _rebuild_cache_indexes()
        except Exception as e:
            logger.error(f"Error reading tool cache: {e}")
            _tool_cache = []
            _rebuild_cache_indexes()
        
        return _tool_cache
    
    async def write_tool_cache(self, tools: List[Dict[str, Any]]) -> bool:
        """
        Write tools to the cache file.
        
        Args:
            tools: List of tool dictionaries to cache
            
        Returns:
            True if successful, False otherwise
        """
        global _tool_cache, _cache_last_loaded
        
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            async with aiofiles.open(self.cache_file, 'w') as f:
                await f.write(json.dumps(tools, indent=2))
            _tool_cache = tools
            _rebuild_cache_indexes()  # Rebuild O(1) lookup indexes
            _cache_last_loaded = datetime.now(timezone.utc)
            logger.info(f"Wrote {len(tools)} tools to cache (indexed by name and id)")
            return True
        except Exception as e:
            logger.error(f"Error writing tool cache: {e}")
            return False
    
    def get_cached_tools(self) -> List[Dict[str, Any]]:
        """Get the in-memory cached tools without reading from disk."""
        return _tool_cache
    
    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name using O(1) dictionary lookup.
        
        Args:
            name: The tool name to look up
            
        Returns:
            Tool dictionary if found, None otherwise
        """
        return _tool_cache_by_name.get(name)
    
    def get_tool_by_id(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by ID using O(1) dictionary lookup.
        
        Args:
            tool_id: The tool ID to look up
            
        Returns:
            Tool dictionary if found, None otherwise
        """
        return _tool_cache_by_id.get(tool_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "tool_count": len(_tool_cache),
            "last_loaded": _cache_last_loaded.isoformat() if _cache_last_loaded else None,
            "cache_file": self.cache_file,
            "cache_exists": os.path.exists(self.cache_file)
        }
    
    @staticmethod
    def is_letta_core_tool(tool: Dict[str, Any]) -> bool:
        """
        Check if a tool is a Letta core tool that should be excluded.
        
        Args:
            tool: Tool dictionary to check
            
        Returns:
            True if this is a Letta core tool
        """
        tool_type = tool.get("tool_type", "")
        source = tool.get("source", "")
        name = tool.get("name", "")
        
        # Check tool_type
        if tool_type == "letta_core":
            return True
        
        # Check source patterns
        letta_sources = ["letta", "letta_core", "core", "builtin"]
        if source.lower() in letta_sources:
            return True
        
        # Check name patterns for known Letta core tools
        letta_core_names = [
            "send_message", "conversation_search", "archival_memory_search",
            "archival_memory_insert", "core_memory_append", "core_memory_replace",
            "pause_heartbeats", "message_chatgpt"
        ]
        if name in letta_core_names:
            return True
        
        return False
    
    def filter_mcp_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter tools to only include MCP tools (exclude Letta core tools).
        
        Args:
            tools: List of tool dictionaries
            
        Returns:
            Filtered list containing only MCP tools
        """
        filtered = []
        for tool in tools:
            tool_type = tool.get("tool_type", "")
            is_letta_core = self.is_letta_core_tool(tool)
            is_mcp_tool = (tool_type == "external_mcp" or 
                         (not is_letta_core and tool_type == "custom"))
            
            if is_mcp_tool:
                filtered.append(tool)
        
        return filtered


# Singleton instance for convenience
_service_instance: Optional[ToolCacheService] = None


def get_tool_cache_service(cache_dir: str = None) -> ToolCacheService:
    """Get or create the singleton ToolCacheService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ToolCacheService(cache_dir)
    return _service_instance
