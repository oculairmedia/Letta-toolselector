"""
Tool Search Service

Wraps the search_tools functionality for use by blueprints.
Includes TTL-based caching for search results.
"""

import logging
import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Callable
from cachetools import TTLCache
import threading

logger = logging.getLogger(__name__)

# Module state - search function injected at startup
_search_tools_func: Optional[Callable] = None

# Cache configuration from environment
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "60"))
SEARCH_CACHE_MAX_SIZE = int(os.getenv("SEARCH_CACHE_MAX_SIZE", "1000"))
SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"

# Thread-safe cache for search results
_search_cache: TTLCache = TTLCache(maxsize=SEARCH_CACHE_MAX_SIZE, ttl=SEARCH_CACHE_TTL_SECONDS)
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}


def _make_cache_key(query: str, limit: int, reranker_config: Optional[Dict[str, Any]]) -> str:
    """Create a consistent cache key from search parameters."""
    config_str = json.dumps(reranker_config, sort_keys=True) if reranker_config else ""
    key_content = f"{query}:{limit}:{config_str}"
    return hashlib.md5(key_content.encode()).hexdigest()


class ToolSearchService:
    """Service for searching tools."""
    
    @staticmethod
    def configure(search_tools_func: Callable):
        """
        Configure the search service with the search function.
        
        Args:
            search_tools_func: The search_tools function from weaviate module
        """
        global _search_tools_func
        _search_tools_func = search_tools_func
        logger.info("ToolSearchService configured with search_tools function")
    
    @staticmethod
    def search(query: str, limit: int = 10, reranker_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for tools matching the query.
        
        Uses TTL-based caching to avoid repeated searches for identical queries.
        Cache is configurable via environment variables:
        - SEARCH_CACHE_TTL_SECONDS: Cache TTL (default: 60)
        - SEARCH_CACHE_MAX_SIZE: Max cache entries (default: 1000)
        - SEARCH_CACHE_ENABLED: Enable/disable cache (default: true)
        
        Args:
            query: Search query string
            limit: Maximum number of results
            reranker_config: Optional reranker configuration
            
        Returns:
            List of matching tool dictionaries
        """
        global _cache_stats
        
        if not _search_tools_func:
            logger.error("ToolSearchService not configured - search_tools function not set")
            return []
        
        # Generate cache key (needed for both check and store)
        cache_key = _make_cache_key(query, limit, reranker_config) if SEARCH_CACHE_ENABLED else None
        
        # Check cache first (if enabled)
        if SEARCH_CACHE_ENABLED and cache_key:
            with _cache_lock:
                if cache_key in _search_cache:
                    _cache_stats["hits"] += 1
                    logger.debug(f"Search cache hit for query: '{query[:50]}...' (limit={limit})")
                    return _search_cache[cache_key]
                _cache_stats["misses"] += 1
        
        try:
            results = _search_tools_func(query=query, limit=limit, reranker_config=reranker_config)
            
            # Cache the results
            if SEARCH_CACHE_ENABLED and cache_key and results:
                with _cache_lock:
                    _search_cache[cache_key] = results
                logger.debug(f"Cached search results for query: '{query[:50]}...' ({len(results)} results)")
            
            return results
        except Exception as e:
            logger.error(f"Error in tool search: {e}")
            return []
    
    @staticmethod
    def is_configured() -> bool:
        """Check if the service is properly configured."""
        return _search_tools_func is not None
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with _cache_lock:
            return {
                "enabled": SEARCH_CACHE_ENABLED,
                "ttl_seconds": SEARCH_CACHE_TTL_SECONDS,
                "max_size": SEARCH_CACHE_MAX_SIZE,
                "current_size": len(_search_cache),
                "hits": _cache_stats["hits"],
                "misses": _cache_stats["misses"],
                "hit_rate": (
                    _cache_stats["hits"] / (_cache_stats["hits"] + _cache_stats["misses"])
                    if (_cache_stats["hits"] + _cache_stats["misses"]) > 0
                    else 0.0
                )
            }
    
    @staticmethod
    def clear_cache():
        """Clear the search cache. Call after tool index refresh."""
        global _cache_stats
        with _cache_lock:
            _search_cache.clear()
            _cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("Search cache cleared")


def configure_search_service(search_tools_func: Callable):
    """Convenience function to configure the search service."""
    ToolSearchService.configure(search_tools_func)
