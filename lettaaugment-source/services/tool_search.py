"""
Tool Search Service

Wraps the search_tools functionality for use by blueprints.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Module state - search function injected at startup
_search_tools_func: Optional[Callable] = None


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
        
        Args:
            query: Search query string
            limit: Maximum number of results
            reranker_config: Optional reranker configuration
            
        Returns:
            List of matching tool dictionaries
        """
        if not _search_tools_func:
            logger.error("ToolSearchService not configured - search_tools function not set")
            return []
        
        try:
            return _search_tools_func(query=query, limit=limit, reranker_config=reranker_config)
        except Exception as e:
            logger.error(f"Error in tool search: {e}")
            return []
    
    @staticmethod
    def is_configured() -> bool:
        """Check if the service is properly configured."""
        return _search_tools_func is not None


def configure_search_service(search_tools_func: Callable):
    """Convenience function to configure the search service."""
    ToolSearchService.configure(search_tools_func)
