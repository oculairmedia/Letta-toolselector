"""
Search Service

This module provides a unified interface for tool search operations including:
- Weaviate vector search
- Letta native search
- Hybrid search with fallback
- Reranking support (vLLM and Ollama)
- Query expansion

Key Functions:
- configure: Set up the search service with dependencies
- search: Unified search interface with provider abstraction
- search_with_reranking: Search with explicit reranking control
- rerank: Rerank a list of documents against a query

The service abstracts away the complexity of multiple search providers
and reranking implementations, providing a clean API for consumers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Protocol, runtime_checkable

import httpx

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class RerankerConfig:
    """Configuration for the reranker service."""
    enabled: bool = True
    provider: str = "vllm"  # "vllm" or "ollama"
    url: str = "http://100.81.139.20:11435/rerank"
    model: str = "qwen3-reranker-4b"
    timeout: float = 30.0
    initial_limit: int = 30  # Number of candidates to retrieve for reranking
    top_k: int = 10  # Number of results after reranking
    instruction: str = "Given a query, rank the following documents by relevance."


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""
    enabled: bool = True
    use_universal: bool = True  # Use universal (schema-based) vs legacy expansion


@dataclass
class SearchConfig:
    """Configuration for the search service."""
    provider: str = "weaviate"  # "weaviate", "letta", or "hybrid"
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)


# ============================================================================
# Reranker Protocol and Implementations
# ============================================================================

@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int
    relevance_score: float
    document: Optional[str] = None


@runtime_checkable
class RerankerClient(Protocol):
    """Protocol for reranker implementations."""
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[RerankResult]:
        """Rerank documents against a query."""
        ...


class VLLMReranker:
    """Reranker using vLLM's native cross-encoder endpoint."""
    
    def __init__(
        self,
        url: str = "http://100.81.139.20:11435/rerank",
        model: str = "qwen3-reranker-4b",
        timeout: float = 30.0
    ):
        self.url = url
        self.model = model
        self.timeout = timeout
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[RerankResult]:
        """Rerank documents using vLLM's rerank endpoint."""
        if not documents:
            return []
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_k": min(top_k, len(documents))
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append(RerankResult(
                        index=item["index"],
                        relevance_score=item["relevance_score"],
                        document=documents[item["index"]] if item["index"] < len(documents) else None
                    ))
                return results
                
        except httpx.TimeoutException:
            logger.warning(f"vLLM reranker timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"vLLM reranker error: {e}")
            raise


class OllamaReranker:
    """Reranker using Ollama adapter (generative approach)."""
    
    def __init__(
        self,
        url: str = "http://ollama-reranker-adapter:8080/rerank",
        timeout: float = 30.0,
        instruction: str = "Given a query, rank the following documents by relevance."
    ):
        self.url = url
        self.timeout = timeout
        self.instruction = instruction
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[RerankResult]:
        """Rerank documents using Ollama adapter."""
        if not documents:
            return []
        
        payload = {
            "query": query,
            "documents": documents,
            "k": min(top_k, len(documents)),
            "instruction": self.instruction
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append(RerankResult(
                        index=item["index"],
                        relevance_score=item["relevance_score"],
                        document=documents[item["index"]] if item["index"] < len(documents) else None
                    ))
                return results
                
        except httpx.TimeoutException:
            logger.warning(f"Ollama reranker timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Ollama reranker error: {e}")
            raise


def create_reranker(config: RerankerConfig) -> RerankerClient:
    """Factory function to create a reranker based on config."""
    if config.provider.lower() == "vllm":
        return VLLMReranker(
            url=config.url,
            model=config.model,
            timeout=config.timeout
        )
    elif config.provider.lower() == "ollama":
        return OllamaReranker(
            url=config.url,
            timeout=config.timeout,
            instruction=config.instruction
        )
    else:
        raise ValueError(f"Unknown reranker provider: {config.provider}")


# ============================================================================
# Module State (for dependency injection)
# ============================================================================

_weaviate_client: Any = None
_letta_sdk_client_func: Optional[Callable] = None
_config: SearchConfig = SearchConfig()
_reranker: Optional[RerankerClient] = None

# Query expansion functions (imported lazily)
_expand_query_func: Optional[Callable] = None
_expand_query_with_analysis_func: Optional[Callable] = None


def configure(
    weaviate_client: Any = None,
    letta_sdk_client_func: Optional[Callable] = None,
    config: Optional[SearchConfig] = None,
    reranker: Optional[RerankerClient] = None
):
    """
    Configure the search service with required dependencies.
    
    Args:
        weaviate_client: Weaviate client instance
        letta_sdk_client_func: Function to get Letta SDK client
        config: SearchConfig with provider, reranker, and expansion settings
        reranker: Optional pre-configured reranker (otherwise created from config)
    """
    global _weaviate_client, _letta_sdk_client_func, _config, _reranker
    global _expand_query_func, _expand_query_with_analysis_func
    
    _weaviate_client = weaviate_client
    _letta_sdk_client_func = letta_sdk_client_func
    
    if config:
        _config = config
    
    # Create reranker from config if not provided
    if reranker:
        _reranker = reranker
    elif _config.reranker.enabled:
        try:
            _reranker = create_reranker(_config.reranker)
            logger.info(f"Created {_config.reranker.provider} reranker")
        except Exception as e:
            logger.warning(f"Failed to create reranker: {e}")
            _reranker = None
    
    # Import query expansion functions
    _import_query_expansion()
    
    logger.info(f"Search service configured: provider={_config.provider}, "
                f"reranking={'enabled' if _config.reranker.enabled else 'disabled'}, "
                f"expansion={'enabled' if _config.expansion.enabled else 'disabled'}")


def _import_query_expansion():
    """Lazily import query expansion functions."""
    global _expand_query_func, _expand_query_with_analysis_func
    
    if not _config.expansion.enabled:
        return
    
    # Try universal expansion first (preferred)
    if _config.expansion.use_universal:
        try:
            from universal_query_expansion import (
                expand_query_universally,
                expand_query_with_analysis
            )
            _expand_query_func = expand_query_universally
            _expand_query_with_analysis_func = expand_query_with_analysis
            logger.info("Loaded universal query expansion")
            return
        except ImportError as e:
            logger.debug(f"Universal query expansion not available: {e}")
    
    # Fall back to legacy expansion
    try:
        from query_expansion import expand_search_query, expand_search_query_with_metadata
        _expand_query_func = expand_search_query
        _expand_query_with_analysis_func = expand_search_query_with_metadata
        logger.info("Loaded legacy query expansion")
    except ImportError as e:
        logger.debug(f"Legacy query expansion not available: {e}")


# ============================================================================
# Search Functions
# ============================================================================

@dataclass
class ToolSearchResult:
    """Result from a tool search operation."""
    name: str
    description: str
    tool_id: Optional[str] = None
    tool_type: Optional[str] = None
    mcp_server_name: Optional[str] = None
    score: float = 0.0
    rerank_score: Optional[float] = None
    distance: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    enhanced_description: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


def expand_query(query: str) -> str:
    """
    Expand a search query for better tool discovery.
    
    Uses query expansion to detect operations/intents and add relevant keywords.
    
    Args:
        query: Original search query
        
    Returns:
        Expanded query string
    """
    if not _config.expansion.enabled or not _expand_query_func:
        return query
    
    try:
        expanded = _expand_query_func(query)
        if expanded != query:
            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query


async def search(
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    enable_reranking: Optional[bool] = None,
    enable_expansion: Optional[bool] = None,
    provider: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Unified tool search interface.
    
    Searches for tools using the configured provider (weaviate, letta, or hybrid).
    Supports optional reranking and query expansion.
    
    Args:
        query: Search query describing the tool you're looking for
        limit: Maximum number of results to return
        min_score: Minimum relevance score (0-1) to include
        enable_reranking: Override reranking setting (None = use config)
        enable_expansion: Override expansion setting (None = use config)
        provider: Override search provider (None = use config)
        
    Returns:
        List of tool dicts with search results
    """
    search_provider = provider or _config.provider
    use_reranking = enable_reranking if enable_reranking is not None else _config.reranker.enabled
    use_expansion = enable_expansion if enable_expansion is not None else _config.expansion.enabled
    
    # Apply query expansion if enabled
    search_query = query
    if use_expansion:
        search_query = expand_query(query)
    
    # Execute search based on provider
    if search_provider == "letta":
        return await _search_via_letta(search_query, limit, min_score)
    
    elif search_provider == "hybrid":
        # Try Letta first, fallback to Weaviate
        try:
            return await _search_via_letta(search_query, limit, min_score)
        except Exception as e:
            logger.warning(f"Letta search failed, falling back to Weaviate: {e}")
            return await _search_via_weaviate(search_query, limit, use_reranking)
    
    else:  # "weaviate" (default)
        return await _search_via_weaviate(search_query, limit, use_reranking)


async def _search_via_letta(query: str, limit: int, min_score: float) -> List[Dict[str, Any]]:
    """Search using Letta's native tools.search() API."""
    if not _letta_sdk_client_func:
        raise RuntimeError("Letta SDK not available for tool search")
    
    sdk_client = _letta_sdk_client_func()
    results = await sdk_client.search_tools_with_scores(
        query=query,
        limit=limit,
        min_score=min_score
    )
    logger.info(f"Letta search for '{query}' returned {len(results)} results")
    return results


async def _search_via_weaviate(query: str, limit: int, use_reranking: bool) -> List[Dict[str, Any]]:
    """Search using Weaviate vector database."""
    # Import here to avoid circular imports
    from weaviate_tool_search_with_reranking import search_tools_with_reranking
    
    # Run synchronous Weaviate search in thread pool
    results = await asyncio.to_thread(
        search_tools_with_reranking,
        query=query,
        limit=limit,
        use_reranking=use_reranking
    )
    logger.info(f"Weaviate search for '{query}' returned {len(results)} results")
    return results


async def rerank(
    query: str,
    documents: List[str],
    top_k: int = 10
) -> List[RerankResult]:
    """
    Rerank a list of documents against a query.
    
    Uses the configured reranker (vLLM or Ollama) to reorder documents
    by relevance to the query.
    
    Args:
        query: The search query
        documents: List of document strings to rerank
        top_k: Number of top results to return
        
    Returns:
        List of RerankResult with index, score, and document
        
    Raises:
        RuntimeError: If reranker is not configured
    """
    if not _reranker:
        raise RuntimeError("Reranker not configured")
    
    return await _reranker.rerank(query, documents, top_k)


async def search_with_reranking(
    query: str,
    limit: int = 10,
    rerank_initial_limit: int = 30,
    rerank_property: str = "enhanced_description",
    enable_expansion: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Search with explicit reranking control.
    
    This is a more explicit interface when you need fine control over
    reranking parameters.
    
    Args:
        query: Search query
        limit: Number of final results to return
        rerank_initial_limit: Number of candidates to retrieve for reranking
        rerank_property: Property to use for reranking (not currently used)
        enable_expansion: Override expansion setting
        
    Returns:
        List of tool dicts with rerank scores
    """
    use_expansion = enable_expansion if enable_expansion is not None else _config.expansion.enabled
    
    # Apply query expansion
    search_query = query
    if use_expansion:
        search_query = expand_query(query)
    
    # Import here to avoid circular imports
    from weaviate_tool_search_with_reranking import search_tools_with_reranking
    
    # Run synchronous Weaviate search in thread pool
    results = await asyncio.to_thread(
        search_tools_with_reranking,
        query=search_query,
        limit=limit,
        use_reranking=True,
        rerank_initial_limit=rerank_initial_limit,
        rerank_property=rerank_property
    )
    
    return results


def get_config() -> SearchConfig:
    """Get the current search configuration."""
    return _config


def get_reranker() -> Optional[RerankerClient]:
    """Get the current reranker instance."""
    return _reranker


def is_configured() -> bool:
    """Check if the search service is properly configured."""
    return _weaviate_client is not None or _letta_sdk_client_func is not None


# ============================================================================
# Utility Functions
# ============================================================================

def format_tool_for_reranking(tool: Dict[str, Any]) -> str:
    """
    Format a tool dict into a document string for reranking.
    
    Creates a structured representation of the tool that works well
    with reranking models.
    
    Args:
        tool: Tool dictionary with name, description, etc.
        
    Returns:
        Formatted document string
    """
    name = tool.get('name', '')
    description = tool.get('description', '')
    mcp_server = tool.get('mcp_server_name', '')
    tags = tool.get('tags', [])
    
    # Extract service name from tags if not in mcp_server
    service = mcp_server
    if not service and tags:
        for tag in tags:
            if isinstance(tag, str) and tag.startswith('mcp:'):
                service = tag[4:]
                break
    
    # Extract action keywords from tool name
    name_parts = name.lower().replace('_', ' ').replace('-', ' ').split()
    action_keywords = [p for p in name_parts if p in [
        'create', 'read', 'update', 'delete', 'list', 'search', 'get', 'set',
        'add', 'remove', 'edit', 'manage', 'find', 'query', 'sync', 'upload',
        'download', 'export', 'import', 'send', 'receive', 'start', 'stop'
    ]]
    
    # Build document format
    doc_parts = [f"Tool Name: {name}"]
    if service:
        doc_parts.append(f"Service: {service}")
    if action_keywords:
        doc_parts.append(f"Actions: {', '.join(action_keywords)}")
    if description:
        # Truncate description to first 500 chars for efficiency
        desc_truncated = description[:500] + ('...' if len(description) > 500 else '')
        doc_parts.append(f"Description: {desc_truncated}")
    
    return '\n'.join(doc_parts)


async def test_reranker() -> Dict[str, Any]:
    """
    Test if the reranker is properly configured and working.
    
    Returns:
        Status report dict with test results
    """
    status = {
        "reranking_enabled": _config.reranker.enabled,
        "provider": _config.reranker.provider,
        "url": _config.reranker.url,
        "test_passed": False,
        "error": None
    }
    
    if not _config.reranker.enabled:
        status["error"] = "Reranking is disabled"
        return status
    
    if not _reranker:
        status["error"] = "Reranker not initialized"
        return status
    
    try:
        # Try a simple rerank operation
        test_docs = [
            "Tool for creating new documents",
            "Tool for searching files",
            "Tool for deleting records"
        ]
        results = await rerank("create document", test_docs, top_k=3)
        
        if results:
            status["test_passed"] = True
            status["num_results"] = len(results)
            status["top_score"] = results[0].relevance_score if results else None
        else:
            status["error"] = "No results returned"
            
    except Exception as e:
        status["error"] = str(e)
    
    return status
