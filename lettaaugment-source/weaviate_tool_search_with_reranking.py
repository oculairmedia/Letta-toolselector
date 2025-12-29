"""
Weaviate Tool Search with Reranking Support
Enhanced version that supports two-stage retrieval with Ollama-based reranking
and automatic query expansion for better multifunctional tool discovery.
"""
import weaviate
from weaviate.classes.query import MetadataQuery, HybridFusion, Rerank
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import requests
import json
from specialized_embedding import (
    is_qwen3_format_enabled,
    get_search_instruction,
    get_detailed_instruct,
    format_query_for_qwen3,
)
from qwen3_reranker_utils import DEFAULT_RERANK_INSTRUCTION
from weaviate_client_manager import get_client_manager

# Import query expansion modules for multifunctional tool discovery
# We support two modes: universal (schema-based) and legacy (hardcoded)
QUERY_EXPANSION_AVAILABLE = False
UNIVERSAL_EXPANSION_AVAILABLE = False

# Try universal expansion first (preferred)
try:
    from universal_query_expansion import (
        expand_query_universally,
        expand_query_with_analysis,
        get_universal_expander,
    )
    UNIVERSAL_EXPANSION_AVAILABLE = True
    QUERY_EXPANSION_AVAILABLE = True
    print("Universal query expansion loaded (schema-based, dynamic)")
except ImportError as e:
    print(f"Universal query expansion not available: {e}")

# Fall back to legacy expansion if universal not available
if not UNIVERSAL_EXPANSION_AVAILABLE:
    try:
        from query_expansion import (
            expand_search_query,
            expand_search_query_with_metadata,
        )
        QUERY_EXPANSION_AVAILABLE = True
        print("Legacy query expansion loaded (hardcoded mappings)")
    except ImportError as e:
        print(f"Warning: No query expansion available: {e}")

# Provide fallback implementations if neither is available
if not QUERY_EXPANSION_AVAILABLE:
    def expand_search_query(query: str) -> str:
        return query
    
    def expand_search_query_with_metadata(query: str):
        return type('ExpandedQuery', (), {
            'original_query': query,
            'expanded_query': query,
            'detected_operations': [],
            'detected_domains': [],
            'added_keywords': [],
            'confidence': 0.0
        })()
    
    def expand_query_universally(query: str, tool_cache_path=None) -> str:
        return query
    
    def expand_query_with_analysis(query: str, tool_cache_path=None):
        return type('ExpandedQuery', (), {
            'original_query': query,
            'expanded_query': query,
            'detected_intents': [],
            'matched_tool_capabilities': [],
            'added_keywords': [],
            'confidence': 0.0
        })()


# Environment variable to enable/disable query expansion
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
# Environment variable to prefer universal expansion over legacy
USE_UNIVERSAL_EXPANSION = os.getenv("USE_UNIVERSAL_EXPANSION", "true").lower() == "true"
# Environment variable to enable reranking by default for all searches
ENABLE_RERANKING_BY_DEFAULT = os.getenv("ENABLE_RERANKING_BY_DEFAULT", "true").lower() == "true"

# Reranker configuration - supports both vLLM (recommended) and Ollama adapter
# RERANKER_PROVIDER: "vllm" (default, faster & better scores) or "ollama"
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "vllm").lower()
# For vLLM: http://100.81.139.20:11435/rerank (native cross-encoder, ~5x faster)
# For Ollama adapter: http://ollama-reranker-adapter:8080/rerank (generative approach)
RERANKER_URL = os.getenv("RERANKER_URL", "http://100.81.139.20:11435/rerank")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "qwen3-reranker-4b")
RERANKER_TIMEOUT = float(os.getenv("RERANKER_TIMEOUT", "30.0"))

# Persistent HTTP client for reranker (avoids TCP connection overhead per request)
import httpx
_reranker_client: httpx.Client = None

def get_reranker_client() -> httpx.Client:
    """Get or create persistent HTTP client for reranker calls."""
    global _reranker_client
    if _reranker_client is None:
        _reranker_client = httpx.Client(
            timeout=RERANKER_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    return _reranker_client

def close_reranker_client():
    """Close the reranker client. Call on app shutdown."""
    global _reranker_client
    if _reranker_client is not None:
        _reranker_client.close()
        _reranker_client = None

def init_client():
    """Initialize Weaviate client using v4 API."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set. Vectorizer might fail.")

    # Determine Weaviate host and port
    weaviate_http_host = os.getenv("WEAVIATE_HTTP_HOST", "weaviate")
    weaviate_http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    weaviate_grpc_host = os.getenv("WEAVIATE_GRPC_HOST", "weaviate")
    weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    print(f"Attempting to connect to Weaviate -> HTTP: {weaviate_http_host}:{weaviate_http_port}, GRPC: {weaviate_grpc_host}:{weaviate_grpc_port}")
    
    client_headers = {}
    if openai_api_key:
        client_headers["X-OpenAI-Api-Key"] = openai_api_key
    else:
        print("No OpenAI API key provided to Weaviate client; vectorization may rely on Weaviate's config or fail if OpenAI is default.")

    try:
        client = weaviate.connect_to_custom(
            http_host=weaviate_http_host,
            http_port=weaviate_http_port,
            http_secure=False,
            grpc_host=weaviate_grpc_host,
            grpc_port=weaviate_grpc_port,
            grpc_secure=False,
            headers=client_headers
        )
        
        print(f"Connected to Weaviate: {client.is_ready()}")
        return client
        
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        raise


def search_tools_with_reranking(
    query: str,
    limit: int = 10,
    use_reranking: bool = True,
    rerank_initial_limit: int = 30,
    rerank_property: str = "enhanced_description",
    enable_query_expansion: Optional[bool] = None
) -> list:
    """
    Search tools with optional reranking support and automatic query expansion.
    
    Query expansion improves discovery of multifunctional tools (like CRUD tools)
    by detecting operation intent (create, read, update, delete) and domain 
    (book, page, issue, agent) from the query, then injecting relevant keywords.
    
    Example: "create a book" becomes "create book crud content_crud make add new 
             documentation wiki bookstack manage" to find both specific and 
             unified tools.
    
    Args:
        query: Search query
        limit: Number of final results to return
        use_reranking: Whether to use reranking (requires reranker module)
        rerank_initial_limit: Number of candidates to retrieve for reranking
        rerank_property: Property to use for reranking
        enable_query_expansion: Override for query expansion (None = use env var)
    
    Returns:
        List of tools with scores
    """
    # Check if reranking is enabled globally
    enable_reranking = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    use_reranking = use_reranking and enable_reranking
    
    # Determine if query expansion should be used
    use_expansion = enable_query_expansion if enable_query_expansion is not None else ENABLE_QUERY_EXPANSION
    
    # Apply query expansion for better multifunctional tool discovery
    original_query = query
    expansion_metadata = None
    if use_expansion and QUERY_EXPANSION_AVAILABLE:
        # Prefer universal expansion (schema-based, dynamic)
        if UNIVERSAL_EXPANSION_AVAILABLE and USE_UNIVERSAL_EXPANSION:
            expansion_metadata = expand_query_with_analysis(query)
            query = expansion_metadata.expanded_query
            if expansion_metadata.added_keywords:
                print(f"Universal query expansion: '{original_query}' -> added {len(expansion_metadata.added_keywords)} keywords")
                print(f"  Intents detected: {[i.value if hasattr(i, 'value') else str(i) for i in expansion_metadata.detected_intents]}")
                print(f"  Matched tools: {expansion_metadata.matched_tool_capabilities[:5]}")
        else:
            # Fall back to legacy expansion (hardcoded)
            expansion_metadata = expand_search_query_with_metadata(query)
            query = expansion_metadata.expanded_query
            if expansion_metadata.added_keywords:
                print(f"Legacy query expansion: '{original_query}' -> added {len(expansion_metadata.added_keywords)} keywords")
                detected_ops = getattr(expansion_metadata, 'detected_operations', [])
                detected_doms = getattr(expansion_metadata, 'detected_domains', [])
                print(f"  Operations detected: {[op.value if hasattr(op, 'value') else str(op) for op in detected_ops]}")
                print(f"  Domains detected: {detected_doms}")
    
    # Get reranking parameters from environment
    if use_reranking:
        rerank_initial_limit = int(os.getenv("RERANK_INITIAL_LIMIT", str(rerank_initial_limit)))
        rerank_top_k = int(os.getenv("RERANK_TOP_K", str(limit)))
    
    try:
        # Use connection pool for better performance (avoids 50-200ms connection overhead per search)
        manager = get_client_manager()
        with manager.get_client() as client:
            try:
                # Get the Tool collection
                collection = client.collections.get("Tool")
                
                # Prepare query for Qwen3 embeddings without contamination
                cleaned_query = format_query_for_qwen3(query)
                hybrid_query = cleaned_query
                if is_qwen3_format_enabled():
                    hybrid_query = get_detailed_instruct(get_search_instruction(), cleaned_query)

                # Build base query
                if use_reranking:
                    print(f"Using client-side reranking: retrieving {rerank_initial_limit} candidates for top-{limit} results")
                    
                    # Get initial results without Weaviate reranking to avoid panics
                    result = collection.query.hybrid(
                        query=hybrid_query,
                        alpha=0.75,  # 75% vector search, 25% keyword search
                        limit=rerank_initial_limit,  # Get more candidates for reranking
                        fusion_type=HybridFusion.RELATIVE_SCORE,
                        query_properties=["name^2", "enhanced_description^2", "description^1.5", "tags"],
                        return_metadata=MetadataQuery(score=True)
                    )
                    
                    # Apply client-side reranking
                    if result and hasattr(result, 'objects') and result.objects:
                        try:
                            # Prepare documents for reranking with improved format
                            documents = []
                            for obj in result.objects:
                                name = obj.properties.get('name', '')
                                description = obj.properties.get('description', '')
                                tags = obj.properties.get('tags', [])
                                mcp_server = obj.properties.get('mcp_server_name', '')
                                
                                # Extract service name from tags (e.g., "mcp:huly" -> "huly")
                                service = mcp_server
                                if not service and tags:
                                    for tag in tags:
                                        if tag.startswith('mcp:'):
                                            service = tag[4:]  # Remove "mcp:" prefix
                                            break
                                
                                # Extract action keywords from tool name (e.g., "huly_create_issue" -> ["create", "issue"])
                                name_parts = name.lower().replace('_', ' ').replace('-', ' ').split()
                                action_keywords = [p for p in name_parts if p in [
                                    'create', 'read', 'update', 'delete', 'list', 'search', 'get', 'set',
                                    'add', 'remove', 'edit', 'manage', 'find', 'query', 'sync', 'upload',
                                    'download', 'export', 'import', 'send', 'receive', 'start', 'stop'
                                ]]
                                
                                # Build improved document format for reranker
                                doc_parts = [f"Tool Name: {name}"]
                                if service:
                                    doc_parts.append(f"Service: {service}")
                                if action_keywords:
                                    doc_parts.append(f"Actions: {', '.join(action_keywords)}")
                                if description:
                                    # Truncate description to first 500 chars for reranker efficiency
                                    desc_truncated = description[:500] + ('...' if len(description) > 500 else '')
                                    doc_parts.append(f"Description: {desc_truncated}")
                                
                                doc_text = '\n'.join(doc_parts)
                                documents.append(doc_text.strip())
                            
                            if documents:
                                # Call reranker using persistent client (avoids TCP overhead)
                                client = get_reranker_client()
                                
                                # Build payload based on provider
                                if RERANKER_PROVIDER == "vllm":
                                    # vLLM format: /v1/rerank endpoint
                                    # Embed instruction in query since vLLM doesn't have separate instruction field
                                    instructed_query = f"{DEFAULT_RERANK_INSTRUCTION}\n\nQuery: {cleaned_query}"
                                    payload = {
                                        "model": RERANKER_MODEL,
                                        "query": instructed_query,
                                        "documents": documents,
                                        "top_k": min(limit, len(documents)),
                                    }
                                else:
                                    # Ollama adapter format (default)
                                    payload = {
                                        "query": cleaned_query,
                                        "documents": documents,
                                        "k": min(limit, len(documents)),
                                        "instruction": DEFAULT_RERANK_INSTRUCTION,
                                    }
                                
                                response = client.post(RERANKER_URL, json=payload)
                                if response.status_code == 200:
                                    rerank_data = response.json()
                                    rerank_results = rerank_data.get('results', [])
                                    
                                    # Create mapping from rerank indices to original objects with scores
                                    scored_objects = []
                                    for rerank_result in rerank_results:
                                        original_idx = rerank_result['index']
                                        if original_idx < len(result.objects):
                                            obj = result.objects[original_idx]
                                            score = rerank_result['relevance_score']
                                            scored_objects.append((score, obj))
                                    
                                    # Sort by rerank score and take top-k  
                                    scored_objects.sort(key=lambda x: x[0], reverse=True)
                                    
                                    # Store rerank scores on the objects for later use
                                    for score, obj in scored_objects[:limit]:
                                        if not hasattr(obj, 'rerank_score'):
                                            obj.rerank_score = score
                                    
                                    result.objects = [obj for _, obj in scored_objects[:limit]]
                                    
                                    print(f"Client-side reranking completed: {len(scored_objects)} results reranked")
                                else:
                                    print(f"Reranker request failed: {response.status_code}, falling back to original order")
                                    result.objects = result.objects[:limit]
                        except Exception as e:
                            print(f"Client-side reranking failed: {e}, falling back to original order")
                            result.objects = result.objects[:limit]
                    else:
                        print("No results to rerank, proceeding with standard search")
                        use_reranking = False  # Fall back to standard search
                        
                else:
                    print(f"Standard search without reranking for {limit} results")
                    
                    # Standard hybrid search without reranking
                    result = collection.query.hybrid(
                        query=hybrid_query,
                        alpha=0.75,
                        limit=limit,
                        fusion_type=HybridFusion.RELATIVE_SCORE,
                        query_properties=["name^2", "enhanced_description^2", "description^1.5", "tags"],
                        return_metadata=MetadataQuery(score=True)
                    )

                # Process results
                tools = []
                if result and hasattr(result, 'objects'):
                    for obj in result.objects:
                        tool_data = obj.properties
                        
                        # Enhanced descriptions are now included in the search and available in results
                        
                        # Handle scoring
                        if hasattr(obj, 'rerank_score'):
                            # Client-side reranking was used
                            tool_data["rerank_score"] = obj.rerank_score
                            tool_data["score"] = obj.rerank_score  # Also set score for compatibility
                            score = obj.rerank_score
                        elif hasattr(obj, 'metadata') and obj.metadata is not None:
                            # Standard Weaviate scoring
                            score = getattr(obj.metadata, 'score', 0.5)
                            
                            tool_data["distance"] = 1 - (score if score is not None else 0.5)
                            tool_data["score"] = score if score is not None else 0.5
                        else:
                            tool_data["distance"] = 0.5
                            tool_data["score"] = 0.5
                        
                        tools.append(tool_data)
                
                if use_reranking:
                    print(f"Reranking complete: returned {len(tools)} tools")
                
                return tools
                
            except Exception as e:
                print(f"Error in collection query: {e}")
                # Fallback to non-reranked search if reranking fails
                if use_reranking:
                    print("Falling back to standard search without reranking")
                    return search_tools_with_reranking(
                        query=query,
                        limit=limit,
                        use_reranking=False
                    )
                return []
            
    except Exception as e:
        print(f"Error in search_tools_with_reranking: {e}")
        return []
    # Note: No need to close client - connection pool manages lifecycle

def search_tools(query: str, limit: int = 10, reranker_config: Optional[Dict[str, Any]] = None) -> list:
    """
    Backward-compatible wrapper that uses reranking if enabled.
    Maintains the original API signature while adding reranking capabilities.
    
    Reranking is enabled by default via ENABLE_RERANKING_BY_DEFAULT env var (default: true).
    Can be overridden by passing reranker_config={'enabled': False}.
    """
    # Determine if reranking should be used
    # Default to ENABLE_RERANKING_BY_DEFAULT env var (true by default)
    use_reranking = ENABLE_RERANKING_BY_DEFAULT
    
    # Allow explicit override via reranker_config
    if reranker_config is not None:
        use_reranking = reranker_config.get('enabled', use_reranking)
        
    return search_tools_with_reranking(
        query=query,
        limit=limit,
        use_reranking=use_reranking
    )

def test_reranking_capability() -> Dict[str, Any]:
    """
    Test if reranking is properly configured and working.
    Returns a status report.
    """
    status = {
        "reranking_enabled": os.getenv("ENABLE_RERANKING", "false").lower() == "true",
        "rerank_initial_limit": int(os.getenv("RERANK_INITIAL_LIMIT", "30")),
        "rerank_top_k": int(os.getenv("RERANK_TOP_K", "10")),
        "test_passed": False,
        "error": None
    }
    
    if not status["reranking_enabled"]:
        status["error"] = "Reranking is disabled (ENABLE_RERANKING=false)"
        return status
    
    try:
        # Try a simple search with reranking
        results = search_tools_with_reranking(
            query="test tool search",
            limit=5,
            use_reranking=True
        )
        
        if results:
            # Check if rerank scores are present
            has_rerank_scores = any('rerank_score' in r for r in results)
            status["test_passed"] = True
            status["has_rerank_scores"] = has_rerank_scores
            status["num_results"] = len(results)
        else:
            status["error"] = "No results returned"
            
    except Exception as e:
        status["error"] = str(e)
    
    return status
# ------------------------------------------------------------------
# Compatibility Adapter
# Some code (dashboard-backend/main.py, integration_layer.py) imports:
#   from weaviate_tool_search_with_reranking import WeaviateToolSearch
# The original refactor only exposed functions. We provide a lightweight
# adapter class so existing imports work without modifying the backend.
# ------------------------------------------------------------------
class WeaviateToolSearch:
    """
    Lightweight adapter providing an object-oriented interface around the
    module-level search functions. Keeps backward compatibility with older
    code that expected a class.
    """
    def __init__(self, reranker_config: Optional[Dict[str, Any]] = None):
        self.reranker_config = reranker_config or {}

    def search(self, query: str, limit: int = 10) -> list:
        """Primary search entrypoint (may use reranking based on config)."""
        return search_tools(query=query, limit=limit, reranker_config=self.reranker_config)

    def search_with_reranking(self, query: str, limit: int = 10) -> list:
        """Force reranking path (ignores internal enabled flag)."""
        return search_tools_with_reranking(query=query, limit=limit, use_reranking=True)

    def search_standard(self, query: str, limit: int = 10) -> list:
        """Force standard (non-reranked) hybrid search."""
        return search_tools_with_reranking(query=query, limit=limit, use_reranking=False)

    def test_reranking(self) -> Dict[str, Any]:
        """Return reranking capability status."""
        return test_reranking_capability()

# Provide a module-level convenience instance (optional usage)
default_tool_search = WeaviateToolSearch()
if __name__ == "__main__":
    # Test the search with reranking
    import json
    
    print("Testing Weaviate tool search with reranking...")
    print("=" * 50)
    
    # Test reranking capability
    print("\n1. Testing reranking configuration:")
    status = test_reranking_capability()
    print(json.dumps(status, indent=2))
    
    # Test actual search
    print("\n2. Testing search with 'ghost blog' query:")
    results = search_tools("ghost blog", limit=5)
    
    for i, tool in enumerate(results, 1):
        print(f"\n{i}. {tool.get('name', 'Unknown')}")
        print(f"   Score: {tool.get('score', 'N/A')}")
        if 'rerank_score' in tool:
            print(f"   Rerank Score: {tool['rerank_score']}")
        print(f"   Description: {tool.get('description', 'N/A')[:100]}...")
