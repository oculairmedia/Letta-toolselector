"""
Weaviate Tool Search with Reranking Support
Enhanced version that supports two-stage retrieval with Ollama-based reranking
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
    rerank_property: str = "enhanced_description"
) -> list:
    """
    Search tools with optional reranking support.
    
    Args:
        query: Search query
        limit: Number of final results to return
        use_reranking: Whether to use reranking (requires reranker module)
        rerank_initial_limit: Number of candidates to retrieve for reranking
        rerank_property: Property to use for reranking
    
    Returns:
        List of tools with scores
    """
    client = None
    
    # Check if reranking is enabled globally
    enable_reranking = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    use_reranking = use_reranking and enable_reranking
    
    # Get reranking parameters from environment
    if use_reranking:
        rerank_initial_limit = int(os.getenv("RERANK_INITIAL_LIMIT", str(rerank_initial_limit)))
        rerank_top_k = int(os.getenv("RERANK_TOP_K", str(limit)))
    
    try:
        client = init_client()
        
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
                        # Prepare documents for reranking
                        documents = []
                        for obj in result.objects:
                            # Create structured document text for better reranking
                            name = obj.properties.get('name', '')
                            description = obj.properties.get('description', '')
                            tags = obj.properties.get('tags', [])
                            tool_type = obj.properties.get('tool_type', '')
                            
                            # Format with more context for the reranker
                            doc_text = f"Tool: {name}\nType: {tool_type}\nDescription: {description}"
                            if tags:
                                doc_text += f"\nCategories: {', '.join(tags)}"
                            documents.append(doc_text.strip())
                        
                        if documents:
                            # Call our reranker adapter with task-specific instruction
                            import httpx
                            reranker_url = "http://ollama-reranker-adapter:8080/rerank"
                            
                            # Add task-specific context to the query for better reranking
                            instructed_query = f"Find tools that can: {query}"
                            
                            payload = {
                                "query": instructed_query,  # Use instructed query for better context
                                "documents": documents,
                                "k": min(limit, len(documents))
                            }
                            
                            response = httpx.post(reranker_url, json=payload, timeout=30.0)
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
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

def search_tools(query: str, limit: int = 10, reranker_config: Optional[Dict[str, Any]] = None) -> list:
    """
    Backward-compatible wrapper that uses reranking if enabled.
    Maintains the original API signature while adding reranking capabilities.
    """
    # Determine if reranking should be used
    use_reranking = False  # Default to NO reranking if no config provided
    if reranker_config is not None:
        use_reranking = reranker_config.get('enabled', False)
        
    return search_tools_with_reranking(
        query=query,
        limit=limit,
        use_reranking=use_reranking  # Will check ENABLE_RERANKING env var internally
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