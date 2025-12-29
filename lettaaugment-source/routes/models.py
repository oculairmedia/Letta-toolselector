"""
Models Blueprint

Provides endpoints for model listing and health checks.

Routes:
- GET /api/v1/models/embedding - Get available embedding models
- GET /api/v1/models/reranker - Get available reranker models
- GET /api/v1/embedding/health - Get embedding health status
- GET /api/v1/search/test - Test search functionality
"""

from quart import Blueprint, request, jsonify
import logging
import os
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# Create the blueprint
models_bp = Blueprint('models', __name__)

# Module state - injected via configure()
_search_tools_func: Optional[Callable] = None
_bm25_vector_override_service: Optional[Any] = None


def configure(
    search_tools_func: Optional[Callable] = None,
    bm25_vector_override_service: Optional[Any] = None
):
    """
    Configure the models blueprint with required dependencies.
    
    Args:
        search_tools_func: Function to search tools
        bm25_vector_override_service: BM25/vector override service for search testing
    """
    global _search_tools_func, _bm25_vector_override_service
    
    _search_tools_func = search_tools_func
    _bm25_vector_override_service = bm25_vector_override_service
    
    logger.info("Models blueprint configured")


# =============================================================================
# Embedding Models
# =============================================================================

@models_bp.route('/api/v1/models/embedding', methods=['GET'])
async def get_embedding_models():
    """Get available embedding models from all configured providers."""
    try:
        models = []
        
        # OpenAI embedding models
        openai_models = [
            {
                "id": "text-embedding-3-small",
                "name": "OpenAI Text Embedding 3 Small",
                "provider": "openai",
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k": 0.00002,
                "recommended": True
            },
            {
                "id": "text-embedding-3-large", 
                "name": "OpenAI Text Embedding 3 Large",
                "provider": "openai",
                "dimensions": 3072,
                "max_tokens": 8191,
                "cost_per_1k": 0.00013,
                "recommended": False
            },
            {
                "id": "text-embedding-ada-002",
                "name": "OpenAI Text Embedding Ada 002",
                "provider": "openai", 
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_1k": 0.0001,
                "recommended": False
            }
        ]
        
        # Add OpenAI models if API key is available
        if os.getenv('OPENAI_API_KEY'):
            models.extend(openai_models)
        
        # Ollama embedding models - get from Ollama endpoint
        try:
            ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
            ollama_url = f"http://{ollama_host}:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        ollama_data = await response.json()
                        for model in ollama_data.get('models', []):
                            # Filter for embedding models (heuristic based on name)
                            model_name = model.get('name', '')
                            if any(keyword in model_name.lower() for keyword in ['embed', 'embedding', 'bge', 'e5']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name}",
                                    "provider": "ollama",
                                    "dimensions": "variable",
                                    "max_tokens": "variable", 
                                    "cost_per_1k": 0.0,
                                    "recommended": False,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama embedding models: {str(e)}")
        
        # Add fallback local models if no models found
        if not models:
            models = [
                {
                    "id": "all-MiniLM-L6-v2",
                    "name": "All MiniLM L6 v2 (Local)",
                    "provider": "local",
                    "dimensions": 384,
                    "max_tokens": 512,
                    "cost_per_1k": 0.0,
                    "recommended": True
                }
            ]
        
        return jsonify({
            "success": True,
            "data": {
                "models": models,
                "total": len(models),
                "providers": list(set(m["provider"] for m in models))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Reranker Models
# =============================================================================

@models_bp.route('/api/v1/models/reranker', methods=['GET'])
async def get_reranker_models():
    """Get available reranker models from all configured providers."""
    try:
        models = []
        
        # Ollama reranker models
        try:
            ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80') 
            ollama_url = f"http://{ollama_host}:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        ollama_data = await response.json()
                        for model in ollama_data.get('models', []):
                            model_name = model.get('name', '')
                            # Filter for reranker models
                            if any(keyword in model_name.lower() for keyword in ['rerank', 'cross-encoder', 'bge-rerank', 'colbert']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name}",
                                    "provider": "ollama",
                                    "type": "cross-encoder",
                                    "cost_per_1k": 0.0,
                                    "recommended": True,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
                            # Also include general models that can be used for reranking
                            elif any(keyword in model_name.lower() for keyword in ['mistral', 'llama', 'qwen']):
                                models.append({
                                    "id": model_name,
                                    "name": f"Ollama {model_name} (General)",
                                    "provider": "ollama",
                                    "type": "generative",
                                    "cost_per_1k": 0.0,
                                    "recommended": False,
                                    "size": model.get('size', 0),
                                    "modified_at": model.get('modified_at')
                                })
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama reranker models: {str(e)}")
        
        # Add some common reranker models as fallback
        if not models:
            models = [
                {
                    "id": "mistral:7b",
                    "name": "Mistral 7B (Fallback)",
                    "provider": "ollama",
                    "type": "generative",
                    "cost_per_1k": 0.0,
                    "recommended": True
                },
                {
                    "id": "llama2:7b",
                    "name": "Llama 2 7B (Fallback)",
                    "provider": "ollama", 
                    "type": "generative",
                    "cost_per_1k": 0.0,
                    "recommended": False
                }
            ]
        
        return jsonify({
            "success": True,
            "data": {
                "models": models,
                "total": len(models),
                "providers": list(set(m["provider"] for m in models)),
                "types": list(set(m["type"] for m in models))
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting reranker models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Embedding Health
# =============================================================================

@models_bp.route('/api/v1/embedding/health', methods=['GET'])
async def get_embedding_health():
    """Get comprehensive embedding model health and status information."""
    try:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()

        if not embedding_provider:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'unknown',
                    'provider': 'not_configured',
                    'model': 'not_configured',
                    'availability': False,
                    'response_time_ms': 0,
                    'last_checked': datetime.now().isoformat(),
                    'error_message': 'No embedding provider configured'
                }
            })

        # Start timing
        start_time = time.time()
        status = 'healthy'
        error_message = None
        availability = False
        model_info = {}

        try:
            if embedding_provider == "openai":
                # Test OpenAI embedding model
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")

                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

                # Test with a small embedding request
                response = await asyncio.to_thread(
                    openai.embeddings.create,
                    input="test embedding health check",
                    model=embedding_model
                )

                if response and response.data:
                    availability = True
                    model_info = {
                        'dimensions': len(response.data[0].embedding) if response.data else 0,
                        'max_tokens': 8191 if 'small' in embedding_model else 8191,
                        'cost_per_1k': 0.00002 if 'small' in embedding_model else 0.00013
                    }
                else:
                    status = 'error'
                    error_message = 'No response from OpenAI API'

            elif embedding_provider == "ollama":
                # Test Ollama embedding model
                ollama_host = os.getenv("OLLAMA_EMBEDDING_HOST", "localhost")
                ollama_port = int(os.getenv("OLLAMA_EMBEDDING_PORT", "11434"))
                embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

                async with aiohttp.ClientSession() as session:
                    url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
                    payload = {
                        "model": embedding_model,
                        "prompt": "test embedding health check"
                    }

                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'embedding' in result:
                                availability = True
                                model_info = {
                                    'dimensions': len(result['embedding']) if result.get('embedding') else 0,
                                    'max_tokens': 2048,  # Default for most Ollama models
                                }
                            else:
                                status = 'error'
                                error_message = 'Invalid response format from Ollama'
                        else:
                            status = 'error'
                            error_message = f'Ollama API returned status {response.status}'
            else:
                status = 'error'
                error_message = f'Unsupported embedding provider: {embedding_provider}'

        except asyncio.TimeoutError:
            status = 'error'
            error_message = 'Connection timeout'
        except Exception as e:
            status = 'warning' if availability else 'error'
            error_message = str(e)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Determine final status based on response time
        if status == 'healthy':
            if response_time_ms > 5000:  # 5 seconds
                status = 'warning'
                error_message = 'High response time detected'
            elif response_time_ms > 10000:  # 10 seconds
                status = 'error'
                error_message = 'Very high response time'

        # Get model name
        model_name = 'unknown'
        if embedding_provider == "openai":
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        elif embedding_provider == "ollama":
            model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # Simulate performance metrics (in real implementation, this would come from monitoring)
        performance_metrics = {
            'avg_response_time': response_time_ms,
            'success_rate': 0.98 if availability else 0.0,
            'total_requests': 1250,  # Simulated
            'failed_requests': 25 if not availability else 2  # Simulated
        }

        health_data = {
            'status': status,
            'provider': embedding_provider,
            'model': model_name,
            'availability': availability,
            'response_time_ms': response_time_ms,
            'last_checked': datetime.now().isoformat(),
            'error_message': error_message,
            'performance_metrics': performance_metrics,
            'model_info': model_info
        }

        return jsonify({
            'success': True,
            'data': health_data
        })

    except Exception as e:
        logger.error(f"Embedding health check error: {e}")
        return jsonify({
            'success': False,
            'error': f'Health check failed: {str(e)}'
        }), 500


# =============================================================================
# Search Test
# =============================================================================

@models_bp.route('/api/v1/search/test', methods=['GET'])
async def search_test():
    """Test search functionality with parameter overrides and detailed rankings."""
    try:
        query = request.args.get('query')
        if not query:
            return jsonify({"success": False, "error": "Query parameter is required"}), 400
            
        # Parse optional override parameters
        limit = request.args.get('limit', 10, type=int)
        alpha = request.args.get('alpha', type=float)  # Weaviate hybrid search alpha
        distance_metric = request.args.get('distance_metric')  # e.g., 'cosine', 'euclidean'
        reranker_enabled = request.args.get('reranker_enabled', 'true').lower() == 'true'
        reranker_model = request.args.get('reranker_model')
        
        # Build configuration overrides
        weaviate_overrides = {}
        if alpha is not None:
            weaviate_overrides['alpha'] = alpha
        if distance_metric:
            weaviate_overrides['distance_metric'] = distance_metric
            
        reranker_overrides = {
            'enabled': reranker_enabled
        }
        if reranker_model:
            reranker_overrides['model'] = reranker_model
            
        # Perform search with overrides
        start_time = time.time()
        
        # Use the BM25/vector override service if needed
        if weaviate_overrides and _bm25_vector_override_service:
            search_results = _bm25_vector_override_service(
                query, 
                limit=limit, 
                overrides=weaviate_overrides,
                reranker_config=reranker_overrides if reranker_enabled else None
            )
        elif _search_tools_func:
            search_results = _search_tools_func(query, limit=limit, reranker_config=reranker_overrides)
        else:
            return jsonify({"success": False, "error": "Search not configured"}), 503
        
        search_time = time.time() - start_time
        
        # Format results with detailed rankings
        formatted_results = []
        for i, result in enumerate(search_results):
            formatted_result = {
                "tool": {
                    "id": result.get('id', ''),
                    "name": result.get('name', ''),
                    "description": result.get('description', ''),
                    "source": result.get('source', 'unknown'),
                    "category": result.get('category'),
                    "tags": result.get('tags', [])
                },
                "score": result.get('rerank_score', 0) if reranker_enabled and result.get('rerank_score') is not None else result.get('score', 0),
                "rank": i + 1,
                "reasoning": result.get('reasoning', ''),
                "vector_score": result.get('vector_score', 0),
                "keyword_score": result.get('keyword_score', 0),
                "rerank_score": result.get('rerank_score', 0) if reranker_enabled else None,
                "original_rank": result.get('original_rank', i + 1)
            }
            formatted_results.append(formatted_result)
        
        response_data = {
            "query": query,
            "results": formatted_results,
            "metadata": {
                "total_found": len(formatted_results),
                "search_time": round(search_time, 3),
                "parameters_used": {
                    "limit": limit,
                    "alpha": alpha,
                    "distance_metric": distance_metric,
                    "reranker_enabled": reranker_enabled,
                    "reranker_model": reranker_model
                },
                "weaviate_overrides": weaviate_overrides,
                "reranker_overrides": reranker_overrides
            }
        }
        
        logger.info(f"Test search completed for query: {query} with overrides")
        return jsonify({"success": True, "data": response_data})
        
    except Exception as e:
        logger.error(f"Error during test search: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
