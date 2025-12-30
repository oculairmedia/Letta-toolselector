"""
Ollama Routes Blueprint

Handles Ollama-related endpoints:
- /api/v1/ollama/models - List available models
"""

from quart import Blueprint, request, jsonify
import logging
import os
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

# Create blueprint
ollama_bp = Blueprint('ollama', __name__, url_prefix='/api/v1/ollama')


def configure():
    """Configure the ollama routes (placeholder for future dependencies)."""
    pass


def _get_fallback_models():
    """Get fallback models including configured embedding model"""
    configured_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'llama2:7b')
    fallback_models = [configured_model, "mistral:7b", "llama2:7b", "codellama:7b"]
    # Remove duplicates while preserving order
    return list(dict.fromkeys(fallback_models))


@ollama_bp.route('/models', methods=['GET'])
async def get_ollama_models():
    """Get available models from Ollama instance."""
    try:
        # Get base URL from environment configuration
        ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
        base_url = f"http://{ollama_host}:11434"
        
        # Try to query Ollama API for models
        logger.info(f"Fetching Ollama models from: {base_url}")
        
        async with aiohttp.ClientSession() as session:
            # Query the Ollama API tags endpoint
            ollama_url = f"{base_url}/api/tags"
            async with session.get(ollama_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    # Format models for frontend
                    formatted_models = []
                    for model in models:
                        formatted_model = {
                            "name": model.get('name', ''),
                            "size": model.get('size', 0),
                            "modified_at": model.get('modified_at', ''),
                            "digest": model.get('digest', ''),
                            "details": model.get('details', {})
                        }
                        formatted_models.append(formatted_model)
                    
                    logger.info(f"Successfully fetched {len(formatted_models)} models from Ollama")
                    return jsonify({
                        "success": True, 
                        "data": {
                            "models": formatted_models,
                            "base_url": base_url,
                            "total": len(formatted_models)
                        }
                    })
                else:
                    logger.warning(f"Ollama API returned status {response.status}")
                    return jsonify({
                        "success": False, 
                        "error": f"Ollama API returned status {response.status}",
                        "fallback_models": _get_fallback_models()
                    }), 503
                    
    except asyncio.TimeoutError:
        logger.error("Timeout connecting to Ollama API")
        return jsonify({
            "success": False, 
            "error": "Timeout connecting to Ollama instance",
            "fallback_models": ["mistral:7b", "llama2:7b", "codellama:7b"]
        }), 503
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return jsonify({
            "success": False, 
            "error": f"Failed to fetch models: {str(e)}",
            "fallback_models": _get_fallback_models()
        }), 503
