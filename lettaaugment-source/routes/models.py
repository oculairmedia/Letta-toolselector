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
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Create the blueprint
models_bp = Blueprint('models', __name__)

# Module state - injected via configure()
_get_embedding_models_func: Optional[Callable] = None
_get_reranker_models_func: Optional[Callable] = None
_get_embedding_health_func: Optional[Callable] = None
_search_test_func: Optional[Callable] = None


def configure(
    get_embedding_models_func: Optional[Callable] = None,
    get_reranker_models_func: Optional[Callable] = None,
    get_embedding_health_func: Optional[Callable] = None,
    search_test_func: Optional[Callable] = None
):
    """
    Configure the models blueprint with required dependencies.
    """
    global _get_embedding_models_func, _get_reranker_models_func
    global _get_embedding_health_func, _search_test_func
    
    _get_embedding_models_func = get_embedding_models_func
    _get_reranker_models_func = get_reranker_models_func
    _get_embedding_health_func = get_embedding_health_func
    _search_test_func = search_test_func
    
    logger.info("Models blueprint configured")


@models_bp.route('/api/v1/models/embedding', methods=['GET'])
async def get_embedding_models():
    """Get available embedding models."""
    if not _get_embedding_models_func:
        return jsonify({"error": "Embedding models not configured"}), 503
    return await _get_embedding_models_func()


@models_bp.route('/api/v1/models/reranker', methods=['GET'])
async def get_reranker_models():
    """Get available reranker models."""
    if not _get_reranker_models_func:
        return jsonify({"error": "Reranker models not configured"}), 503
    return await _get_reranker_models_func()


@models_bp.route('/api/v1/embedding/health', methods=['GET'])
async def get_embedding_health():
    """Get embedding health status."""
    if not _get_embedding_health_func:
        return jsonify({"error": "Embedding health not configured"}), 503
    return await _get_embedding_health_func()


@models_bp.route('/api/v1/search/test', methods=['GET'])
async def search_test():
    """Test search functionality."""
    if not _search_test_func:
        return jsonify({"error": "Search test not configured"}), 503
    return await _search_test_func()
