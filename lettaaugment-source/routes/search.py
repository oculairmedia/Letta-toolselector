"""
Search Routes Blueprint

Handles search parameter management endpoints:
- /api/v1/search/parameter-schemas
- /api/v1/search/supported-distance-metrics
- /api/v1/search/parameter-sets
- /api/v1/search/statistics
"""

from quart import Blueprint, request, jsonify
import logging
import time

logger = logging.getLogger(__name__)

# Create blueprint
search_bp = Blueprint('search', __name__, url_prefix='/api/v1/search')

# Module state - to be configured
_bm25_vector_override_service = None


def configure(bm25_vector_override_service=None):
    """
    Configure the search routes with required dependencies.
    
    Args:
        bm25_vector_override_service: BM25/Vector override service instance
    """
    global _bm25_vector_override_service
    _bm25_vector_override_service = bm25_vector_override_service


@search_bp.route('/parameter-schemas', methods=['GET'])
async def get_search_parameter_schemas():
    """LDTS-79: Get parameter validation schemas for BM25 and vector parameters"""
    logger.info("Received request for /api/v1/search/parameter-schemas")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        return jsonify({
            "bm25_parameters": _bm25_vector_override_service.get_bm25_parameter_schema(),
            "vector_parameters": _bm25_vector_override_service.get_vector_parameter_schema(),
            "supported_distance_metrics": _bm25_vector_override_service.get_supported_distance_metrics()
        })
        
    except Exception as e:
        logger.error(f"Failed to get parameter schemas: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get parameter schemas: {str(e)}"}), 500


@search_bp.route('/supported-distance-metrics', methods=['GET'])
async def get_supported_distance_metrics():
    """LDTS-79: Get list of supported vector distance metrics"""
    logger.info("Received request for /api/v1/search/supported-distance-metrics")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        metrics = _bm25_vector_override_service.get_supported_distance_metrics()
        
        return jsonify({
            "supported_distance_metrics": metrics,
            "count": len(metrics),
            "default_metric": "cosine"
        })
        
    except Exception as e:
        logger.error(f"Failed to get supported distance metrics: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get distance metrics: {str(e)}"}), 500


@search_bp.route('/parameter-sets', methods=['POST'])
async def create_search_parameter_set():
    """LDTS-79: Create a new search parameter set with BM25 and vector overrides"""
    logger.info("Received request for /api/v1/search/parameter-sets [POST]")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No parameter set data provided"}), 400
        
        # Extract required fields
        name = data.get("name")
        if not name:
            return jsonify({"error": "Parameter set name is required"}), 400
        
        description = data.get("description", "")
        
        parameter_set_id = _bm25_vector_override_service.create_parameter_set(
            name=name,
            description=description,
            bm25_params=data.get("bm25_parameters", []),
            vector_params=data.get("vector_parameters", []),
            hybrid_alpha=data.get("hybrid_alpha"),
            fusion_type=data.get("fusion_type", "relative_score"),
            metadata=data.get("metadata")
        )
        
        logger.info(f"Created search parameter set: {name} ({parameter_set_id})")
        
        return jsonify({
            "parameter_set_id": parameter_set_id,
            "name": name,
            "description": description,
            "created": True
        })
        
    except Exception as e:
        logger.error(f"Failed to create search parameter set: {e}", exc_info=True)
        return jsonify({"error": f"Failed to create parameter set: {str(e)}"}), 500


@search_bp.route('/parameter-sets', methods=['GET'])
async def list_search_parameter_sets():
    """LDTS-79: List all search parameter sets"""
    logger.info("Received request for /api/v1/search/parameter-sets [GET]")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        # Get query parameter for active_only (default: True)
        args = request.args
        active_only = args.get('active_only', 'true').lower() == 'true'
        
        parameter_sets = _bm25_vector_override_service.list_parameter_sets(active_only=active_only)
        
        return jsonify({
            "parameter_sets": parameter_sets,
            "total_count": len(parameter_sets),
            "active_only": active_only
        })
        
    except Exception as e:
        logger.error(f"Failed to list parameter sets: {e}", exc_info=True)
        return jsonify({"error": f"Failed to list parameter sets: {str(e)}"}), 500


@search_bp.route('/parameter-sets/<parameter_set_id>', methods=['GET'])
async def get_search_parameter_set(parameter_set_id: str):
    """LDTS-79: Get a specific search parameter set"""
    logger.info(f"Received request for /api/v1/search/parameter-sets/{parameter_set_id} [GET]")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        parameter_set = _bm25_vector_override_service.get_parameter_set(parameter_set_id)
        
        if not parameter_set:
            return jsonify({"error": f"Parameter set not found: {parameter_set_id}"}), 404
        
        return jsonify({
            "parameter_set_id": parameter_set.parameter_set_id,
            "name": parameter_set.name,
            "description": parameter_set.description,
            "bm25_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.bm25_overrides
            ],
            "vector_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.vector_overrides
            ],
            "hybrid_alpha": parameter_set.hybrid_alpha,
            "fusion_type": parameter_set.fusion_type,
            "created_at": parameter_set.created_at.isoformat(),
            "active": parameter_set.active,
            "metadata": parameter_set.metadata
        })
        
    except Exception as e:
        logger.error(f"Failed to get parameter set {parameter_set_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get parameter set: {str(e)}"}), 500


@search_bp.route('/statistics', methods=['GET'])
async def get_search_override_statistics():
    """LDTS-79: Get BM25/Vector override service statistics"""
    logger.info("Received request for /api/v1/search/statistics")
    
    try:
        if not _bm25_vector_override_service:
            return jsonify({"error": "Search service not configured"}), 503
            
        statistics = _bm25_vector_override_service.get_statistics()
        
        return jsonify({
            "service_statistics": statistics,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to get search statistics: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get statistics: {str(e)}"}), 500
