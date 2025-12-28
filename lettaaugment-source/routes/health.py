"""
Health Blueprint

Provides health check endpoints for monitoring the API server status.

Routes:
- GET /api/v1/health - Health check (v1)
- GET /api/health - Health check (legacy)
"""

from quart import Blueprint, jsonify
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Create the blueprint
health_bp = Blueprint('health', __name__)

# Module state - injected via configure()
_get_health_status_func = None


def configure(get_health_status_func=None):
    """
    Configure the health blueprint with required dependencies.
    
    Args:
        get_health_status_func: Function that returns health status dict
    """
    global _get_health_status_func
    _get_health_status_func = get_health_status_func
    logger.info("Health blueprint configured")


@health_bp.route('/api/v1/health', methods=['GET'])
async def health_check_v1():
    """Health check endpoint for the API server (v1)."""
    return await health_check()


@health_bp.route('/api/health', methods=['GET'])
async def health_check():
    """
    Health check endpoint for the API server.
    
    Returns status of:
    - Weaviate connection
    - Tool cache
    - MCP servers cache
    """
    if _get_health_status_func:
        # Delegate to the configured health status function
        return await _get_health_status_func()
    
    # Fallback minimal response if not configured
    return jsonify({
        "status": "OK",
        "version": "1.0.0",
        "message": "Health blueprint not fully configured",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200
