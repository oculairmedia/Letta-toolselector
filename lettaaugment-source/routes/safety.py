"""
Safety Blueprint

Provides safety status and validation endpoints.

Routes:
- GET /api/v1/safety/status - Get safety system status
- POST /api/v1/safety/validate-operation - Validate operation safety
- GET /api/v1/safety/emergency-status - Get emergency safety status
"""

from quart import Blueprint, request, jsonify
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Create the blueprint
safety_bp = Blueprint('safety', __name__)

# Module state - injected via configure()
_get_safety_status_func: Optional[Callable] = None
_validate_operation_func: Optional[Callable] = None
_get_emergency_status_func: Optional[Callable] = None


def configure(
    get_safety_status_func: Optional[Callable] = None,
    validate_operation_func: Optional[Callable] = None,
    get_emergency_status_func: Optional[Callable] = None
):
    """
    Configure the safety blueprint with required dependencies.
    """
    global _get_safety_status_func, _validate_operation_func, _get_emergency_status_func
    
    _get_safety_status_func = get_safety_status_func
    _validate_operation_func = validate_operation_func
    _get_emergency_status_func = get_emergency_status_func
    
    logger.info("Safety blueprint configured")


@safety_bp.route('/api/v1/safety/status', methods=['GET'])
async def get_safety_status():
    """Get comprehensive safety system status."""
    if not _get_safety_status_func:
        return jsonify({"error": "Safety status not configured"}), 503
    return await _get_safety_status_func()


@safety_bp.route('/api/v1/safety/validate-operation', methods=['POST'])
async def validate_operation():
    """Validate if a specific operation is safe to perform."""
    if not _validate_operation_func:
        return jsonify({"error": "Operation validation not configured"}), 503
    return await _validate_operation_func()


@safety_bp.route('/api/v1/safety/emergency-status', methods=['GET'])
async def get_emergency_status():
    """Get emergency safety status for critical monitoring."""
    if not _get_emergency_status_func:
        return jsonify({"error": "Emergency status not configured"}), 503
    return await _get_emergency_status_func()
