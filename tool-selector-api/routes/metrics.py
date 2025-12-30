"""
Prometheus metrics endpoint for the Tool Selector API.
"""

from quart import Blueprint, Response
from metrics import get_metrics, get_content_type

metrics_bp = Blueprint('metrics', __name__)


@metrics_bp.route('/metrics', methods=['GET'])
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return Response(
        get_metrics(),
        mimetype=get_content_type()
    )
