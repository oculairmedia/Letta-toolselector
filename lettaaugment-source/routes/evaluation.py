"""
Evaluation Blueprint

Provides endpoints for evaluation, analytics, and A/B comparison testing.

Routes:
- POST /api/v1/evaluations - Submit an evaluation
- GET /api/v1/evaluations - Get evaluations
- GET /api/v1/analytics - Get analytics data
- POST /api/v1/rerank/compare - Compare reranker configurations
- POST /api/v1/ab-comparison/run - Run A/B comparison
- GET /api/v1/ab-comparison/results - Get A/B comparison results
- GET /api/v1/ab-comparison/results/<id> - Get specific A/B comparison result
"""

from quart import Blueprint, request, jsonify
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Create the blueprint
evaluation_bp = Blueprint('evaluation', __name__)

# Module state - injected via configure()
_submit_evaluation_func: Optional[Callable] = None
_get_evaluations_func: Optional[Callable] = None
_get_analytics_func: Optional[Callable] = None
_compare_rerank_func: Optional[Callable] = None
_run_ab_comparison_func: Optional[Callable] = None
_get_ab_results_func: Optional[Callable] = None
_get_ab_result_by_id_func: Optional[Callable] = None


def configure(
    submit_evaluation_func: Optional[Callable] = None,
    get_evaluations_func: Optional[Callable] = None,
    get_analytics_func: Optional[Callable] = None,
    compare_rerank_func: Optional[Callable] = None,
    run_ab_comparison_func: Optional[Callable] = None,
    get_ab_results_func: Optional[Callable] = None,
    get_ab_result_by_id_func: Optional[Callable] = None
):
    """
    Configure the evaluation blueprint with required dependencies.
    """
    global _submit_evaluation_func, _get_evaluations_func, _get_analytics_func
    global _compare_rerank_func, _run_ab_comparison_func
    global _get_ab_results_func, _get_ab_result_by_id_func
    
    _submit_evaluation_func = submit_evaluation_func
    _get_evaluations_func = get_evaluations_func
    _get_analytics_func = get_analytics_func
    _compare_rerank_func = compare_rerank_func
    _run_ab_comparison_func = run_ab_comparison_func
    _get_ab_results_func = get_ab_results_func
    _get_ab_result_by_id_func = get_ab_result_by_id_func
    
    logger.info("Evaluation blueprint configured")


@evaluation_bp.route('/api/v1/evaluations', methods=['POST'])
async def submit_evaluation():
    """Submit an evaluation rating."""
    if not _submit_evaluation_func:
        return jsonify({"error": "Submit evaluation not configured"}), 503
    return await _submit_evaluation_func()


@evaluation_bp.route('/api/v1/evaluations', methods=['GET'])
async def get_evaluations():
    """Get evaluations with optional filtering."""
    if not _get_evaluations_func:
        return jsonify({"error": "Get evaluations not configured"}), 503
    return await _get_evaluations_func()


@evaluation_bp.route('/api/v1/analytics', methods=['GET'])
async def get_analytics():
    """Get analytics data."""
    if not _get_analytics_func:
        return jsonify({"error": "Analytics not configured"}), 503
    return await _get_analytics_func()


@evaluation_bp.route('/api/v1/rerank/compare', methods=['POST'])
async def compare_rerank():
    """Compare reranker configurations."""
    if not _compare_rerank_func:
        return jsonify({"error": "Rerank compare not configured"}), 503
    return await _compare_rerank_func()


@evaluation_bp.route('/api/v1/ab-comparison/run', methods=['POST'])
async def run_ab_comparison():
    """Run A/B comparison tests."""
    if not _run_ab_comparison_func:
        return jsonify({"error": "A/B comparison not configured"}), 503
    return await _run_ab_comparison_func()


@evaluation_bp.route('/api/v1/ab-comparison/results', methods=['GET'])
async def get_ab_results():
    """Get A/B comparison results."""
    if not _get_ab_results_func:
        return jsonify({"error": "A/B results not configured"}), 503
    return await _get_ab_results_func()


@evaluation_bp.route('/api/v1/ab-comparison/results/<comparison_id>', methods=['GET'])
async def get_ab_result_by_id(comparison_id):
    """Get specific A/B comparison result."""
    if not _get_ab_result_by_id_func:
        return jsonify({"error": "A/B result lookup not configured"}), 503
    return await _get_ab_result_by_id_func(comparison_id)
