"""
Cost Control Routes Blueprint

Handles cost control and budget management endpoints:
- /api/v1/cost-control/status - System status
- /api/v1/cost-control/summary - Cost summary by period
- /api/v1/cost-control/budget - Get/Set/Delete budget limits
- /api/v1/cost-control/alerts - Get cost alerts
- /api/v1/cost-control/record - Record manual cost
- /api/v1/cost-control/estimate - Estimate operation cost
- /api/v1/cost-control/reset - Reset period costs
"""

from quart import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

# Create blueprint
cost_control_bp = Blueprint('cost_control', __name__, url_prefix='/api/v1/cost-control')

# Module state - to be configured
_get_cost_manager = None
_CostCategory = None
_BudgetPeriod = None
_AlertLevel = None


def configure(get_cost_manager=None, CostCategory=None, BudgetPeriod=None, AlertLevel=None):
    """
    Configure the cost control routes with required dependencies.
    
    Args:
        get_cost_manager: Function to get the cost manager instance
        CostCategory: CostCategory enum class
        BudgetPeriod: BudgetPeriod enum class
        AlertLevel: AlertLevel enum class
    """
    global _get_cost_manager, _CostCategory, _BudgetPeriod, _AlertLevel
    _get_cost_manager = get_cost_manager
    _CostCategory = CostCategory
    _BudgetPeriod = BudgetPeriod
    _AlertLevel = AlertLevel


def _check_configured():
    """Check if the blueprint is properly configured."""
    if not _get_cost_manager:
        return False, "Cost control not configured"
    return True, None


@cost_control_bp.route('/status', methods=['GET'])
async def get_cost_control_status():
    """Get overall cost control system status"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        manager = _get_cost_manager()
        
        # Get budget status
        budget_status = await manager.get_budget_status()
        
        # Get recent alerts
        recent_alerts = await manager.get_recent_alerts(hours=24)
        alert_count = len(recent_alerts)
        critical_alerts = len([a for a in recent_alerts if a.level in [_AlertLevel.CRITICAL, _AlertLevel.EMERGENCY]])
        
        # Get daily summary
        daily_summary = await manager.get_cost_summary(_BudgetPeriod.DAILY)
        
        return jsonify({
            'success': True,
            'data': {
                'budget_status': budget_status,
                'daily_summary': daily_summary.to_dict(),
                'alert_summary': {
                    'total_alerts_24h': alert_count,
                    'critical_alerts_24h': critical_alerts
                },
                'system_status': 'healthy' if critical_alerts == 0 else 'warning'
            }
        })
        
    except Exception as e:
        logger.error(f"Cost control status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/summary', methods=['GET'])
async def get_cost_summary():
    """Get cost summary for specified period"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        period_str = request.args.get('period', 'daily').lower()
        
        try:
            period = _BudgetPeriod(period_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {period_str}. Valid periods: {[p.value for p in _BudgetPeriod]}'
            }), 400
        
        manager = _get_cost_manager()
        summary = await manager.get_cost_summary(period)
        
        return jsonify({
            'success': True,
            'data': summary.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Cost summary error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/budget', methods=['GET'])
async def get_budget_limits():
    """Get all budget limits and their status"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        manager = _get_cost_manager()
        budget_status = await manager.get_budget_status()
        
        return jsonify({
            'success': True,
            'data': budget_status
        })
        
    except Exception as e:
        logger.error(f"Budget limits error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/budget', methods=['POST'])
async def set_budget_limit():
    """Set or update a budget limit"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['period', 'limit']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Parse period
        try:
            period = _BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in _BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = _CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in _CostCategory]}'
                }), 400
        
        # Parse limit
        try:
            limit = float(data['limit'])
            if limit < 0:
                raise ValueError("Limit must be non-negative")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid limit value. Must be a non-negative number.'
            }), 400
        
        manager = _get_cost_manager()
        
        # Set budget limit
        budget_key = await manager.set_budget_limit(
            category=category,
            period=period,
            limit=limit,
            hard_limit=data.get('hard_limit', False),
            alert_thresholds=data.get('alert_thresholds', [0.5, 0.8, 0.95]),
            enabled=data.get('enabled', True)
        )
        
        return jsonify({
            'success': True,
            'data': {
                'budget_key': budget_key,
                'message': f'Budget limit set for {category.value if category else "overall"} - {period.value}'
            }
        })
        
    except Exception as e:
        logger.error(f"Set budget limit error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/budget', methods=['DELETE'])
async def remove_budget_limit():
    """Remove a budget limit"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        data = await request.get_json()
        
        # Validate required fields
        if 'period' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: period'
            }), 400
        
        # Parse period
        try:
            period = _BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in _BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = _CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in _CostCategory]}'
                }), 400
        
        manager = _get_cost_manager()
        removed = await manager.remove_budget_limit(category, period)
        
        if removed:
            return jsonify({
                'success': True,
                'data': {
                    'message': f'Budget limit removed for {category.value if category else "overall"} - {period.value}'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Budget limit not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Remove budget limit error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/alerts', methods=['GET'])
async def get_cost_alerts():
    """Get recent cost alerts"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        hours = int(request.args.get('hours', 24))
        if hours < 1 or hours > 168:  # Limit to 1 hour - 1 week
            hours = 24
            
        manager = _get_cost_manager()
        alerts = await manager.get_recent_alerts(hours=hours)
        
        # Convert alerts to dict format
        alert_data = [alert.to_dict() for alert in alerts]
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alert_data,
                'count': len(alert_data),
                'hours_requested': hours
            }
        })
        
    except Exception as e:
        logger.error(f"Cost alerts error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/record', methods=['POST'])
async def record_manual_cost():
    """Manually record a cost entry"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['category', 'operation', 'cost']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Parse category
        try:
            category = _CostCategory(data['category'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in _CostCategory]}'
            }), 400
        
        # Parse cost
        try:
            cost = float(data['cost'])
            if cost < 0:
                raise ValueError("Cost must be non-negative")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid cost value. Must be a non-negative number.'
            }), 400
        
        manager = _get_cost_manager()
        
        # Record the cost
        allowed = await manager.record_cost(
            category=category,
            operation=data['operation'],
            cost=cost,
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'success': True,
            'data': {
                'recorded': True,
                'allowed': allowed,
                'message': 'Cost recorded successfully' if allowed else 'Cost recorded but operation would be blocked by hard limits'
            }
        })
        
    except Exception as e:
        logger.error(f"Record manual cost error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/estimate', methods=['POST'])
async def estimate_operation_cost():
    """Estimate cost for a planned operation"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        data = await request.get_json()
        
        if 'operation_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: operation_type'
            }), 400
        
        operation_type = data['operation_type']
        params = data.get('params', {})
        
        manager = _get_cost_manager()
        estimated_cost = await manager.estimate_operation_cost(operation_type, **params)
        
        return jsonify({
            'success': True,
            'data': {
                'operation_type': operation_type,
                'estimated_cost': estimated_cost,
                'currency': 'USD',
                'params_used': params
            }
        })
        
    except Exception as e:
        logger.error(f"Estimate operation cost error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@cost_control_bp.route('/reset', methods=['POST'])
async def reset_period_costs():
    """Reset costs for a specific period and category (admin function)"""
    try:
        ok, err = _check_configured()
        if not ok:
            return jsonify({'success': False, 'error': err}), 503
        
        data = await request.get_json()
        
        if 'period' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: period'
            }), 400
        
        # Parse period
        try:
            period = _BudgetPeriod(data['period'].lower())
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid period: {data["period"]}. Valid periods: {[p.value for p in _BudgetPeriod]}'
            }), 400
        
        # Parse category (optional)
        category = None
        if 'category' in data and data['category']:
            try:
                category = _CostCategory(data['category'].lower())
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'Invalid category: {data["category"]}. Valid categories: {[c.value for c in _CostCategory]}'
                }), 400
        
        manager = _get_cost_manager()
        await manager.reset_period_costs(category, period)
        
        return jsonify({
            'success': True,
            'data': {
                'message': f'Costs reset for {category.value if category else "all categories"} in period {period.value}'
            }
        })
        
    except Exception as e:
        logger.error(f"Reset period costs error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
