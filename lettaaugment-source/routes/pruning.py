"""
API routes for scheduled pruning management.

Provides endpoints to:
- View scheduler status
- Trigger manual pruning runs
- Configure scheduler settings
"""

import logging
from quart import Blueprint, jsonify, request

from pruning_scheduler import get_pruning_scheduler, PruningSchedulerConfig

# Alias for external configuration
get_scheduler = get_pruning_scheduler

logger = logging.getLogger(__name__)

pruning_bp = Blueprint('pruning', __name__, url_prefix='/api/v1/pruning')


@pruning_bp.route('/status', methods=['GET'])
async def get_scheduler_status():
    """Get current pruning scheduler status."""
    scheduler = get_pruning_scheduler()
    return jsonify(scheduler.get_status())


@pruning_bp.route('/run', methods=['POST'])
async def trigger_pruning_run():
    """
    Trigger a manual pruning run.
    
    Query params:
        dry_run: bool (optional) - Override dry_run setting
        
    Body (optional):
        {
            "agent_ids": ["id1", "id2"],  // Specific agents to prune (optional)
            "dry_run": true/false          // Override dry_run setting
        }
    """
    scheduler = get_pruning_scheduler()
    
    if not scheduler.is_configured():
        return jsonify({
            "error": "Scheduler not configured",
            "message": "The pruning scheduler has not been configured with required functions"
        }), 503
    
    # Get dry_run override from query or body
    dry_run = None
    
    # Check query param first
    dry_run_param = request.args.get('dry_run')
    if dry_run_param is not None:
        dry_run = dry_run_param.lower() == 'true'
    
    # Body can override
    try:
        body = await request.get_json() or {}
        if 'dry_run' in body:
            dry_run = bool(body['dry_run'])
    except Exception:
        body = {}
    
    logger.info(f"Manual pruning run triggered (dry_run={dry_run})")
    
    try:
        result = await scheduler.run_now(dry_run=dry_run)
        
        return jsonify({
            "success": result.error is None,
            "dry_run": result.dry_run,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "summary": {
                "agents_processed": result.agents_processed,
                "agents_skipped": result.agents_skipped,
                "agents_failed": result.agents_failed,
                "total_tools_pruned": result.total_tools_pruned,
            },
            "error": result.error,
            "results": [
                {
                    "agent_id": r.agent_id,
                    "success": r.success,
                    "mcp_tools_before": r.mcp_tools_before,
                    "mcp_tools_after": r.mcp_tools_after,
                    "tools_pruned": r.tools_pruned,
                    "tools_protected": r.tools_protected,
                    "skipped_reason": r.skipped_reason,
                    "error": r.error,
                }
                for r in result.results
            ]
        })
        
    except Exception as e:
        logger.exception(f"Error during manual pruning run: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@pruning_bp.route('/config', methods=['GET'])
async def get_scheduler_config():
    """Get current scheduler configuration."""
    scheduler = get_pruning_scheduler()
    config = scheduler.config
    
    return jsonify({
        "enabled": config.enabled,
        "interval_hours": config.interval_hours,
        "drop_rate": config.drop_rate,
        "dry_run": config.dry_run,
        "min_mcp_tools": config.min_mcp_tools,
        "skip_agents": list(config.skip_agents),
        "batch_size": config.batch_size,
        "batch_delay_seconds": config.batch_delay_seconds,
    })


@pruning_bp.route('/config', methods=['PATCH'])
async def update_scheduler_config():
    """
    Update scheduler configuration (runtime only, not persisted).
    
    Body:
        {
            "enabled": bool,
            "interval_hours": float,
            "drop_rate": float,
            "dry_run": bool,
            "min_mcp_tools": int,
            "skip_agents": ["agent_id1", "agent_id2"],
            "batch_size": int,
            "batch_delay_seconds": float
        }
    """
    scheduler = get_pruning_scheduler()
    
    try:
        body = await request.get_json() or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    config = scheduler.config
    updated_fields = []
    
    if 'enabled' in body:
        config.enabled = bool(body['enabled'])
        updated_fields.append('enabled')
        
    if 'interval_hours' in body:
        config.interval_hours = float(body['interval_hours'])
        updated_fields.append('interval_hours')
        
    if 'drop_rate' in body:
        rate = float(body['drop_rate'])
        if not 0.0 <= rate <= 1.0:
            return jsonify({"error": "drop_rate must be between 0.0 and 1.0"}), 400
        config.drop_rate = rate
        updated_fields.append('drop_rate')
        
    if 'dry_run' in body:
        config.dry_run = bool(body['dry_run'])
        updated_fields.append('dry_run')
        
    if 'min_mcp_tools' in body:
        min_tools = int(body['min_mcp_tools'])
        if min_tools < 0:
            return jsonify({"error": "min_mcp_tools must be >= 0"}), 400
        config.min_mcp_tools = min_tools
        updated_fields.append('min_mcp_tools')
        
    if 'skip_agents' in body:
        config.skip_agents = set(body['skip_agents'])
        updated_fields.append('skip_agents')
        
    if 'batch_size' in body:
        batch_size = int(body['batch_size'])
        if batch_size < 1:
            return jsonify({"error": "batch_size must be >= 1"}), 400
        config.batch_size = batch_size
        updated_fields.append('batch_size')
        
    if 'batch_delay_seconds' in body:
        config.batch_delay_seconds = float(body['batch_delay_seconds'])
        updated_fields.append('batch_delay_seconds')
    
    logger.info(f"Scheduler config updated: {updated_fields}")
    
    return jsonify({
        "success": True,
        "updated_fields": updated_fields,
        "config": {
            "enabled": config.enabled,
            "interval_hours": config.interval_hours,
            "drop_rate": config.drop_rate,
            "dry_run": config.dry_run,
            "min_mcp_tools": config.min_mcp_tools,
            "skip_agents": list(config.skip_agents),
            "batch_size": config.batch_size,
            "batch_delay_seconds": config.batch_delay_seconds,
        }
    })


@pruning_bp.route('/start', methods=['POST'])
async def start_scheduler():
    """Start the background scheduler."""
    scheduler = get_pruning_scheduler()
    
    if not scheduler.is_configured():
        return jsonify({
            "error": "Scheduler not configured",
            "message": "The pruning scheduler has not been configured with required functions"
        }), 503
    
    if scheduler._running:
        return jsonify({
            "success": True,
            "message": "Scheduler already running",
            "status": scheduler.get_status()
        })
    
    # Enable if not already
    scheduler.config.enabled = True
    await scheduler.start()
    
    return jsonify({
        "success": True,
        "message": "Scheduler started",
        "status": scheduler.get_status()
    })


@pruning_bp.route('/stop', methods=['POST'])
async def stop_scheduler():
    """Stop the background scheduler."""
    scheduler = get_pruning_scheduler()
    
    if not scheduler._running:
        return jsonify({
            "success": True,
            "message": "Scheduler not running",
            "status": scheduler.get_status()
        })
    
    await scheduler.stop()
    
    return jsonify({
        "success": True,
        "message": "Scheduler stopped",
        "status": scheduler.get_status()
    })
