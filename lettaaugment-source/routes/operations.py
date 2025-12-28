"""
Operations Routes Blueprint

Handles system operations, maintenance, and logging endpoints:
- /api/v1/maintenance/* - System maintenance operations
- /api/v1/logs/* - Log viewing and management
- /api/v1/environment/* - Environment variable management
"""

from quart import Blueprint, request, jsonify
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprints
maintenance_bp = Blueprint('maintenance', __name__, url_prefix='/api/v1/maintenance')
logs_bp = Blueprint('logs', __name__, url_prefix='/api/v1/logs')
environment_bp = Blueprint('environment', __name__, url_prefix='/api/v1/environment')

# Module state - to be configured
_start_time = None
_cache_dir = None
_log_config_change = None
_test_weaviate_connection = None
_test_ollama_connection = None
_test_letta_connection = None
_get_tool_count_from_cache = None
_get_cache_size = None
_get_last_sync_time = None
_get_weaviate_index_status = None
_get_memory_usage = None
_get_disk_usage = None
_get_cpu_info = None
_get_log_file_size = None
_get_recent_error_count = None
_get_recent_warning_count = None
_perform_cleanup_operation = None
_perform_optimization = None
_get_log_entries = None
_perform_log_analysis = None
_get_error_log_entries = None
_clear_log_files = None
_export_log_data = None


def configure(
    start_time=None,
    cache_dir=None,
    log_config_change=None,
    test_weaviate_connection=None,
    test_ollama_connection=None,
    test_letta_connection=None,
    get_tool_count_from_cache=None,
    get_cache_size=None,
    get_last_sync_time=None,
    get_weaviate_index_status=None,
    get_memory_usage=None,
    get_disk_usage=None,
    get_cpu_info=None,
    get_log_file_size=None,
    get_recent_error_count=None,
    get_recent_warning_count=None,
    perform_cleanup_operation=None,
    perform_optimization=None,
    get_log_entries=None,
    perform_log_analysis=None,
    get_error_log_entries=None,
    clear_log_files=None,
    export_log_data=None
):
    """Configure the operations routes with required dependencies."""
    global _start_time, _cache_dir, _log_config_change
    global _test_weaviate_connection, _test_ollama_connection, _test_letta_connection
    global _get_tool_count_from_cache, _get_cache_size, _get_last_sync_time, _get_weaviate_index_status
    global _get_memory_usage, _get_disk_usage, _get_cpu_info, _get_log_file_size
    global _get_recent_error_count, _get_recent_warning_count
    global _perform_cleanup_operation, _perform_optimization
    global _get_log_entries, _perform_log_analysis, _get_error_log_entries, _clear_log_files, _export_log_data
    
    _start_time = start_time or time.time()
    _cache_dir = cache_dir
    _log_config_change = log_config_change
    _test_weaviate_connection = test_weaviate_connection
    _test_ollama_connection = test_ollama_connection
    _test_letta_connection = test_letta_connection
    _get_tool_count_from_cache = get_tool_count_from_cache
    _get_cache_size = get_cache_size
    _get_last_sync_time = get_last_sync_time
    _get_weaviate_index_status = get_weaviate_index_status
    _get_memory_usage = get_memory_usage
    _get_disk_usage = get_disk_usage
    _get_cpu_info = get_cpu_info
    _get_log_file_size = get_log_file_size
    _get_recent_error_count = get_recent_error_count
    _get_recent_warning_count = get_recent_warning_count
    _perform_cleanup_operation = perform_cleanup_operation
    _perform_optimization = perform_optimization
    _get_log_entries = get_log_entries
    _perform_log_analysis = perform_log_analysis
    _get_error_log_entries = get_error_log_entries
    _clear_log_files = clear_log_files
    _export_log_data = export_log_data


# =============================================================================
# Maintenance Endpoints
# =============================================================================

@maintenance_bp.route('/status', methods=['GET'])
async def get_maintenance_status():
    """Get system maintenance status and health information."""
    try:
        maintenance_status = {
            "system": {
                "uptime_seconds": int(time.time() - _start_time) if _start_time else 0,
                "memory_usage": _get_memory_usage() if _get_memory_usage else {},
                "disk_usage": _get_disk_usage() if _get_disk_usage else {},
                "cpu_info": _get_cpu_info() if _get_cpu_info else {},
                "last_restart": datetime.fromtimestamp(_start_time).isoformat() if _start_time else None
            },
            "services": {
                "weaviate": await _test_weaviate_connection({"url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')}) if _test_weaviate_connection else {"available": False},
                "ollama": await _test_ollama_connection({"host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}) if _test_ollama_connection else {"available": False},
                "letta_api": await _test_letta_connection() if _test_letta_connection else {"available": False}
            },
            "database": {
                "tool_count": await _get_tool_count_from_cache() if _get_tool_count_from_cache else 0,
                "cache_size": _get_cache_size() if _get_cache_size else 0,
                "last_sync": _get_last_sync_time() if _get_last_sync_time else None,
                "index_status": await _get_weaviate_index_status() if _get_weaviate_index_status else {}
            },
            "logs": {
                "log_level": os.getenv('LOG_LEVEL', 'INFO'),
                "log_size": _get_log_file_size() if _get_log_file_size else 0,
                "error_count": await _get_recent_error_count() if _get_recent_error_count else 0,
                "warning_count": await _get_recent_warning_count() if _get_recent_warning_count else 0
            }
        }

        # Determine overall health status
        health_status = "healthy"
        issues = []

        # Check service availability
        if not maintenance_status["services"]["weaviate"].get("available", False):
            health_status = "degraded"
            issues.append("Weaviate service unavailable")

        if not maintenance_status["services"]["ollama"].get("available", False):
            if os.getenv('USE_OLLAMA_EMBEDDINGS', '').lower() == 'true':
                health_status = "degraded"
                issues.append("Ollama service unavailable (but configured for embeddings)")
            else:
                issues.append("Ollama service unavailable (not critical)")

        # Check resource usage
        memory_usage = maintenance_status["system"]["memory_usage"].get("percent", 0)
        if memory_usage > 90:
            health_status = "critical"
            issues.append(f"High memory usage: {memory_usage}%")
        elif memory_usage > 80:
            health_status = "warning"
            issues.append(f"Elevated memory usage: {memory_usage}%")

        disk_usage = maintenance_status["system"]["disk_usage"].get("percent", 0)
        if disk_usage > 95:
            health_status = "critical"
            issues.append(f"Critical disk usage: {disk_usage}%")
        elif disk_usage > 85:
            health_status = "warning"
            issues.append(f"High disk usage: {disk_usage}%")

        maintenance_status["health"] = {
            "status": health_status,
            "issues": issues,
            "last_check": datetime.now().isoformat()
        }

        return jsonify({
            "success": True,
            "data": maintenance_status
        })

    except Exception as e:
        logger.error(f"Error getting maintenance status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@maintenance_bp.route('/cleanup', methods=['POST'])
async def perform_system_cleanup():
    """Perform system cleanup operations."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        operations = data.get('operations', [])
        dry_run = data.get('dry_run', False)

        available_operations = {
            "clear_cache": "Clear runtime cache files",
            "rotate_logs": "Rotate application log files",
            "cleanup_temp": "Remove temporary files",
            "compress_old_logs": "Compress old log files",
            "cleanup_backups": "Remove old backup files (>30 days)",
            "optimize_database": "Optimize Weaviate database",
            "clear_audit_logs": "Clear old audit log entries"
        }

        if not operations:
            operations = list(available_operations.keys())

        cleanup_results = {
            "operations": [],
            "total_space_freed": 0,
            "errors": []
        }

        for operation in operations:
            if operation not in available_operations:
                cleanup_results["errors"].append(f"Unknown operation: {operation}")
                continue

            try:
                if _perform_cleanup_operation:
                    result = await _perform_cleanup_operation(operation, dry_run)
                else:
                    result = {"success": False, "details": "Cleanup not configured"}
                    
                cleanup_results["operations"].append({
                    "name": operation,
                    "description": available_operations[operation],
                    "status": "completed" if result.get("success") else "failed",
                    "details": result.get("details", ""),
                    "space_freed": result.get("space_freed", 0),
                    "files_affected": result.get("files_affected", 0)
                })
                cleanup_results["total_space_freed"] += result.get("space_freed", 0)

            except Exception as e:
                cleanup_results["errors"].append(f"Error in {operation}: {str(e)}")
                cleanup_results["operations"].append({
                    "name": operation,
                    "description": available_operations[operation],
                    "status": "failed",
                    "details": str(e),
                    "space_freed": 0,
                    "files_affected": 0
                })

        # Log cleanup operation
        if _log_config_change:
            await _log_config_change(
                action="system_cleanup",
                config_type="maintenance",
                changes={"operations": operations, "dry_run": dry_run},
                metadata={"results": cleanup_results}
            )

        return jsonify({
            "success": True,
            "data": {
                "cleanup_results": cleanup_results,
                "dry_run": dry_run,
                "completed_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error performing system cleanup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@maintenance_bp.route('/restart', methods=['POST'])
async def restart_system_components():
    """Restart system components (simulated - requires external orchestration)."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        components = data.get('components', [])
        force = data.get('force', False)

        available_components = {
            "api_server": "API Server (requires external restart)",
            "sync_service": "Tool Sync Service",
            "mcp_server": "MCP Protocol Server",
            "weaviate": "Weaviate Database (external service)",
            "ollama": "Ollama Service (external service)"
        }

        if not components:
            return jsonify({
                "success": False,
                "error": "No components specified for restart"
            }), 400

        restart_results = {
            "components": [],
            "warnings": []
        }

        for component in components:
            if component not in available_components:
                restart_results["warnings"].append(f"Unknown component: {component}")
                continue

            if component == "api_server":
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "requires_external_action",
                    "message": "API server restart requires external orchestration (Docker, systemd, etc.)"
                })
            elif component in ["weaviate", "ollama"]:
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "external_service",
                    "message": f"{component.capitalize()} is an external service - restart manually or via orchestration"
                })
            else:
                restart_results["components"].append({
                    "name": component,
                    "description": available_components[component],
                    "status": "simulated",
                    "message": f"{component} restart simulated - implement actual restart logic"
                })

        # Log restart request
        if _log_config_change:
            await _log_config_change(
                action="component_restart",
                config_type="maintenance",
                changes={"components": components, "force": force},
                metadata={"results": restart_results}
            )

        return jsonify({
            "success": True,
            "data": {
                "restart_results": restart_results,
                "requested_at": datetime.now().isoformat(),
                "note": "Most restart operations require external orchestration"
            }
        })

    except Exception as e:
        logger.error(f"Error restarting system components: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@maintenance_bp.route('/optimize', methods=['POST'])
async def optimize_system():
    """Optimize system performance."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        operations = data.get('operations', [])

        available_optimizations = {
            "rebuild_cache": "Rebuild tool cache from Letta API",
            "reindex_weaviate": "Trigger Weaviate reindexing",
            "compact_database": "Compact database files",
            "optimize_embeddings": "Optimize embedding storage",
            "cleanup_duplicates": "Remove duplicate tool entries"
        }

        if not operations:
            operations = list(available_optimizations.keys())

        optimization_results = {
            "operations": [],
            "performance_impact": {},
            "warnings": []
        }

        for operation in operations:
            if operation not in available_optimizations:
                optimization_results["warnings"].append(f"Unknown optimization: {operation}")
                continue

            try:
                if _perform_optimization:
                    result = await _perform_optimization(operation)
                else:
                    result = {"success": False, "details": "Optimization not configured"}
                    
                optimization_results["operations"].append({
                    "name": operation,
                    "description": available_optimizations[operation],
                    "status": "completed" if result.get("success") else "failed",
                    "details": result.get("details", ""),
                    "performance_impact": result.get("performance_impact", {})
                })

            except Exception as e:
                optimization_results["warnings"].append(f"Error in {operation}: {str(e)}")

        # Log optimization
        if _log_config_change:
            await _log_config_change(
                action="system_optimization",
                config_type="maintenance",
                changes={"operations": operations},
                metadata={"results": optimization_results}
            )

        return jsonify({
            "success": True,
            "data": {
                "optimization_results": optimization_results,
                "completed_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error optimizing system: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Log Endpoints
# =============================================================================

@logs_bp.route('', methods=['GET'])
async def get_logs():
    """Get application logs with filtering and pagination."""
    try:
        # Query parameters
        level = request.args.get('level', 'all')
        lines = min(int(request.args.get('lines', 100)), 10000)
        search = request.args.get('search', '')
        from_time = request.args.get('from', '')
        to_time = request.args.get('to', '')

        if _get_log_entries:
            log_entries = await _get_log_entries(
                level=level,
                lines=lines,
                search=search,
                from_time=from_time,
                to_time=to_time
            )
        else:
            log_entries = []

        return jsonify({
            "success": True,
            "data": {
                "logs": log_entries,
                "total_lines": len(log_entries),
                "filters": {
                    "level": level,
                    "lines": lines,
                    "search": search,
                    "from_time": from_time,
                    "to_time": to_time
                }
            }
        })
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@logs_bp.route('/analysis', methods=['GET'])
async def analyze_logs():
    """Analyze logs for patterns, errors, and insights."""
    try:
        timeframe = request.args.get('timeframe', '24h')
        include_details = request.args.get('include_details', 'false').lower() == 'true'

        if _perform_log_analysis:
            analysis = await _perform_log_analysis(timeframe, include_details)
        else:
            analysis = {"message": "Log analysis not configured"}

        return jsonify({
            "success": True,
            "data": analysis
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@logs_bp.route('/errors', methods=['GET'])
async def get_error_logs():
    """Get error logs with detailed analysis."""
    try:
        hours = min(int(request.args.get('hours', 24)), 168)
        include_stack_trace = request.args.get('include_stack_trace', 'true').lower() == 'true'
        group_by = request.args.get('group_by', 'none')

        if _get_error_log_entries:
            error_logs = await _get_error_log_entries(
                hours=hours,
                include_stack_trace=include_stack_trace,
                group_by=group_by
            )
        else:
            error_logs = {"errors": [], "message": "Error log retrieval not configured"}

        return jsonify({
            "success": True,
            "data": error_logs
        })
    except Exception as e:
        logger.error(f"Error getting error logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@logs_bp.route('/clear', methods=['POST'])
async def clear_logs():
    """Clear log files with backup option."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        backup = data.get('backup', True)
        older_than_days = data.get('older_than_days', 0)

        if _clear_log_files:
            result = await _clear_log_files(backup=backup, older_than_days=older_than_days)
        else:
            result = {"success": False, "message": "Log clearing not configured"}

        # Log the action
        if _log_config_change:
            await _log_config_change(
                action="clear_logs",
                config_type="logging",
                changes={"backup": backup, "older_than_days": older_than_days},
                metadata={"result": result}
            )

        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@logs_bp.route('/export', methods=['POST'])
async def export_logs():
    """Export logs in various formats (JSON, CSV, text)."""
    try:
        data = await request.get_json()
        format_type = data.get('format', 'json')
        filters = data.get('filters', {})

        if _export_log_data:
            export_result = await _export_log_data(format_type, filters)
        else:
            export_result = {"success": False, "message": "Log export not configured"}

        return jsonify({
            "success": True,
            "data": export_result
        })
    except Exception as e:
        logger.error(f"Error exporting logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Environment Endpoints
# =============================================================================

# Safe environment variables that can be exposed
SAFE_ENV_VARS = {
    # Tool Selector Configuration
    'MAX_TOTAL_TOOLS', 'MAX_MCP_TOOLS', 'MIN_MCP_TOOLS', 'DEFAULT_DROP_RATE',
    'EXCLUDE_LETTA_CORE_TOOLS', 'EXCLUDE_OFFICIAL_TOOLS', 'MANAGE_ONLY_MCP_TOOLS',
    'MIN_SCORE_DEFAULT', 'SEMANTIC_WEIGHT', 'KEYWORD_WEIGHT',
    # Ollama Configuration
    'OLLAMA_EMBEDDING_HOST', 'OLLAMA_PORT', 'OLLAMA_TIMEOUT', 'OLLAMA_BASE_URL',
    'OLLAMA_EMBEDDING_MODEL', 'USE_OLLAMA_EMBEDDINGS', 'OLLAMA_DEFAULT_MODEL',
    'OLLAMA_TEMPERATURE', 'OLLAMA_CONTEXT_LENGTH', 'OLLAMA_NUM_PARALLEL',
    'OLLAMA_NUM_CTX', 'OLLAMA_NUM_GPU', 'OLLAMA_LOW_VRAM',
    # Weaviate Configuration
    'WEAVIATE_URL', 'WEAVIATE_TIMEOUT', 'WEAVIATE_RETRIES', 'WEAVIATE_CLASS_NAME',
    'WEAVIATE_VECTOR_INDEX', 'WEAVIATE_DISTANCE_METRIC', 'WEAVIATE_ALPHA',
    'WEAVIATE_LIMIT', 'WEAVIATE_AUTOCUT', 'WEAVIATE_EF_CONSTRUCTION', 'WEAVIATE_EF',
    'WEAVIATE_MAX_CONNECTIONS', 'WEAVIATE_VECTOR_CACHE_MAX_OBJECTS', 'WEAVIATE_CLEANUP_INTERVAL',
    # Embedding Configuration
    'EMBEDDING_PROVIDER', 'OPENAI_EMBEDDING_MODEL', 'EMBEDDING_DIMENSION',
    # Reranker Configuration
    'RERANKER_ENABLED', 'RERANKER_PROVIDER', 'RERANKER_MODEL', 'RERANKER_URL',
    # Logging Configuration
    'LOG_LEVEL', 'LOG_FORMAT', 'LOG_FILE',
    # Server Configuration
    'API_PORT', 'API_HOST', 'DEBUG_MODE',
    # Letta API Configuration (safe parts)
    'LETTA_API_URL', 'LETTA_TIMEOUT'
}


@environment_bp.route('', methods=['GET'])
async def get_environment_variables():
    """Get current environment variables (filtered for security)."""
    try:
        env_vars = {}
        for var in SAFE_ENV_VARS:
            value = os.getenv(var)
            if value is not None:
                env_vars[var] = value

        # Group by category
        grouped_vars = {
            "tool_selector": {},
            "ollama": {},
            "weaviate": {},
            "embedding": {},
            "reranker": {},
            "logging": {},
            "server": {},
            "letta_api": {}
        }

        for var, value in env_vars.items():
            if var.startswith('MAX_') or var.startswith('MIN_') or var.startswith('DEFAULT_') or var.startswith('EXCLUDE_') or var.startswith('MANAGE_') or var in ['SEMANTIC_WEIGHT', 'KEYWORD_WEIGHT']:
                grouped_vars["tool_selector"][var] = value
            elif var.startswith('OLLAMA_') or var == 'USE_OLLAMA_EMBEDDINGS':
                grouped_vars["ollama"][var] = value
            elif var.startswith('WEAVIATE_'):
                grouped_vars["weaviate"][var] = value
            elif var.startswith('EMBEDDING_') or var == 'OPENAI_EMBEDDING_MODEL':
                grouped_vars["embedding"][var] = value
            elif var.startswith('RERANKER_'):
                grouped_vars["reranker"][var] = value
            elif var.startswith('LOG_'):
                grouped_vars["logging"][var] = value
            elif var.startswith('API_') or var == 'DEBUG_MODE':
                grouped_vars["server"][var] = value
            elif var.startswith('LETTA_'):
                grouped_vars["letta_api"][var] = value

        return jsonify({
            "success": True,
            "data": {
                "variables": env_vars,
                "grouped": grouped_vars,
                "total_count": len(env_vars)
            }
        })
    except Exception as e:
        logger.error(f"Error getting environment variables: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@environment_bp.route('/validate', methods=['POST'])
async def validate_environment():
    """Validate environment variable values."""
    try:
        data = await request.get_json()
        variables = data.get('variables', {})

        validation_results = []
        for var_name, var_value in variables.items():
            result = {"name": var_name, "value": var_value, "valid": True, "errors": [], "warnings": []}

            # Basic validation rules
            if var_name in ['MAX_TOTAL_TOOLS', 'MAX_MCP_TOOLS', 'MIN_MCP_TOOLS']:
                try:
                    val = int(var_value)
                    if val < 1:
                        result["valid"] = False
                        result["errors"].append("Value must be at least 1")
                    elif val > 100:
                        result["warnings"].append("Very high value may impact performance")
                except ValueError:
                    result["valid"] = False
                    result["errors"].append("Value must be an integer")

            elif var_name in ['DEFAULT_DROP_RATE', 'WEAVIATE_ALPHA']:
                try:
                    val = float(var_value)
                    if val < 0 or val > 1:
                        result["valid"] = False
                        result["errors"].append("Value must be between 0 and 1")
                except ValueError:
                    result["valid"] = False
                    result["errors"].append("Value must be a number")

            elif var_name in ['OLLAMA_PORT', 'WEAVIATE_TIMEOUT', 'LETTA_TIMEOUT']:
                try:
                    val = int(var_value)
                    if val < 1:
                        result["valid"] = False
                        result["errors"].append("Value must be positive")
                except ValueError:
                    result["valid"] = False
                    result["errors"].append("Value must be an integer")

            validation_results.append(result)

        all_valid = all(r["valid"] for r in validation_results)

        return jsonify({
            "success": True,
            "data": {
                "results": validation_results,
                "all_valid": all_valid
            }
        })
    except Exception as e:
        logger.error(f"Error validating environment: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@environment_bp.route('/template', methods=['GET'])
async def get_environment_template():
    """Get environment variable template with descriptions."""
    try:
        template = {
            "tool_selector": {
                "MAX_TOTAL_TOOLS": {"default": "30", "description": "Maximum total tools per agent", "type": "integer"},
                "MAX_MCP_TOOLS": {"default": "20", "description": "Maximum MCP tools per agent", "type": "integer"},
                "MIN_MCP_TOOLS": {"default": "7", "description": "Minimum MCP tools to keep", "type": "integer"},
                "DEFAULT_DROP_RATE": {"default": "0.6", "description": "Default tool drop rate (0-1)", "type": "float"}
            },
            "ollama": {
                "OLLAMA_EMBEDDING_HOST": {"default": "192.168.50.80", "description": "Ollama server host", "type": "string"},
                "OLLAMA_PORT": {"default": "11434", "description": "Ollama server port", "type": "integer"},
                "OLLAMA_EMBEDDING_MODEL": {"default": "nomic-embed-text", "description": "Embedding model name", "type": "string"},
                "USE_OLLAMA_EMBEDDINGS": {"default": "false", "description": "Use Ollama for embeddings", "type": "boolean"}
            },
            "weaviate": {
                "WEAVIATE_URL": {"default": "http://weaviate:8080/", "description": "Weaviate server URL", "type": "url"},
                "WEAVIATE_TIMEOUT": {"default": "30", "description": "Connection timeout in seconds", "type": "integer"},
                "WEAVIATE_ALPHA": {"default": "0.75", "description": "Hybrid search alpha (0-1)", "type": "float"}
            }
        }

        return jsonify({
            "success": True,
            "data": template
        })
    except Exception as e:
        logger.error(f"Error getting environment template: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
