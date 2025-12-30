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
import os
import time

logger = logging.getLogger(__name__)

# Create the blueprint
safety_bp = Blueprint('safety', __name__)


def configure():
    """
    Configure the safety blueprint.
    
    No external dependencies needed - all safety checks use environment variables.
    """
    logger.info("Safety blueprint configured")


# =============================================================================
# Safety Helper Functions
# =============================================================================

def check_letta_api_isolation() -> bool:
    """Check that Letta API writes are disabled"""
    try:
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        
        return safety_mode == "testing" and read_only_mode and no_attach_mode
    except Exception:
        return False


def check_agent_modification_blocked() -> bool:
    """Check that agent modifications are blocked"""
    try:
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        block_agent_ops = os.getenv("LDTS_BLOCK_AGENT_MODIFICATIONS", "true").lower() == "true"
        
        return (safety_mode == "testing" and no_attach_mode) or block_agent_ops
    except Exception:
        return False


def check_tool_attachment_blocked() -> bool:
    """Check that tool attachment/detachment is blocked"""
    try:
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        
        return no_attach_mode or read_only_mode
    except Exception:
        return False


def check_database_read_only() -> bool:
    """Check that database is in read-only mode"""
    try:
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        database_mode = os.getenv("LDTS_DATABASE_MODE", "testing")
        db_url = os.getenv("DATABASE_URL", "")
        
        return read_only_mode and database_mode == "testing" and "prod" not in db_url.lower()
    except Exception:
        return False


def check_sandbox_mode() -> bool:
    """Check that we're operating in sandbox mode"""
    try:
        return os.getenv("LDTS_SAFETY_MODE", "production") == "testing"
    except Exception:
        return False


def has_production_indicators(operation: str, context: dict) -> bool:
    """Check if operation has production indicators"""
    production_flags = [
        "agent_id" in context,
        "production" in str(context).lower(),
        "prod" in str(context).lower(),
        "modify" in operation.lower(),
        "create" in operation.lower() and "agent" in operation.lower()
    ]
    return any(production_flags)


def validate_operation_safety(operation: str, context: dict) -> bool:
    """Validate if operation is safe"""
    blocked_ops = ["attach_tool", "detach_tool", "modify_agent", "create_agent", "delete_agent"]
    allowed_ops = ["search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"]
    
    if operation in blocked_ops:
        return False
    
    if operation in allowed_ops:
        return not has_production_indicators(operation, context)
    
    return False


def has_production_access() -> bool:
    """Check if production access is detected"""
    try:
        letta_api_url = os.getenv("LETTA_API_URL", "")
        return "prod" in letta_api_url.lower() or "production" in letta_api_url.lower()
    except Exception:
        return False


def are_critical_operations_blocked() -> bool:
    """Check if critical operations are properly blocked"""
    try:
        no_attach = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        read_only = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        return no_attach and read_only
    except Exception:
        return False


def get_block_reason(operation: str, context: dict) -> str:
    """Get reason why operation is blocked"""
    if operation in ["attach_tool", "detach_tool"]:
        return "Tool attachment/detachment operations are blocked for safety"
    elif operation in ["modify_agent", "create_agent", "delete_agent"]:
        return "Agent modification operations are blocked for safety"
    elif has_production_indicators(operation, context):
        return "Operation contains production indicators"
    else:
        return "Operation not in allowed list"


def suggest_safe_alternatives(operation: str) -> list:
    """Suggest safe alternatives for blocked operations"""
    alternatives = {
        "attach_tool": ["search for tools", "evaluate tool relevance", "create benchmarks"],
        "detach_tool": ["analyze tool usage", "create removal recommendations", "document changes"],
        "modify_agent": ["analyze agent configuration", "create modification proposals", "document requirements"],
        "create_agent": ["design agent specifications", "validate configurations", "create templates"]
    }
    return alternatives.get(operation, ["use read-only operations", "create analysis reports"])


def generate_safety_recommendations(isolation_checks: dict, safety_config: dict) -> list:
    """Generate safety recommendations based on current status"""
    recommendations = []
    
    if not isolation_checks.get("letta_api_isolated", False):
        recommendations.append("Enable Letta API isolation by setting LDTS_SAFETY_MODE=testing")
    
    if not isolation_checks.get("agent_modifications_blocked", False):
        recommendations.append("Block agent modifications with LDTS_NO_ATTACH_MODE=true")
    
    if not isolation_checks.get("tool_attachments_blocked", False):
        recommendations.append("Block tool operations with LDTS_READ_ONLY_MODE=true")
    
    if not isolation_checks.get("database_read_only", False):
        recommendations.append("Enable database read-only mode with LDTS_DATABASE_MODE=testing")
    
    if not safety_config.get("production_isolated", False):
        recommendations.append("Ensure production isolation with LDTS_PRODUCTION_ISOLATED=true")
    
    if safety_config.get("safety_mode") == "production":
        recommendations.append("Switch to testing mode with LDTS_SAFETY_MODE=testing")
    
    if not recommendations:
        recommendations.append("All safety measures are properly configured")
    
    return recommendations


def get_immediate_actions(emergency_status: dict) -> list:
    """Get immediate actions required for emergency conditions"""
    actions = []
    
    if emergency_status.get("emergency_lockdown", False):
        actions.append("EMERGENCY: System is in lockdown mode - review emergency procedures")
    
    if emergency_status.get("read_only_compromised", False):
        actions.append("CRITICAL: Enable read-only mode immediately with LDTS_READ_ONLY_MODE=true")
    
    if emergency_status.get("production_access_detected", False):
        actions.append("CRITICAL: Production access detected - isolate system immediately")
    
    if emergency_status.get("safety_mode_production", False):
        actions.append("HIGH: Switch to testing mode with LDTS_SAFETY_MODE=testing")
    
    if not emergency_status.get("critical_operations_blocked", False):
        actions.append("HIGH: Block critical operations with safety environment variables")
    
    return actions or ["Monitor system status and maintain current safety measures"]


# =============================================================================
# Safety Routes
# =============================================================================

@safety_bp.route('/api/v1/safety/status', methods=['GET'])
async def get_safety_status():
    """Get comprehensive safety system status."""
    try:
        # Check environment safety configurations
        safety_config = {
            "safety_mode": os.getenv("LDTS_SAFETY_MODE", "production"),
            "read_only_mode": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true",
            "no_attach_mode": os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true",
            "emergency_lockdown": os.getenv("LDTS_EMERGENCY_LOCKDOWN", "false").lower() == "true",
            "production_isolated": os.getenv("LDTS_PRODUCTION_ISOLATED", "true").lower() == "true"
        }
        
        # Perform isolation checks
        isolation_checks = {
            "letta_api_isolated": check_letta_api_isolation(),
            "agent_modifications_blocked": check_agent_modification_blocked(),
            "tool_attachments_blocked": check_tool_attachment_blocked(),
            "database_read_only": check_database_read_only(),
            "sandbox_mode_active": check_sandbox_mode()
        }
        
        # Calculate overall safety score
        passed_checks = sum(1 for passed in isolation_checks.values() if passed)
        total_checks = len(isolation_checks)
        safety_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Determine overall status
        if safety_score == 100:
            overall_status = "SECURE"
            status_message = "All safety checks passed"
        elif safety_score >= 80:
            overall_status = "WARNING" 
            status_message = "Some safety checks failed - review required"
        else:
            overall_status = "CRITICAL"
            status_message = "Multiple safety failures detected - immediate attention required"
        
        # Get blocked operations
        blocked_operations = [
            "attach_tool", "detach_tool", "modify_agent", "create_agent",
            "delete_agent", "update_memory", "production_write"
        ]
        
        allowed_operations = [
            "search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"
        ]
        
        return jsonify({
            "success": True,
            "data": {
                "overall_status": overall_status,
                "status_message": status_message,
                "safety_score": safety_score,
                "safety_config": safety_config,
                "isolation_checks": isolation_checks,
                "allowed_operations": allowed_operations,
                "blocked_operations": blocked_operations,
                "emergency_triggers": [
                    "read_only_mode_disabled",
                    "production_isolation_compromised",
                    "critical_safety_violation",
                    "multiple_safety_check_failures"
                ],
                "recommendations": generate_safety_recommendations(isolation_checks, safety_config),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting safety status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_status": {
                "overall_status": "ERROR",
                "status_message": "Safety status check failed - assuming unsafe conditions",
                "safety_score": 0
            }
        }), 500


@safety_bp.route('/api/v1/safety/validate-operation', methods=['POST'])
async def validate_operation():
    """Validate if a specific operation is safe to perform."""
    try:
        data = await request.get_json()
        
        if not data or 'operation' not in data:
            return jsonify({"success": False, "error": "Missing 'operation' in request"}), 400
        
        operation = data['operation']
        context = data.get('context', {})
        
        # Perform safety validation
        is_safe = validate_operation_safety(operation, context)
        
        # Get detailed validation result
        validation_result = {
            "operation": operation,
            "is_safe": is_safe,
            "validation_checks": {
                "operation_allowed": operation in [
                    "search", "rerank", "evaluate", "configure_test", "analytics", "benchmark"
                ],
                "operation_blocked": operation in [
                    "attach_tool", "detach_tool", "modify_agent", "create_agent",
                    "delete_agent", "update_memory", "production_write"
                ],
                "read_only_mode_active": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true",
                "safety_mode_testing": os.getenv("LDTS_SAFETY_MODE", "production") == "testing",
                "no_production_indicators": not has_production_indicators(operation, context)
            },
            "context": context,
            "timestamp": time.time()
        }
        
        if not is_safe:
            validation_result["block_reason"] = get_block_reason(operation, context)
            validation_result["alternatives"] = suggest_safe_alternatives(operation)
        
        return jsonify({"success": True, "data": validation_result})
        
    except Exception as e:
        logger.error(f"Error validating operation: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@safety_bp.route('/api/v1/safety/emergency-status', methods=['GET'])
async def get_emergency_status():
    """Get emergency safety status for critical monitoring."""
    try:
        # Quick emergency checks
        emergency_status = {
            "emergency_lockdown": os.getenv("LDTS_EMERGENCY_LOCKDOWN", "false").lower() == "true",
            "read_only_compromised": os.getenv("LDTS_READ_ONLY_MODE", "true").lower() != "true",
            "production_access_detected": has_production_access(),
            "safety_mode_production": os.getenv("LDTS_SAFETY_MODE", "production") == "production",
            "critical_operations_blocked": are_critical_operations_blocked()
        }
        
        # Count emergency conditions
        emergency_conditions = sum(1 for condition in emergency_status.values() if condition)
        
        if emergency_conditions == 0:
            alert_level = "NORMAL"
            message = "No emergency conditions detected"
        elif emergency_conditions == 1:
            alert_level = "CAUTION"
            message = "One emergency condition detected - monitoring required"
        else:
            alert_level = "EMERGENCY"
            message = f"{emergency_conditions} emergency conditions detected - immediate action required"
        
        return jsonify({
            "success": True,
            "data": {
                "alert_level": alert_level,
                "message": message,
                "emergency_conditions_count": emergency_conditions,
                "emergency_status": emergency_status,
                "immediate_actions_required": get_immediate_actions(emergency_status),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting emergency status: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "emergency_fallback": {
                "alert_level": "CRITICAL",
                "message": "Emergency status check failed - assume emergency conditions"
            }
        }), 500
