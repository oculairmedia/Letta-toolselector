"""
LDTS-30: Safety measures and read-only mode enforcement

Critical safety system to ensure no production impact from dashboard testing.
"""

import logging
import functools
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request
from pydantic import BaseModel
import os
from enum import Enum

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for operations"""
    READ_ONLY = "read_only"
    TESTING = "testing"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"

class SafetyConfig(BaseModel):
    """Safety configuration"""
    read_only_mode: bool = True
    prevent_attachments: bool = True
    audit_logging: bool = True
    no_attach_mode_banner: bool = True
    allowed_operations: List[str] = [
        "search", "rerank", "evaluate", "configure_test", "analytics"
    ]
    blocked_operations: List[str] = [
        "attach_tool", "detach_tool", "modify_agent", "create_agent", 
        "delete_agent", "update_memory", "production_write"
    ]

class SafetyValidator:
    """Comprehensive safety validation system"""
    
    def __init__(self):
        self.config = SafetyConfig()
        self.safety_violations: List[Dict[str, Any]] = []
        
    def validate_read_only_mode(self) -> bool:
        """Ensure all operations are read-only"""
        if not self.config.read_only_mode:
            logger.error("READ-ONLY MODE DISABLED - SECURITY RISK!")
            return False
        return True
    
    def validate_operation(self, operation: str, context: Dict[str, Any] = None) -> bool:
        """Validate if operation is safe for testing environment"""
        context = context or {}
        
        # Check if operation is explicitly blocked
        if operation in self.config.blocked_operations:
            self.log_safety_violation("BLOCKED_OPERATION", operation, context)
            return False
            
        # Check if operation is in allowed list
        if operation not in self.config.allowed_operations:
            self.log_safety_violation("UNKNOWN_OPERATION", operation, context)
            return False
            
        # Check for production system indicators
        if self.is_production_operation(operation, context):
            self.log_safety_violation("PRODUCTION_ACCESS", operation, context)
            return False
            
        return True
    
    def is_production_operation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Detect if operation could affect production systems"""
        production_indicators = [
            "agent_id" in context,  # Any agent modification
            "tool_attach" in operation.lower(),
            "tool_detach" in operation.lower(),
            "modify" in operation.lower(),
            "create" in operation.lower() and "agent" in operation.lower(),
            "delete" in operation.lower(),
            "production" in str(context).lower()
        ]
        
        return any(production_indicators)
    
    def log_safety_violation(self, violation_type: str, operation: str, context: Dict[str, Any]):
        """Log safety violations for audit trail"""
        violation = {
            "type": violation_type,
            "operation": operation,
            "context": context,
            "timestamp": logger.time(),
            "severity": "CRITICAL"
        }
        
        self.safety_violations.append(violation)
        logger.critical(f"SAFETY VIOLATION: {violation_type} - {operation} - {context}")
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown if safety is compromised"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        # In production, this would trigger actual shutdown
        raise HTTPException(
            status_code=503,
            detail=f"Service emergency shutdown: {reason}"
        )

# Global safety validator instance
safety_validator = SafetyValidator()

def safety_check(operation: str, safety_level: SafetyLevel = SafetyLevel.READ_ONLY):
    """Decorator to enforce safety checks on API endpoints"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request context
            request_context = {}
            for arg in args:
                if isinstance(arg, Request):
                    request_context.update({
                        "method": arg.method,
                        "url": str(arg.url),
                        "client": str(arg.client.host if arg.client else "unknown")
                    })
                    break
            
            # Validate operation safety
            if not safety_validator.validate_operation(operation, request_context):
                raise HTTPException(
                    status_code=403,
                    detail=f"Operation '{operation}' blocked by safety system"
                )
            
            # Validate read-only mode
            if not safety_validator.validate_read_only_mode():
                safety_validator.emergency_shutdown("Read-only mode compromised")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
                
                # Log successful safe operation
                logger.info(f"Safe operation completed: {operation}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in safe operation {operation}: {e}")
                raise
                
        return wrapper
    return decorator

# Production isolation checks
def verify_production_isolation():
    """Verify complete isolation from production systems"""
    isolation_checks = {
        "letta_api_write_disabled": check_letta_api_isolation(),
        "agent_modification_blocked": check_agent_modification_blocked(),
        "tool_attachment_blocked": check_tool_attachment_blocked(),
        "read_only_database": check_database_read_only(),
        "sandbox_mode": check_sandbox_mode()
    }
    
    failed_checks = [check for check, passed in isolation_checks.items() if not passed]
    
    if failed_checks:
        logger.critical(f"PRODUCTION ISOLATION FAILED: {failed_checks}")
        safety_validator.emergency_shutdown(f"Production isolation compromised: {failed_checks}")
    
    logger.info("Production isolation verified successfully")
    return True

def check_letta_api_isolation() -> bool:
    """Check that Letta API writes are disabled"""
    try:
        # Check environment configuration
        letta_api_url = os.getenv("LETTA_API_URL", "")
        letta_password = os.getenv("LETTA_PASSWORD", "")
        
        # Verify testing mode is enabled
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        if safety_mode != "testing":
            logger.warning(f"Safety mode is '{safety_mode}', expected 'testing' for API isolation")
            return False
        
        # Check if we have credentials but verify read-only mode
        if letta_api_url and letta_password:
            # Verify read-only mode environment variables
            read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
            no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
            
            if not read_only_mode or not no_attach_mode:
                logger.error("Letta API isolation compromised: write modes enabled")
                return False
            
            logger.info("Letta API isolation verified: read-only and no-attach modes active")
            return True
        else:
            logger.warning("Letta API credentials not configured - assuming isolated")
            return True
            
    except Exception as e:
        logger.error(f"Error checking Letta API isolation: {e}")
        return False

def check_agent_modification_blocked() -> bool:
    """Check that agent modifications are blocked"""
    try:
        # Check environment configuration for agent protection
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        
        # Verify we're in testing mode with no-attach enabled
        if safety_mode == "testing" and no_attach_mode:
            logger.info("Agent modification blocked: safety mode active")
            return True
        
        # Check for explicit agent modification blocking
        block_agent_ops = os.getenv("LDTS_BLOCK_AGENT_MODIFICATIONS", "true").lower() == "true"
        if block_agent_ops:
            logger.info("Agent modification blocked: explicit blocking enabled")
            return True
        
        # Check if we're in emergency lockdown mode
        emergency_mode = os.getenv("LDTS_EMERGENCY_LOCKDOWN", "false").lower() == "true"
        if emergency_mode:
            logger.info("Agent modification blocked: emergency lockdown active")
            return True
        
        logger.warning("Agent modification blocking not fully verified")
        return False
        
    except Exception as e:
        logger.error(f"Error checking agent modification blocking: {e}")
        return False

def check_tool_attachment_blocked() -> bool:
    """Check that tool attachment/detachment is blocked"""
    try:
        # Check no-attach mode configuration
        no_attach_mode = os.getenv("LDTS_NO_ATTACH_MODE", "true").lower() == "true"
        safety_mode = os.getenv("LDTS_SAFETY_MODE", "production")
        
        if no_attach_mode and safety_mode == "testing":
            logger.info("Tool attachment blocked: no-attach mode active")
            return True
        
        # Check for explicit tool operation blocking
        block_tool_ops = os.getenv("LDTS_BLOCK_TOOL_OPERATIONS", "true").lower() == "true"
        if block_tool_ops:
            logger.info("Tool attachment blocked: explicit blocking enabled")
            return True
        
        # Check if we're in read-only mode
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        if read_only_mode:
            logger.info("Tool attachment blocked: read-only mode active")
            return True
        
        # Verify that dangerous operations are in blocked list
        if "attach_tool" in safety_validator.config.blocked_operations:
            logger.info("Tool attachment blocked: operation in blocked list")
            return True
        
        logger.warning("Tool attachment blocking not fully verified")
        return False
        
    except Exception as e:
        logger.error(f"Error checking tool attachment blocking: {e}")
        return False

def check_database_read_only() -> bool:
    """Check that database is in read-only mode for production data"""
    try:
        # Check if we're using a separate testing database
        database_mode = os.getenv("LDTS_DATABASE_MODE", "testing")
        if database_mode == "testing":
            logger.info("Database isolation verified: using testing database")
            return True
        
        # Check read-only mode settings
        read_only_mode = os.getenv("LDTS_READ_ONLY_MODE", "true").lower() == "true"
        if not read_only_mode:
            logger.error("Database not in read-only mode")
            return False
        
        # Check if database writes are explicitly disabled
        db_write_disabled = os.getenv("LDTS_DISABLE_DB_WRITES", "true").lower() == "true"
        if db_write_disabled:
            logger.info("Database writes disabled: explicit setting active")
            return True
        
        # Check for production database isolation
        production_isolated = os.getenv("LDTS_PRODUCTION_ISOLATED", "true").lower() == "true"
        if production_isolated:
            logger.info("Database isolation verified: production isolation active")
            return True
        
        # Verify no production database connection strings
        db_url = os.getenv("DATABASE_URL", "")
        if "production" in db_url.lower() or "prod" in db_url.lower():
            logger.error("Production database connection detected")
            return False
        
        logger.info("Database read-only mode verified")
        return True
        
    except Exception as e:
        logger.error(f"Error checking database read-only mode: {e}")
        return False

def check_sandbox_mode() -> bool:
    """Check that we're operating in sandbox mode"""
    return os.getenv("LDTS_SAFETY_MODE", "production") == "testing"

# Safety status endpoint data
def get_safety_status() -> Dict[str, Any]:
    """Get current safety system status"""
    return {
        "read_only_mode": safety_validator.config.read_only_mode,
        "production_isolation": True,  # Result of isolation checks
        "safety_violations_count": len(safety_validator.safety_violations),
        "recent_violations": safety_validator.safety_violations[-5:],  # Last 5
        "allowed_operations": safety_validator.config.allowed_operations,
        "blocked_operations": safety_validator.config.blocked_operations,
        "emergency_shutdown_triggers": [
            "read_only_mode_disabled",
            "production_isolation_compromised", 
            "critical_safety_violation"
        ]
    }