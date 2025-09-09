"""
LDTS-71: Enforce no-attach/no-write mode via config flags and server banner

Configuration flags and visual indicators to enforce and display no-attach/no-write mode
for testing safety.
"""

import logging
import os
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

class SafetyModeConfig(Enum):
    """Safety mode configuration options"""
    FULL_PRODUCTION = "full_production"  # All operations allowed (dangerous)
    READ_WRITE_TESTING = "read_write_testing"  # Read/write but no agent attach
    READ_ONLY_TESTING = "read_only_testing"  # Only read operations
    NO_ATTACH_MODE = "no_attach_mode"  # No tool attachment/detachment
    EMERGENCY_LOCKDOWN = "emergency_lockdown"  # All operations blocked

class NoAttachModeConfig(BaseModel):
    """Configuration for no-attach/no-write mode"""
    mode: SafetyModeConfig = SafetyModeConfig.NO_ATTACH_MODE
    show_banner: bool = True
    banner_message: str = "🔒 NO-ATTACH MODE: Tool attachment/detachment disabled for testing safety"
    banner_color: str = "#ff6b35"  # Orange warning color
    enforce_no_attach: bool = True
    enforce_no_write: bool = True
    allow_read_operations: bool = True
    allow_testing_endpoints: bool = True
    emergency_contact: str = "system-admin"
    
class NoAttachModeManager:
    """Manager for no-attach/no-write mode enforcement"""
    
    def __init__(self):
        self.config = self._load_config()
        self.mode_violations = []
        self.banner_shown_count = 0
        
        logger.info(f"NoAttachModeManager initialized: mode={self.config.mode.value}")
    
    def _load_config(self) -> NoAttachModeConfig:
        """Load configuration from environment variables"""
        mode_str = os.getenv("LDTS_SAFETY_MODE", "no_attach_mode")
        try:
            mode = SafetyModeConfig(mode_str)
        except ValueError:
            logger.warning(f"Invalid safety mode '{mode_str}', defaulting to no_attach_mode")
            mode = SafetyModeConfig.NO_ATTACH_MODE
        
        return NoAttachModeConfig(
            mode=mode,
            show_banner=os.getenv("LDTS_SHOW_BANNER", "true").lower() == "true",
            banner_message=os.getenv("LDTS_BANNER_MESSAGE", 
                "🔒 NO-ATTACH MODE: Tool attachment/detachment disabled for testing safety"),
            enforce_no_attach=os.getenv("LDTS_ENFORCE_NO_ATTACH", "true").lower() == "true",
            enforce_no_write=os.getenv("LDTS_ENFORCE_NO_WRITE", "true").lower() == "true",
            allow_testing_endpoints=os.getenv("LDTS_ALLOW_TESTING", "true").lower() == "true",
            emergency_contact=os.getenv("LDTS_EMERGENCY_CONTACT", "system-admin")
        )
    
    def is_operation_allowed(self, operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if operation is allowed under current safety mode"""
        context = context or {}
        
        # Emergency lockdown blocks everything
        if self.config.mode == SafetyModeConfig.EMERGENCY_LOCKDOWN:
            return {
                "allowed": False,
                "reason": "emergency_lockdown",
                "message": "All operations blocked due to emergency lockdown"
            }
        
        # Check attach/detach operations
        if self._is_attach_operation(operation, context):
            if not self._is_attach_allowed():
                return {
                    "allowed": False,
                    "reason": "no_attach_mode",
                    "message": "Tool attachment/detachment blocked by no-attach mode"
                }
        
        # Check write operations  
        if self._is_write_operation(operation, context):
            if not self._is_write_allowed():
                return {
                    "allowed": False,
                    "reason": "no_write_mode", 
                    "message": "Write operations blocked by safety mode"
                }
        
        # Check testing operations
        if self._is_testing_operation(operation, context):
            if not self.config.allow_testing_endpoints:
                return {
                    "allowed": False,
                    "reason": "testing_disabled",
                    "message": "Testing endpoints disabled"
                }
        
        return {
            "allowed": True,
            "reason": "operation_permitted",
            "mode": self.config.mode.value
        }
    
    def _is_attach_operation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Check if operation involves tool attachment/detachment"""
        attach_indicators = [
            "attach" in operation.lower(),
            "detach" in operation.lower(), 
            "tool_attachment" in operation.lower(),
            "modify_agent_tools" in operation.lower(),
            "agent_id" in context and any(key in context for key in ["tool_ids", "attach_tools", "detach_tools"])
        ]
        return any(attach_indicators)
    
    def _is_write_operation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Check if operation involves writing/modification"""
        write_indicators = [
            "create" in operation.lower(),
            "update" in operation.lower(),
            "modify" in operation.lower(),
            "delete" in operation.lower(),
            "write" in operation.lower(),
            "save" in operation.lower(),
            "production" in str(context).lower()
        ]
        return any(write_indicators)
    
    def _is_testing_operation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Check if operation is a testing endpoint"""
        testing_indicators = [
            "test" in operation.lower(),
            "evaluate" in operation.lower(),
            "benchmark" in operation.lower(),
            "validate" in operation.lower(),
            "/test" in context.get("path", ""),
            "/evaluate" in context.get("path", "")
        ]
        return any(testing_indicators)
    
    def _is_attach_allowed(self) -> bool:
        """Check if attach operations are allowed"""
        if self.config.mode == SafetyModeConfig.FULL_PRODUCTION:
            return True  # Dangerous but allowed in production mode
        return not self.config.enforce_no_attach
    
    def _is_write_allowed(self) -> bool:
        """Check if write operations are allowed"""
        allowed_modes = [
            SafetyModeConfig.FULL_PRODUCTION,
            SafetyModeConfig.READ_WRITE_TESTING
        ]
        if self.config.mode in allowed_modes:
            return True
        return not self.config.enforce_no_write
    
    def get_banner_html(self) -> str:
        """Generate HTML banner for no-attach mode"""
        if not self.config.show_banner:
            return ""
        
        self.banner_shown_count += 1
        
        mode_icons = {
            SafetyModeConfig.NO_ATTACH_MODE: "🔒",
            SafetyModeConfig.READ_ONLY_TESTING: "👀", 
            SafetyModeConfig.READ_WRITE_TESTING: "⚠️",
            SafetyModeConfig.EMERGENCY_LOCKDOWN: "🚨",
            SafetyModeConfig.FULL_PRODUCTION: "⚡"
        }
        
        mode_colors = {
            SafetyModeConfig.NO_ATTACH_MODE: "#ff6b35",  # Orange
            SafetyModeConfig.READ_ONLY_TESTING: "#2196F3",  # Blue
            SafetyModeConfig.READ_WRITE_TESTING: "#FF9800",  # Amber  
            SafetyModeConfig.EMERGENCY_LOCKDOWN: "#f44336",  # Red
            SafetyModeConfig.FULL_PRODUCTION: "#4CAF50"  # Green (dangerous)
        }
        
        icon = mode_icons.get(self.config.mode, "🔒")
        color = mode_colors.get(self.config.mode, self.config.banner_color)
        
        banner_html = f"""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: {color};
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: bold;
            font-size: 16px;
            z-index: 9999;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        ">
            {icon} {self.config.banner_message}
            <span style="margin-left: 20px; font-size: 14px;">
                Mode: {self.config.mode.value.upper()}
            </span>
            <span style="margin-left: 20px; font-size: 12px;">
                Emergency Contact: {self.config.emergency_contact}
            </span>
        </div>
        <div style="height: 60px;"></div> <!-- Spacer for fixed banner -->
        """
        
        return banner_html
    
    def get_banner_data(self) -> Dict[str, Any]:
        """Get banner data for API responses"""
        return {
            "show_banner": self.config.show_banner,
            "banner_message": self.config.banner_message,
            "banner_color": self.config.banner_color,
            "mode": self.config.mode.value,
            "safety_indicators": {
                "no_attach_enforced": self.config.enforce_no_attach,
                "no_write_enforced": self.config.enforce_no_write,
                "testing_allowed": self.config.allow_testing_endpoints
            },
            "emergency_contact": self.config.emergency_contact,
            "banner_shown_count": self.banner_shown_count
        }
    
    def set_emergency_lockdown(self, reason: str, triggered_by: str = "system"):
        """Trigger emergency lockdown mode"""
        self.config.mode = SafetyModeConfig.EMERGENCY_LOCKDOWN
        self.config.banner_message = f"🚨 EMERGENCY LOCKDOWN: {reason}"
        self.config.banner_color = "#f44336"  # Red
        
        violation = {
            "type": "EMERGENCY_LOCKDOWN",
            "reason": reason,
            "triggered_by": triggered_by,
            "timestamp": logger.time(),
            "previous_mode": self.config.mode.value
        }
        
        self.mode_violations.append(violation)
        logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED: {reason} (triggered by: {triggered_by})")
    
    def get_mode_status(self) -> Dict[str, Any]:
        """Get current mode status and statistics"""
        return {
            "current_mode": self.config.mode.value,
            "configuration": {
                "enforce_no_attach": self.config.enforce_no_attach,
                "enforce_no_write": self.config.enforce_no_write,
                "allow_read_operations": self.config.allow_read_operations,
                "allow_testing_endpoints": self.config.allow_testing_endpoints,
                "show_banner": self.config.show_banner
            },
            "statistics": {
                "banner_shown_count": self.banner_shown_count,
                "mode_violations": len(self.mode_violations),
                "recent_violations": self.mode_violations[-5:]  # Last 5
            },
            "safety_summary": {
                "attach_operations_allowed": self._is_attach_allowed(),
                "write_operations_allowed": self._is_write_allowed(),
                "testing_operations_allowed": self.config.allow_testing_endpoints,
                "emergency_lockdown_active": self.config.mode == SafetyModeConfig.EMERGENCY_LOCKDOWN
            },
            "banner_data": self.get_banner_data()
        }

# Global no-attach mode manager
no_attach_manager = NoAttachModeManager()