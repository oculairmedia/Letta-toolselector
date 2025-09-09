"""
LDTS-70: Read-only guard middleware for testing endpoints

Middleware to enforce read-only mode for all testing endpoints with comprehensive
request filtering and safety validation.
"""

import logging
import time
from typing import Dict, Any, Set
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)

class ReadOnlyGuardMiddleware:
    """Middleware to enforce read-only mode for testing endpoints"""
    
    # HTTP methods that are always allowed (read operations)
    SAFE_METHODS: Set[str] = {"GET", "HEAD", "OPTIONS"}
    
    # HTTP methods that require validation (potential write operations)
    DANGEROUS_METHODS: Set[str] = {"POST", "PUT", "PATCH", "DELETE"}
    
    # Endpoints that are completely blocked in read-only mode
    BLOCKED_ENDPOINTS: Set[str] = {
        "/api/v1/agents/modify",
        "/api/v1/agents/create", 
        "/api/v1/agents/delete",
        "/api/v1/tools/attach",
        "/api/v1/tools/detach",
        "/api/v1/memory/modify",
        "/api/v1/production/write"
    }
    
    # Read-only testing endpoints that are allowed with POST/PUT
    ALLOWED_TESTING_ENDPOINTS: Set[str] = {
        "/api/v1/search/test",
        "/api/v1/rerank/test", 
        "/api/v1/config/validate",
        "/api/v1/config/validate-yaml",
        "/api/v1/config/estimate-cost",
        "/api/v1/evaluation/test",
        "/api/v1/benchmark/test",
        "/api/v1/safety/validate-operation",
        "/api/v1/rate-limit/test",
        "/api/v1/audit/export"
    }
    
    def __init__(self, enforce_read_only: bool = True, log_violations: bool = True):
        """
        Initialize read-only guard middleware
        
        Args:
            enforce_read_only: Whether to enforce read-only mode (default: True)
            log_violations: Whether to log blocked requests (default: True)
        """
        self.enforce_read_only = enforce_read_only
        self.log_violations = log_violations
        self.violation_count = 0
        self.blocked_requests: list = []
        
        logger.info(f"ReadOnlyGuardMiddleware initialized: enforce={enforce_read_only}")
    
    async def __call__(self, request: Request, call_next):
        """Process request through read-only guard"""
        start_time = time.time()
        
        try:
            # Skip middleware if read-only mode is disabled
            if not self.enforce_read_only:
                return await call_next(request)
            
            # Validate request against read-only rules
            violation_result = await self._validate_request(request)
            
            if violation_result["blocked"]:
                return await self._create_blocked_response(request, violation_result)
            
            # Request is safe - proceed
            response = await call_next(request)
            
            # Add read-only mode headers
            response.headers["X-LDTS-Read-Only-Mode"] = "enforced"
            response.headers["X-LDTS-Safety-Level"] = "testing"
            
            # Log successful safe request
            duration = (time.time() - start_time) * 1000
            logger.debug(f"Safe request processed: {request.method} {request.url.path} ({duration:.1f}ms)")
            
            return response
            
        except Exception as e:
            logger.error(f"ReadOnlyGuardMiddleware error: {e}")
            # In case of middleware error, default to blocking for safety
            return await self._create_error_response(request, str(e))
    
    async def _validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate request against read-only rules"""
        method = request.method.upper()
        path = request.url.path
        
        # Always allow safe HTTP methods
        if method in self.SAFE_METHODS:
            return {"blocked": False, "reason": "safe_method", "method": method}
        
        # Block explicitly dangerous endpoints
        if path in self.BLOCKED_ENDPOINTS:
            return {
                "blocked": True, 
                "reason": "blocked_endpoint", 
                "method": method, 
                "path": path
            }
        
        # Allow specific testing endpoints even with POST/PUT
        if path in self.ALLOWED_TESTING_ENDPOINTS:
            return {
                "blocked": False, 
                "reason": "allowed_testing_endpoint", 
                "method": method,
                "path": path
            }
        
        # Check request body for dangerous operations
        if method in self.DANGEROUS_METHODS:
            body_check = await self._validate_request_body(request)
            if body_check["dangerous"]:
                return {
                    "blocked": True,
                    "reason": "dangerous_request_body",
                    "method": method,
                    "path": path,
                    "details": body_check["details"]
                }
        
        # Default: allow requests that pass all checks
        return {"blocked": False, "reason": "passed_validation", "method": method}
    
    async def _validate_request_body(self, request: Request) -> Dict[str, Any]:
        """Validate request body for dangerous operations"""
        try:
            # Clone request to avoid consuming body
            body = await request.body()
            
            if not body:
                return {"dangerous": False, "details": "empty_body"}
            
            # Try to parse JSON body
            try:
                body_json = json.loads(body.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Non-JSON body - apply basic string checks
                body_str = body.decode('utf-8', errors='ignore').lower()
                return self._check_dangerous_strings(body_str)
            
            # Check JSON body for dangerous operations
            return self._check_dangerous_json(body_json)
            
        except Exception as e:
            logger.warning(f"Could not validate request body: {e}")
            # If we can't validate, assume it's safe (testing endpoint)
            return {"dangerous": False, "details": f"validation_error: {str(e)}"}
    
    def _check_dangerous_strings(self, body_str: str) -> Dict[str, Any]:
        """Check string body for dangerous keywords"""
        dangerous_keywords = [
            "attach_tool", "detach_tool", "modify_agent", "create_agent", 
            "delete_agent", "update_memory", "production_write", "agent_id",
            "tool_attachment", "agent_modification"
        ]
        
        found_keywords = [kw for kw in dangerous_keywords if kw in body_str]
        
        if found_keywords:
            return {
                "dangerous": True,
                "details": f"dangerous_keywords: {found_keywords}"
            }
        
        return {"dangerous": False, "details": "string_body_safe"}
    
    def _check_dangerous_json(self, body_json: Any) -> Dict[str, Any]:
        """Check JSON body for dangerous operations"""
        if not isinstance(body_json, dict):
            return {"dangerous": False, "details": "non_dict_json"}
        
        # Check for agent-related operations
        if "agent_id" in body_json:
            return {
                "dangerous": True,
                "details": "contains_agent_id"
            }
        
        # Check for tool operations
        tool_operations = ["attach_tool", "detach_tool", "tool_ids", "tool_attachment"]
        found_ops = [op for op in tool_operations if op in body_json]
        if found_ops:
            return {
                "dangerous": True,
                "details": f"tool_operations: {found_ops}"
            }
        
        # Check for production indicators
        production_indicators = ["production", "modify", "create", "delete", "update"]
        body_str = json.dumps(body_json).lower()
        found_indicators = [ind for ind in production_indicators if ind in body_str]
        
        if found_indicators:
            return {
                "dangerous": True,
                "details": f"production_indicators: {found_indicators}"
            }
        
        return {"dangerous": False, "details": "json_body_safe"}
    
    async def _create_blocked_response(self, request: Request, violation_result: Dict[str, Any]) -> JSONResponse:
        """Create response for blocked request"""
        self.violation_count += 1
        
        blocked_info = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "client": str(request.client.host if request.client else "unknown"),
            "reason": violation_result["reason"],
            "details": violation_result.get("details", "")
        }
        
        self.blocked_requests.append(blocked_info)
        
        if self.log_violations:
            logger.warning(f"BLOCKED REQUEST: {blocked_info}")
        
        error_response = {
            "error": "Request blocked by read-only guard",
            "reason": violation_result["reason"],
            "method": request.method,
            "path": request.url.path,
            "message": "This endpoint is blocked in read-only testing mode",
            "safety_mode": "read_only_testing",
            "violation_id": self.violation_count,
            "timestamp": blocked_info["timestamp"]
        }
        
        if "details" in violation_result:
            error_response["details"] = violation_result["details"]
        
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=error_response,
            headers={
                "X-LDTS-Read-Only-Mode": "enforced",
                "X-LDTS-Safety-Level": "read_only_violation",
                "X-LDTS-Violation-Count": str(self.violation_count)
            }
        )
    
    async def _create_error_response(self, request: Request, error_msg: str) -> JSONResponse:
        """Create response for middleware errors"""
        logger.error(f"ReadOnlyGuardMiddleware internal error: {error_msg}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Read-only guard middleware error",
                "message": "Safety middleware encountered an error - blocking request for safety",
                "details": error_msg,
                "safety_mode": "error_fallback"
            },
            headers={
                "X-LDTS-Read-Only-Mode": "error",
                "X-LDTS-Safety-Level": "error_blocking"
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            "enforce_read_only": self.enforce_read_only,
            "violation_count": self.violation_count,
            "blocked_requests": len(self.blocked_requests),
            "recent_violations": self.blocked_requests[-10:],  # Last 10
            "safe_methods": list(self.SAFE_METHODS),
            "dangerous_methods": list(self.DANGEROUS_METHODS),
            "blocked_endpoints": list(self.BLOCKED_ENDPOINTS),
            "allowed_testing_endpoints": list(self.ALLOWED_TESTING_ENDPOINTS)
        }

# Global middleware instance
readonly_guard = ReadOnlyGuardMiddleware()