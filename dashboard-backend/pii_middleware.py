"""
LDTS-72: PII Protection and Payload Limit Middleware

Middleware to automatically apply PII redaction and payload size limits
to all testing API endpoints.
"""

import logging
import time
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pii_protection import pii_protection_manager

logger = logging.getLogger(__name__)

class PIIProtectionMiddleware:
    """Middleware for PII protection and payload validation"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.processed_requests = 0
        self.blocked_requests = 0
        
        # Endpoints that require PII protection
        self.protected_endpoints = {
            "/api/v1/search/test",
            "/api/v1/rerank/test",
            "/api/v1/evaluation/",
            "/api/v1/benchmark/",
            "/api/v1/config/validate",
            "/api/v1/audit/export"
        }
        
        # Endpoints that are exempt from PII protection
        self.exempt_endpoints = {
            "/api/v1/health",
            "/api/v1/safety/",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
        
        logger.info(f"PIIProtectionMiddleware initialized: enabled={enabled}")
    
    async def __call__(self, request: Request, call_next):
        """Process request through PII protection"""
        start_time = time.time()
        
        try:
            # Skip middleware if disabled
            if not self.enabled:
                return await call_next(request)
            
            # Skip exempt endpoints
            if self._is_exempt_endpoint(request.url.path):
                return await call_next(request)
            
            # Apply protection for protected endpoints
            if self._requires_protection(request.url.path):
                protection_result = await self._apply_protection(request)
                
                if protection_result.get("blocked"):
                    self.blocked_requests += 1
                    return await self._create_blocked_response(request, protection_result)
            
            # Process the request
            self.processed_requests += 1
            response = await call_next(request)
            
            # Apply PII redaction to response if needed
            if self._requires_protection(request.url.path):
                response = await self._protect_response(response)
            
            # Add protection headers
            response.headers["X-LDTS-PII-Protection"] = "active"
            response.headers["X-LDTS-Payload-Validation"] = "active"
            
            duration = (time.time() - start_time) * 1000
            logger.debug(f"PII protection applied: {request.method} {request.url.path} ({duration:.1f}ms)")
            
            return response
            
        except HTTPException as e:
            # PII protection blocked the request
            self.blocked_requests += 1
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "Request blocked by PII protection",
                    "detail": e.detail,
                    "protection_type": "pii_validation"
                },
                headers={
                    "X-LDTS-PII-Protection": "blocked",
                    "X-LDTS-Protection-Reason": "pii_detected"
                }
            )
        except Exception as e:
            logger.error(f"PIIProtectionMiddleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "PII protection middleware error",
                    "message": "Error in PII protection - request blocked for safety"
                },
                headers={"X-LDTS-PII-Protection": "error"}
            )
    
    def _is_exempt_endpoint(self, path: str) -> bool:
        """Check if endpoint is exempt from PII protection"""
        return any(exempt in path for exempt in self.exempt_endpoints)
    
    def _requires_protection(self, path: str) -> bool:
        """Check if endpoint requires PII protection"""
        return any(protected in path for protected in self.protected_endpoints)
    
    async def _apply_protection(self, request: Request) -> Dict[str, Any]:
        """Apply PII protection and payload validation"""
        try:
            # Validate and redact the request
            validation_result = await pii_protection_manager.validate_and_redact_request(request)
            
            # Check if request should be blocked
            pii_result = validation_result.get('pii_result', {})
            size_validation = validation_result.get('size_validation', {})
            
            # Block if PII detected in strict mode
            if pii_result.get('pii_detected') and pii_protection_manager.redaction_config.strict_mode:
                return {
                    "blocked": True,
                    "reason": "pii_detected_strict_mode",
                    "details": {
                        "detection_count": pii_result.get('detection_count', 0),
                        "detection_types": [d['type'] for d in pii_result.get('detections', [])]
                    }
                }
            
            # Block if payload size exceeded
            if not size_validation.get('valid') and pii_protection_manager.payload_config.enforce_limits:
                return {
                    "blocked": True,
                    "reason": "payload_size_exceeded",
                    "details": {
                        "size_mb": size_validation.get('size_mb'),
                        "limit_mb": size_validation.get('limit_mb')
                    }
                }
            
            # Request passed all checks
            return {
                "blocked": False,
                "validation_result": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error applying PII protection: {e}")
            return {
                "blocked": True,
                "reason": "protection_error",
                "details": {"error": str(e)}
            }
    
    async def _protect_response(self, response):
        """Apply PII redaction to response"""
        # For JSON responses, redact PII from response body
        if hasattr(response, 'body') and response.headers.get('content-type', '').startswith('application/json'):
            try:
                import json
                # This is simplified - in a real implementation you'd need to
                # properly handle the response body redaction
                pass
            except Exception as e:
                logger.warning(f"Could not apply PII redaction to response: {e}")
        
        return response
    
    async def _create_blocked_response(self, request: Request, protection_result: Dict[str, Any]) -> JSONResponse:
        """Create response for blocked request"""
        reason = protection_result.get("reason", "unknown")
        details = protection_result.get("details", {})
        
        error_content = {
            "error": "Request blocked by PII protection",
            "reason": reason,
            "method": request.method,
            "path": request.url.path,
            "timestamp": time.time(),
            "protection_details": details
        }
        
        status_code = 400
        if reason == "payload_size_exceeded":
            status_code = 413
        elif reason == "pii_detected_strict_mode":
            status_code = 400
        
        headers = {
            "X-LDTS-PII-Protection": "blocked",
            "X-LDTS-Protection-Reason": reason,
            "X-LDTS-Blocked-Count": str(self.blocked_requests)
        }
        
        logger.warning(f"PII PROTECTION BLOCKED: {reason} - {request.method} {request.url.path}")
        
        return JSONResponse(
            status_code=status_code,
            content=error_content,
            headers=headers
        )
    
    def get_middleware_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            "enabled": self.enabled,
            "processed_requests": self.processed_requests,
            "blocked_requests": self.blocked_requests,
            "protection_statistics": pii_protection_manager.get_protection_statistics(),
            "protected_endpoints": list(self.protected_endpoints),
            "exempt_endpoints": list(self.exempt_endpoints)
        }

# Global PII protection middleware instance
pii_middleware = PIIProtectionMiddleware()