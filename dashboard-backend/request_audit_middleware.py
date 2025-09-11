"""
LDTS-74: Add per-request audit logs for all testing endpoints

Per-request audit logging middleware that captures comprehensive details
for all testing API endpoints with structured logging and correlation tracking.
"""

import logging
import time
import uuid
import json
from typing import Dict, Any, Optional, Set, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import aiofiles
from centralized_audit_logger import centralized_audit_logger
from audit_logging import audit_logger, AuditEventType, AuditLevel

logger = logging.getLogger(__name__)

class RequestAuditMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive per-request audit logging"""
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.request_log_file = Path("audit_logs/request_audit.log")
        self.request_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Endpoints that require detailed audit logging
        self.audited_endpoints = {
            "/api/v1/search/test",
            "/api/v1/rerank/test", 
            "/api/v1/config/validate",
            "/api/v1/config/validate-yaml",
            "/api/v1/config/estimate-cost",
            "/api/v1/evaluation/",
            "/api/v1/benchmark/",
            "/api/v1/safety/",
            "/api/v1/audit/"
        }
        
        # Sensitive parameters to redact in logs
        self.sensitive_params = {
            "password", "token", "secret", "key", "auth", "credential",
            "api_key", "private", "confidential"
        }
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info(f"RequestAuditMiddleware initialized: enabled={enabled}")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive audit logging"""
        
        if not self.enabled:
            return await call_next(request)
        
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        start_time = time.time()
        request_data = await self._extract_request_data(request, correlation_id)
        
        # Check if this endpoint requires audit logging
        requires_audit = self._requires_detailed_audit(request.url.path)
        
        if requires_audit:
            # Log request start
            await self._log_request_start(request_data)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract response data
            response_data = await self._extract_response_data(response, processing_time)
            
            # Create comprehensive audit log entry
            if requires_audit:
                audit_entry = await self._create_audit_entry(
                    request_data, response_data, correlation_id, processing_time
                )
                
                # Write to multiple audit systems
                await self._write_audit_logs(audit_entry)
                
                # Add to centralized audit logger
                await self._log_to_centralized_audit(audit_entry)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, response.status_code)
            
            # Add audit headers to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Audit-Logged"] = "true" if requires_audit else "false"
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            
            return response
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Log error
            error_data = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time": processing_time
            }
            
            if requires_audit:
                await self._log_request_error(request_data, error_data, correlation_id)
            
            self.error_count += 1
            
            # Re-raise the exception
            raise
    
    async def _extract_request_data(self, request: Request, correlation_id: str) -> Dict[str, Any]:
        """Extract comprehensive request data for audit logging"""
        
        # Basic request info
        request_data = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": {
                "host": request.client.host if request.client else None,
                "port": request.client.port if request.client else None
            },
            "cookies": dict(request.cookies)
        }
        
        # Extract request body if present
        try:
            body = await request.body()
            if body:
                try:
                    # Try to parse as JSON
                    request_data["body"] = json.loads(body.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Store as text if not JSON
                    request_data["body_text"] = body.decode('utf-8', errors='ignore')[:1000]
                
                request_data["body_size"] = len(body)
        except Exception as e:
            request_data["body_error"] = str(e)
        
        # Redact sensitive information
        request_data = self._redact_sensitive_data(request_data)
        
        return request_data
    
    async def _extract_response_data(self, response: Response, processing_time: float) -> Dict[str, Any]:
        """Extract response data for audit logging"""
        
        response_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "processing_time": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Try to extract response body size
        if hasattr(response, 'body') and response.body:
            response_data["body_size"] = len(response.body)
        
        return response_data
    
    def _requires_detailed_audit(self, path: str) -> bool:
        """Check if endpoint requires detailed audit logging"""
        return any(endpoint in path for endpoint in self.audited_endpoints)
    
    def _redact_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from request data"""
        
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_params):
                    redacted[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    redacted[key] = self._redact_sensitive_data(value)
                elif isinstance(value, list):
                    redacted[key] = [self._redact_sensitive_data(item) if isinstance(item, dict) else item for item in value]
                else:
                    redacted[key] = value
            return redacted
        else:
            return data
    
    async def _create_audit_entry(self, 
                                request_data: Dict[str, Any], 
                                response_data: Dict[str, Any],
                                correlation_id: str,
                                processing_time: float) -> Dict[str, Any]:
        """Create comprehensive audit log entry"""
        
        return {
            "audit_type": "per_request_audit",
            "correlation_id": correlation_id,
            "request": request_data,
            "response": response_data,
            "performance": {
                "processing_time": processing_time,
                "timestamp_start": request_data["timestamp"],
                "timestamp_end": response_data["timestamp"]
            },
            "classification": {
                "endpoint_type": self._classify_endpoint(request_data["path"]),
                "risk_level": self._assess_request_risk(request_data, response_data),
                "requires_review": self._requires_manual_review(request_data, response_data)
            },
            "compliance": {
                "pii_detected": await self._check_pii_presence(request_data),
                "security_headers_present": self._check_security_headers(response_data),
                "rate_limit_applied": "X-RateLimit-Remaining" in response_data.get("headers", {})
            }
        }
    
    def _classify_endpoint(self, path: str) -> str:
        """Classify endpoint type for audit purposes"""
        if "/search/" in path:
            return "search_operation"
        elif "/rerank/" in path:
            return "rerank_operation"
        elif "/config/" in path:
            return "configuration_operation"
        elif "/evaluation/" in path:
            return "evaluation_operation"
        elif "/safety/" in path:
            return "safety_operation"
        elif "/audit/" in path:
            return "audit_operation"
        else:
            return "other_operation"
    
    def _assess_request_risk(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> str:
        """Assess risk level of the request"""
        
        # High risk indicators
        if response_data["status_code"] >= 500:
            return "HIGH"  # Server errors
        elif response_data["status_code"] >= 400:
            return "MEDIUM"  # Client errors
        elif "/safety/" in request_data["path"]:
            return "MEDIUM"  # Safety operations
        elif request_data.get("body_size", 0) > 1000000:  # > 1MB
            return "MEDIUM"  # Large payloads
        else:
            return "LOW"
    
    def _requires_manual_review(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> bool:
        """Determine if request requires manual review"""
        
        review_indicators = [
            response_data["status_code"] >= 500,  # Server errors
            "/safety/emergency" in request_data["path"],  # Emergency operations
            request_data.get("body_size", 0) > 5000000,  # Very large payloads
            response_data.get("processing_time", 0) > 30.0  # Very slow requests
        ]
        
        return any(review_indicators)
    
    async def _check_pii_presence(self, request_data: Dict[str, Any]) -> bool:
        """Check if request contains PII"""
        # This would integrate with the PII protection system
        # Simplified check for common PII indicators
        request_str = json.dumps(request_data).lower()
        pii_indicators = ["email", "phone", "ssn", "credit", "personal"]
        return any(indicator in request_str for indicator in pii_indicators)
    
    def _check_security_headers(self, response_data: Dict[str, Any]) -> bool:
        """Check if security headers are present"""
        headers = response_data.get("headers", {})
        security_headers = ["X-Content-Type-Options", "X-Frame-Options", "X-XSS-Protection"]
        return any(header in headers for header in security_headers)
    
    async def _log_request_start(self, request_data: Dict[str, Any]):
        """Log request start"""
        log_entry = {
            "event": "request_start",
            "correlation_id": request_data["correlation_id"],
            "method": request_data["method"],
            "path": request_data["path"],
            "timestamp": request_data["timestamp"],
            "client_ip": request_data["client"]["host"]
        }
        
        await self._write_to_request_log(log_entry)
    
    async def _log_request_error(self, 
                                request_data: Dict[str, Any], 
                                error_data: Dict[str, Any],
                                correlation_id: str):
        """Log request error"""
        
        error_entry = {
            "event": "request_error",
            "correlation_id": correlation_id,
            "request": request_data,
            "error": error_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self._write_to_request_log(error_entry)
        
        # Also log to main audit logger
        await audit_logger.log_event(
            event_type=AuditEventType.ERROR_EVENT,
            operation=f"{request_data['method']} {request_data['path']}",
            level=AuditLevel.ERROR,
            correlation_id=correlation_id,
            error_details=error_data,
            parameters=request_data
        )
    
    async def _write_audit_logs(self, audit_entry: Dict[str, Any]):
        """Write audit entry to multiple log destinations"""
        
        # Write to request-specific log
        await self._write_to_request_log(audit_entry)
        
        # Write to main audit logger
        await audit_logger.log_api_request(
            method=audit_entry["request"]["method"],
            endpoint=audit_entry["request"]["path"],
            parameters=audit_entry["request"].get("query_params", {}),
            response_status=audit_entry["response"]["status_code"],
            response_time_ms=int(audit_entry["performance"]["processing_time"] * 1000),
            user_context={
                "client_ip": audit_entry["request"]["client"]["host"],
                "user_agent": audit_entry["request"]["headers"].get("user-agent", "")
            },
            request_context={
                "correlation_id": audit_entry["correlation_id"],
                "risk_level": audit_entry["classification"]["risk_level"]
            }
        )
    
    async def _log_to_centralized_audit(self, audit_entry: Dict[str, Any]):
        """Log to centralized audit logger"""
        
        # Add request audit to centralized audit chain
        await centralized_audit_logger._add_to_audit_chain(
            entry_type="api_request",
            content={
                "correlation_id": audit_entry["correlation_id"],
                "endpoint": audit_entry["request"]["path"],
                "method": audit_entry["request"]["method"],
                "status_code": audit_entry["response"]["status_code"],
                "processing_time": audit_entry["performance"]["processing_time"],
                "risk_level": audit_entry["classification"]["risk_level"],
                "timestamp": audit_entry["request"]["timestamp"]
            },
            risk_level=audit_entry["classification"]["risk_level"]
        )
    
    async def _write_to_request_log(self, entry: Dict[str, Any]):
        """Write entry to request audit log file"""
        try:
            async with aiofiles.open(self.request_log_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write request audit log: {e}")
    
    def _update_performance_metrics(self, processing_time: float, status_code: int):
        """Update performance tracking metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if status_code >= 400:
            self.error_count += 1
    
    def get_middleware_statistics(self) -> Dict[str, Any]:
        """Get middleware performance statistics"""
        avg_processing_time = (self.total_processing_time / self.request_count 
                             if self.request_count > 0 else 0.0)
        
        error_rate = (self.error_count / self.request_count 
                     if self.request_count > 0 else 0.0)
        
        return {
            "enabled": self.enabled,
            "total_requests_audited": self.request_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "audited_endpoints": list(self.audited_endpoints),
            "log_file_path": str(self.request_log_file),
            "sensitive_params_count": len(self.sensitive_params)
        }
    
    async def get_recent_audit_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        entries = []
        
        try:
            if self.request_log_file.exists():
                async with aiofiles.open(self.request_log_file, 'r', encoding='utf-8') as f:
                    lines = await f.readlines()
                
                # Get the most recent entries
                for line in lines[-limit:]:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to read audit entries: {e}")
        
        return entries

# Global request audit middleware instance
request_audit_middleware = RequestAuditMiddleware(None)