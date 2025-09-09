"""
LDTS-31: Audit logging system for testing activities

Comprehensive audit trail for all dashboard operations with structured logging,
user activity tracking, and compliance reporting.
"""

import logging
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import asyncio
import aiofiles
from pydantic import BaseModel
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    SEARCH_QUERY = "search_query"
    RERANK_OPERATION = "rerank_operation"
    CONFIG_VALIDATION = "config_validation"
    CONFIG_CHANGE = "config_change"
    EVALUATION_SUBMIT = "evaluation_submit"
    SAFETY_VIOLATION = "safety_violation"
    API_REQUEST = "api_request"
    USER_SESSION = "user_session"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"

class AuditLevel(Enum):
    """Audit log levels"""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AuditEvent(BaseModel):
    """Structured audit event model"""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    level: AuditLevel
    user_context: Dict[str, Any]
    operation: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

class UserActivitySummary(BaseModel):
    """User activity summary for reporting"""
    user_id: str
    session_id: str
    start_time: str
    end_time: str
    total_operations: int
    operations_by_type: Dict[str, int]
    errors_count: int
    performance_summary: Dict[str, Any]

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, log_file_path: str = "audit.log", max_file_size_mb: int = 100):
        self.log_file_path = Path(log_file_path)
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.current_session = {}
        self.event_buffer = []
        self.buffer_size = 100
        
        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize structured logger
        self.struct_logger = structlog.get_logger("audit")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        operation: str,
        level: AuditLevel = AuditLevel.INFO,
        user_context: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            level=level,
            operation=operation,
            user_context=user_context or {},
            parameters=parameters or {},
            result=result or {},
            performance_metrics=performance_metrics or {},
            correlation_id=correlation_id,
            session_id=session_id,
            ip_address=request_context.get("ip_address") if request_context else None,
            user_agent=request_context.get("user_agent") if request_context else None,
            error_details=error_details
        )
        
        # Log to structured logger
        await self._write_structured_log(event)
        
        # Add to buffer for batch processing
        self.event_buffer.append(event)
        
        # Flush buffer if needed
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_buffer()
    
    async def _write_structured_log(self, event: AuditEvent):
        """Write event to structured log"""
        log_data = event.dict()
        
        # Use appropriate log level
        if event.level == AuditLevel.DEBUG:
            self.struct_logger.debug("audit_event", **log_data)
        elif event.level == AuditLevel.INFO:
            self.struct_logger.info("audit_event", **log_data)
        elif event.level == AuditLevel.WARNING:
            self.struct_logger.warning("audit_event", **log_data)
        elif event.level == AuditLevel.ERROR:
            self.struct_logger.error("audit_event", **log_data)
        elif event.level == AuditLevel.CRITICAL:
            self.struct_logger.critical("audit_event", **log_data)
    
    async def _flush_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self.event_buffer:
            return
        
        try:
            # Write to file
            async with aiofiles.open(self.log_file_path, 'a', encoding='utf-8') as f:
                for event in self.event_buffer:
                    await f.write(json.dumps(event.dict()) + '\n')
            
            # Clear buffer
            self.event_buffer.clear()
            
            # Check file size and rotate if needed
            await self._rotate_log_if_needed()
            
        except Exception as e:
            logger.error("Failed to flush audit buffer", error=str(e))
    
    async def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds max size"""
        try:
            if self.log_file_path.exists() and self.log_file_path.stat().st_size > self.max_file_size:
                # Create backup filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.log_file_path.with_suffix(f'.{timestamp}.log')
                
                # Move current log to backup
                self.log_file_path.rename(backup_path)
                
                logger.info(f"Rotated audit log to {backup_path}")
                
        except Exception as e:
            logger.error("Failed to rotate audit log", error=str(e))
    
    async def log_api_request(
        self,
        method: str,
        endpoint: str,
        parameters: Dict[str, Any],
        response_status: int,
        response_time_ms: int,
        user_context: Dict[str, Any],
        request_context: Dict[str, Any],
        error_details: Optional[Dict[str, Any]] = None
    ):
        """Log API request with full context"""
        
        level = AuditLevel.ERROR if response_status >= 400 else AuditLevel.INFO
        
        await self.log_event(
            event_type=AuditEventType.API_REQUEST,
            operation=f"{method} {endpoint}",
            level=level,
            user_context=user_context,
            parameters={
                "method": method,
                "endpoint": endpoint,
                "parameters": parameters,
                "response_status": response_status
            },
            performance_metrics={
                "response_time_ms": response_time_ms
            },
            request_context=request_context,
            error_details=error_details
        )
    
    async def log_search_operation(
        self,
        query: str,
        search_type: str,
        results_count: int,
        search_time_ms: int,
        reranked: bool,
        user_context: Dict[str, Any],
        parameters: Dict[str, Any],
        correlation_id: str
    ):
        """Log search operation with detailed metrics"""
        
        await self.log_event(
            event_type=AuditEventType.SEARCH_QUERY,
            operation=f"search_{search_type}",
            user_context=user_context,
            parameters={
                "query": query,
                "search_type": search_type,
                "reranked": reranked,
                **parameters
            },
            result={
                "results_count": results_count,
                "reranked": reranked
            },
            performance_metrics={
                "search_time_ms": search_time_ms
            },
            correlation_id=correlation_id
        )
    
    async def log_configuration_change(
        self,
        config_section: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        user_context: Dict[str, Any],
        validation_result: Dict[str, Any]
    ):
        """Log configuration changes"""
        
        await self.log_event(
            event_type=AuditEventType.CONFIG_CHANGE,
            operation=f"config_update_{config_section}",
            level=AuditLevel.WARNING,  # Config changes are important
            user_context=user_context,
            parameters={
                "config_section": config_section,
                "old_config": old_config,
                "new_config": new_config
            },
            result=validation_result
        )
    
    async def log_safety_violation(
        self,
        violation_type: str,
        operation: str,
        context: Dict[str, Any],
        user_context: Dict[str, Any]
    ):
        """Log safety violations"""
        
        await self.log_event(
            event_type=AuditEventType.SAFETY_VIOLATION,
            operation=operation,
            level=AuditLevel.CRITICAL,
            user_context=user_context,
            parameters={
                "violation_type": violation_type,
                "operation": operation,
                "context": context
            },
            error_details={
                "severity": "CRITICAL",
                "requires_investigation": True
            }
        )
    
    async def log_evaluation_submission(
        self,
        query: str,
        results_evaluated: int,
        evaluation_scores: List[float],
        user_context: Dict[str, Any],
        evaluation_metadata: Dict[str, Any]
    ):
        """Log manual evaluation submissions"""
        
        await self.log_event(
            event_type=AuditEventType.EVALUATION_SUBMIT,
            operation="submit_evaluation",
            user_context=user_context,
            parameters={
                "query": query,
                "results_evaluated": results_evaluated,
                "evaluation_metadata": evaluation_metadata
            },
            result={
                "evaluation_scores": evaluation_scores,
                "average_score": sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0
            }
        )
    
    async def get_audit_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Retrieve audit logs with filtering"""
        
        logs = []
        
        try:
            if not self.log_file_path.exists():
                return logs
            
            async with aiofiles.open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = await f.readlines()
            
            for line in lines[-limit:]:  # Get recent entries
                try:
                    log_data = json.loads(line.strip())
                    event = AuditEvent(**log_data)
                    
                    # Apply filters
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    if event_type and event.event_type != event_type:
                        continue
                    if user_id and event.user_context.get("user_id") != user_id:
                        continue
                    
                    logs.append(event)
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse audit log line", error=str(e))
                    continue
            
            return logs
            
        except Exception as e:
            logger.error("Failed to retrieve audit logs", error=str(e))
            return []
    
    async def generate_activity_summary(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Optional[UserActivitySummary]:
        """Generate user activity summary"""
        
        try:
            # Get relevant logs
            logs = await self.get_audit_logs(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                limit=1000
            )
            
            if not logs:
                return None
            
            # Filter by session if specified
            if session_id:
                logs = [log for log in logs if log.session_id == session_id]
            
            if not logs:
                return None
            
            # Calculate summary statistics
            operations_by_type = {}
            errors_count = 0
            total_response_time = 0
            response_time_count = 0
            
            for log in logs:
                # Count operations by type
                op_type = log.event_type.value
                operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
                
                # Count errors
                if log.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                    errors_count += 1
                
                # Accumulate response times
                response_time = log.performance_metrics.get("response_time_ms") or log.performance_metrics.get("search_time_ms")
                if response_time:
                    total_response_time += response_time
                    response_time_count += 1
            
            # Calculate performance summary
            performance_summary = {
                "total_operations": len(logs),
                "error_rate": errors_count / len(logs) if logs else 0,
                "average_response_time_ms": total_response_time / response_time_count if response_time_count else 0
            }
            
            return UserActivitySummary(
                user_id=user_id,
                session_id=session_id or "all_sessions",
                start_time=logs[0].timestamp if logs else "",
                end_time=logs[-1].timestamp if logs else "",
                total_operations=len(logs),
                operations_by_type=operations_by_type,
                errors_count=errors_count,
                performance_summary=performance_summary
            )
            
        except Exception as e:
            logger.error("Failed to generate activity summary", error=str(e), user_id=user_id)
            return None
    
    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old audit logs"""
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_time.isoformat()
            
            # Read current logs
            if not self.log_file_path.exists():
                return
            
            kept_logs = []
            
            async with aiofiles.open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = await f.readlines()
            
            for line in lines:
                try:
                    log_data = json.loads(line.strip())
                    if log_data.get("timestamp", "") > cutoff_iso:
                        kept_logs.append(line)
                except json.JSONDecodeError:
                    continue
            
            # Write back only recent logs
            async with aiofiles.open(self.log_file_path, 'w', encoding='utf-8') as f:
                await f.writelines(kept_logs)
            
            logger.info(f"Cleaned up old audit logs, kept {len(kept_logs)} entries")
            
        except Exception as e:
            logger.error("Failed to cleanup old logs", error=str(e))

# Global audit logger instance
audit_logger = AuditLogger("dashboard-backend/audit.log")

@contextmanager
def audit_context(
    operation: str,
    event_type: AuditEventType = AuditEventType.API_REQUEST,
    user_context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
):
    """Context manager for audit logging"""
    
    start_time = time.time()
    correlation_id = correlation_id or str(uuid.uuid4())
    
    try:
        yield correlation_id
        
        # Log successful operation
        end_time = time.time()
        asyncio.create_task(audit_logger.log_event(
            event_type=event_type,
            operation=operation,
            level=AuditLevel.INFO,
            user_context=user_context or {},
            performance_metrics={
                "duration_ms": int((end_time - start_time) * 1000)
            },
            correlation_id=correlation_id
        ))
        
    except Exception as e:
        # Log failed operation
        end_time = time.time()
        asyncio.create_task(audit_logger.log_event(
            event_type=event_type,
            operation=operation,
            level=AuditLevel.ERROR,
            user_context=user_context or {},
            performance_metrics={
                "duration_ms": int((end_time - start_time) * 1000)
            },
            correlation_id=correlation_id,
            error_details={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        ))
        raise

# Convenience functions
async def log_api_request(method: str, endpoint: str, **kwargs):
    """Log API request"""
    await audit_logger.log_api_request(method, endpoint, **kwargs)

async def log_search_operation(query: str, **kwargs):
    """Log search operation"""
    await audit_logger.log_search_operation(query, **kwargs)

async def log_safety_violation(violation_type: str, operation: str, **kwargs):
    """Log safety violation"""
    await audit_logger.log_safety_violation(violation_type, operation, **kwargs)