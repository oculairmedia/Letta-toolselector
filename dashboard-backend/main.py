"""
LDTS Reranker Testing Dashboard Backend API

FastAPI server with async support for ML workloads.
Provides testing endpoints for search, reranking, and configuration management.

LDTS-26: Set up FastAPI server with async support for ML workloads
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import os
from pathlib import Path

# Import existing LDTS services
import sys
sys.path.append(str(Path(__file__).parent.parent / "lettaaugment-source"))

from api_server import app as existing_app
from weaviate_tool_search_with_reranking import WeaviateToolSearch
from safety import safety_check, SafetyLevel, verify_production_isolation, get_safety_status
from search_endpoints import router as search_router
from missing_endpoints import router as missing_endpoints_router
from integration_layer import integration_layer
from config_validation import validate_configuration, validate_yaml_config, ConfigValidationResponse
from audit_logging import AuditLogger, AuditEventType, audit_logger
from rate_limiting import rate_limiter, check_rate_limit, get_request_context, validate_request_size
from readonly_middleware import readonly_guard
from no_attach_mode import no_attach_manager, SafetyModeConfig
from pii_middleware import pii_middleware
from pii_protection import pii_protection_manager
from centralized_audit_logger import centralized_audit_logger, AuditIntegrityLevel
from request_audit_middleware import RequestAuditMiddleware
from reranker_config_manager import (
    get_reranker_manager, initialize_reranker_manager, 
    RerankerStatus, RerankResult
)
from reranker_latency_manager import (
    initialize_latency_manager, get_latency_manager,
    LatencyBudget, FallbackMode, LatencyBudgetStatus
)
from evaluation_metrics import (
    evaluation_metrics_service, EvaluationQuery, EvaluationResult,
    MetricType, EvaluationSummary, MetricResult
)
from evaluation_persistence import (
    evaluation_persistence_service, EvaluationRun, StoredEvaluationResult,
    EvaluationComparison, EvaluationStatus
)
from production_isolation import (
    production_isolation_service, ProductionResource, IsolationRule,
    SafetyValidation, EnvironmentType, IsolationLevel, ValidationResult
)
from weaviate_overrides import (
    weaviate_override_service, WeaviateOverride, OverrideSet,
    OverrideApplication, OverrideScope, ParameterType
)
from bm25_vector_overrides import (
    bm25_vector_override_service, BM25Override, VectorDistanceOverride,
    SearchParameterSet, BM25ParameterType, VectorParameterType, VectorDistanceMetric
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting LDTS Dashboard Backend...")
    
    # Initialize ML workload resources
    await initialize_ml_resources()
    
    yield
    
    logger.info("Shutting down LDTS Dashboard Backend...")
    await cleanup_ml_resources()

# Create FastAPI app with async lifespan
app = FastAPI(
    title="LDTS Reranker Testing Dashboard API",
    description="Testing and evaluation interface for Letta Tool Selector reranking system",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LDTS-70: Add read-only guard middleware for testing endpoints
app.middleware("http")(readonly_guard)

# LDTS-72: Add PII protection and payload size limit middleware
app.middleware("http")(pii_middleware)

# LDTS-74: Add per-request audit logging middleware for testing endpoints
app.add_middleware(RequestAuditMiddleware, enabled=True)

# Include search endpoints router
app.include_router(search_router)

# Include missing endpoints router for frontend compatibility
app.include_router(missing_endpoints_router)

# Global state for ML resources
ml_resources: Dict[str, Any] = {}

async def initialize_ml_resources():
    """Initialize ML workload resources"""
    try:
        logger.info("Initializing ML resources...")
        
        # CRITICAL: Verify production isolation first
        verify_production_isolation()
        
        # Initialize integration layer with existing LDTS services
        await integration_layer.initialize()
        
        # Audit logging and rate limiting are already initialized
        # No need to call initialize methods
        
        # Initialize reranker manager
        reranker_config = {
            "ollama_reranker": {
                "enabled": True,
                "model": "qwen3-reranker-4b",
                "base_url": "http://localhost:11434",
                "max_batch_size": 16,
                "timeout_seconds": 30,
                "temperature": 0.1,
                "top_k": 100,
                "threshold": 0.0
            }
        }
        reranker_initialized = await initialize_reranker_manager(reranker_config)
        
        # Initialize reranker latency manager
        latency_config = {
            "max_latency_ms": 5000,
            "warning_threshold": 0.8,
            "fallback_mode": "original_order",
            "enable_circuit_breaker": True,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_reset_time": 30,
            "adaptive_budget": True,
            "min_budget_ms": 1000,
            "max_budget_ms": 10000
        }
        latency_manager = initialize_latency_manager(latency_config)
        
        # Store integration layer reference
        ml_resources["integration_layer"] = integration_layer
        ml_resources["weaviate_search"] = integration_layer.weaviate_client
        ml_resources["embedding_providers"] = integration_layer.embedding_providers
        ml_resources["rerankers"] = integration_layer.reranker_services
        ml_resources["audit_logger"] = audit_logger
        ml_resources["rate_limiter"] = rate_limiter
        ml_resources["reranker_manager"] = get_reranker_manager() if reranker_initialized else None
        ml_resources["latency_manager"] = latency_manager
        ml_resources["evaluation_metrics"] = evaluation_metrics_service
        ml_resources["evaluation_persistence"] = evaluation_persistence_service
        ml_resources["production_isolation"] = production_isolation_service
        ml_resources["weaviate_overrides"] = weaviate_override_service
        ml_resources["bm25_vector_overrides"] = bm25_vector_override_service
        
        # Initialize evaluation persistence database
        await evaluation_persistence_service.initialize_database()
        
        logger.info("ML resources initialized successfully")
        logger.info("SAFETY MODE: Read-only testing mode enforced")
        logger.info(f"Integration status: {integration_layer.get_service_status()}")
        
        # Log system startup
        await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            operation="dashboard_startup",
            user_context={"user_id": "system"},
            parameters={"version": "0.1.0", "safety_mode": "read_only_testing"}
        )
    except Exception as e:
        logger.error(f"Failed to initialize ML resources: {e}")
        raise

async def cleanup_ml_resources():
    """Cleanup ML resources on shutdown"""
    logger.info("Cleaning up ML resources...")
    
    # Log system shutdown
    if ml_resources.get("audit_logger"):
        try:
            await audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                operation="dashboard_shutdown",
                user_id="system",
                details={"reason": "normal_shutdown"}
            )
            await audit_logger.shutdown()
        except Exception as e:
            logger.error(f"Failed to log shutdown event: {e}")
    
    # Shutdown rate limiter
    if ml_resources.get("rate_limiter"):
        try:
            await rate_limiter.shutdown()
        except Exception as e:
            logger.error(f"Failed to shutdown rate limiter: {e}")
    
    ml_resources.clear()

# Health check endpoint
@app.get("/api/v1/health")
@safety_check("health_check", SafetyLevel.READ_ONLY)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ldts-dashboard-backend", 
        "version": "0.1.0",
        "timestamp": time.time(),
        "ml_resources_loaded": len(ml_resources) > 0,
        "safety_mode": "read_only_testing",
        "no_attach_mode": no_attach_manager.config.mode.value,
        "banner_active": no_attach_manager.config.show_banner
    }

# LDTS-30: Safety status endpoint
@app.get("/api/v1/safety/status")
@safety_check("safety_status", SafetyLevel.READ_ONLY)
async def safety_status():
    """Get current safety system status"""
    return get_safety_status()

# LDTS-70: Read-only guard middleware status endpoint
@app.get("/api/v1/safety/readonly-guard/status")
@safety_check("readonly_guard_status", SafetyLevel.READ_ONLY)
async def readonly_guard_status():
    """Get read-only guard middleware statistics"""
    return readonly_guard.get_statistics()

# LDTS-70: Read-only guard middleware control endpoint
@app.post("/api/v1/safety/readonly-guard/configure")
@safety_check("readonly_guard_configure", SafetyLevel.READ_ONLY)
async def configure_readonly_guard(enforce_read_only: bool, log_violations: bool = True):
    """Configure read-only guard middleware settings"""
    readonly_guard.enforce_read_only = enforce_read_only
    readonly_guard.log_violations = log_violations
    
    return {
        "message": "Read-only guard configuration updated",
        "enforce_read_only": enforce_read_only,
        "log_violations": log_violations,
        "timestamp": time.time()
    }

# LDTS-71: No-attach mode status endpoint
@app.get("/api/v1/safety/no-attach/status")
@safety_check("no_attach_status", SafetyLevel.READ_ONLY)
async def no_attach_status():
    """Get no-attach mode status and configuration"""
    return no_attach_manager.get_mode_status()

# LDTS-71: Safety mode banner endpoint
@app.get("/api/v1/safety/banner", response_class=HTMLResponse)
async def get_safety_banner():
    """Get safety mode banner HTML"""
    banner_html = no_attach_manager.get_banner_html()
    if not banner_html:
        return HTMLResponse(content="<div>No banner active</div>")
    return HTMLResponse(content=banner_html)

# LDTS-71: Safety mode banner data endpoint  
@app.get("/api/v1/safety/banner/data")
@safety_check("banner_data", SafetyLevel.READ_ONLY)
async def get_banner_data():
    """Get safety mode banner data for frontend integration"""
    return no_attach_manager.get_banner_data()

# LDTS-71: Emergency lockdown endpoint
@app.post("/api/v1/safety/emergency-lockdown")
@safety_check("emergency_lockdown", SafetyLevel.READ_ONLY)
async def trigger_emergency_lockdown(reason: str, triggered_by: str = "manual"):
    """Trigger emergency lockdown mode"""
    no_attach_manager.set_emergency_lockdown(reason, triggered_by)
    
    # Log the emergency lockdown
    await audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_EVENT,
        operation="emergency_lockdown",
        user_id=triggered_by,
        details={"reason": reason, "mode": "emergency_lockdown"}
    )
    
    return {
        "message": "Emergency lockdown activated",
        "reason": reason,
        "triggered_by": triggered_by,
        "mode": SafetyModeConfig.EMERGENCY_LOCKDOWN.value,
        "timestamp": time.time()
    }

# LDTS-71: Validate operation against no-attach mode
@app.post("/api/v1/safety/validate-operation-mode")
@safety_check("validate_operation_mode", SafetyLevel.READ_ONLY)
async def validate_operation_mode(operation: str, context: Dict[str, Any] = None):
    """Validate operation against current safety mode"""
    result = no_attach_manager.is_operation_allowed(operation, context or {})
    
    return {
        "operation": operation,
        "context": context,
        "result": result,
        "current_mode": no_attach_manager.config.mode.value,
        "timestamp": time.time()
    }

# LDTS-72: PII protection status endpoint
@app.get("/api/v1/safety/pii-protection/status")
@safety_check("pii_protection_status", SafetyLevel.READ_ONLY)
async def pii_protection_status():
    """Get PII protection status and statistics"""
    return pii_protection_manager.get_protection_statistics()

# LDTS-72: PII protection middleware statistics
@app.get("/api/v1/safety/pii-protection/middleware-stats")
@safety_check("pii_middleware_stats", SafetyLevel.READ_ONLY)
async def pii_middleware_statistics():
    """Get PII protection middleware statistics"""
    return pii_middleware.get_middleware_statistics()

# LDTS-72: Configure PII protection
@app.post("/api/v1/safety/pii-protection/configure")
@safety_check("pii_protection_configure", SafetyLevel.READ_ONLY)
async def configure_pii_protection(
    pii_config: Optional[Dict[str, Any]] = None,
    payload_config: Optional[Dict[str, Any]] = None
):
    """Configure PII protection settings"""
    pii_protection_manager.update_configuration(pii_config, payload_config)
    
    return {
        "message": "PII protection configuration updated",
        "pii_config": pii_config,
        "payload_config": payload_config,
        "current_config": pii_protection_manager.get_protection_statistics()['configuration'],
        "timestamp": time.time()
    }

# LDTS-72: Test PII detection
@app.post("/api/v1/safety/pii-protection/test")
@safety_check("pii_protection_test", SafetyLevel.READ_ONLY)
async def test_pii_detection(test_text: str):
    """Test PII detection on provided text"""
    result = pii_protection_manager._detect_and_redact_pii(test_text, "test_endpoint")
    
    return {
        "test_text_length": len(test_text),
        "pii_detected": result['pii_detected'],
        "detection_count": result.get('detection_count', 0),
        "detections": result.get('detections', []),
        "redacted_content": result['redacted_content'],
        "timestamp": time.time()
    }

# LDTS-72: Clear PII protection statistics
@app.post("/api/v1/safety/pii-protection/clear-stats")
@safety_check("pii_protection_clear_stats", SafetyLevel.READ_ONLY)
async def clear_pii_statistics():
    """Clear PII protection statistics (for maintenance)"""
    pii_protection_manager.clear_statistics()
    
    return {
        "message": "PII protection statistics cleared",
        "timestamp": time.time()
    }

# LDTS-73: Centralized audit logger endpoints

@app.get("/api/v1/audit/centralized/status")
@safety_check("centralized_audit_status", SafetyLevel.READ_ONLY)
async def centralized_audit_status():
    """Get centralized audit logger status"""
    return {
        "integrity_level": centralized_audit_logger.integrity_level.value,
        "audit_chain_length": len(centralized_audit_logger.audit_chain),
        "configuration_changes": len(centralized_audit_logger.config_change_history),
        "integrity_violations": len(centralized_audit_logger.integrity_violations),
        "watched_sections": list(centralized_audit_logger.watched_config_sections)
    }

@app.get("/api/v1/audit/centralized/configuration-history")
@safety_check("audit_config_history", SafetyLevel.READ_ONLY)
async def get_configuration_history(
    component: Optional[str] = None,
    section: Optional[str] = None,
    limit: int = Query(default=50, le=500)
):
    """Get configuration change history"""
    history = await centralized_audit_logger.get_configuration_history(
        component=component,
        section=section,
        limit=limit
    )
    
    return {
        "configuration_changes": [change.dict() for change in history],
        "total_changes": len(history),
        "filters": {
            "component": component,
            "section": section,
            "limit": limit
        }
    }

@app.get("/api/v1/audit/centralized/risk-summary")
@safety_check("audit_risk_summary", SafetyLevel.READ_ONLY)
async def get_configuration_risk_summary():
    """Get configuration risk analysis summary"""
    return await centralized_audit_logger.get_configuration_risk_summary()

@app.post("/api/v1/audit/centralized/verify-integrity")
@safety_check("audit_verify_integrity", SafetyLevel.READ_ONLY)
async def verify_audit_integrity():
    """Verify audit chain integrity"""
    verification_result = await centralized_audit_logger.verify_audit_integrity()
    
    return {
        "verification_result": verification_result,
        "timestamp": time.time(),
        "status": "VERIFIED" if verification_result["chain_valid"] else "COMPROMISED"
    }

@app.post("/api/v1/audit/centralized/log-config-change")
@safety_check("audit_log_config_change", SafetyLevel.READ_ONLY)
async def log_configuration_change_endpoint(
    component: str,
    section: str,
    old_config: Dict[str, Any],
    new_config: Dict[str, Any],
    user_context: Dict[str, Any],
    validation_result: Dict[str, Any]
):
    """Log a configuration change with hashing"""
    change_id = await centralized_audit_logger.log_configuration_change(
        component=component,
        section=section,
        old_config=old_config,
        new_config=new_config,
        user_context=user_context,
        validation_result=validation_result
    )
    
    return {
        "change_id": change_id,
        "message": "Configuration change logged successfully",
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/centralized/configuration-hashes")
@safety_check("audit_config_hashes", SafetyLevel.READ_ONLY)
async def get_configuration_hashes():
    """Get current configuration hashes"""
    return {
        "configuration_hashes": centralized_audit_logger.configuration_hashes,
        "hash_count": len(centralized_audit_logger.configuration_hashes),
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/centralized/integrity-violations")
@safety_check("audit_integrity_violations", SafetyLevel.READ_ONLY)
async def get_integrity_violations():
    """Get audit integrity violations"""
    return {
        "integrity_violations": centralized_audit_logger.integrity_violations,
        "violation_count": len(centralized_audit_logger.integrity_violations),
        "timestamp": time.time()
    }

# LDTS-74: Per-request audit logging endpoints

@app.get("/api/v1/audit/requests/statistics")
@safety_check("request_audit_statistics", SafetyLevel.READ_ONLY)
async def get_request_audit_statistics():
    """Get request audit middleware statistics"""
    # Access the middleware instance through the app
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, RequestAuditMiddleware):
            # This is simplified - in production we'd store reference to middleware
            break
    
    return {
        "message": "Request audit middleware statistics",
        "note": "Statistics are tracked in middleware instance",
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/requests/recent")
@safety_check("request_audit_recent", SafetyLevel.READ_ONLY)
async def get_recent_request_audits(limit: int = Query(default=50, le=500)):
    """Get recent request audit entries"""
    
    # Create a temporary instance to access the log reading functionality
    temp_middleware = RequestAuditMiddleware(None)
    recent_entries = await temp_middleware.get_recent_audit_entries(limit=limit)
    
    return {
        "audit_entries": recent_entries,
        "count": len(recent_entries),
        "limit": limit,
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/requests/by-correlation/{correlation_id}")
@safety_check("request_audit_by_correlation", SafetyLevel.READ_ONLY)
async def get_audit_by_correlation(correlation_id: str):
    """Get audit entry by correlation ID"""
    
    temp_middleware = RequestAuditMiddleware(None)
    recent_entries = await temp_middleware.get_recent_audit_entries(limit=1000)
    
    # Find entries with matching correlation ID
    matching_entries = [
        entry for entry in recent_entries 
        if entry.get("correlation_id") == correlation_id
    ]
    
    return {
        "correlation_id": correlation_id,
        "audit_entries": matching_entries,
        "count": len(matching_entries),
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/requests/by-endpoint")
@safety_check("request_audit_by_endpoint", SafetyLevel.READ_ONLY) 
async def get_audit_by_endpoint(endpoint: str, limit: int = Query(default=50, le=500)):
    """Get audit entries for specific endpoint"""
    
    temp_middleware = RequestAuditMiddleware(None)
    recent_entries = await temp_middleware.get_recent_audit_entries(limit=limit * 2)  # Get more to filter
    
    # Filter by endpoint
    matching_entries = [
        entry for entry in recent_entries 
        if endpoint in entry.get("request", {}).get("path", "")
    ]
    
    return {
        "endpoint": endpoint,
        "audit_entries": matching_entries[:limit],
        "count": len(matching_entries),
        "timestamp": time.time()
    }

@app.get("/api/v1/audit/requests/performance-summary")
@safety_check("request_audit_performance", SafetyLevel.READ_ONLY)
async def get_request_performance_summary():
    """Get request performance summary from audit logs"""
    
    temp_middleware = RequestAuditMiddleware(None) 
    recent_entries = await temp_middleware.get_recent_audit_entries(limit=1000)
    
    # Calculate performance metrics
    total_requests = len(recent_entries)
    processing_times = []
    status_codes = {}
    endpoints = {}
    error_count = 0
    
    for entry in recent_entries:
        if "performance" in entry:
            processing_times.append(entry["performance"]["processing_time"])
        
        if "response" in entry:
            status_code = entry["response"]["status_code"]
            status_codes[status_code] = status_codes.get(status_code, 0) + 1
            
            if status_code >= 400:
                error_count += 1
        
        if "request" in entry:
            endpoint = entry["request"]["path"]
            endpoints[endpoint] = endpoints.get(endpoint, 0) + 1
    
    # Calculate averages
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    error_rate = error_count / total_requests if total_requests > 0 else 0
    
    return {
        "summary": {
            "total_requests": total_requests,
            "average_processing_time": avg_processing_time,
            "error_count": error_count,
            "error_rate": error_rate
        },
        "status_code_distribution": status_codes,
        "top_endpoints": dict(sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:10]),
        "performance_percentiles": {
            "p50": sorted(processing_times)[len(processing_times)//2] if processing_times else 0,
            "p95": sorted(processing_times)[int(len(processing_times)*0.95)] if processing_times else 0,
            "p99": sorted(processing_times)[int(len(processing_times)*0.99)] if processing_times else 0
        } if processing_times else {},
        "timestamp": time.time()
    }

# LDTS-75: Expose reranker configuration in testing API

@app.get("/api/v1/reranker/config")
@safety_check("reranker_config", SafetyLevel.READ_ONLY)
async def get_reranker_configuration():
    """Get all reranker configurations"""
    try:
        if not ml_resources.get("reranker_manager"):
            return {
                "error": "Reranker manager not available",
                "available_rerankers": [],
                "configurations": {},
                "timestamp": time.time()
            }
        
        reranker_manager = ml_resources["reranker_manager"]
        available_rerankers = reranker_manager.get_available_rerankers()
        
        configurations = {}
        for reranker_name in available_rerankers:
            configurations[reranker_name] = reranker_manager.get_reranker_config(reranker_name)
        
        return {
            "available_rerankers": available_rerankers,
            "default_reranker": reranker_manager.default_reranker,
            "configurations": configurations,
            "initialized": reranker_manager._initialized,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get reranker configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reranker configuration: {str(e)}")

@app.get("/api/v1/reranker/config/{reranker_name}")
@safety_check("reranker_config_specific", SafetyLevel.READ_ONLY)
async def get_specific_reranker_config(reranker_name: str):
    """Get configuration for a specific reranker"""
    try:
        if not ml_resources.get("reranker_manager"):
            raise HTTPException(status_code=503, detail="Reranker manager not available")
        
        reranker_manager = ml_resources["reranker_manager"]
        config = reranker_manager.get_reranker_config(reranker_name)
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Reranker '{reranker_name}' not found")
        
        return {
            "reranker_name": reranker_name,
            "configuration": config,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get specific reranker configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reranker configuration: {str(e)}")

@app.get("/api/v1/reranker/status")
@safety_check("reranker_status", SafetyLevel.READ_ONLY)
async def get_reranker_status():
    """Get status of all rerankers"""
    try:
        if not ml_resources.get("reranker_manager"):
            return {
                "reranker_manager_available": False,
                "reranker_statuses": {},
                "timestamp": time.time()
            }
        
        reranker_manager = ml_resources["reranker_manager"]
        statuses = await reranker_manager.health_check_all()
        
        return {
            "reranker_manager_available": True,
            "reranker_statuses": {name: status.value for name, status in statuses.items()},
            "default_reranker": reranker_manager.default_reranker,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get reranker status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get reranker status: {str(e)}")

@app.post("/api/v1/reranker/test")
@safety_check("reranker_test", SafetyLevel.READ_ONLY)
async def test_reranker(
    query: str,
    documents: List[str],
    reranker_name: Optional[str] = None,
    original_scores: Optional[List[float]] = None
):
    """Test reranking functionality"""
    try:
        if not ml_resources.get("reranker_manager"):
            raise HTTPException(status_code=503, detail="Reranker manager not available")
        
        # Validate input
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")
        
        if len(documents) > 100:
            raise HTTPException(status_code=400, detail="Too many documents (max: 100)")
        
        if original_scores and len(original_scores) != len(documents):
            raise HTTPException(status_code=400, detail="Original scores length must match documents length")
        
        reranker_manager = ml_resources["reranker_manager"]
        
        # Perform reranking
        start_time = time.time()
        result = await reranker_manager.rerank(
            query=query,
            documents=documents,
            original_scores=original_scores,
            reranker_name=reranker_name
        )
        processing_time = time.time() - start_time
        
        # Build response with reranked documents
        reranked_documents = [documents[i] for i in result.reranked_indices]
        
        return {
            "query": query,
            "reranking_result": {
                "reranked_indices": result.reranked_indices,
                "reranked_scores": result.reranked_scores,
                "reranked_documents": reranked_documents,
                "original_scores": result.original_scores,
                "provider": result.provider,
                "model": result.model,
                "processing_time": result.processing_time,
                "document_count": result.document_count,
                "metadata": result.metadata
            },
            "test_metadata": {
                "reranker_used": reranker_name or reranker_manager.default_reranker,
                "total_processing_time": processing_time,
                "documents_provided": len(documents),
                "original_scores_provided": original_scores is not None
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranker test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranker test failed: {str(e)}")

# LDTS-76: Latency management endpoints
@app.get("/api/v1/reranker/latency/status")
@safety_check("latency_status", SafetyLevel.READ_ONLY)
async def get_latency_status():
    """Get latency manager status and statistics"""
    try:
        if not ml_resources.get("latency_manager"):
            return {
                "latency_manager_available": False,
                "message": "Latency manager not initialized",
                "timestamp": time.time()
            }
        
        latency_manager = ml_resources["latency_manager"]
        stats = latency_manager.get_statistics()
        
        return {
            "latency_manager_available": True,
            "statistics": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get latency status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latency status: {str(e)}")

@app.get("/api/v1/reranker/latency/config")
@safety_check("latency_config", SafetyLevel.READ_ONLY)
async def get_latency_config():
    """Get current latency configuration"""
    try:
        if not ml_resources.get("latency_manager"):
            raise HTTPException(status_code=503, detail="Latency manager not available")
        
        latency_manager = ml_resources["latency_manager"]
        
        return {
            "config": {
                "default_budget_seconds": latency_manager.default_budget_seconds,
                "circuit_breaker_threshold": latency_manager.circuit_breaker_threshold,
                "circuit_breaker_timeout": latency_manager.circuit_breaker_timeout,
                "fallback_enabled": latency_manager.fallback_enabled,
                "enable_adaptive_budgets": latency_manager.enable_adaptive_budgets,
                "adaptive_percentile": latency_manager.adaptive_percentile,
                "adaptive_window_size": latency_manager.adaptive_window_size
            },
            "circuit_breaker_states": {
                provider: {
                    "state": state["state"].value,
                    "failure_count": state["failure_count"],
                    "last_failure_time": state["last_failure_time"]
                } for provider, state in latency_manager.circuit_breakers.items()
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latency config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latency config: {str(e)}")

@app.post("/api/v1/reranker/latency/configure")
@safety_check("latency_configure", SafetyLevel.READ_ONLY)
async def configure_latency_manager(
    default_budget_seconds: Optional[float] = None,
    circuit_breaker_threshold: Optional[int] = None,
    circuit_breaker_timeout: Optional[float] = None,
    fallback_enabled: Optional[bool] = None,
    enable_adaptive_budgets: Optional[bool] = None
):
    """Update latency manager configuration"""
    try:
        if not ml_resources.get("latency_manager"):
            raise HTTPException(status_code=503, detail="Latency manager not available")
        
        latency_manager = ml_resources["latency_manager"]
        updates = {}
        
        if default_budget_seconds is not None:
            if default_budget_seconds <= 0:
                raise HTTPException(status_code=400, detail="Default budget must be positive")
            latency_manager.default_budget_seconds = default_budget_seconds
            updates["default_budget_seconds"] = default_budget_seconds
            
        if circuit_breaker_threshold is not None:
            if circuit_breaker_threshold < 1:
                raise HTTPException(status_code=400, detail="Circuit breaker threshold must be at least 1")
            latency_manager.circuit_breaker_threshold = circuit_breaker_threshold
            updates["circuit_breaker_threshold"] = circuit_breaker_threshold
            
        if circuit_breaker_timeout is not None:
            if circuit_breaker_timeout <= 0:
                raise HTTPException(status_code=400, detail="Circuit breaker timeout must be positive")
            latency_manager.circuit_breaker_timeout = circuit_breaker_timeout
            updates["circuit_breaker_timeout"] = circuit_breaker_timeout
            
        if fallback_enabled is not None:
            latency_manager.fallback_enabled = fallback_enabled
            updates["fallback_enabled"] = fallback_enabled
            
        if enable_adaptive_budgets is not None:
            latency_manager.enable_adaptive_budgets = enable_adaptive_budgets
            updates["enable_adaptive_budgets"] = enable_adaptive_budgets
        
        return {
            "message": "Latency manager configuration updated",
            "updates": updates,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure latency manager: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure latency manager: {str(e)}")

@app.post("/api/v1/reranker/latency/reset-circuits")
@safety_check("reset_circuits", SafetyLevel.READ_ONLY)
async def reset_circuit_breakers():
    """Reset all circuit breakers"""
    try:
        if not ml_resources.get("latency_manager"):
            raise HTTPException(status_code=503, detail="Latency manager not available")
        
        latency_manager = ml_resources["latency_manager"]
        reset_count = 0
        
        for provider_name in latency_manager.circuit_breakers:
            latency_manager.reset_circuit_breaker(provider_name)
            reset_count += 1
        
        return {
            "message": f"Reset {reset_count} circuit breakers",
            "reset_count": reset_count,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breakers: {str(e)}")

@app.post("/api/v1/reranker/test-with-budget")
@safety_check("reranker_test_with_budget", SafetyLevel.READ_ONLY)
async def test_reranker_with_budget(
    query: str,
    documents: List[str],
    reranker_name: Optional[str] = None,
    original_scores: Optional[List[float]] = None,
    budget_seconds: Optional[float] = None
):
    """Test reranking functionality with latency budget enforcement"""
    try:
        if not ml_resources.get("reranker_manager"):
            raise HTTPException(status_code=503, detail="Reranker manager not available")
            
        if not ml_resources.get("latency_manager"):
            raise HTTPException(status_code=503, detail="Latency manager not available")
        
        # Validate input
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")
        
        if len(documents) > 100:
            raise HTTPException(status_code=400, detail="Too many documents (max: 100)")
        
        if original_scores and len(original_scores) != len(documents):
            raise HTTPException(status_code=400, detail="Original scores length must match documents length")
        
        if budget_seconds is not None and budget_seconds <= 0:
            raise HTTPException(status_code=400, detail="Budget must be positive")
        
        reranker_manager = ml_resources["reranker_manager"]
        latency_manager = ml_resources["latency_manager"]
        
        # Perform reranking with budget enforcement
        start_time = time.time()
        
        async with latency_manager.enforce_budget(
            operation_name="reranker_test",
            provider_name=reranker_name or reranker_manager.default_reranker,
            budget_seconds=budget_seconds
        ) as budget_context:
            result = await reranker_manager.rerank(
                query=query,
                documents=documents,
                original_scores=original_scores,
                reranker_name=reranker_name
            )
        
        total_processing_time = time.time() - start_time
        
        # Build response with reranked documents
        reranked_documents = [documents[i] for i in result.reranked_indices]
        
        return {
            "query": query,
            "reranking_result": {
                "reranked_indices": result.reranked_indices,
                "reranked_scores": result.reranked_scores,
                "reranked_documents": reranked_documents,
                "original_scores": result.original_scores,
                "provider": result.provider,
                "model": result.model,
                "processing_time": result.processing_time,
                "document_count": result.document_count,
                "metadata": result.metadata
            },
            "budget_enforcement": {
                "budget_used": budget_context.elapsed_time,
                "budget_seconds": budget_context.budget_seconds,
                "budget_exceeded": budget_context.budget_exceeded,
                "fallback_triggered": budget_context.fallback_triggered,
                "circuit_breaker_triggered": budget_context.circuit_breaker_triggered,
                "operation_name": budget_context.operation_name
            },
            "test_metadata": {
                "reranker_used": reranker_name or reranker_manager.default_reranker,
                "total_processing_time": total_processing_time,
                "documents_provided": len(documents),
                "original_scores_provided": original_scores is not None,
                "budget_requested": budget_seconds
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranker test with budget failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranker test with budget failed: {str(e)}")

# LDTS-80: Evaluation metrics endpoints
@app.get("/api/v1/evaluation/metrics/supported")
@safety_check("metrics_supported", SafetyLevel.READ_ONLY)
async def get_supported_metrics():
    """Get list of supported evaluation metrics"""
    try:
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        metrics_service = ml_resources["evaluation_metrics"]
        supported_metrics = metrics_service.get_supported_metrics()
        statistics = metrics_service.get_statistics()
        
        return {
            "supported_metrics": supported_metrics,
            "default_k_values": metrics_service.default_k_values,
            "service_statistics": statistics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get supported metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported metrics: {str(e)}")

@app.post("/api/v1/evaluation/compute")
@safety_check("evaluation_compute", SafetyLevel.READ_ONLY)
async def compute_evaluation_metrics(
    queries: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None,
    cache_results: bool = True
):
    """
    Compute evaluation metrics for ranking results
    
    Args:
        queries: List of evaluation queries with ground truth
        results: List of ranking results to evaluate
        k_values: K values for @K metrics (optional)
        metrics: Specific metrics to compute (optional, default: all)
        cache_results: Whether to cache computation results
    """
    try:
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        # Validate input
        if not queries:
            raise HTTPException(status_code=400, detail="Queries list cannot be empty")
        
        if not results:
            raise HTTPException(status_code=400, detail="Results list cannot be empty")
        
        # Convert dictionaries to evaluation objects
        evaluation_queries = []
        for q in queries:
            try:
                eval_query = EvaluationQuery(
                    query_id=q["query_id"],
                    query_text=q["query_text"],
                    relevant_doc_ids=q["relevant_doc_ids"],
                    relevant_scores=q.get("relevant_scores"),
                    metadata=q.get("metadata")
                )
                evaluation_queries.append(eval_query)
            except KeyError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field in query: {e}"
                )
        
        evaluation_results = []
        for r in results:
            try:
                eval_result = EvaluationResult(
                    query_id=r["query_id"],
                    ranked_doc_ids=r["ranked_doc_ids"],
                    scores=r.get("scores"),
                    processing_time=r.get("processing_time"),
                    metadata=r.get("metadata")
                )
                evaluation_results.append(eval_result)
            except KeyError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field in result: {e}"
                )
        
        # Convert metric names to MetricType enums
        metric_types = None
        if metrics:
            try:
                metric_types = [MetricType(m) for m in metrics]
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metric type: {e}"
                )
        
        # Compute evaluation metrics
        metrics_service = ml_resources["evaluation_metrics"]
        evaluation_summary = metrics_service.evaluate_ranking(
            queries=evaluation_queries,
            results=evaluation_results,
            k_values=k_values,
            metrics=metric_types,
            cache_results=cache_results
        )
        
        # Convert to JSON-serializable format
        def serialize_metric_result(mr):
            if hasattr(mr, '__dict__'):
                return asdict(mr)
            return mr
        
        serialized_individual = []
        for result in evaluation_summary.individual_results:
            serialized_metrics = {}
            for metric_name, metric_data in result["metrics"].items():
                if isinstance(metric_data, dict):
                    # Handle @K metrics
                    serialized_metrics[metric_name] = {
                        str(k): serialize_metric_result(v) for k, v in metric_data.items()
                    }
                else:
                    # Handle non-@K metrics
                    serialized_metrics[metric_name] = serialize_metric_result(metric_data)
            
            serialized_individual.append({
                "query_id": result["query_id"],
                "metrics": serialized_metrics
            })
        
        return {
            "evaluation_summary": {
                "total_queries": evaluation_summary.total_queries,
                "individual_results": serialized_individual,
                "aggregate_metrics": evaluation_summary.aggregate_metrics,
                "k_values": evaluation_summary.k_values,
                "processing_time": evaluation_summary.processing_time,
                "timestamp": evaluation_summary.timestamp,
                "metadata": evaluation_summary.metadata
            },
            "service_info": {
                "cache_used": cache_results,
                "metrics_computed": metrics or "all",
                "k_values_used": k_values or metrics_service.default_k_values
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation computation failed: {str(e)}")

@app.post("/api/v1/evaluation/compare")
@safety_check("evaluation_compare", SafetyLevel.READ_ONLY)
async def compare_ranking_systems(
    queries: List[Dict[str, Any]],
    system_a_results: List[Dict[str, Any]],
    system_b_results: List[Dict[str, Any]],
    system_a_name: str = "System A",
    system_b_name: str = "System B",
    k_values: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None
):
    """
    Compare two ranking systems using evaluation metrics
    
    Args:
        queries: Ground truth queries
        system_a_results: Results from first system
        system_b_results: Results from second system
        system_a_name: Name for first system
        system_b_name: Name for second system
        k_values: K values for @K metrics
        metrics: Specific metrics to compute
    """
    try:
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        # Validate input
        if not queries:
            raise HTTPException(status_code=400, detail="Queries list cannot be empty")
        
        if not system_a_results:
            raise HTTPException(status_code=400, detail="System A results cannot be empty")
        
        if not system_b_results:
            raise HTTPException(status_code=400, detail="System B results cannot be empty")
        
        # Convert to evaluation objects
        evaluation_queries = [
            EvaluationQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_doc_ids=q["relevant_doc_ids"],
                relevant_scores=q.get("relevant_scores"),
                metadata=q.get("metadata")
            ) for q in queries
        ]
        
        system_a_eval_results = [
            EvaluationResult(
                query_id=r["query_id"],
                ranked_doc_ids=r["ranked_doc_ids"],
                scores=r.get("scores"),
                processing_time=r.get("processing_time"),
                metadata=r.get("metadata")
            ) for r in system_a_results
        ]
        
        system_b_eval_results = [
            EvaluationResult(
                query_id=r["query_id"],
                ranked_doc_ids=r["ranked_doc_ids"],
                scores=r.get("scores"),
                processing_time=r.get("processing_time"),
                metadata=r.get("metadata")
            ) for r in system_b_results
        ]
        
        # Convert metric names if provided
        metric_types = None
        if metrics:
            metric_types = [MetricType(m) for m in metrics]
        
        # Evaluate both systems
        metrics_service = ml_resources["evaluation_metrics"]
        
        system_a_summary = metrics_service.evaluate_ranking(
            queries=evaluation_queries,
            results=system_a_eval_results,
            k_values=k_values,
            metrics=metric_types,
            cache_results=True
        )
        
        system_b_summary = metrics_service.evaluate_ranking(
            queries=evaluation_queries,
            results=system_b_eval_results,
            k_values=k_values,
            metrics=metric_types,
            cache_results=True
        )
        
        # Compute improvements/differences
        metric_comparisons = {}
        for metric_name in system_a_summary.aggregate_metrics:
            if metric_name in system_b_summary.aggregate_metrics:
                a_value = system_a_summary.aggregate_metrics[metric_name]
                b_value = system_b_summary.aggregate_metrics[metric_name]
                
                difference = b_value - a_value
                relative_improvement = (difference / a_value * 100) if a_value != 0 else 0
                
                metric_comparisons[metric_name] = {
                    f"{system_a_name}_value": a_value,
                    f"{system_b_name}_value": b_value,
                    "absolute_difference": difference,
                    "relative_improvement_percent": relative_improvement,
                    "winner": system_b_name if b_value > a_value else system_a_name if a_value > b_value else "tie"
                }
        
        return {
            "comparison_summary": {
                "system_a_name": system_a_name,
                "system_b_name": system_b_name,
                "total_queries": system_a_summary.total_queries,
                "metric_comparisons": metric_comparisons,
                "processing_time": {
                    f"{system_a_name}_time": system_a_summary.processing_time,
                    f"{system_b_name}_time": system_b_summary.processing_time
                }
            },
            "detailed_results": {
                system_a_name: {
                    "aggregate_metrics": system_a_summary.aggregate_metrics,
                    "total_queries": system_a_summary.total_queries
                },
                system_b_name: {
                    "aggregate_metrics": system_b_summary.aggregate_metrics,
                    "total_queries": system_b_summary.total_queries
                }
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"System comparison failed: {str(e)}")

@app.get("/api/v1/evaluation/metrics/statistics")
@safety_check("metrics_statistics", SafetyLevel.READ_ONLY)
async def get_metrics_statistics():
    """Get evaluation metrics service statistics"""
    try:
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        metrics_service = ml_resources["evaluation_metrics"]
        statistics = metrics_service.get_statistics()
        
        return {
            "service_statistics": statistics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics statistics: {str(e)}")

@app.post("/api/v1/evaluation/metrics/clear-cache")
@safety_check("metrics_clear_cache", SafetyLevel.READ_ONLY)
async def clear_metrics_cache():
    """Clear evaluation metrics computation cache"""
    try:
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        metrics_service = ml_resources["evaluation_metrics"]
        cache_size_before = len(metrics_service.computation_cache)
        
        metrics_service.clear_cache()
        
        return {
            "message": "Evaluation metrics cache cleared",
            "cache_entries_cleared": cache_size_before,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear metrics cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear metrics cache: {str(e)}")

# LDTS-81: Evaluation persistence endpoints
@app.post("/api/v1/evaluation/runs/create")
@safety_check("evaluation_create_run", SafetyLevel.READ_ONLY)
async def create_evaluation_run(
    run_name: str,
    description: str = "",
    configuration: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Create a new evaluation run"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        if not run_name.strip():
            raise HTTPException(status_code=400, detail="Run name cannot be empty")
        
        persistence_service = ml_resources["evaluation_persistence"]
        run_id = await persistence_service.create_evaluation_run(
            run_name=run_name,
            description=description,
            configuration=configuration,
            metadata=metadata
        )
        
        return {
            "run_id": run_id,
            "run_name": run_name,
            "description": description,
            "status": EvaluationStatus.PENDING.value,
            "created_at": time.time(),
            "message": "Evaluation run created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create evaluation run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation run: {str(e)}")

@app.post("/api/v1/evaluation/runs/{run_id}/execute")
@safety_check("evaluation_execute_run", SafetyLevel.READ_ONLY)
async def execute_evaluation_run(
    run_id: str,
    queries: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None
):
    """Execute evaluation run and store results"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        if not ml_resources.get("evaluation_metrics"):
            raise HTTPException(status_code=503, detail="Evaluation metrics service not available")
        
        persistence_service = ml_resources["evaluation_persistence"]
        metrics_service = ml_resources["evaluation_metrics"]
        
        # Validate run exists
        run = await persistence_service.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        if run.status != EvaluationStatus.PENDING:
            raise HTTPException(status_code=400, detail="Run is not in pending status")
        
        # Update run status to running
        await persistence_service.update_run_status(run_id, EvaluationStatus.RUNNING)
        
        try:
            # Convert to evaluation objects
            evaluation_queries = [
                EvaluationQuery(
                    query_id=q["query_id"],
                    query_text=q["query_text"],
                    relevant_doc_ids=q["relevant_doc_ids"],
                    relevant_scores=q.get("relevant_scores"),
                    metadata=q.get("metadata")
                ) for q in queries
            ]
            
            evaluation_results = [
                EvaluationResult(
                    query_id=r["query_id"],
                    ranked_doc_ids=r["ranked_doc_ids"],
                    scores=r.get("scores"),
                    processing_time=r.get("processing_time"),
                    metadata=r.get("metadata")
                ) for r in results
            ]
            
            # Convert metric names if provided
            metric_types = None
            if metrics:
                metric_types = [MetricType(m) for m in metrics]
            
            # Compute evaluation metrics
            evaluation_summary = metrics_service.evaluate_ranking(
                queries=evaluation_queries,
                results=evaluation_results,
                k_values=k_values,
                metrics=metric_types,
                cache_results=True
            )
            
            # Store results
            result_ids = await persistence_service.store_evaluation_summary(
                run_id=run_id,
                summary=evaluation_summary,
                queries=queries,
                results=results
            )
            
            # Update run status to completed
            await persistence_service.update_run_status(run_id, EvaluationStatus.COMPLETED)
            
            return {
                "run_id": run_id,
                "status": EvaluationStatus.COMPLETED.value,
                "total_queries": evaluation_summary.total_queries,
                "processing_time": evaluation_summary.processing_time,
                "aggregate_metrics": evaluation_summary.aggregate_metrics,
                "result_ids": result_ids,
                "message": "Evaluation run completed successfully"
            }
            
        except Exception as e:
            # Update run status to failed
            await persistence_service.update_run_status(
                run_id, EvaluationStatus.FAILED, str(e)
            )
            raise
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute evaluation run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute evaluation run: {str(e)}")

@app.get("/api/v1/evaluation/runs")
@safety_check("evaluation_list_runs", SafetyLevel.READ_ONLY)
async def list_evaluation_runs(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """List evaluation runs with filtering and pagination"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        # Validate parameters
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must be non-negative")
        
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = EvaluationStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Parse date filters
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO 8601.")
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO 8601.")
        
        persistence_service = ml_resources["evaluation_persistence"]
        runs, total_count = await persistence_service.list_evaluation_runs(
            limit=limit,
            offset=offset,
            status_filter=status_filter,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        # Convert to serializable format
        serialized_runs = []
        for run in runs:
            serialized_runs.append({
                "run_id": run.run_id,
                "run_name": run.run_name,
                "description": run.description,
                "status": run.status.value,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "total_queries": run.total_queries,
                "k_values": run.k_values,
                "metrics_computed": run.metrics_computed,
                "processing_time": run.processing_time,
                "error_message": run.error_message,
                "metadata": run.metadata
            })
        
        return {
            "runs": serialized_runs,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total_count": total_count,
                "has_next": (offset + limit) < total_count,
                "has_prev": offset > 0
            },
            "filters": {
                "status": status,
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list evaluation runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list evaluation runs: {str(e)}")

@app.get("/api/v1/evaluation/runs/{run_id}")
@safety_check("evaluation_get_run", SafetyLevel.READ_ONLY)
async def get_evaluation_run(run_id: str):
    """Get detailed information about an evaluation run"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        persistence_service = ml_resources["evaluation_persistence"]
        run = await persistence_service.get_evaluation_run(run_id)
        
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        # Get aggregate metrics if completed
        aggregate_metrics = {}
        if run.status == EvaluationStatus.COMPLETED:
            aggregate_metrics = await persistence_service.get_run_aggregate_metrics(run_id)
        
        return {
            "run": {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "description": run.description,
                "status": run.status.value,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "total_queries": run.total_queries,
                "k_values": run.k_values,
                "metrics_computed": run.metrics_computed,
                "configuration": run.configuration,
                "processing_time": run.processing_time,
                "error_message": run.error_message,
                "metadata": run.metadata
            },
            "aggregate_metrics": aggregate_metrics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation run: {str(e)}")

@app.get("/api/v1/evaluation/runs/{run_id}/results")
@safety_check("evaluation_get_run_results", SafetyLevel.READ_ONLY)
async def get_run_results(
    run_id: str,
    limit: int = 100,
    offset: int = 0,
    query_filter: Optional[str] = None
):
    """Get results for a specific evaluation run"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        # Validate parameters
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        persistence_service = ml_resources["evaluation_persistence"]
        
        # Check if run exists
        run = await persistence_service.get_evaluation_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        results, total_count = await persistence_service.get_run_results(
            run_id=run_id,
            limit=limit,
            offset=offset,
            query_filter=query_filter
        )
        
        # Convert to serializable format
        serialized_results = []
        for result in results:
            serialized_results.append({
                "result_id": result.result_id,
                "query_id": result.query_id,
                "query_text": result.query_text,
                "relevant_doc_ids": result.relevant_doc_ids,
                "ranked_doc_ids": result.ranked_doc_ids,
                "metrics": result.metrics,
                "scores": result.scores,
                "processing_time": result.processing_time,
                "created_at": result.created_at.isoformat(),
                "metadata": result.metadata
            })
        
        return {
            "run_id": run_id,
            "results": serialized_results,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total_count": total_count,
                "has_next": (offset + limit) < total_count,
                "has_prev": offset > 0
            },
            "filters": {
                "query_filter": query_filter
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get run results: {str(e)}")

@app.post("/api/v1/evaluation/runs/{run_a_id}/compare/{run_b_id}")
@safety_check("evaluation_compare_runs", SafetyLevel.READ_ONLY)
async def compare_evaluation_runs(run_a_id: str, run_b_id: str, store_comparison: bool = True):
    """Compare two evaluation runs"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        persistence_service = ml_resources["evaluation_persistence"]
        comparison = await persistence_service.compare_runs(
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            store_comparison=store_comparison
        )
        
        return {
            "comparison": {
                "comparison_id": comparison.comparison_id,
                "run_a_id": comparison.run_a_id,
                "run_b_id": comparison.run_b_id,
                "run_a_name": comparison.run_a_name,
                "run_b_name": comparison.run_b_name,
                "metric_improvements": comparison.metric_improvements,
                "statistical_significance": comparison.statistical_significance,
                "created_at": comparison.created_at.isoformat(),
                "summary": comparison.summary
            },
            "stored": store_comparison,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare evaluation runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare evaluation runs: {str(e)}")

@app.delete("/api/v1/evaluation/runs/{run_id}")
@safety_check("evaluation_delete_run", SafetyLevel.READ_ONLY)
async def delete_evaluation_run(run_id: str):
    """Delete an evaluation run and all associated results"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        persistence_service = ml_resources["evaluation_persistence"]
        deleted = await persistence_service.delete_run(run_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        return {
            "message": f"Evaluation run {run_id} deleted successfully",
            "run_id": run_id,
            "deleted": True,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation run: {str(e)}")

@app.post("/api/v1/evaluation/maintenance/cleanup")
@safety_check("evaluation_cleanup", SafetyLevel.READ_ONLY)
async def cleanup_old_runs(keep_days: int = 30):
    """Clean up old evaluation runs"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        if keep_days < 1:
            raise HTTPException(status_code=400, detail="keep_days must be at least 1")
        
        persistence_service = ml_resources["evaluation_persistence"]
        deleted_count = await persistence_service.cleanup_old_runs(keep_days)
        
        return {
            "message": f"Cleaned up {deleted_count} old evaluation runs",
            "deleted_count": deleted_count,
            "keep_days": keep_days,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup old runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old runs: {str(e)}")

@app.get("/api/v1/evaluation/persistence/statistics")
@safety_check("evaluation_persistence_statistics", SafetyLevel.READ_ONLY)
async def get_persistence_statistics():
    """Get evaluation persistence service statistics"""
    try:
        if not ml_resources.get("evaluation_persistence"):
            raise HTTPException(status_code=503, detail="Evaluation persistence service not available")
        
        persistence_service = ml_resources["evaluation_persistence"]
        statistics = await persistence_service.get_statistics()
        
        return {
            "service_statistics": statistics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get persistence statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get persistence statistics: {str(e)}")

# LDTS-65: Production isolation and safety validation endpoints
@app.post("/api/v1/safety/isolation/validate")
@safety_check("isolation_validate", SafetyLevel.READ_ONLY)
async def validate_resource_access(
    resource_identifier: str,
    resource_type: str,
    operation: str = "access",
    context: Optional[Dict[str, Any]] = None
):
    """Validate if access to a resource is safe and allowed"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        if not resource_identifier.strip():
            raise HTTPException(status_code=400, detail="Resource identifier cannot be empty")
        
        if not resource_type.strip():
            raise HTTPException(status_code=400, detail="Resource type cannot be empty")
        
        isolation_service = ml_resources["production_isolation"]
        validation = await isolation_service.validate_resource_access(
            resource_identifier=resource_identifier,
            resource_type=resource_type,
            operation=operation,
            context=context or {}
        )
        
        # Convert to serializable format
        return {
            "validation": {
                "validation_id": validation.validation_id,
                "resource_identifier": validation.resource_identifier,
                "resource_type": validation.resource_type,
                "environment_detected": validation.environment_detected.value,
                "validation_result": validation.validation_result.value,
                "matched_rules": validation.matched_rules,
                "risk_score": validation.risk_score,
                "timestamp": validation.timestamp.isoformat(),
                "details": validation.details,
                "recommendations": validation.recommendations
            },
            "isolation_level": isolation_service.isolation_level.value,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resource validation failed: {str(e)}")

@app.post("/api/v1/safety/isolation/validate-config")
@safety_check("isolation_validate_config", SafetyLevel.READ_ONLY)
async def validate_configuration_safety(config: Dict[str, Any]):
    """Validate configuration for production safety"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        if not config:
            raise HTTPException(status_code=400, detail="Configuration cannot be empty")
        
        isolation_service = ml_resources["production_isolation"]
        issues = await isolation_service.validate_configuration(config)
        
        return {
            "configuration_safe": len(issues) == 0,
            "issues_found": issues,
            "issue_count": len(issues),
            "recommendations": [
                "Review all flagged configuration values",
                "Ensure no production credentials or endpoints are present",
                "Use environment-specific configuration files",
                "Implement proper secret management"
            ] if issues else ["Configuration appears safe for testing"],
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration validation failed: {str(e)}")

@app.get("/api/v1/safety/isolation/rules")
@safety_check("isolation_get_rules", SafetyLevel.READ_ONLY)
async def get_isolation_rules():
    """Get all production isolation rules"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        isolation_service = ml_resources["production_isolation"]
        rules = isolation_service.get_isolation_rules()
        
        return {
            "rules": rules,
            "total_rules": len(rules),
            "isolation_level": isolation_service.isolation_level.value,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get isolation rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get isolation rules: {str(e)}")

@app.post("/api/v1/safety/isolation/rules")
@safety_check("isolation_add_rule", SafetyLevel.READ_ONLY)
async def add_isolation_rule(
    rule_name: str,
    resource_type: str,
    pattern: str,
    action: str,
    description: str,
    enabled: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add custom production isolation rule"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        # Validate inputs
        if not rule_name.strip():
            raise HTTPException(status_code=400, detail="Rule name cannot be empty")
        
        if not pattern.strip():
            raise HTTPException(status_code=400, detail="Pattern cannot be empty")
        
        # Validate pattern is valid regex
        try:
            import re
            re.compile(pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {str(e)}")
        
        # Validate action
        try:
            validation_result = ValidationResult(action)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid action. Must be one of: {[v.value for v in ValidationResult]}"
            )
        
        # Create rule
        rule_id = f"custom_{int(time.time())}"
        rule = IsolationRule(
            rule_id=rule_id,
            rule_name=rule_name,
            resource_type=resource_type,
            pattern=pattern,
            action=validation_result,
            description=description,
            enabled=enabled,
            metadata=metadata
        )
        
        isolation_service = ml_resources["production_isolation"]
        isolation_service.add_isolation_rule(rule)
        
        return {
            "message": "Isolation rule added successfully",
            "rule": {
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "resource_type": rule.resource_type,
                "pattern": rule.pattern,
                "action": rule.action.value,
                "description": rule.description,
                "enabled": rule.enabled,
                "metadata": rule.metadata
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add isolation rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add isolation rule: {str(e)}")

@app.get("/api/v1/safety/isolation/resources")
@safety_check("isolation_get_resources", SafetyLevel.READ_ONLY)
async def get_production_resources():
    """Get all registered production resources"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        isolation_service = ml_resources["production_isolation"]
        resources = isolation_service.get_production_resources()
        
        return {
            "resources": resources,
            "total_resources": len(resources),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get production resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get production resources: {str(e)}")

@app.post("/api/v1/safety/isolation/resources")
@safety_check("isolation_add_resource", SafetyLevel.READ_ONLY)
async def add_production_resource(
    resource_type: str,
    identifier: str,
    environment: str,
    criticality: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Register a known production resource"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        # Validate inputs
        if not identifier.strip():
            raise HTTPException(status_code=400, detail="Resource identifier cannot be empty")
        
        # Validate environment
        try:
            env_type = EnvironmentType(environment)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid environment. Must be one of: {[e.value for e in EnvironmentType]}"
            )
        
        # Validate criticality
        if criticality not in ["high", "medium", "low"]:
            raise HTTPException(status_code=400, detail="Criticality must be high, medium, or low")
        
        # Create resource
        resource_id = f"resource_{int(time.time())}"
        resource = ProductionResource(
            resource_id=resource_id,
            resource_type=resource_type,
            identifier=identifier,
            environment=env_type,
            criticality=criticality,
            description=description,
            metadata=metadata
        )
        
        isolation_service = ml_resources["production_isolation"]
        isolation_service.add_production_resource(resource)
        
        return {
            "message": "Production resource registered successfully",
            "resource": {
                "resource_id": resource.resource_id,
                "resource_type": resource.resource_type,
                "identifier": resource.identifier,
                "environment": resource.environment.value,
                "criticality": resource.criticality,
                "description": resource.description,
                "metadata": resource.metadata
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add production resource: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add production resource: {str(e)}")

@app.put("/api/v1/safety/isolation/level")
@safety_check("isolation_update_level", SafetyLevel.READ_ONLY)
async def update_isolation_level(isolation_level: str):
    """Update production isolation enforcement level"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        # Validate isolation level
        try:
            level = IsolationLevel(isolation_level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid isolation level. Must be one of: {[l.value for l in IsolationLevel]}"
            )
        
        isolation_service = ml_resources["production_isolation"]
        old_level = isolation_service.isolation_level
        isolation_service.update_isolation_level(level)
        
        return {
            "message": f"Isolation level updated from {old_level.value} to {level.value}",
            "old_level": old_level.value,
            "new_level": level.value,
            "cache_cleared": True,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update isolation level: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update isolation level: {str(e)}")

@app.post("/api/v1/safety/isolation/cache/clear")
@safety_check("isolation_clear_cache", SafetyLevel.READ_ONLY)
async def clear_isolation_cache():
    """Clear production isolation validation cache"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        isolation_service = ml_resources["production_isolation"]
        cache_size_before = len(isolation_service.validation_cache)
        
        isolation_service.clear_validation_cache()
        
        return {
            "message": "Isolation validation cache cleared",
            "cache_entries_cleared": cache_size_before,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear isolation cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear isolation cache: {str(e)}")

@app.get("/api/v1/safety/isolation/statistics")
@safety_check("isolation_statistics", SafetyLevel.READ_ONLY)
async def get_isolation_statistics():
    """Get production isolation service statistics"""
    try:
        if not ml_resources.get("production_isolation"):
            raise HTTPException(status_code=503, detail="Production isolation service not available")
        
        isolation_service = ml_resources["production_isolation"]
        statistics = isolation_service.get_statistics()
        
        return {
            "service_statistics": statistics,
            "environment_types": [e.value for e in EnvironmentType],
            "isolation_levels": [l.value for l in IsolationLevel],
            "validation_results": [v.value for v in ValidationResult],
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get isolation statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get isolation statistics: {str(e)}")

# LDTS-78: Weaviate overrides endpoints
@app.post("/api/v1/config/weaviate/override-sets")
@safety_check("weaviate_create_override_set", SafetyLevel.READ_ONLY)
async def create_weaviate_override_set(
    name: str,
    description: str,
    overrides: List[Dict[str, Any]],
    scope: str = "request",
    created_by: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Create a new Weaviate parameter override set"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        if not name.strip():
            raise HTTPException(status_code=400, detail="Override set name cannot be empty")
        
        if not overrides:
            raise HTTPException(status_code=400, detail="At least one override must be provided")
        
        # Validate scope
        try:
            override_scope = OverrideScope(scope)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scope. Must be one of: {[s.value for s in OverrideScope]}"
            )
        
        # Validate all overrides before creating the set
        override_service = ml_resources["weaviate_overrides"]
        validation_errors = []
        
        for i, override_data in enumerate(overrides):
            errors = override_service.validate_override_data(override_data)
            if errors:
                validation_errors.extend([f"Override {i+1}: {error}" for error in errors])
        
        if validation_errors:
            raise HTTPException(status_code=400, detail={"validation_errors": validation_errors})
        
        # Create override set
        override_set_id = override_service.create_override_set(
            name=name,
            description=description,
            overrides=overrides,
            scope=override_scope,
            created_by=created_by,
            metadata=metadata
        )
        
        return {
            "message": "Override set created successfully",
            "override_set_id": override_set_id,
            "name": name,
            "scope": scope,
            "overrides_count": len(overrides),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create override set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create override set: {str(e)}")

@app.get("/api/v1/config/weaviate/override-sets")
@safety_check("weaviate_list_override_sets", SafetyLevel.READ_ONLY)
async def list_weaviate_override_sets(
    scope: Optional[str] = None,
    active_only: bool = True
):
    """List all Weaviate parameter override sets"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        # Validate scope if provided
        scope_filter = None
        if scope:
            try:
                scope_filter = OverrideScope(scope)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid scope. Must be one of: {[s.value for s in OverrideScope]}"
                )
        
        override_service = ml_resources["weaviate_overrides"]
        override_sets = override_service.list_override_sets(
            scope_filter=scope_filter,
            active_only=active_only
        )
        
        # Convert to serializable format
        serialized_sets = []
        for override_set in override_sets:
            serialized_sets.append({
                "override_set_id": override_set.override_set_id,
                "name": override_set.name,
                "description": override_set.description,
                "scope": override_set.scope.value,
                "created_at": override_set.created_at.isoformat(),
                "created_by": override_set.created_by,
                "active": override_set.active,
                "overrides_count": len(override_set.overrides),
                "metadata": override_set.metadata
            })
        
        return {
            "override_sets": serialized_sets,
            "total_count": len(serialized_sets),
            "filters": {
                "scope": scope,
                "active_only": active_only
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list override sets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list override sets: {str(e)}")

@app.get("/api/v1/config/weaviate/override-sets/{override_set_id}")
@safety_check("weaviate_get_override_set", SafetyLevel.READ_ONLY)
async def get_weaviate_override_set(override_set_id: str):
    """Get detailed information about a Weaviate override set"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        override_set = override_service.get_override_set(override_set_id)
        
        if not override_set:
            raise HTTPException(status_code=404, detail="Override set not found")
        
        # Convert overrides to serializable format
        serialized_overrides = []
        for override in override_set.overrides:
            serialized_overrides.append({
                "parameter_name": override.parameter_name,
                "parameter_type": override.parameter_type.value,
                "original_value": override.original_value,
                "override_value": override.override_value,
                "scope": override.scope.value,
                "description": override.description,
                "enabled": override.enabled,
                "metadata": override.metadata
            })
        
        return {
            "override_set": {
                "override_set_id": override_set.override_set_id,
                "name": override_set.name,
                "description": override_set.description,
                "scope": override_set.scope.value,
                "created_at": override_set.created_at.isoformat(),
                "created_by": override_set.created_by,
                "active": override_set.active,
                "metadata": override_set.metadata,
                "overrides": serialized_overrides
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get override set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get override set: {str(e)}")

@app.put("/api/v1/config/weaviate/override-sets/{override_set_id}")
@safety_check("weaviate_update_override_set", SafetyLevel.READ_ONLY)
async def update_weaviate_override_set(
    override_set_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    active: Optional[bool] = None
):
    """Update Weaviate override set properties"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        updated = override_service.update_override_set(
            override_set_id=override_set_id,
            name=name,
            description=description,
            active=active
        )
        
        if not updated:
            raise HTTPException(status_code=404, detail="Override set not found")
        
        return {
            "message": "Override set updated successfully",
            "override_set_id": override_set_id,
            "updates": {
                "name": name,
                "description": description,
                "active": active
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update override set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update override set: {str(e)}")

@app.delete("/api/v1/config/weaviate/override-sets/{override_set_id}")
@safety_check("weaviate_delete_override_set", SafetyLevel.READ_ONLY)
async def delete_weaviate_override_set(override_set_id: str):
    """Delete a Weaviate override set"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        deleted = override_service.delete_override_set(override_set_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Override set not found")
        
        return {
            "message": f"Override set {override_set_id} deleted successfully",
            "override_set_id": override_set_id,
            "deleted": True,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete override set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete override set: {str(e)}")

@app.post("/api/v1/config/weaviate/apply-overrides")
@safety_check("weaviate_apply_overrides", SafetyLevel.READ_ONLY)
async def apply_weaviate_overrides(
    base_config: Dict[str, Any],
    override_set_id: Optional[str] = None,
    request_overrides: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None
):
    """Apply Weaviate parameter overrides to a base configuration"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        if not base_config:
            raise HTTPException(status_code=400, detail="Base configuration cannot be empty")
        
        override_service = ml_resources["weaviate_overrides"]
        application = override_service.apply_overrides(
            base_config=base_config,
            override_set_id=override_set_id,
            request_overrides=request_overrides,
            session_id=session_id
        )
        
        return {
            "application": {
                "application_id": application.application_id,
                "override_set_id": application.override_set_id,
                "original_config": application.original_config,
                "modified_config": application.modified_config,
                "applied_overrides": application.applied_overrides,
                "skipped_overrides": application.skipped_overrides,
                "warnings": application.warnings,
                "timestamp": application.timestamp.isoformat()
            },
            "summary": {
                "overrides_applied": len(application.applied_overrides),
                "overrides_skipped": len(application.skipped_overrides),
                "warnings_count": len(application.warnings),
                "configuration_modified": application.original_config != application.modified_config
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply overrides: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply overrides: {str(e)}")

@app.get("/api/v1/config/weaviate/parameter-schemas")
@safety_check("weaviate_get_schemas", SafetyLevel.READ_ONLY)
async def get_weaviate_parameter_schemas():
    """Get Weaviate parameter schemas for validation"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        schemas = override_service.get_parameter_schemas()
        
        return {
            "parameter_schemas": schemas,
            "parameter_types": [t.value for t in ParameterType],
            "override_scopes": [s.value for s in OverrideScope],
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get parameter schemas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameter schemas: {str(e)}")

@app.post("/api/v1/config/weaviate/validate-override")
@safety_check("weaviate_validate_override", SafetyLevel.READ_ONLY)
async def validate_weaviate_override(override_data: Dict[str, Any]):
    """Validate a single Weaviate parameter override"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        errors = override_service.validate_override_data(override_data)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "override_data": override_data,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate override: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate override: {str(e)}")

@app.get("/api/v1/config/weaviate/statistics")
@safety_check("weaviate_override_statistics", SafetyLevel.READ_ONLY)
async def get_weaviate_override_statistics():
    """Get Weaviate override service statistics"""
    try:
        if not ml_resources.get("weaviate_overrides"):
            raise HTTPException(status_code=503, detail="Weaviate override service not available")
        
        override_service = ml_resources["weaviate_overrides"]
        statistics = override_service.get_statistics()
        
        return {
            "service_statistics": statistics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get override statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get override statistics: {str(e)}")

# LDTS-79: BM25 Parameter Overrides and Vector Distance Selection
@app.post("/api/v1/search/parameter-sets")
@safety_check("bm25_vector_create_parameter_set", SafetyLevel.READ_ONLY)
async def create_search_parameter_set(parameter_set_data: Dict[str, Any]):
    """Create a new search parameter set with BM25 and vector overrides"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        
        # Extract required fields
        name = parameter_set_data.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Parameter set name is required")
        
        description = parameter_set_data.get("description", "")
        
        parameter_set_id = override_service.create_parameter_set(
            name=name,
            description=description,
            bm25_params=parameter_set_data.get("bm25_parameters", []),
            vector_params=parameter_set_data.get("vector_parameters", []),
            hybrid_alpha=parameter_set_data.get("hybrid_alpha"),
            fusion_type=parameter_set_data.get("fusion_type", "relative_score"),
            metadata=parameter_set_data.get("metadata")
        )
        
        logger.info(f"Created search parameter set: {name} ({parameter_set_id})")
        
        return {
            "parameter_set_id": parameter_set_id,
            "name": name,
            "description": description,
            "created": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create search parameter set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create parameter set: {str(e)}")

@app.get("/api/v1/search/parameter-sets")
@safety_check("bm25_vector_list_parameter_sets", SafetyLevel.READ_ONLY)
async def list_search_parameter_sets(active_only: bool = True):
    """List all search parameter sets"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        parameter_sets = override_service.list_parameter_sets(active_only=active_only)
        
        return {
            "parameter_sets": parameter_sets,
            "total_count": len(parameter_sets),
            "active_only": active_only
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list parameter sets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list parameter sets: {str(e)}")

@app.get("/api/v1/search/parameter-sets/{parameter_set_id}")
@safety_check("bm25_vector_get_parameter_set", SafetyLevel.READ_ONLY)
async def get_search_parameter_set(parameter_set_id: str):
    """Get a specific search parameter set"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        parameter_set = override_service.get_parameter_set(parameter_set_id)
        
        if not parameter_set:
            raise HTTPException(status_code=404, detail=f"Parameter set not found: {parameter_set_id}")
        
        return {
            "parameter_set_id": parameter_set.parameter_set_id,
            "name": parameter_set.name,
            "description": parameter_set.description,
            "bm25_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.bm25_overrides
            ],
            "vector_overrides": [
                {
                    "parameter_type": override.parameter_type.value,
                    "value": override.value,
                    "description": override.description,
                    "enabled": override.enabled,
                    "validation_range": override.validation_range
                } for override in parameter_set.vector_overrides
            ],
            "hybrid_alpha": parameter_set.hybrid_alpha,
            "fusion_type": parameter_set.fusion_type,
            "created_at": parameter_set.created_at.isoformat(),
            "active": parameter_set.active,
            "metadata": parameter_set.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get parameter set {parameter_set_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameter set: {str(e)}")

@app.post("/api/v1/search/apply-parameter-set")
@safety_check("bm25_vector_apply_parameter_set", SafetyLevel.READ_ONLY)
async def apply_search_parameter_set(application_data: Dict[str, Any]):
    """Apply a search parameter set to a base configuration"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        parameter_set_id = application_data.get("parameter_set_id")
        base_config = application_data.get("base_config", {})
        
        if not parameter_set_id:
            raise HTTPException(status_code=400, detail="parameter_set_id is required")
        
        override_service = ml_resources["bm25_vector_overrides"]
        modified_config = override_service.apply_parameter_set(parameter_set_id, base_config)
        
        logger.info(f"Applied parameter set {parameter_set_id} to base configuration")
        
        return {
            "parameter_set_id": parameter_set_id,
            "base_config": base_config,
            "modified_config": modified_config,
            "applied": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply parameter set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply parameter set: {str(e)}")

@app.delete("/api/v1/search/parameter-sets/{parameter_set_id}")
@safety_check("bm25_vector_delete_parameter_set", SafetyLevel.DANGEROUS)
async def delete_search_parameter_set(parameter_set_id: str):
    """Delete a search parameter set"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        deleted = override_service.delete_parameter_set(parameter_set_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Parameter set not found: {parameter_set_id}")
        
        logger.info(f"Deleted parameter set: {parameter_set_id}")
        
        return {
            "parameter_set_id": parameter_set_id,
            "deleted": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete parameter set {parameter_set_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete parameter set: {str(e)}")

@app.get("/api/v1/search/supported-distance-metrics")
@safety_check("bm25_vector_distance_metrics", SafetyLevel.READ_ONLY)
async def get_supported_distance_metrics():
    """Get list of supported vector distance metrics"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        metrics = override_service.get_supported_distance_metrics()
        
        return {
            "supported_distance_metrics": metrics,
            "count": len(metrics),
            "default_metric": "cosine"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get supported distance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get distance metrics: {str(e)}")

@app.get("/api/v1/search/parameter-schemas")
@safety_check("bm25_vector_parameter_schemas", SafetyLevel.READ_ONLY)
async def get_search_parameter_schemas():
    """Get parameter validation schemas for BM25 and vector parameters"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        
        return {
            "bm25_parameters": override_service.get_bm25_parameter_schema(),
            "vector_parameters": override_service.get_vector_parameter_schema(),
            "supported_distance_metrics": override_service.get_supported_distance_metrics()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get parameter schemas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get parameter schemas: {str(e)}")

@app.get("/api/v1/search/statistics")
@safety_check("bm25_vector_statistics", SafetyLevel.READ_ONLY)
async def get_search_override_statistics():
    """Get BM25/Vector override service statistics"""
    try:
        if not ml_resources.get("bm25_vector_overrides"):
            raise HTTPException(status_code=503, detail="BM25/Vector override service not available")
        
        override_service = ml_resources["bm25_vector_overrides"]
        statistics = override_service.get_statistics()
        
        return {
            "service_statistics": statistics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get search override statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get search override statistics: {str(e)}")

# LDTS-30: Operation validation endpoint
@app.post("/api/v1/safety/validate-operation")
@safety_check("validate_operation", SafetyLevel.READ_ONLY)
async def validate_operation_endpoint(operation: str, context: Optional[Dict[str, Any]] = None):
    """Validate if an operation is safe"""
    from safety import safety_validator
    
    is_safe = safety_validator.validate_operation(operation, context or {})
    return {
        "operation": operation,
        "is_safe": is_safe,
        "context": context,
        "message": "Operation allowed" if is_safe else "Operation blocked by safety system"
    }

# LDTS-29: Integration status endpoint
@app.get("/api/v1/integration/status")
@safety_check("integration_status", SafetyLevel.READ_ONLY)
async def integration_status():
    """Get integration layer status"""
    return {
        "integration": integration_layer.get_service_status(),
        "ml_resources_loaded": len(ml_resources),
        "services": {
            "weaviate": ml_resources.get("weaviate_search") is not None,
            "embedding_providers": len(ml_resources.get("embedding_providers", {})),
            "rerankers": len(ml_resources.get("rerankers", {}))
        }
    }

# LDTS-28: Configuration validation endpoint
@app.post("/api/v1/config/validate", response_model=ConfigValidationResponse)
@safety_check("config_validation", SafetyLevel.READ_ONLY)
async def validate_config_endpoint(config: Dict[str, Any]):
    """Validate dashboard configuration"""
    try:
        validation_result = validate_configuration(config)
        
        logger.info(f"Configuration validation: valid={validation_result.valid}, "
                   f"errors={len(validation_result.errors)}, "
                   f"warnings={len(validation_result.warnings)}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# LDTS-28: YAML configuration validation endpoint
@app.post("/api/v1/config/validate-yaml", response_model=ConfigValidationResponse)
@safety_check("config_validation_yaml", SafetyLevel.READ_ONLY)
async def validate_yaml_config_endpoint(yaml_content: str):
    """Validate YAML configuration content"""
    try:
        validation_result = validate_yaml_config(yaml_content)
        
        logger.info(f"YAML validation: valid={validation_result.valid}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"YAML validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"YAML validation failed: {str(e)}")

# LDTS-28: Configuration cost estimation endpoint
@app.post("/api/v1/config/estimate-cost")
@safety_check("config_cost_estimation", SafetyLevel.READ_ONLY)
async def estimate_config_cost(config: Dict[str, Any]):
    """Estimate costs for configuration"""
    try:
        validation_result = validate_configuration(config)
        
        if validation_result.cost_estimate:
            return {
                "valid": validation_result.valid,
                "cost_estimate": validation_result.cost_estimate,
                "performance_impact": validation_result.performance_impact,
                "errors": len(validation_result.errors),
                "warnings": len(validation_result.warnings)
            }
        else:
            return {
                "valid": False,
                "error": "Cannot estimate cost for invalid configuration",
                "errors": [error.dict() for error in validation_result.errors]
            }
            
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")

# LDTS-31: Audit logging endpoints
@app.get("/api/v1/audit/events")
@safety_check("audit_events_list", SafetyLevel.READ_ONLY)
async def get_audit_events(
    limit: int = Query(default=100, le=1000), 
    event_type: Optional[str] = None,
    operation: Optional[str] = None,
    user_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get audit events with filtering"""
    try:
        filters = {}
        if event_type:
            filters["event_type"] = event_type
        if operation:
            filters["operation"] = operation
        if user_id:
            filters["user_id"] = user_id
        if start_time:
            filters["start_time"] = start_time
        if end_time:
            filters["end_time"] = end_time
        
        events = await audit_logger.get_events(limit=limit, filters=filters)
        
        return {
            "events": events,
            "total": len(events),
            "filters": filters
        }
    except Exception as e:
        logger.error(f"Failed to retrieve audit events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit events: {str(e)}")

@app.get("/api/v1/audit/summary")
@safety_check("audit_summary", SafetyLevel.READ_ONLY)
async def get_audit_summary():
    """Get audit activity summary"""
    try:
        summary = await audit_logger.get_activity_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to generate audit summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audit summary: {str(e)}")

@app.post("/api/v1/audit/export")
@safety_check("audit_export", SafetyLevel.READ_ONLY)
async def export_audit_log(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    format: str = "json"
):
    """Export audit logs for compliance reporting"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        # Get filtered events
        filters = {}
        if start_time:
            filters["start_time"] = start_time
        if end_time:
            filters["end_time"] = end_time
        
        events = await audit_logger.get_events(limit=10000, filters=filters)
        
        # Log the export operation
        await audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            operation="audit_log_export",
            user_id="system",
            details={"format": format, "event_count": len(events), "filters": filters}
        )
        
        if format == "json":
            return {"export_data": events, "format": "json", "count": len(events)}
        else:
            # Convert to CSV format (simplified)
            csv_data = "timestamp,event_type,operation,user_id,result,duration_ms\n"
            for event in events:
                csv_data += f"{event.get('timestamp','')},{event.get('event_type','')},{event.get('operation','')},{event.get('user_id','')},{event.get('result','')},{event.get('duration_ms','')}\n"
            
            return {"export_data": csv_data, "format": "csv", "count": len(events)}
            
    except Exception as e:
        logger.error(f"Failed to export audit log: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export audit log: {str(e)}")

# LDTS-32: Rate limiting endpoints
@app.get("/api/v1/rate-limit/status")
@safety_check("rate_limit_status", SafetyLevel.READ_ONLY)
async def get_rate_limit_status():
    """Get overall rate limiting system status"""
    try:
        status = rate_limiter.get_system_stats()
        return status
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit status: {str(e)}")

@app.get("/api/v1/rate-limit/client/{client_id}")
@safety_check("rate_limit_client_stats", SafetyLevel.READ_ONLY)
async def get_client_rate_limit_stats(client_id: str):
    """Get rate limiting statistics for a specific client"""
    try:
        stats = rate_limiter.get_client_stats(client_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get client rate limit stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get client stats: {str(e)}")

@app.get("/api/v1/rate-limit/clients")
@safety_check("rate_limit_clients_list", SafetyLevel.READ_ONLY)
async def list_rate_limit_clients(limit: int = Query(default=50, le=1000)):
    """List all clients with their rate limiting statistics"""
    try:
        clients_stats = []
        for client_id, quota in list(rate_limiter.client_quotas.items())[:limit]:
            stats = rate_limiter.get_client_stats(client_id)
            clients_stats.append(stats)
        
        return {
            "clients": clients_stats,
            "total_clients": len(rate_limiter.client_quotas),
            "showing": len(clients_stats)
        }
    except Exception as e:
        logger.error(f"Failed to list rate limit clients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list clients: {str(e)}")

@app.post("/api/v1/rate-limit/test")
@safety_check("rate_limit_test", SafetyLevel.READ_ONLY)
async def test_rate_limit(request: Request):
    """Test rate limiting functionality"""
    try:
        # Check rate limit
        await check_rate_limit(request, "general")
        
        # Get client stats
        client_id = rate_limiter.get_client_id(request)
        stats = rate_limiter.get_client_stats(client_id)
        
        return {
            "message": "Request allowed",
            "client_id": client_id,
            "client_stats": stats,
            "timestamp": time.time()
        }
    except HTTPException as e:
        # Return rate limit error details
        return {
            "message": "Request blocked",
            "error": e.detail,
            "status_code": e.status_code,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Rate limit test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rate limit test failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LDTS Reranker Testing Dashboard API",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Different port from existing LDTS API server
        reload=True,
        log_level="info"
    )