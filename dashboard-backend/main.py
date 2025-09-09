"""
LDTS Reranker Testing Dashboard Backend API

FastAPI server with async support for ML workloads.
Provides testing endpoints for search, reranking, and configuration management.

LDTS-26: Set up FastAPI server with async support for ML workloads
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os
from pathlib import Path

# Import existing LDTS services
import sys
sys.path.append(str(Path(__file__).parent.parent / "lettaaugment-source"))

from api_server import app as existing_app
from weaviate_tool_search_with_reranking import WeaviateToolSearch
from safety import safety_check, SafetyLevel, verify_production_isolation, get_safety_status
from search_endpoints import router as search_router
from integration_layer import integration_layer
from config_validation import validate_configuration, validate_yaml_config, ConfigValidationResponse
from audit_logging import AuditLogger, AuditEventType, audit_logger
from rate_limiting import rate_limiter, check_rate_limit, get_request_context, validate_request_size

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

# Include search endpoints router
app.include_router(search_router)

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
        
        # Initialize audit logging system
        await audit_logger.initialize()
        
        # Initialize rate limiting system
        await rate_limiter.initialize()
        
        # Store integration layer reference
        ml_resources["integration_layer"] = integration_layer
        ml_resources["weaviate_search"] = integration_layer.weaviate_client
        ml_resources["embedding_providers"] = integration_layer.embedding_providers
        ml_resources["rerankers"] = integration_layer.reranker_services
        ml_resources["audit_logger"] = audit_logger
        ml_resources["rate_limiter"] = rate_limiter
        
        logger.info("ML resources initialized successfully")
        logger.info("SAFETY MODE: Read-only testing mode enforced")
        logger.info(f"Integration status: {integration_layer.get_service_status()}")
        
        # Log system startup
        await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            operation="dashboard_startup",
            user_id="system",
            details={"version": "0.1.0", "safety_mode": "read_only_testing"}
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
        "safety_mode": "read_only_testing"
    }

# LDTS-30: Safety status endpoint
@app.get("/api/v1/safety/status")
@safety_check("safety_status", SafetyLevel.READ_ONLY)
async def safety_status():
    """Get current safety system status"""
    return get_safety_status()

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