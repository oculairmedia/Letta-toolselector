from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import time
from typing import Dict, Any

from config.settings import settings
from app.middleware.rate_limiting import RateLimitingMiddleware
from app.middleware.safety import SafetyMiddleware
from app.routers import search, rerank, config as config_router, health
from app.services.ldts_client import LDTSClient
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global services
ldts_client: LDTSClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting LDTS Reranker Testing Dashboard Backend...")
    
    # Initialize LDTS client
    global ldts_client
    ldts_client = LDTSClient()
    await ldts_client.initialize()
    
    # Verify connectivity to LDTS services
    try:
        health_status = await ldts_client.check_health()
        logger.info(f"LDTS services health check: {health_status}")
    except Exception as e:
        logger.warning(f"LDTS services health check failed: {e}")
    
    logger.info("Backend startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")
    if ldts_client:
        await ldts_client.close()
    logger.info("Backend shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

# Add middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

# Add trusted host middleware in production
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # Configure appropriately for production
    )

# Add safety middleware (read-only enforcement)
app.add_middleware(SafetyMiddleware)

# Add rate limiting middleware
if settings.ENABLE_RATE_LIMITING:
    app.add_middleware(RateLimitingMiddleware)

# Dependency to get LDTS client
async def get_ldts_client() -> LDTSClient:
    """Dependency to get the LDTS client instance."""
    if ldts_client is None:
        raise HTTPException(
            status_code=503, 
            detail="LDTS client not initialized"
        )
    return ldts_client

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "type": "http_error",
                "timestamp": time.time()
            }
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": "internal_error",
            "timestamp": time.time()
        }
    )

# Include routers
app.include_router(health.router, prefix=f"{settings.API_V1_STR}")
app.include_router(config_router.router, prefix=f"{settings.API_V1_STR}")
app.include_router(search.router, prefix=f"{settings.API_V1_STR}")
app.include_router(rerank.router, prefix=f"{settings.API_V1_STR}")

# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "api_docs": f"{settings.API_V1_STR}/docs",
        "health": f"{settings.API_V1_STR}/health",
        "read_only_mode": settings.READ_ONLY_MODE,
        "timestamp": time.time()
    }

# API version info
@app.get(f"{settings.API_V1_STR}/info")
async def api_info() -> Dict[str, Any]:
    """Get API version and configuration information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_version": settings.API_V1_STR,
        "read_only_mode": settings.READ_ONLY_MODE,
        "rate_limiting_enabled": settings.ENABLE_RATE_LIMITING,
        "reranking_enabled": settings.ENABLE_RERANKING,
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENABLE_ACCESS_LOGS
    )