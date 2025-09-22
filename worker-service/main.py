from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
import requests
from requests.adapters import HTTPAdapter

# Ensure the repository root and current directory are importable
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (CURRENT_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import settings  # noqa: E402  pylint: disable=wrong-import-position
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    FindToolsRequest,
    FindToolsResponse,
    HealthResponse,
)
from tool_selector_client import attach_tools  # noqa: E402  pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Letta Tools Worker Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=settings.pool_connections,
        pool_maxsize=settings.pool_maxsize,
        max_retries=settings.pool_max_retries,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = _build_session()


def _log_debug(message: str) -> None:
    logger.debug(message)


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    """Ensure pooled HTTP resources are released cleanly."""

    session.close()


@app.post("/find_tools", response_model=FindToolsResponse)
async def find_tools_endpoint(request: FindToolsRequest) -> FindToolsResponse:
    """Proxy the find_tools request using the persistent HTTP session."""

    logger.info(
        "Processing find_tools request (agent_id=%s, limit=%s, min_score=%s)",
        request.agent_id,
        request.limit,
        request.min_score,
    )

    try:
        result = await asyncio.to_thread(
            attach_tools,
            query=request.query,
            agent_id=request.agent_id,
            keep_tools=request.keep_tools,
            limit=request.limit,
            min_score=request.min_score,
            request_heartbeat=request.request_heartbeat,
            session=session,
            logger=_log_debug,
        )
    except Exception as exc:  # pragma: no cover - defensive safety net
        logger.exception("Unhandled exception while processing find_tools")
        raise HTTPException(status_code=500, detail="Unhandled worker service error") from exc

    logger.info("find_tools completed with status=%s", result.get("status"))
    return FindToolsResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple readiness probe used by Docker health checks."""

    return HealthResponse(status="healthy", service=settings.service_name)


@app.get("/config")
async def current_config() -> dict:
    """Expose a trimmed view of runtime configuration for debugging."""

    return {
        "log_level": settings.log_level,
        "service_name": settings.service_name,
        "pool_connections": settings.pool_connections,
        "pool_maxsize": settings.pool_maxsize,
        "pool_max_retries": settings.pool_max_retries,
    }
