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
    AttachToolRequest,
    DetachToolRequest,
    FindToolsRequest,
    FindToolsResponse,
    HealthResponse,
    InspectToolRequest,
    ListAgentToolsRequest,
    LookupToolRequest,
)
from tool_selector_client import attach_tools  # noqa: E402  pylint: disable=wrong-import-position
from letta_tool_utils import (  # noqa: E402  pylint: disable=wrong-import-position
    build_tool_selector_headers,
    get_tool_selector_base_url,
    get_tool_selector_timeout,
)

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


# ── Granular tool management endpoints ──────────────────────────────────



def _api_base_url() -> str:
    return get_tool_selector_base_url().rstrip("/")


def _api_request_kwargs() -> dict:
    """Common kwargs for proxied requests to the API server."""
    return {
        "headers": {**build_tool_selector_headers(), "Content-Type": "application/json"},
        "timeout": get_tool_selector_timeout(),
    }


def _proxy_get(path: str, params: dict | None = None) -> dict:
    """Synchronous GET proxy to the API server."""
    url = f"{_api_base_url()}/{path.lstrip('/')}"
    kwargs = _api_request_kwargs()
    if params:
        kwargs["params"] = params
    resp = session.get(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _proxy_post(path: str, json_body: dict) -> dict:
    """Synchronous POST proxy to the API server."""
    url = f"{_api_base_url()}/{path.lstrip('/')}"
    kwargs = _api_request_kwargs()
    kwargs["json"] = json_body
    resp = session.post(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


@app.post("/lookup_tool")
async def lookup_tool_endpoint(request: LookupToolRequest) -> dict:
    """Proxy tool lookup to GET /api/v1/tools/lookup."""
    logger.info("Processing lookup_tool (name=%s, id=%s, fuzzy=%s)", request.tool_name, request.tool_id, request.fuzzy)
    try:
        params: dict = {}
        if request.tool_name:
            params["tool_name"] = request.tool_name
        if request.tool_id:
            params["tool_id"] = request.tool_id
        if request.fuzzy:
            params["fuzzy"] = "true"
            params["limit"] = str(request.limit)
        result = await asyncio.to_thread(_proxy_get, "api/v1/tools/lookup", params)
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        try:
            return exc.response.json()  # type: ignore[union-attr]
        except Exception:
            raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled exception in lookup_tool")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/attach_tool")
async def attach_tool_endpoint(request: AttachToolRequest) -> dict:
    """Proxy direct tool attach to POST /api/v1/tools/direct-attach."""
    logger.info("Processing attach_tool (agent=%s, tools=%d, pin=%s)", request.agent_id, len(request.tools), request.pin)
    try:
        payload = {
            "agent_id": request.agent_id,
            "tools": [t.model_dump(exclude_none=True) for t in request.tools],
            "pin": request.pin,
        }
        result = await asyncio.to_thread(_proxy_post, "api/v1/tools/direct-attach", payload)
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        try:
            return exc.response.json()  # type: ignore[union-attr]
        except Exception:
            raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled exception in attach_tool")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detach_tool")
async def detach_tool_endpoint(request: DetachToolRequest) -> dict:
    """Proxy direct tool detach to POST /api/v1/tools/direct-detach."""
    logger.info("Processing detach_tool (agent=%s, tools=%d, unpin=%s)", request.agent_id, len(request.tools), request.unpin)
    try:
        payload = {
            "agent_id": request.agent_id,
            "tools": [t.model_dump(exclude_none=True) for t in request.tools],
            "unpin": request.unpin,
        }
        result = await asyncio.to_thread(_proxy_post, "api/v1/tools/direct-detach", payload)
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        try:
            return exc.response.json()  # type: ignore[union-attr]
        except Exception:
            raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled exception in detach_tool")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/list_agent_tools")
async def list_agent_tools_endpoint(request: ListAgentToolsRequest) -> dict:
    """Proxy agent tool listing to GET /api/v1/tools/agent/{agent_id}."""
    logger.info("Processing list_agent_tools (agent=%s, filter=%s)", request.agent_id, request.filter)
    try:
        params: dict = {}
        if request.filter != "all":
            params["filter"] = request.filter
        if request.include_schema:
            params["include_schema"] = "true"
        result = await asyncio.to_thread(
            _proxy_get, f"api/v1/tools/agent/{request.agent_id}", params
        )
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        try:
            return exc.response.json()  # type: ignore[union-attr]
        except Exception:
            raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled exception in list_agent_tools")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/inspect_tool")
async def inspect_tool_endpoint(request: InspectToolRequest) -> dict:
    """Proxy tool inspection to GET /api/v1/tools/inspect/{tool_name_or_id}."""
    logger.info("Processing inspect_tool (tool=%s)", request.tool_name_or_id)
    try:
        result = await asyncio.to_thread(
            _proxy_get, f"api/v1/tools/inspect/{request.tool_name_or_id}"
        )
        return result
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        try:
            return exc.response.json()  # type: ignore[union-attr]
        except Exception:
            raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Unhandled exception in inspect_tool")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
