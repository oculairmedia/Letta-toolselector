"""Common utilities for interacting with the Tool Selector API.

This module centralises the shared logic used by both the legacy
``find_tools.py`` script and the new FastAPI worker service so that we
maintain consistent sanitisation, payload construction, and response
handling regardless of the execution environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

import requests
from requests import Response, Session

from letta_tool_utils import (
    build_tool_selector_headers,
    get_find_tools_id_with_fallback,
    get_tool_selector_base_url,
    get_tool_selector_timeout,
)


DEFAULT_LIMIT = 10
DEFAULT_MIN_SCORE = 50.0
MIN_LIMIT = 1
MAX_LIMIT = int(os.getenv("FIND_TOOLS_MAX_LIMIT", "25"))
MIN_SCORE_RANGE = (0.0, 100.0)

_LogFn = Optional[Callable[[str], None]]


def _default_log(message: str) -> None:
    print(f"[tool_selector] {message}", file=sys.stderr)


def _log(line: str, logger: _LogFn = None) -> None:
    if logger:
        logger(line)
    else:
        _default_log(line)


def get_attach_endpoint() -> str:
    base_url = get_tool_selector_base_url().rstrip("/")
    return urljoin(f"{base_url}/", "api/v1/tools/attach")


def sanitize_limit(value: Optional[Any], logger: _LogFn = None) -> int:
    try:
        limit = int(value) if value is not None else DEFAULT_LIMIT
    except (TypeError, ValueError):
        _log(f"Invalid limit '{value}', using default {DEFAULT_LIMIT}", logger)
        limit = DEFAULT_LIMIT
    return max(MIN_LIMIT, min(limit, MAX_LIMIT))


def sanitize_min_score(value: Optional[Any], logger: _LogFn = None) -> float:
    try:
        score = float(value) if value is not None else DEFAULT_MIN_SCORE
    except (TypeError, ValueError):
        _log(f"Invalid min_score '{value}', using default {DEFAULT_MIN_SCORE}", logger)
        score = DEFAULT_MIN_SCORE

    lower, upper = MIN_SCORE_RANGE
    if score < lower or score > upper:
        score = max(lower, min(score, upper))
        if logger:
            _log(
                f"min_score outside range {MIN_SCORE_RANGE}, clamped to {score}",
                logger,
            )
    return score


def prepare_keep_tools(
    keep_tools: Optional[str],
    agent_id: Optional[str],
    logger: _LogFn = None,
) -> list[str]:
    keep_tool_ids: list[str] = []

    find_tools_id = get_find_tools_id_with_fallback(agent_id=agent_id)
    if find_tools_id:
        keep_tool_ids.append(find_tools_id)
    else:
        _log(
            "Warning: could not resolve find_tools ID; proceeding without auto-preserve entry",
            logger,
        )

    if keep_tools:
        for item in keep_tools.split(","):
            tool_id = item.strip()
            if tool_id and tool_id not in keep_tool_ids:
                keep_tool_ids.append(tool_id)

    return keep_tool_ids


def build_payload(
    *,
    query: Optional[str],
    agent_id: Optional[str],
    keep_tools: Optional[str],
    limit: Optional[Any],
    min_score: Optional[Any],
    request_heartbeat: bool = False,
    logger: _LogFn = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "limit": sanitize_limit(limit, logger=logger),
        "min_score": sanitize_min_score(min_score, logger=logger),
        "keep_tools": prepare_keep_tools(keep_tools, agent_id, logger=logger),
        "request_heartbeat": bool(request_heartbeat),
    }

    if query is not None and query != "":
        payload["query"] = query
    if agent_id:
        payload["agent_id"] = agent_id

    return payload


def send_attach_request(
    payload: Dict[str, Any],
    *,
    session: Optional[Session] = None,
    logger: _LogFn = None,
) -> Response:
    endpoint = get_attach_endpoint()
    timeout = get_tool_selector_timeout()
    headers = build_tool_selector_headers()

    _log(
        f"POST {endpoint} (limit={payload.get('limit')}, min_score={payload.get('min_score')})",
        logger,
    )

    if session is not None:
        return session.post(endpoint, json=payload, timeout=timeout, headers=headers)

    return requests.post(endpoint, json=payload, timeout=timeout, headers=headers)


def build_success_response(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "message": result.get("message", "Tools updated successfully."),
        "details": result.get("details"),
    }


def build_error_response(
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = {
        "status": "error",
        "message": message,
    }
    if details:
        response["details"] = details
    return response


def format_attach_result(response: Response, logger: _LogFn = None) -> Dict[str, Any]:
    try:
        result = response.json()
    except ValueError:
        _log("Received non-JSON response from tool selector API", logger)
        result = {}

    if response.status_code == 200 and isinstance(result, dict) and result.get("success"):
        return build_success_response(result)

    if isinstance(result, dict):
        error_message = result.get("error") or f"HTTP {response.status_code}"
        error_details = result.get("details")
    else:
        error_message = f"HTTP {response.status_code}"
        error_details = None

    _log(f"Tool attach failed: {error_message}", logger)
    return build_error_response(error_message, error_details)


def attach_tools(
    *,
    query: Optional[str],
    agent_id: Optional[str],
    keep_tools: Optional[str],
    limit: Optional[Any],
    min_score: Optional[Any],
    request_heartbeat: bool,
    session: Optional[Session] = None,
    logger: _LogFn = None,
) -> Dict[str, Any]:
    payload = build_payload(
        query=query,
        agent_id=agent_id,
        keep_tools=keep_tools,
        limit=limit,
        min_score=min_score,
        request_heartbeat=request_heartbeat,
        logger=logger,
    )

    try:
        response = send_attach_request(payload, session=session, logger=logger)
    except requests.exceptions.Timeout:
        _log("Tool attach request timed out", logger)
        return build_error_response("Tool attach request timed out.")
    except requests.exceptions.RequestException as exc:
        _log(f"Tool attach request failed: {exc}", logger)
        return build_error_response("Failed to contact tool selector API.")

    try:
        return format_attach_result(response, logger=logger)
    except Exception as exc:  # pragma: no cover - defensive guard
        _log(f"Unexpected error while formatting response: {exc}", logger)
        return build_error_response(
            "Unexpected error while formatting response.",
            {"exception": str(exc)},
        )


def attach_tools_as_json(**kwargs: Any) -> str:
    """Helper to preserve existing script behaviour."""
    result = attach_tools(**kwargs)
    return json.dumps(result, ensure_ascii=False)
