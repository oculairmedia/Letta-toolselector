import json
import os
import sys
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests import Response

from letta_tool_utils import (
    get_find_tools_id_with_fallback,
    get_tool_selector_base_url,
    build_tool_selector_headers,
    get_tool_selector_timeout,
)


DEFAULT_LIMIT = 10
DEFAULT_MIN_SCORE = 50.0
MIN_LIMIT = 1
MAX_LIMIT = int(os.getenv('FIND_TOOLS_MAX_LIMIT', '25'))
MIN_SCORE_RANGE = (0.0, 100.0)
ATTACH_ENDPOINT = urljoin(get_tool_selector_base_url() + '/', 'api/v1/tools/attach')
REQUEST_HEADERS = build_tool_selector_headers()
REQUEST_TIMEOUT = get_tool_selector_timeout()


def _log(message: str) -> None:
    print(f"[find_tools] {message}", file=sys.stderr)


def _sanitize_limit(value: Optional[Any]) -> int:
    try:
        limit = int(value) if value is not None else DEFAULT_LIMIT
    except (TypeError, ValueError):
        _log(f"Invalid limit '{value}', using default {DEFAULT_LIMIT}")
        limit = DEFAULT_LIMIT
    return max(MIN_LIMIT, min(limit, MAX_LIMIT))


def _sanitize_min_score(value: Optional[Any]) -> float:
    try:
        score = float(value) if value is not None else DEFAULT_MIN_SCORE
    except (TypeError, ValueError):
        _log(f"Invalid min_score '{value}', using default {DEFAULT_MIN_SCORE}")
        score = DEFAULT_MIN_SCORE
    lower, upper = MIN_SCORE_RANGE
    return max(lower, min(score, upper))


def _prepare_keep_tools(keep_tools: Optional[str], agent_id: Optional[str]) -> list:
    keep_tool_ids = []

    find_tools_id = get_find_tools_id_with_fallback(agent_id=agent_id)
    if find_tools_id:
        keep_tool_ids.append(find_tools_id)
    else:
        _log("Warning: could not resolve find_tools ID; proceeding without auto-preserve entry")

    if keep_tools:
        for item in keep_tools.split(','):
            tool_id = item.strip()
            if tool_id and tool_id not in keep_tool_ids:
                keep_tool_ids.append(tool_id)

    return keep_tool_ids


def _request_attach(payload: Dict[str, Any]) -> Response:
    _log(f"POST {ATTACH_ENDPOINT} (limit={payload.get('limit')}, min_score={payload.get('min_score')})")
    return requests.post(
        ATTACH_ENDPOINT,
        headers=REQUEST_HEADERS,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )


def _build_success_response(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "message": result.get("message", "Tools updated successfully."),
        "details": result.get("details"),
    }


def _build_error_response(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = {
        "status": "error",
        "message": message,
    }
    if details:
        response["details"] = details
    return response


def Find_tools(
    query: str = None,
    agent_id: str = None,
    keep_tools: str = None,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    request_heartbeat: bool = False,
) -> str:
    """
    Silently manage tools for the agent.

    Args:
        query (str): Your search query - what kind of tool are you looking for?
        agent_id (str): Your agent ID
        keep_tools (str): Comma-separated list of tool IDs to preserve
        limit (int): Maximum number of tools to find (default: 10)
        min_score (float): Minimum match score 0-100 (default: 50.0)
        request_heartbeat (bool): Whether to request an immediate heartbeat (default: False)

    Returns:
        str: JSON string describing the outcome of the tool update
    """

    sanitized_limit = _sanitize_limit(limit)
    sanitized_min_score = _sanitize_min_score(min_score)
    keep_tool_ids = _prepare_keep_tools(keep_tools, agent_id)

    payload: Dict[str, Any] = {
        "limit": sanitized_limit,
        "min_score": sanitized_min_score,
        "keep_tools": keep_tool_ids,
        "request_heartbeat": bool(request_heartbeat),
    }

    if query is not None and query != "":
        payload["query"] = query
    if agent_id:
        payload["agent_id"] = agent_id

    try:
        response = _request_attach(payload)
    except requests.exceptions.Timeout:
        _log("Tool attach request timed out")
        return json.dumps(_build_error_response("Tool attach request timed out."), ensure_ascii=False)
    except requests.exceptions.RequestException as exc:
        _log(f"Tool attach request failed: {exc}")
        return json.dumps(_build_error_response("Failed to contact tool selector API."), ensure_ascii=False)

    try:
        result = response.json()
    except ValueError:
        _log("Received non-JSON response from tool selector API")
        result = {}

    try:
        if response.status_code == 200 and result.get("success"):
            summary = _build_success_response(result)
            return json.dumps(summary, ensure_ascii=False)

        error_message = result.get("error") or f"HTTP {response.status_code}"
        error_details = result.get("details") if isinstance(result, dict) else None
        _log(f"Tool attach failed: {error_message}")
        return json.dumps(_build_error_response(error_message, error_details), ensure_ascii=False)

    except Exception as exc:  # pragma: no cover - defensive guard
        _log(f"Unexpected error while formatting response: {exc}")
        return json.dumps(
            _build_error_response("Unexpected error while formatting response.", {"exception": str(exc)}),
            ensure_ascii=False,
        )

if __name__ == "__main__":
    # Get args from sys.argv without using argparse
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i+1].startswith('--'):
                args[key] = sys.argv[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    # Convert types
    limit_value = args.get('limit')
    min_score_value = args.get('min_score')
    request_heartbeat = args.get('request_heartbeat', 'false').lower() == 'true'

    result = Find_tools(
        query=args.get('query'),
        agent_id=args.get('agent_id'),
        keep_tools=args.get('keep_tools'),
        limit=limit_value,
        min_score=min_score_value,
        request_heartbeat=request_heartbeat
    )
    print(result)
