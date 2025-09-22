import json
import os
import requests
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urljoin

from letta_tool_utils import (
    get_find_tools_id_with_fallback,
    get_tool_selector_base_url,
    build_tool_selector_headers,
    get_tool_selector_timeout,
)


DEFAULT_LIMIT = 10
DEFAULT_MIN_SCORE = 50.0
MAX_LIMIT = int(os.getenv('FIND_TOOLS_MAX_LIMIT', '25'))
MIN_LIMIT = 1
MIN_SCORE_RANGE = (0.0, 100.0)
ATTACH_ENDPOINT = urljoin(get_tool_selector_base_url() + '/', 'api/v1/tools/attach')
REQUEST_HEADERS = build_tool_selector_headers()
REQUEST_TIMEOUT = get_tool_selector_timeout()

def _log(message: str) -> None:
    print(f"[find_tools_enhanced] {message}", file=sys.stderr)


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


def _prepare_keep_tools(keep_tools: Optional[str], agent_id: Optional[str]) -> List[str]:
    keep_tool_ids: List[str] = []

    find_tools_id = get_find_tools_id_with_fallback(agent_id=agent_id)
    if find_tools_id:
        keep_tool_ids.append(find_tools_id)
    else:
        _log("Warning: could not resolve find_tools ID; continuing without auto-preserve entry")

    if keep_tools:
        for item in keep_tools.split(','):
            tool_id = item.strip()
            if tool_id and tool_id not in keep_tool_ids:
                keep_tool_ids.append(tool_id)

    return keep_tool_ids


def _request_attach(payload: Dict[str, Any]) -> requests.Response:
    _log(
        "POST %s (limit=%s, min_score=%s)" % (
            ATTACH_ENDPOINT,
            payload.get('limit'),
            payload.get('min_score'),
        )
    )
    return requests.post(
        ATTACH_ENDPOINT,
        headers=REQUEST_HEADERS,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )


def _build_simple_success_response(result: Dict[str, Any]) -> str:
    payload = {
        "status": "success",
        "message": result.get("message", "Tools updated successfully."),
        "details": result.get("details"),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_error_response(message: str, details: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {
        "status": "error",
        "message": message,
    }
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def Find_tools(
    query: str = None,
    agent_id: str = None,
    keep_tools: str = None,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    request_heartbeat: bool = False,
    detailed_response: bool = False,
) -> str:
    """
    Intelligently manage tools for the agent with detailed feedback.

    Args:
        query (str): Your search query - what kind of tool are you looking for?
        agent_id (str): Your agent ID
        keep_tools (str): Comma-separated list of tool IDs to preserve
        limit (int): Maximum number of tools to find (default: 10)
        min_score (float): Minimum match score 0-100 (default: 50.0)
        request_heartbeat (bool): Whether to request an immediate heartbeat (default: False)
        detailed_response (bool): Return detailed information about tool changes (default: False)

    Returns:
        str: Detailed response with tool changes or simple success message
    """
    keep_tool_ids = _prepare_keep_tools(keep_tools, agent_id)

    # Track operation metadata
    operation_id = f"op_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sanitized_limit = _sanitize_limit(limit)
    sanitized_min_score = _sanitize_min_score(min_score)

    payload = {
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
        error_payload = _build_error_response("Tool attach request timed out.")
        return error_payload
    except requests.exceptions.RequestException as exc:
        _log(f"Tool attach request failed: {exc}")
        error_payload = _build_error_response("Failed to contact tool selector API.")
        return error_payload

    try:
        result = response.json()
    except ValueError:
        _log("Received non-JSON response from tool selector API")
        result = {}

    print(f"[{operation_id}] Tool Attach Operation", file=sys.stderr)
    print(f"Query: {query}", file=sys.stderr)
    print(f"Agent: {agent_id}", file=sys.stderr)
    print(f"Response:", file=sys.stderr)
    print(json.dumps(result, indent=2), file=sys.stderr)

    try:
        if response.status_code == 200 and result.get("success"):
            if detailed_response and "details" in result:
                details = result.get("details", {})
                response_data = {
                    "status": "success",
                    "operation_id": operation_id,
                    "summary": _build_summary_from_details(details),
                    "details": {
                        "attached_tools": details.get("successful_attachments", []),
                        "detached_tools": details.get("detached_tools", []),
                        "kept_tools": details.get("preserved_tools", []),
                        "failed_attachments": details.get("failed_attachments", []),
                        "statistics": {
                            "total_attached": len(details.get("successful_attachments", [])),
                            "total_detached": len(details.get("detached_tools", [])),
                            "total_kept": len(details.get("preserved_tools", [])),
                            "processed": details.get("processed_count", 0),
                            "passed_filter": details.get("passed_filter_count", 0)
                        }
                    },
                    "recommendations": _generate_recommendations_from_details(details, query)
                }
                return json.dumps(response_data, indent=2)

            return _build_simple_success_response(result)

        error = result.get("error", f"HTTP {response.status_code}")
        error_payload = {
            "status": "error",
            "operation_id": operation_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        if detailed_response:
            return json.dumps(error_payload, indent=2)
        return _build_error_response(error, result.get("details") if isinstance(result, dict) else None)

    except Exception as exc:  # pragma: no cover - defensive guard
        _log(f"[{operation_id}] Unexpected error while formatting response: {exc}")
        if detailed_response:
            return json.dumps({
                "status": "error",
                "operation_id": operation_id,
                "error": "Unexpected error while formatting response.",
                "exception": str(exc),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        return _build_error_response("Unexpected error while formatting response.", {"exception": str(exc)})


def _build_summary(data: Dict[str, Any]) -> str:
    """Build a human-readable summary of the tool changes."""
    attached = data.get("attached", [])
    detached = data.get("detached", [])
    kept = data.get("kept", [])
    
    summary_parts = []
    if attached:
        tool_names = [t.get("name", t.get("id", "unknown")) for t in attached[:3]]
        more = f" and {len(attached) - 3} more" if len(attached) > 3 else ""
        summary_parts.append(f"Attached {len(attached)} tools: {', '.join(tool_names)}{more}")
    
    if detached:
        summary_parts.append(f"Detached {len(detached)} tools")
    
    if kept:
        summary_parts.append(f"Kept {len(kept)} existing tools")
    
    return ". ".join(summary_parts) if summary_parts else "No tool changes made."


def _build_summary_from_details(details: Dict[str, Any]) -> str:
    """Build a human-readable summary from actual API response details."""
    attached = details.get("successful_attachments", [])
    detached = details.get("detached_tools", [])
    preserved = details.get("preserved_tools", [])
    
    summary_parts = []
    if attached:
        tool_names = [t.get("name", t.get("tool_id", "unknown")) for t in attached[:3]]
        more = f" and {len(attached) - 3} more" if len(attached) > 3 else ""
        summary_parts.append(f"Attached {len(attached)} tools: {', '.join(tool_names)}{more}")
    
    if detached:
        summary_parts.append(f"Detached {len(detached)} tools")
    
    if preserved:
        summary_parts.append(f"Preserved {len(preserved)} existing tools")
    
    return ". ".join(summary_parts) if summary_parts else "No tool changes made."


def _generate_recommendations(data: Dict[str, Any], query: Optional[str]) -> List[str]:
    """Generate recommendations based on the operation results."""
    recommendations = []
    
    search_results = data.get("search_results", [])
    attached = data.get("attached", [])
    
    # Check if we found what we were looking for
    if query and not attached:
        recommendations.append(f"No tools matched '{query}'. Try a broader search term or check available tool categories.")
    
    # Suggest related tools
    if search_results:
        high_score_tools = [t for t in search_results if t.get("score", 0) > 80 and t not in attached]
        if high_score_tools:
            tool_names = [t.get("name", "unknown") for t in high_score_tools[:2]]
            recommendations.append(f"Consider also: {', '.join(tool_names)} (high relevance scores)")
    
    # Warn about tool limits
    total_tools = len(attached) + len(data.get("kept", []))
    if total_tools > 15:
        recommendations.append(f"You have {total_tools} tools attached. Consider detaching unused tools for better performance.")
    
    return recommendations


def _generate_recommendations_from_details(details: Dict[str, Any], query: Optional[str]) -> List[str]:
    """Generate recommendations based on actual API response details."""
    recommendations = []
    
    attached = details.get("successful_attachments", [])
    failed = details.get("failed_attachments", [])
    processed = details.get("processed_count", 0)
    passed_filter = details.get("passed_filter_count", 0)
    
    # Check if we found what we were looking for
    if query and not attached and processed > 0:
        recommendations.append(f"No tools matched '{query}' with your criteria. Try lowering min_score or using broader search terms.")
    
    # Warn about failed attachments
    if failed:
        recommendations.append(f"{len(failed)} tools failed to attach. Check agent permissions or tool availability.")
    
    # Suggest filter adjustments
    if processed > 0 and passed_filter < processed / 2:
        recommendations.append(f"Only {passed_filter} of {processed} tools passed the score filter. Consider lowering min_score for more results.")
    
    # Performance tip
    total_tools = len(attached) + len(details.get("preserved_tools", []))
    if total_tools > 15:
        recommendations.append(f"You have {total_tools} tools attached. Consider detaching unused tools for better performance.")
    
    return recommendations


# Enhanced version with tool dependency support
def Find_tools_with_rules(query: str = None, agent_id: str = None, keep_tools: str = None,
                         limit: int = 10, min_score: float = 50.0, request_heartbeat: bool = False,
                         apply_rules: bool = True) -> str:
    """
    Enhanced version that supports tool dependency rules.
    
    Additional Args:
        apply_rules (bool): Apply tool dependency and exclusion rules (default: True)
    """
    # Tool dependency rules
    TOOL_DEPENDENCIES = {
        "data_analysis": ["file_reader", "csv_parser"],
        "web_scraper": ["web_search", "html_parser"],
        "code_executor": ["syntax_checker", "security_scanner"]
    }
    
    # Tool exclusions (mutually exclusive tools)
    TOOL_EXCLUSIONS = {
        "tool_v1": ["tool_v2"],
        "local_file_system": ["cloud_storage"],
    }
    
    # First, run the standard find_tools
    initial_result = Find_tools(
        query=query,
        agent_id=agent_id,
        keep_tools=keep_tools,
        limit=limit,
        min_score=min_score,
        request_heartbeat=False,  # We'll handle this after applying rules
        detailed_response=True
    )
    
    if not apply_rules:
        return initial_result
    
    try:
        result_data = json.loads(initial_result)
        if result_data.get("status") != "success":
            return initial_result
        
        # Apply dependency rules
        attached_tools = result_data["details"]["attached_tools"]
        additional_tools = []
        
        for tool in attached_tools:
            tool_name = tool.get("name", "")
            if tool_name in TOOL_DEPENDENCIES:
                deps = TOOL_DEPENDENCIES[tool_name]
                additional_tools.extend(deps)
        
        # Apply exclusion rules
        tools_to_remove = []
        for tool in attached_tools:
            tool_name = tool.get("name", "")
            if tool_name in TOOL_EXCLUSIONS:
                exclusions = TOOL_EXCLUSIONS[tool_name]
                tools_to_remove.extend(exclusions)
        
        # If we need to make additional changes, run find_tools again
        if additional_tools or tools_to_remove:
            # Build new keep_tools list
            current_tools = [t.get("id") for t in attached_tools + result_data["details"]["kept_tools"]]
            for remove in tools_to_remove:
                current_tools = [t for t in current_tools if not t.endswith(remove)]
            
            # Add dependencies
            if additional_tools:
                dependency_query = " OR ".join(additional_tools)
                return Find_tools(
                    query=dependency_query,
                    agent_id=agent_id,
                    keep_tools=",".join(current_tools),
                    limit=len(additional_tools),
                    min_score=30.0,  # Lower threshold for dependencies
                    request_heartbeat=request_heartbeat,
                    detailed_response=True
                )
        
        # If no additional changes needed, return original result
        if request_heartbeat:
            # Trigger heartbeat separately
            _request_heartbeat(agent_id)
        
        return initial_result
        
    except json.JSONDecodeError:
        # If we can't parse the result, return as-is
        return initial_result


def _request_heartbeat(agent_id: str):
    """Request an immediate heartbeat for the agent."""
    # This would call the appropriate Letta API endpoint
    # For now, we'll just log it
    print(f"Requesting heartbeat for agent {agent_id}", file=sys.stderr)


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
    limit = int(args.get('limit', '10'))
    min_score = float(args.get('min_score', '50.0'))
    request_heartbeat = args.get('request_heartbeat', 'false').lower() == 'true'
    detailed_response = args.get('detailed', 'false').lower() == 'true'
    apply_rules = args.get('apply_rules', 'true').lower() == 'true'

    # Use enhanced version if rules are requested
    if apply_rules and 'no_rules' not in args:
        result = Find_tools_with_rules(
            query=args.get('query'),
            agent_id=args.get('agent_id'),
            keep_tools=args.get('keep_tools'),
            limit=limit,
            min_score=min_score,
            request_heartbeat=request_heartbeat,
            apply_rules=True
        )
    else:
        result = Find_tools(
            query=args.get('query'),
            agent_id=args.get('agent_id'),
            keep_tools=args.get('keep_tools'),
            limit=limit,
            min_score=min_score,
            request_heartbeat=request_heartbeat,
            detailed_response=detailed_response
        )
    
    print(result)
