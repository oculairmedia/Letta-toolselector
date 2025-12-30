"""
Structured audit logging for tool management operations.

Emits structured JSON events for tool attach/detach operations
that can be ingested by Loki, Elasticsearch, or other log aggregators.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

# Create a dedicated logger for audit events
audit_logger = logging.getLogger("tool_selector.audit")
audit_logger.setLevel(logging.INFO)

# Ensure audit logs are always INFO level regardless of root logger level
audit_logger.propagate = False

# Add console handler if not already present
if not audit_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    # Use JSON formatter for structured logging
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)


class AuditAction(str, Enum):
    """Tool management actions for audit trail."""
    ATTACH = "attach"
    DETACH = "detach"
    PRUNE = "prune"
    PROTECT = "protect"  # Tool was protected from detachment


class AuditSource(str, Enum):
    """Source of the tool management operation."""
    API_ATTACH = "api_attach_endpoint"
    API_PRUNE = "api_prune_endpoint"
    WEBHOOK = "webhook_trigger"
    SCHEDULED_JOB = "scheduled_pruning"
    MANUAL = "manual_operation"


def emit_tool_event(
    action: AuditAction,
    agent_id: str,
    tool_id: str,
    tool_name: Optional[str] = None,
    source: AuditSource = AuditSource.API_ATTACH,
    reason: Optional[str] = None,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """
    Emit a structured audit event for a tool operation.
    
    Args:
        action: The action performed (attach/detach/prune/protect)
        agent_id: Letta agent ID
        tool_id: Tool ID
        tool_name: Human-readable tool name
        source: Where the operation originated
        reason: Why the operation was performed
        correlation_id: ID to trace related operations (e.g. request ID)
        metadata: Additional context (scores, counts, etc.)
        success: Whether the operation succeeded
        error: Error message if failed
    """
    event = {
        "event_type": "tool_management",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action.value,
        "agent_id": agent_id,
        "tool_id": tool_id,
        "tool_name": tool_name or "unknown",
        "source": source.value,
        "success": success,
    }
    
    # Add optional fields
    if reason:
        event["reason"] = reason
    if correlation_id:
        event["correlation_id"] = correlation_id
    if metadata:
        event["metadata"] = metadata
    if error:
        event["error"] = error
    
    # Emit as structured JSON log
    audit_logger.info(json.dumps(event))


def emit_batch_event(
    action: AuditAction,
    agent_id: str,
    tools: List[Dict[str, Any]],
    source: AuditSource = AuditSource.API_ATTACH,
    reason: Optional[str] = None,
    correlation_id: Optional[str] = None,
    success_count: int = 0,
    failure_count: int = 0
) -> None:
    """
    Emit a structured audit event for a batch operation.
    
    Args:
        action: The action performed (attach/detach/prune)
        agent_id: Letta agent ID
        tools: List of tool dicts with id, name, success
        source: Where the operation originated
        reason: Why the operation was performed
        correlation_id: ID to trace related operations
        success_count: Number of successful operations
        failure_count: Number of failed operations
    """
    event = {
        "event_type": "tool_management_batch",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action.value,
        "agent_id": agent_id,
        "source": source.value,
        "tool_count": len(tools),
        "success_count": success_count,
        "failure_count": failure_count,
        "tools": [
            {
                "tool_id": t.get("tool_id") or t.get("id"),
                "tool_name": t.get("name", "unknown"),
                "success": t.get("success", True)
            }
            for t in tools
        ]
    }
    
    # Add optional fields
    if reason:
        event["reason"] = reason
    if correlation_id:
        event["correlation_id"] = correlation_id
    
    # Emit as structured JSON log
    audit_logger.info(json.dumps(event))


def emit_pruning_event(
    agent_id: str,
    tools_before: int,
    tools_after: int,
    tools_detached: List[str],
    tools_protected: List[str],
    drop_rate: float,
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Emit a structured audit event for a pruning operation.
    
    Args:
        agent_id: Letta agent ID
        tools_before: Tool count before pruning
        tools_after: Tool count after pruning
        tools_detached: List of tool IDs detached
        tools_protected: List of tool IDs protected from detachment
        drop_rate: Pruning aggressiveness (0.0-1.0)
        correlation_id: ID to trace related operations
        metadata: Additional context
    """
    event = {
        "event_type": "tool_pruning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "tools_before": tools_before,
        "tools_after": tools_after,
        "tools_detached_count": len(tools_detached),
        "tools_protected_count": len(tools_protected),
        "drop_rate": drop_rate,
        "tools_detached": tools_detached,
        "tools_protected": tools_protected,
    }
    
    if correlation_id:
        event["correlation_id"] = correlation_id
    if metadata:
        event["metadata"] = metadata
    
    # Emit as structured JSON log
    audit_logger.info(json.dumps(event))


def emit_limit_enforcement_event(
    agent_id: str,
    current_tools: int,
    max_limit: int,
    limit_type: str,
    enforcement_action: str,
    correlation_id: Optional[str] = None
) -> None:
    """
    Emit an event when tool limits are enforced.
    
    Args:
        agent_id: Letta agent ID
        current_tools: Current tool count
        max_limit: Maximum allowed tools
        limit_type: Type of limit (MAX_TOTAL_TOOLS, MAX_MCP_TOOLS, etc.)
        enforcement_action: What was done (pruned, rejected, warned)
        correlation_id: ID to trace related operations
    """
    event = {
        "event_type": "limit_enforcement",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "current_tools": current_tools,
        "max_limit": max_limit,
        "limit_type": limit_type,
        "enforcement_action": enforcement_action,
        "over_limit_by": max(0, current_tools - max_limit)
    }
    
    if correlation_id:
        event["correlation_id"] = correlation_id
    
    # Emit as structured JSON log
    audit_logger.info(json.dumps(event))


# Example usage and testing
if __name__ == "__main__":
    # Test event emission
    emit_tool_event(
        action=AuditAction.ATTACH,
        agent_id="agent-123",
        tool_id="tool-456",
        tool_name="document_search",
        source=AuditSource.API_ATTACH,
        reason="Matched user query",
        correlation_id="req-789",
        metadata={"match_score": 87.5, "query": "search documents"},
        success=True
    )
    
    emit_batch_event(
        action=AuditAction.DETACH,
        agent_id="agent-123",
        tools=[
            {"id": "tool-1", "name": "old_tool", "success": True},
            {"id": "tool-2", "name": "unused_tool", "success": True}
        ],
        source=AuditSource.API_PRUNE,
        reason="Pruning to enforce limits",
        correlation_id="req-789",
        success_count=2,
        failure_count=0
    )
    
    emit_pruning_event(
        agent_id="agent-123",
        tools_before=25,
        tools_after=15,
        tools_detached=["tool-1", "tool-2", "tool-3"],
        tools_protected=["find_tools", "send_message"],
        drop_rate=0.6,
        correlation_id="req-789",
        metadata={"threshold_met": True}
    )
