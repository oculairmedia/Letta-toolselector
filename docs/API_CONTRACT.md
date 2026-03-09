# Tool Selector API Contract v1

**Version**: 1.0.0  
**Last Updated**: 2025-11-29  
**Status**: Active

## Overview

This document defines the contract between the **Tool Selector API** (`/api/v1/tools/attach`) and its clients (primarily webhook-receiver). Both sides MUST adhere to this contract to ensure consistent behavior.

---

## POST /api/v1/tools/attach

Automatically finds, attaches, and manages tools for a Letta agent based on a natural language query.

### Request Schema

#### Headers
```
Content-Type: application/json
Accept: application/json
```

#### Body (JSON)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `agent_id` | string (UUID) | **Yes** | - | Target Letta agent ID |
| `query` | string | No | `""` | Natural language query for tool search |
| `limit` | integer | No | `10` | Maximum number of tools to search (1-25) |
| `min_score` | number | No | `35.0` | Minimum relevance score (0-100) to attach |
| `keep_tools` | array[string] | No | `[]` | Tool IDs to preserve (never detach) |
| `auto_prune` | boolean | No | `true` | Whether to auto-prune after attachment |

**Example Request**:
```json
{
  "agent_id": "agent-123e4567-e89b-12d3-a456-426614174000",
  "query": "I need tools to search documents and create charts",
  "limit": 15,
  "min_score": 50.0,
  "keep_tools": ["tool-abc123", "tool-def456"],
  "auto_prune": true
}
```

**Validation Rules**:
- `agent_id`: MUST be a valid UUID string
- `limit`: MUST be between 1 and 25 (configurable via `FIND_TOOLS_MAX_LIMIT`)
- `min_score`: MUST be between 0.0 and 100.0
- `keep_tools`: MUST be an array of valid tool ID strings

---

### Response Schema

#### Success Response (200 OK)

```json
{
  "success": true,
  "message": "Successfully processed 15 candidates (12 passed min_score=50.0%), attached 8 tool(s) to agent agent-123",
  "details": {
    "detached_tools": ["tool-xyz789"],
    "failed_detachments": [],
    "processed_count": 15,
    "passed_filter_count": 12,
    "success_count": 8,
    "failure_count": 0,
    "successful_attachments": [
      {
        "success": true,
        "tool_id": "tool-abc123",
        "name": "document_search",
        "match_score": 87.5
      }
    ],
    "failed_attachments": [],
    "preserved_tools": ["tool-abc123", "tool-def456"],
    "target_agent": "agent-123e4567-e89b-12d3-a456-426614174000"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Overall operation success |
| `message` | string | Human-readable summary |
| `details` | object | Detailed results |
| `details.detached_tools` | array[string] | Tool IDs successfully detached |
| `details.failed_detachments` | array[string] | Tool IDs that failed to detach |
| `details.processed_count` | integer | Total tools from search |
| `details.passed_filter_count` | integer | Tools passing min_score filter |
| `details.success_count` | integer | Tools successfully attached |
| `details.failure_count` | integer | Tools that failed to attach |
| `details.successful_attachments` | array[object] | Details of attached tools |
| `details.failed_attachments` | array[object] | Details of failed attachments |
| `details.preserved_tools` | array[string] | Tool IDs explicitly preserved |
| `details.target_agent` | string | Agent ID operated on |

**Attachment Object Schema**:
```json
{
  "success": true,
  "tool_id": "tool-uuid",
  "name": "tool_name",
  "match_score": 87.5
}
```

#### Error Response (4xx/5xx)

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

**Common Error Codes**:
- `400 Bad Request` - Missing or invalid required fields
- `500 Internal Server Error` - Server-side processing error

---

## Behavior Specification

### Tool Attachment Logic

1. **Search Phase**:
   - Search Weaviate for tools matching the query
   - Return up to `limit` results
   - Filter results by `min_score` threshold

2. **Processing Phase**:
   - Check if tools are already registered with Letta
   - Auto-register unregistered MCP tools
   - Determine which tools to attach

3. **Pre-Attach Pruning Phase** (NEW in v1.1.0):
   - Calculate projected tool counts after attachment
   - If projected total > `MAX_TOTAL_TOOLS` OR projected MCP > `MAX_MCP_TOOLS`:
     - Perform preemptive pruning to make room
     - Calculate how many tools need removal
     - Respect `MIN_MCP_TOOLS` limit (won't go below minimum)
     - Use aggressive drop rate to clear space
     - Re-fetch agent tools after pruning
   - Skip if no query provided (can't rank relevance)

4. **Detachment Phase**:
   - Identify current MCP tools on agent
   - Detach tools NOT in:
     - `keep_tools` list
     - `NEVER_DETACH_TOOLS` / `PROTECTED_TOOLS` config
     - Newly matched tools
   - Respect `MIN_MCP_TOOLS` limit (won't detach below minimum)

5. **Attachment Phase**:
   - Attach filtered, processed tools in parallel
   - Track successes and failures separately

6. **Post-Attach Pruning Phase** (if `auto_prune=true`):
   - Run intelligent pruning based on relevance
   - Apply `DEFAULT_DROP_RATE` to reduce tool count
   - Enforce `MAX_TOTAL_TOOLS` and `MAX_MCP_TOOLS` limits

### Protected Tools

Tools are **never detached** if they match:
1. **Environment config**: `NEVER_DETACH_TOOLS` or `PROTECTED_TOOLS` (comma-separated names)
2. **Request parameter**: `keep_tools` array (by ID)
3. **Newly matched**: Tools just attached in the same request
4. **Core Letta tools**: When `EXCLUDE_LETTA_CORE_TOOLS=true`

### Tool Limits

| Limit | Default | Description |
|-------|---------|-------------|
| `MAX_TOTAL_TOOLS` | 30 | Maximum total tools per agent (all types) |
| `MAX_MCP_TOOLS` | 20 | Maximum external MCP tools per agent |
| `MIN_MCP_TOOLS` | 7 | Minimum MCP tools (pruning disabled below this) |

**Behavior when limits exceeded**:
- **Pre-attach pruning** (NEW in v1.1.0): Proactively removes tools BEFORE attachment if adding new tools would exceed limits
- **Post-attach pruning**: Selectively removes lowest-relevance tools after attachment
- Protected tools are never removed in either phase
- System warns if limits cannot be met due to protected tools

---

## Version History

### v1.1.0 (2025-11-29)
- **NEW**: Pre-attach pruning to enforce limits BEFORE attachment
- Prevents limit violations proactively rather than reactively
- Improved 6-phase workflow (was 5-phase)

### v1.0.0 (2025-11-29)
- Initial contract definition
- Formalized request/response schemas
- Documented protected tools behavior
- Specified tool limit enforcement

---

## References

- **Implementation**: `/opt/stacks/lettatoolsselector/tool-selector-api/api_server.py`
- **Client**: `/opt/stacks/letta-webhook-receiver-new` (webhook receiver)
- **Related**: See `LTSEL-9` in Huly for contract enforcement tracking
