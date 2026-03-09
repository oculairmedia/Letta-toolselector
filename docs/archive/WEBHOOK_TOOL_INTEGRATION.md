# Webhook Receiver and Tool Selector Integration Guide

## System Overview

This document explains how the Letta webhook receiver and tool selector services work together to provide dynamic tool attachment for agents based on conversational context.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  External Service   │     │  Webhook Receiver   │     │   Tool Selector     │
│  (Plane, Matrix,    │────▶│   (Port 5005)       │────▶│    API (8020)       │
│   Bookstack, etc.)  │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                      │                            │
                                      │                            │
                                      ▼                            ▼
                            ┌─────────────────────┐     ┌─────────────────────┐
                            │    Letta Agent      │     │     Weaviate        │
                            │    (Port 8283)      │     │  Vector Database    │
                            └─────────────────────┘     └─────────────────────┘
```

## Integration Flow

### 1. Webhook Reception (Port 5005)

When an external service sends a webhook to the receiver:

```python
POST /webhook
{
    "message": "I need to analyze customer data and create visualizations",
    "agent_id": "agent-xxx",
    "user": "john.doe",
    "metadata": {...}
}
```

### 2. Tool Discovery Process

The webhook receiver follows this sequence:

1. **Context Generation**
   - Calls Graphiti for semantic context (if enabled)
   - Creates memory blocks from external data sources

2. **Existing Tool Retrieval**
   ```python
   # Get current tools from Letta API
   existing_tools = GET http://localhost:8283/v1/agents/{agent_id}/tools
   ```

3. **Tool Attachment Request**
   ```python
   # Send to tool selector API
   POST http://192.168.50.90:8020/api/v1/tools/attach
   {
       "query": "analyze customer data and create visualizations",
       "agent_id": "agent-xxx",
       "keep_tools": ["tool-123", "tool-456"],  # Existing tools
       "limit": 3,
       "min_score": 70.0,
       "request_heartbeat": false
   }
   ```

4. **Response Processing**
   ```json
   {
       "success": true,
       "details": {
           "successful_attachments": [
               {
                   "tool_id": "tool-data-analyzer",
                   "name": "data_analyzer",
                   "match_score": 92
               },
               {
                   "tool_id": "tool-chart-creator",
                   "name": "visualization_tool",
                   "match_score": 88
               }
           ],
           "detached_tools": [],
           "preserved_tools": ["tool-123", "tool-456"],
           "processed_count": 10,
           "passed_filter_count": 2
       }
   }
   ```

## Tool Attachment Behavior

### Automatic Tool Discovery

The system automatically discovers and attaches tools based on:

1. **Semantic Matching**: Uses the webhook message content as a search query
2. **Relevance Scoring**: Only attaches tools with scores ≥ 70%
3. **Limit Control**: Maximum 3 new tools per webhook
4. **Tool Preservation**: Always keeps existing tools attached

### Tool Selection Criteria

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_score` | 70.0 | High confidence matches only |
| `limit` | 3 | Prevent tool overload |
| `keep_tools` | "*" | Preserve all existing tools |
| `request_heartbeat` | false | Async processing |

### Special Tool: find_tools

The system always preserves the `find_tools` tool (ID: `tool-d4a7c168-3123-4b19-91b5-809320fdddf8`), ensuring agents can discover additional tools during conversation.

## Integration Points

### 1. Webhook Receiver (`flask_webhook_receiver.py`)

```python
# Lines 1777-1804
if agent_id and original_prompt_for_logging:
    # Fetch existing tools
    existing_tool_ids = get_agent_tools(agent_id)
    
    # Attach relevant tools based on prompt
    tool_attachment_result = find_attach_tools(
        query=original_prompt_for_logging,
        agent_id=agent_id,
        limit=5,
        keep_tools=actual_keep_tools_str,
        request_heartbeat=True
    )
```

### 2. Tool Manager (`tool_manager.py`)

Handles communication with the tool selector API:
- Formats requests properly
- Parses detailed responses
- Logs tool attachment results
- Handles errors gracefully

### 3. Tool Selector API (Port 8020)

The backend service that:
- Performs semantic search in Weaviate
- Filters tools by score
- Manages tool attachments/detachments
- Returns detailed operation results

## MCP Server Integration (Port 3020)

A separate MCP server exposes the `find_tools` function for agents:

```
┌─────────────────────┐     ┌─────────────────────┐
│   Claude/Agent      │     │  Tool Selector MCP  │
│                     │────▶│    (Port 3020)      │
│                     │ MCP │                     │
└─────────────────────┘     └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │  Tool Selector API  │
                            │    (Port 8020)      │
                            └─────────────────────┘
```

### MCP vs Direct API

| Feature | Direct API (8020) | MCP Server (3020) |
|---------|------------------|-------------------|
| **Used By** | Webhook receivers | Agents via MCP |
| **Protocol** | REST/HTTP | MCP/JSON-RPC |
| **Features** | Basic attachment | Enhanced with rules |
| **Response** | Detailed JSON | Human-friendly + metadata |

## Common Scenarios

### Scenario 1: Project Management Webhook

```
Webhook: "New issue created: Implement user authentication"
↓
Tool Discovery Query: "Implement user authentication"
↓
Tools Attached:
- code_executor (85% match)
- security_scanner (78% match)
- documentation_generator (72% match)
```

### Scenario 2: Data Analysis Request

```
Webhook: "Analyze Q4 sales data and create report"
↓
Tool Discovery Query: "Analyze Q4 sales data and create report"
↓
Tools Attached:
- data_analyzer (95% match)
- report_generator (90% match)
- visualization_tool (82% match)
```

### Scenario 3: Communication Task

```
Webhook: "Schedule team meeting and send invitations"
↓
Tool Discovery Query: "Schedule team meeting and send invitations"
↓
Tools Attached:
- calendar_tool (92% match)
- email_sender (88% match)
- meeting_scheduler (85% match)
```

## Configuration

### Environment Variables

```bash
# Webhook Receiver
LETTA_API_URL=http://localhost:8283
TOOL_SELECTOR_URL=http://192.168.50.90:8020

# Tool Selector
WEAVIATE_URL=http://weaviate:8080
MIN_SCORE_DEFAULT=70.0
MAX_TOOLS_DEFAULT=3
```

### Tool Attachment Settings

```python
# Default configuration in webhook receiver
TOOL_ATTACHMENT_CONFIG = {
    "enabled": True,
    "min_score": 70.0,
    "max_tools": 3,
    "preserve_existing": True,
    "request_heartbeat": False
}
```

## Best Practices

### 1. Query Optimization

- Use the full webhook message as the query
- Include relevant keywords from the context
- Let semantic search handle synonyms

### 2. Tool Limits

- Keep limits reasonable (3-5 tools)
- Higher limits may slow agent responses
- Consider tool dependencies

### 3. Score Thresholds

- 70%+ for production use
- 50-70% for experimental features
- <50% may attach irrelevant tools

### 4. Monitoring

- Check webhook logs for tool attachment results
- Monitor tool usage patterns
- Track attachment success rates

## Troubleshooting

### No Tools Attached

1. Check if query is too specific
2. Verify tool selector service is running
3. Lower min_score threshold
4. Check Weaviate connectivity

### Too Many Tools

1. Increase min_score threshold
2. Reduce limit parameter
3. Review tool relevance

### Tools Not Working

1. Verify tool IDs are correct
2. Check agent permissions
3. Ensure tools are properly registered

## Advanced Features

### Tool Dependencies (MCP Server Only)

```python
TOOL_DEPENDENCIES = {
    "data_analysis": ["file_reader", "csv_parser"],
    "web_scraper": ["web_search", "html_parser"]
}
```

### Tool Exclusions (MCP Server Only)

```python
TOOL_EXCLUSIONS = {
    "local_file_system": ["cloud_storage"],
    "tool_v1": ["tool_v2"]
}
```

### Custom Scoring

The tool selector uses:
- Semantic similarity (Weaviate vectors)
- Keyword matching
- Tool metadata relevance

## Future Enhancements

1. **Unified Protocol**: Migrate webhook receiver to use MCP
2. **Tool Learning**: Track which tools are most useful
3. **Context Awareness**: Use conversation history for better matches
4. **Performance Metrics**: Monitor tool effectiveness
5. **Dynamic Rules**: Adjust attachment rules based on patterns

## Related Documentation

- [Tool Selector Guide](./TOOL_SELECTOR_GUIDE.md)
- [Letta API Documentation](https://docs.letta.com/api-reference)
- [MCP Protocol Specification](https://modelcontextprotocol.io/docs)