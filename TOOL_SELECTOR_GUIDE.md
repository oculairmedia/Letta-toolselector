# Letta Tool Selector Guide

## Overview

The Letta Tool Selector is an intelligent tool management system that enables Letta agents to dynamically discover, attach, and manage tools based on conversational context. It's exposed as an MCP (Model Context Protocol) server and integrates seamlessly with the Letta ecosystem.

## Architecture

### Components

1. **MCP Server** (`toolselector-mcp`): HTTP server on port 3020 that exposes the `find_tools` function
2. **Tool Discovery Service**: Backend service that searches and ranks tools based on semantic similarity
3. **Python Script** (`find_tools.py`): Core logic for tool management
4. **Enhanced Version** (`find_tools_enhanced.py`): Advanced features including detailed responses and tool rules

### Integration Points

- **Letta API**: Uses the tool attachment endpoint at `/api/v1/tools/attach`
- **Weaviate**: Vector database for semantic tool search
- **MCP Protocol**: Standard protocol for tool exposure to AI agents

## Tool Management Features

### Basic Tool Discovery

The `find_tools` function enables agents to search for and attach relevant tools:

```python
find_tools(
    query="web search browser",
    agent_id="agent-xxx",
    limit=5,
    min_score=75.0
)
```

### Advanced Features

#### 1. Detailed Response Mode

Get comprehensive information about tool changes:

```python
find_tools(
    query="data analysis",
    agent_id="agent-xxx",
    detailed_response=True
)
```

Returns:
```json
{
  "status": "success",
  "operation_id": "op_20250115_123456",
  "summary": "Attached 3 tools: pandas_analyzer, data_visualizer, csv_parser",
  "details": {
    "attached_tools": [...],
    "detached_tools": [...],
    "statistics": {...}
  },
  "recommendations": [
    "Consider also: excel_parser, sql_analyzer (high relevance scores)"
  ]
}
```

#### 2. Tool Dependency Rules

Automatically resolve tool dependencies:

```python
# When requesting data_analysis, automatically includes file_reader and csv_parser
TOOL_DEPENDENCIES = {
    "data_analysis": ["file_reader", "csv_parser"],
    "web_scraper": ["web_search", "html_parser"],
    "code_executor": ["syntax_checker", "security_scanner"]
}
```

#### 3. Tool Exclusion Rules

Prevent conflicting tools from being attached simultaneously:

```python
TOOL_EXCLUSIONS = {
    "local_file_system": ["cloud_storage"],
    "tool_v1": ["tool_v2"]
}
```

### Always-Attached Tools

The system always preserves the `find_tools` tool itself (ID: `tool-d4a7c168-3123-4b19-91b5-809320fdddf8`), ensuring agents can always discover new tools.

## Usage Patterns

### 1. Basic Tool Search

```python
# Simple search for web tools
find_tools(query="web search", agent_id="agent-xxx")
```

### 2. Preserving Existing Tools

```python
# Add new tools while keeping specific ones
find_tools(
    query="visualization",
    agent_id="agent-xxx",
    keep_tools="tool-analyzer-123,tool-parser-456"
)
```

### 3. High-Precision Search

```python
# Only attach highly relevant tools
find_tools(
    query="postgresql database",
    agent_id="agent-xxx",
    min_score=85.0,
    limit=3
)
```

### 4. Immediate Tool Usage

```python
# Request heartbeat for immediate tool availability
find_tools(
    query="file operations",
    agent_id="agent-xxx",
    request_heartbeat=True
)
```

## Best Practices

### Query Construction

1. **Use Multiple Keywords**: "csv excel spreadsheet data" vs just "csv"
2. **Include Synonyms**: "chart graph visualization plot"
3. **Specify Domain**: "python code execution" vs just "execution"

### Score Tuning

- **90-100**: Exact matches only
- **70-89**: High confidence matches (recommended for production)
- **50-69**: Moderate matches (good for exploration)
- **30-49**: Low confidence (experimental)

### Performance Optimization

1. **Limit Tools**: Keep active tools under 15 for optimal performance
2. **Regular Cleanup**: Detach unused tools periodically
3. **Batch Operations**: Combine related tool searches

### Tool Workflows

Common patterns for tool combinations:

- **Data Analysis**: file_reader → parser → analyzer → visualizer
- **Web Scraping**: web_search → scraper → parser → storage
- **Code Development**: syntax_checker → executor → debugger → formatter
- **Research**: search → summarizer → note_taker → report_generator

## Troubleshooting

### Common Issues

1. **No Tools Found**
   - Check query specificity
   - Lower min_score threshold
   - Verify tool availability in the system

2. **Too Many Tools Attached**
   - Increase min_score threshold
   - Reduce limit parameter
   - Use keep_tools to preserve only essential ones

3. **Conflicting Tools**
   - Check exclusion rules
   - Manually specify keep_tools
   - Use detailed_response to see what's happening

### Debug Mode

Enable detailed logging by using the enhanced version:

```bash
python find_tools_enhanced.py \
  --query "test" \
  --agent_id "agent-xxx" \
  --detailed true \
  --apply_rules true
```

## API Reference

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | None | Natural language search query |
| agent_id | string | None | Letta agent ID |
| keep_tools | string | None | Comma-separated tool IDs to preserve |
| limit | int | 10 | Maximum tools to find |
| min_score | float | 50.0 | Minimum relevance score (0-100) |
| request_heartbeat | bool | False | Request immediate heartbeat |
| detailed_response | bool | False | Return detailed information |

### Response Format

**Simple Response**:
```
"Tools updated successfully."
```

**Detailed Response**:
```json
{
  "status": "success|error",
  "operation_id": "op_YYYYMMDD_HHMMSS",
  "summary": "Human-readable summary",
  "details": {
    "attached_tools": [],
    "detached_tools": [],
    "kept_tools": [],
    "search_results": [],
    "statistics": {}
  },
  "recommendations": []
}
```

## Future Enhancements

1. **Memory Block Integration**: Track tool usage patterns in agent memory
2. **Multi-Agent Sharing**: Share successful tool combinations across agents
3. **Performance Metrics**: Track tool effectiveness and usage statistics
4. **Smart Caching**: Cache frequently used tool combinations
5. **Version Management**: Handle tool version updates gracefully

## Contributing

To extend the tool selector:

1. Add new dependency rules in `TOOL_DEPENDENCIES`
2. Define exclusion rules in `TOOL_EXCLUSIONS`
3. Implement custom ranking algorithms
4. Add telemetry for usage analytics

## Support

For issues or questions:
- Check logs: `docker logs toolselector-mcp`
- Verify MCP connection: `curl http://localhost:3020/health`
- Test tool discovery: Use the test scripts in this guide