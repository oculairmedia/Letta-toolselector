# Never-Detach Tool Protection

## Overview

This document describes the enhanced auto-pruning system that protects specific tools from being automatically detached during tool management operations.

## Problem

The original auto-pruning system in `api_server.py` would sometimes detach important tools like `find_tools` that are meant to stay permanently attached to agents. This happened because:

1. The `find_tools` tool would call the attachment API
2. The attachment API would trigger auto-pruning after successful attachments
3. The pruning system would consider `find_tools` as a regular MCP tool subject to removal
4. Based on relevance scoring and tool limits, `find_tools` could get detached

## Solution

The auto-pruning system has been enhanced with a **never-detach protection mechanism** that works at multiple levels:

### 1. Environment Variable Configuration

A new environment variable `NEVER_DETACH_TOOLS` allows configuring tools that should never be detached:

```bash
# Default value (protects find_tools)
NEVER_DETACH_TOOLS=find_tools

# Multiple tools (comma-separated)
NEVER_DETACH_TOOLS=find_tools,important_tool,another_tool
```

### 2. Enhanced Tool Categorization

The tool categorization logic in `_perform_tool_pruning()` now identifies protected tools and moves them to the "core tools" category, making them immune to pruning:

```python
# Check if this tool should never be detached
is_never_detach_tool = (
    tool_id in requested_keep_tool_ids or 
    tool_id in requested_newly_matched_tool_ids or
    any(never_detach_name.lower() in tool_name for never_detach_name in NEVER_DETACH_TOOLS)
)

if is_never_detach_tool or (MANAGE_ONLY_MCP_TOOLS and is_letta_core_tool):
    # Protect never-detach tools and keep list tools
    core_tools_on_agent.append(tool)
```

### 3. Multiple Protection Mechanisms

Tools are protected if they meet ANY of these criteria:

- **Tool ID is in the keep list**: Explicitly passed via `keep_tool_ids` parameter
- **Tool ID is newly matched**: Passed via `newly_matched_tool_ids` parameter  
- **Tool name matches never-detach list**: Tool name contains any name from `NEVER_DETACH_TOOLS`

### 4. Fallback Safeguards

An additional safeguard protects any never-detach tools that somehow end up in the MCP tools list:

```python
# Additional safeguard: protect any never-detach tools that might still be in MCP list
for tool in mcp_tools_on_agent_list:
    tool_name = tool.get('name', '').lower()
    if any(never_detach_name.lower() in tool_name for never_detach_name in NEVER_DETACH_TOOLS):
        final_mcp_tool_ids_to_keep.add(tool.get('id'))
        logger.warning(f"Never-detach tool '{tool.get('name')}' found in MCP list - protecting from pruning")
```

## Files Modified

- `/opt/stacks/lettatoolsselector/tool-selector-api/api_server.py`: Main implementation
  - Added `NEVER_DETACH_TOOLS` environment variable
  - Enhanced tool categorization logic
  - Added protection safeguards
  - Improved logging for protected tools

## Usage Examples

### Protect find_tools only (default)
```bash
NEVER_DETACH_TOOLS=find_tools
```

### Protect multiple tools
```bash
NEVER_DETACH_TOOLS=find_tools,search_tools,agent_memory
```

### Disable never-detach protection (not recommended)
```bash
NEVER_DETACH_TOOLS=""
```

## Logging

The system now provides detailed logging about protected tools:

```
INFO - Protected tools (moved to core): ['find_tools', 'important_tool']
WARNING - Never-detach tool 'find_tools' found in MCP list - protecting from pruning
```

## Testing

Two test scripts are provided:

1. `test_env_parsing.py`: Tests environment variable parsing logic
2. `test_never_detach_config.py`: Tests tool categorization logic

## Backward Compatibility

This change is fully backward compatible:

- Default behavior protects `find_tools` automatically
- Existing `keep_tools` parameter continues to work
- No changes required to existing API calls
- All existing functionality is preserved

## Performance Impact

Minimal performance impact:

- Simple string matching operations during tool categorization
- Protection logic runs only during pruning operations
- No additional API calls or database queries

## Future Enhancements

Potential future improvements:

1. **Tool ID-based protection**: Support tool IDs in addition to names
2. **Regular expression patterns**: Support regex patterns for tool name matching
3. **Per-agent configuration**: Allow different never-detach lists per agent
4. **Dynamic configuration**: Allow runtime updates to never-detach lists