# Dynamic Tool ID Lookup Implementation

## Overview
We implemented a dynamic tool ID lookup system for the find_tools tool that adapts to changes in the Letta system.

## Key Components

### 1. `letta_tool_utils.py`
Contains two main functions:

- **`get_find_tools_id(agent_id: Optional[str] = None)`**
  - Dynamically queries the Letta API to find the tool ID for find_tools
  - If agent_id is provided, checks the agent's attached tools first (most reliable)
  - Falls back to checking all tools and MCP servers
  - Returns None if not found

- **`get_find_tools_id_with_fallback(agent_id: Optional[str] = None, fallback_id: Optional[str] = None)`**
  - Wrapper around get_find_tools_id with fallback support
  - Uses hardcoded fallback ID if dynamic lookup fails
  - Default fallback: `tool-e34b5c60-5bd5-4288-a97f-2167ddf3062b`

### 2. Updated Files
- `find_tools.py` - Now uses `get_find_tools_id_with_fallback(agent_id=agent_id)`
- `find_tools_enhanced.py` - Same update
- `webhook_server/app.py` - Uses dynamic lookup with agent_id

### 3. Environment Variables
The MCP server now requires:
- `LETTA_API_URL` - The Letta API endpoint
- `LETTA_PASSWORD` - Authentication token

## How It Works

1. When find_tools is called with an agent_id, it first checks that agent's attached tools
2. If found there, uses that tool ID (most reliable method)
3. If not found or no agent_id provided, searches all tools
4. Also checks MCP servers to understand the tool's origin
5. Falls back to hardcoded ID if all lookups fail

## Example Usage

```python
# With agent ID (recommended)
tool_id = get_find_tools_id_with_fallback(agent_id="agent-123")

# Without agent ID (less reliable)
tool_id = get_find_tools_id_with_fallback()

# With custom fallback
tool_id = get_find_tools_id_with_fallback(
    agent_id="agent-123",
    fallback_id="tool-custom-id"
)
```

## Benefits

1. **Adaptability**: Automatically adapts to tool ID changes in Letta
2. **Reliability**: Uses agent-specific lookup for accuracy
3. **Fallback Support**: Continues working even if API is unavailable
4. **Debugging**: Provides detailed logging for troubleshooting

## Testing

Run the test script to verify the lookup:
```bash
python3 test_agent_lookup.py
```

This will test:
- Agent-based lookup
- Global lookup
- Fallback behavior