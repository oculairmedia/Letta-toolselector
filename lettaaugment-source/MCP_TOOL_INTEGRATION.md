# MCP Tool Integration Technical Documentation

This document provides technical details about the Model Context Protocol (MCP) tool integration in the LettaSearch system, focusing on the tool attachment/detachment process and recent fixes.

## MCP Tool Architecture

The LettaSearch system integrates with MCP servers to provide a wide range of tools to Letta agents. The integration follows this general flow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Letta Agent   │◄────┤  LettaSearch    │◄────┤    MCP Server   │
│                 │     │   API Server    │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Tool Attachment Process

When a tool attachment request is received, the system performs the following steps:

1. **Search for matching tools** based on the query
2. **Register MCP tools** if they're not already registered in Letta
3. **Identify existing MCP tools** on the agent that need to be detached
4. **Detach tools** that aren't in the keep list
5. **Attach new tools** to the agent

### Code Flow

```python
# 1. Get current tools from agent
tools_response = requests.get(f"{LETTA_URL}/agents/{agent_id}/tools", headers=HEADERS)
all_tools = tools_response.json()

# 2. Find MCP tools
mcp_tools = []
for tool in all_tools:
    if tool.get("tool_type") == "external_mcp":
        tool_id = tool.get("id") or tool.get("tool_id")
        if tool_id and tool_id not in seen_tool_ids:
            # Add to mcp_tools list
            
# 3. Search for matching tools
matching_tools = search_tools(query=query, limit=limit)

# 4. Process and register tools
processed_tools = []
for tool in matching_tools:
    # Find in existing Letta tools or register from MCP
    
# 5. Process detachments and attachments
results = process_tools(agent_id, mcp_tools, processed_tools, keep_tools)
```

## Tool Detachment Process

The tool detachment process is critical for maintaining agent performance. Too many tools can cause context window issues, so the system needs to manage which tools are attached at any given time.

### Detachment Logic

```python
async def process_tools(agent_id: str, mcp_tools: list, matching_tools: list, keep_tools: list = None):
    # Create a set of tool IDs to keep
    keep_tool_ids = set()
    # Add explicitly kept tools
    for tool_id in keep_tools:
        if tool_id:
            keep_tool_ids.add(tool_id)
    # Add new tools being attached
    for tool in matching_tools:
        tool_id = tool.get("id") or tool.get("tool_id")
        if tool_id:
            keep_tool_ids.add(tool_id)
    
    # Find tools to detach (current tools that aren't in keep_tool_ids)
    tools_to_detach = []
    for tool in mcp_tools:
        tool_id = tool.get("tool_id") or tool.get("id")
        tool_name = tool.get("name", "Unknown")
        
        # If tool ID is valid and not in the keep list
        if tool_id and tool_id not in keep_tool_ids:
            tools_to_detach.append({
                "id": tool_id,
                "tool_id": tool_id,
                "name": tool_name
            })
    
    # Detach tools sequentially with retry logic
    for tool in tools_to_detach:
        # Detach with retries
```

## Recent Fix: Variable Name Conflict

A critical issue was identified and fixed in the tool detachment process. The issue was caused by a variable name conflict that was overwriting the list of MCP tools from the agent.

### Problem

In the tool registration code, the variable `mcp_tools` was being reused, which was overwriting the list of MCP tools from the agent:

```python
# Original problematic code
mcp_tools_response = requests.get(f"{LETTA_URL}/tools/mcp/servers/{server_name}/tools", headers=HEADERS)
mcp_tools_response.raise_for_status()
mcp_tools = mcp_tools_response.json()  # This was overwriting the mcp_tools list!
```

This caused the system to lose track of which tools were currently attached to the agent, resulting in inconsistent tool detachment.

### Solution

The fix was to rename the variable to avoid the conflict:

```python
# Fixed code
mcp_tools_response = requests.get(f"{LETTA_URL}/tools/mcp/servers/{server_name}/tools", headers=HEADERS)
mcp_tools_response.raise_for_status()
server_mcp_tools = mcp_tools_response.json()  # Renamed to avoid conflict
```

This ensures that the list of MCP tools from the agent is preserved throughout the process, allowing proper identification of tools that need to be detached.

## Additional Improvements

Several other improvements were made to enhance the reliability of the tool detachment process:

1. **Sequential detachment with retry logic**:
   ```python
   max_attempts = 3
   for attempt in range(1, max_attempts + 1):
       result = await detach_tool(session, agent_id, tool_id)
       if result["success"]:
           break
       else:
           # Wait briefly before retrying
           await asyncio.sleep(0.5)
   ```

2. **Timeout handling**:
   ```python
   timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
   async with session.patch(detach_url, headers=HEADERS, timeout=timeout) as response:
       # Process response
   ```

3. **Improved error handling**:
   ```python
   try:
       response_data = await response.json()
   except Exception as json_error:
       # Handle case where response is not JSON
       response_text = await response.text()
       logger.warning(f"Non-JSON response from detach endpoint: {response_text}")
       response_data = {"text": response_text}
   ```

## Testing the Fix

The fix was tested by making multiple requests to attach different sets of tools to an agent:

1. First request: Attach github tools
   - Result: Successfully attached github tools, detaching any previous MCP tools

2. Second request: Attach crawl4ai tools
   - Result: Successfully attached crawl4ai tools, detaching github tools

3. Third request: Attach github tools again
   - Result: Successfully attached github tools, detaching crawl4ai tools

The logs confirmed that the tool detachment was working correctly in all cases.

## Deployment

The fix was deployed by building and pushing a new Docker image:

```bash
docker build -t oculair/lettaaugment:latest .
docker push oculair/lettaaugment:latest
```

And then restarting the container:

```bash
docker stop lettaaugment-prod
docker rm lettaaugment-prod
docker run -d -p 8020:3001 --env-file .env --name lettaaugment-prod oculair/lettaaugment:latest
```

## Conclusion

The MCP tool integration in the LettaSearch system provides a powerful way to enhance Letta agents with dynamic tool management. The recent fix ensures that the tool detachment process works reliably, preventing context window issues and ensuring that agents have access to the most relevant tools for their current task.

The system is designed to be robust, with features like sequential detachment with retry logic, timeout handling, and improved error handling. These enhancements make the system more reliable and easier to maintain.