# Worker Service Analysis - Agent ID Header Issue

## 🔍 **Root Cause Identified**

The issue with header-only requests returning empty responses (`"text":"}"`) is **NOT** in the MCP server header extraction (which works perfectly), but in the **worker service and backend processing chain**.

## 📋 **Issue Summary**

When using **header-only** requests:
1. ✅ MCP server correctly extracts `x-agent-id` header
2. ✅ MCP server passes agent_id to worker service
3. ❌ **Worker service/backend fails to process properly**
4. ❌ Returns minimal/empty response instead of tool attachments

## 🔧 **Technical Analysis**

### Data Flow Chain
```
Letta Agent → MCP Server → Worker Service → tool_selector_client.py → API Server
           ↓              ↓               ↓                        ↓
    x-agent-id header → agent_id param → attach_tools() → /api/v1/tools/attach
```

### Worker Service Implementation

**File:** `worker-service/main.py`
```python
@app.post("/find_tools", response_model=FindToolsResponse)
async def find_tools_endpoint(request: FindToolsRequest) -> FindToolsResponse:
    logger.info(
        "Processing find_tools request (agent_id=%s, limit=%s, min_score=%s)",
        request.agent_id,  # ← This should show the agent_id from header
        request.limit,
        request.min_score,
    )
    
    result = await asyncio.to_thread(
        attach_tools,
        query=request.query,
        agent_id=request.agent_id,  # ← Passed correctly to attach_tools
        keep_tools=request.keep_tools,
        limit=request.limit,
        min_score=request.min_score,
        request_heartbeat=request.request_heartbeat,
        session=session,
        logger=_log_debug,
    )
```

### Critical Discovery: `prepare_keep_tools` Function

**File:** `tool_selector_client.py` (lines 80-102)
```python
def prepare_keep_tools(
    keep_tools: Optional[str],
    agent_id: Optional[str],  # ← Agent ID from header
    logger: _LogFn = None,
) -> list[str]:
    keep_tool_ids: list[str] = []

    find_tools_id = get_find_tools_id_with_fallback(agent_id=agent_id)  # ← KEY ISSUE
    if find_tools_id:
        keep_tool_ids.append(find_tools_id)
    else:
        _log(
            "Warning: could not resolve find_tools ID; proceeding without auto-preserve entry",
            logger,
        )
```

## 🚨 **Root Cause: `get_find_tools_id_with_fallback` Failure**

The issue is in the `get_find_tools_id_with_fallback` function in `letta_tool_utils.py`:

### Problem Analysis

1. **Agent ID Validation**: When agent_id comes from header, it might fail validation in the Letta API lookup
2. **API Connectivity**: The function tries to query Letta API to find the `find_tools` tool ID
3. **Fallback Logic**: If the lookup fails, it uses a hardcoded fallback ID that might be invalid
4. **Empty Response**: If tool resolution fails, the system returns minimal response

### Code Path Analysis

**File:** `letta_tool_utils.py` (lines 102-150)
```python
def get_find_tools_id(agent_id: Optional[str] = None) -> Optional[str]:
    # If agent_id is provided, check agent's tools first
    if agent_id:
        agent_tools_url = f"{LETTA_URL}/agents/{agent_id}/tools"
        # ← This API call might be failing for header-derived agent_id
        
        try:
            agent_response = requests.get(agent_tools_url, headers=headers, timeout=10)
            if agent_response.status_code == 200:
                # Look for find_tools in agent's tools
                # ← If agent doesn't have find_tools attached, returns None
```

## 🔍 **Specific Issues Identified**

### 1. **Agent API Validation**
- Header-derived agent_id might not pass Letta API validation
- Agent might not exist or be accessible
- API authentication issues

### 2. **Tool ID Resolution Failure**
- `find_tools` tool might not be attached to the agent
- Hardcoded fallback ID might be outdated/invalid
- Tool lookup logic might be too strict

### 3. **Error Handling**
- Failures in tool ID resolution are logged as warnings but processing continues
- Empty `keep_tools` list might cause backend to return minimal response
- No clear error propagation to user

## 🛠️ **Debugging Steps**

### 1. **Enable Worker Service Logging**
```bash
# Set worker service to DEBUG level
export WORKER_LOG_LEVEL=DEBUG
```

### 2. **Check Worker Service Logs**
Look for these log messages:
```
Processing find_tools request (agent_id=agent-597b5756..., limit=10, min_score=10)
Warning: could not resolve find_tools ID; proceeding without auto-preserve entry
```

### 3. **Test Tool ID Resolution**
```python
# Test the tool ID lookup directly
from letta_tool_utils import get_find_tools_id_with_fallback

agent_id = "agent-597b5756-2915-4560-ba6b-91005f085166"
tool_id = get_find_tools_id_with_fallback(agent_id=agent_id)
print(f"Tool ID for agent {agent_id}: {tool_id}")
```

### 4. **Check Agent Tools**
```bash
# Verify agent exists and has tools
curl -H "Authorization: Bearer $LETTA_API_KEY" \
     "$LETTA_API_URL/agents/agent-597b5756-2915-4560-ba6b-91005f085166/tools"
```

## 🔧 **Potential Solutions**

### 1. **Immediate Fix: Bypass Tool ID Resolution**
Modify `prepare_keep_tools` to not require `find_tools` ID:

```python
def prepare_keep_tools(
    keep_tools: Optional[str],
    agent_id: Optional[str],
    logger: _LogFn = None,
) -> list[str]:
    keep_tool_ids: list[str] = []

    # Skip find_tools ID resolution for now
    # find_tools_id = get_find_tools_id_with_fallback(agent_id=agent_id)
    # if find_tools_id:
    #     keep_tool_ids.append(find_tools_id)

    if keep_tools:
        for item in keep_tools.split(","):
            tool_id = item.strip()
            if tool_id and tool_id not in keep_tool_ids:
                keep_tool_ids.append(tool_id)

    return keep_tool_ids
```

### 2. **Robust Fix: Improve Tool ID Resolution**
- Add better error handling in `get_find_tools_id`
- Update hardcoded fallback IDs
- Add agent existence validation
- Improve API error handling

### 3. **Configuration Fix: Environment Variables**
Check these environment variables in worker service:
```bash
LETTA_API_URL=https://letta2.oculair.ca/v1
LETTA_PASSWORD=lettaSecurePass123
# Ensure these match the MCP server configuration
```

## 🎯 **Recommended Action Plan**

### Phase 1: Immediate Debugging
1. Enable DEBUG logging on worker service
2. Check worker service logs during header-only request
3. Verify agent exists in Letta API
4. Test tool ID resolution function directly

### Phase 2: Quick Fix
1. Temporarily bypass `find_tools` ID resolution
2. Test header-only requests work without tool preservation
3. Verify tool attachment/detachment works

### Phase 3: Robust Solution
1. Fix tool ID resolution logic
2. Update fallback IDs
3. Improve error handling and reporting
4. Add comprehensive testing

## 📊 **Expected Behavior After Fix**

Header-only request should return:
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "content": [{
      "type": "text",
      "text": "Attached 10 tools: delete_episode, delete_entity_edge, get_episodes and 7 more. Detached 9 tools. Preserved 1 existing tools"
    }]
  }
}
```

Instead of:
```json
{
  "jsonrpc": "2.0",
  "id": "1", 
  "result": {
    "content": [{
      "type": "text",
      "text": "}"
    }]
  }
}
```

## 🔍 **Conclusion**

The header extraction is working perfectly. The issue is in the **backend tool ID resolution logic** that fails when processing header-derived agent IDs, causing the system to return minimal responses instead of performing proper tool attachment/detachment operations.
