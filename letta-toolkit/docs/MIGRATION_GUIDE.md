# Migration Guide

This guide explains how to migrate existing Letta tool management code to use the `letta-toolkit` SDK.

## Overview

The `letta-toolkit` package consolidates tool management patterns from:
- `letta-webhook-receiver` (`/opt/stacks/letta-webhook-receiver-new`)
- `letta-toolselector` (`/opt/stacks/lettatoolsselector`)

Both projects have their own implementations of tool listing, attachment, and protected tool handling. This SDK provides a unified interface.

## Installation in Your Project

### 1. Add Dependency

**Option A: Git-based installation (recommended for now)**

```bash
# In your project directory
pip install git+https://github.com/oculairmedia/Letta-toolselector.git#subdirectory=letta-toolkit
```

**Option B: Local editable install (for development)**

```bash
pip install -e /opt/stacks/lettatoolsselector/letta-toolkit
```

**Option C: Add to requirements.txt**

```
# requirements.txt
letta-toolkit @ git+https://github.com/oculairmedia/Letta-toolselector.git#subdirectory=letta-toolkit
```

### 2. Set Environment Variables

Ensure these are set (add to `.env` or `docker-compose.yml`):

```bash
LETTA_BASE_URL=https://letta.oculair.ca
LETTA_PASSWORD=your-api-key
PROTECTED_TOOLS=find_agents,find_tools
MAX_TOTAL_TOOLS=20
```

## Migration Patterns

### Pattern 1: Replace Direct HTTP Calls

**Before (webhook-receiver style):**

```python
import requests

def get_agent_tools(agent_id: str) -> list:
    url = f"{LETTA_API_URL}/agents/{agent_id}/tools"
    headers = {"Authorization": f"Bearer {LETTA_PASSWORD}"}
    response = requests.get(url, headers=headers, timeout=30)
    return response.json() if response.status_code == 200 else []
```

**After (with SDK):**

```python
from letta_toolkit import list_agent_tools

def get_agent_tools(agent_id: str) -> list:
    return list_agent_tools(agent_id)
```

### Pattern 2: Replace Tool Attachment Logic

**Before:**

```python
def attach_tool(agent_id: str, tool_id: str) -> bool:
    url = f"{LETTA_API_URL}/agents/{agent_id}/tools/attach/{tool_id}"
    try:
        response = requests.patch(url, headers=headers, json={})
        return response.status_code in (200, 201, 409)
    except Exception:
        return False
```

**After:**

```python
from letta_toolkit import attach_tool_to_agent

def attach_tool(agent_id: str, tool_id: str) -> bool:
    result = attach_tool_to_agent(agent_id, tool_id)
    return result.success
```

### Pattern 3: Replace Batch Operations

**Before:**

```python
def attach_multiple_tools(agent_id: str, tool_ids: list) -> dict:
    results = {"success": [], "failed": []}
    for tool_id in tool_ids:
        if attach_tool(agent_id, tool_id):
            results["success"].append(tool_id)
        else:
            results["failed"].append(tool_id)
    return results
```

**After:**

```python
from letta_toolkit import batch_attach_tools

def attach_multiple_tools(agent_id: str, tool_ids: list) -> dict:
    result = batch_attach_tools(agent_id, tool_ids)
    return {
        "success": [r.tool_id for r in result.successful],
        "failed": [r.tool_id for r in result.failed]
    }
```

### Pattern 4: Replace Protected Tools Logic

**Before (webhook-receiver):**

```python
PROTECTED_TOOLS = ["find_agents", "find_tools"]

def ensure_protected_tools(agent_id: str):
    current = get_agent_tools(agent_id)
    current_names = {t["name"] for t in current}
    
    for tool_name in PROTECTED_TOOLS:
        if tool_name not in current_names:
            tool = find_tool_by_name(tool_name)
            if tool:
                attach_tool(agent_id, tool["id"])
```

**After:**

```python
from letta_toolkit import ensure_protected_tools

def handle_protected_tools(agent_id: str):
    result = ensure_protected_tools(agent_id)
    if not result.success:
        logger.warning(f"Failed to attach: {result.failed}")
```

### Pattern 5: Use Custom Configuration

**Before:**

```python
# Scattered configuration
LETTA_URL = os.environ.get("LETTA_BASE_URL", "https://letta.oculair.ca")
LETTA_KEY = os.environ.get("LETTA_PASSWORD", "")
TIMEOUT = int(os.environ.get("LETTA_TIMEOUT", "30"))
```

**After:**

```python
from letta_toolkit.config import LettaConfig, set_config

# Centralized configuration
config = LettaConfig()  # Reads from environment automatically

# Or customize
config = LettaConfig(
    base_url="https://custom.letta.com",
    api_key="custom-key",
    timeout=60,
)
set_config(config)
```

## Migration Checklist

### For webhook-receiver (`/opt/stacks/letta-webhook-receiver-new`)

- [ ] Add `letta-toolkit` to `requirements.txt`
- [ ] Update `webhook_server/tool_inventory.py` to use SDK
- [ ] Update `tool_manager.py` to use SDK
- [ ] Replace direct API calls in `webhook_server/app.py`
- [ ] Remove duplicated helper functions
- [ ] Update tests to use SDK mocks

### For tool-selector (`/opt/stacks/lettatoolsselector`)

- [ ] Add `letta-toolkit` to `requirements.txt`
- [ ] Update `tool_selector_client.py` to use SDK
- [ ] Update `letta_tool_utils.py` to use SDK
- [ ] Replace API calls in main application
- [ ] Update worker service to use SDK

## Gradual Migration Strategy

1. **Phase 1: Install SDK alongside existing code**
   - Add SDK dependency
   - Don't change existing code yet

2. **Phase 2: Migrate read operations first**
   - Replace `list_agent_tools` calls
   - Lower risk - read-only operations

3. **Phase 3: Migrate write operations**
   - Replace `attach_tool` / `detach_tool` calls
   - Replace batch operations

4. **Phase 4: Migrate protected tools logic**
   - Replace custom protected tools handling
   - Remove duplicated code

5. **Phase 5: Clean up**
   - Remove old helper functions
   - Update tests
   - Update documentation

## Backward Compatibility

The SDK is designed to be a drop-in replacement. Key compatibility notes:

| Old Pattern | SDK Equivalent | Notes |
|-------------|----------------|-------|
| `requests.get(url)` | `client.get(path)` | SDK handles base URL |
| `response.status_code == 200` | `result.success` | SDK handles status codes |
| `response.json()` | Return value | SDK parses JSON automatically |
| Manual retry loops | Automatic | SDK has built-in retry |

## Testing During Migration

```python
# Test that SDK works with your Letta instance
from letta_toolkit import list_agent_tools
from letta_toolkit.config import get_config

# Verify config
config = get_config()
print(f"Base URL: {config.base_url}")
print(f"Has API key: {bool(config.api_key)}")

# Test API connectivity
tools = list_agent_tools("your-test-agent-id")
print(f"Retrieved {len(tools)} tools")
```

## Getting Help

If you encounter issues during migration:

1. Check the [API Reference](API_REFERENCE.md) for function signatures
2. Enable debug logging to see API calls:
   ```python
   import logging
   logging.getLogger("letta_toolkit").setLevel(logging.DEBUG)
   ```
3. Review test files in `letta-toolkit/tests/` for usage examples
