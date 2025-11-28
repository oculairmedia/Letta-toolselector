# Letta Toolkit Examples

Practical code examples for common use cases.

## Table of Contents

- [Basic Operations](#basic-operations)
- [Agent Tool Management](#agent-tool-management)
- [Protected Tools](#protected-tools)
- [Webhook Handler Integration](#webhook-handler-integration)
- [Tool Pruning](#tool-pruning)
- [Error Handling](#error-handling)
- [Logging and Debugging](#logging-and-debugging)

---

## Basic Operations

### List All Tools for an Agent

```python
from letta_toolkit import list_agent_tools

agent_id = "agent-abc123"
tools = list_agent_tools(agent_id)

print(f"Agent has {len(tools)} tools:")
for tool in tools:
    print(f"  - {tool['name']} (ID: {tool['id']})")
```

### Find a Tool by Name

```python
from letta_toolkit.tools import get_tool_by_name

tool = get_tool_by_name("find_agents")
if tool:
    print(f"Found tool: {tool['name']}")
    print(f"  ID: {tool['id']}")
    print(f"  Description: {tool.get('description', 'N/A')}")
else:
    print("Tool not found")
```

### Attach a Single Tool

```python
from letta_toolkit import attach_tool_to_agent

result = attach_tool_to_agent("agent-123", "tool-456")

if result.success:
    print(f"Successfully attached tool {result.tool_id}")
else:
    print(f"Failed to attach: {result.error}")
```

### Detach a Single Tool

```python
from letta_toolkit import detach_tool_from_agent

result = detach_tool_from_agent("agent-123", "tool-456")

if result.success:
    print("Tool detached")
```

---

## Agent Tool Management

### Get Tool Count for an Agent

```python
from letta_toolkit import list_agent_tools

def get_tool_count(agent_id: str) -> int:
    """Get the number of tools attached to an agent."""
    tools = list_agent_tools(agent_id)
    return len(tools)

count = get_tool_count("agent-123")
print(f"Agent has {count} tools")
```

### Attach Multiple Tools

```python
from letta_toolkit import batch_attach_tools

tool_ids = ["tool-1", "tool-2", "tool-3", "tool-4"]
result = batch_attach_tools("agent-123", tool_ids)

print(f"Attached: {len(result.successful)} tools")
print(f"Failed: {len(result.failed)} tools")

if not result.all_success:
    for failure in result.failed:
        print(f"  - {failure.tool_id}: {failure.error}")
```

### Detach Multiple Tools

```python
from letta_toolkit import batch_detach_tools

# Detach old/unused tools
tools_to_remove = ["old-tool-1", "old-tool-2"]
result = batch_detach_tools("agent-123", tools_to_remove)

print(f"Detached {len(result.successful)} tools")
```

### Replace Agent's Tools

```python
from letta_toolkit import list_agent_tools, batch_attach_tools, batch_detach_tools

def replace_agent_tools(agent_id: str, new_tool_ids: list[str]) -> dict:
    """Replace all non-protected tools with a new set."""
    from letta_toolkit.config import get_config
    from letta_toolkit.protected import is_tool_protected
    
    config = get_config()
    
    # Get current tools
    current_tools = list_agent_tools(agent_id)
    current_ids = {t['id'] for t in current_tools}
    current_names = {t['id']: t['name'] for t in current_tools}
    
    # Determine what to add and remove
    new_set = set(new_tool_ids)
    to_add = new_set - current_ids
    to_remove = [
        tid for tid in (current_ids - new_set)
        if not is_tool_protected(current_names.get(tid, ""))
    ]
    
    # Execute changes
    add_result = batch_attach_tools(agent_id, list(to_add)) if to_add else None
    remove_result = batch_detach_tools(agent_id, to_remove) if to_remove else None
    
    return {
        "added": len(add_result.successful) if add_result else 0,
        "removed": len(remove_result.successful) if remove_result else 0,
        "add_failures": len(add_result.failed) if add_result else 0,
        "remove_failures": len(remove_result.failed) if remove_result else 0,
    }
```

---

## Protected Tools

### Ensure Protected Tools on Agent

```python
from letta_toolkit import ensure_protected_tools

result = ensure_protected_tools("agent-123")

print(f"Already present: {result.already_present}")
print(f"Newly attached: {[t['name'] for t in result.attached]}")

if result.failed:
    print(f"Failed to attach: {[t['name'] for t in result.failed]}")
```

### Use Custom Protected Tools List

```python
from letta_toolkit import ensure_protected_tools

# Override the default protected tools
custom_protected = ["find_agents", "find_tools", "send_message", "archival_memory_search"]

result = ensure_protected_tools(
    "agent-123",
    protected_tools=custom_protected
)

if result.success:
    print("All protected tools are attached")
```

### Check if Tool is Protected Before Detaching

```python
from letta_toolkit import detach_tool_from_agent, list_agent_tools
from letta_toolkit.protected import is_tool_protected

def safe_detach(agent_id: str, tool_id: str, tool_name: str) -> bool:
    """Detach a tool only if it's not protected."""
    if is_tool_protected(tool_name):
        print(f"Cannot detach {tool_name} - it's protected")
        return False
    
    result = detach_tool_from_agent(agent_id, tool_id)
    return result.success

# Usage
tools = list_agent_tools("agent-123")
for tool in tools:
    if tool['name'].startswith("deprecated_"):
        safe_detach("agent-123", tool['id'], tool['name'])
```

---

## Webhook Handler Integration

### Handle Incoming Webhook with Tool Selection

```python
from letta_toolkit import (
    list_agent_tools,
    batch_attach_tools,
    batch_detach_tools,
    ensure_protected_tools,
)
from letta_toolkit.config import get_config

def handle_webhook(agent_id: str, message: str, recommended_tools: list[str]):
    """
    Handle a webhook that includes recommended tools for the message.
    
    Args:
        agent_id: The Letta agent ID
        message: The incoming message
        recommended_tools: List of tool IDs recommended for this message
    """
    config = get_config()
    max_tools = config.max_total_tools
    
    # 1. Ensure protected tools are attached
    protected_result = ensure_protected_tools(agent_id)
    
    # 2. Get current tool count
    current_tools = list_agent_tools(agent_id)
    current_tool_ids = {t['id'] for t in current_tools}
    
    # 3. Filter recommended tools (exclude already attached)
    tools_to_add = [t for t in recommended_tools if t not in current_tool_ids]
    
    # 4. Check if we need to prune
    total_after_add = len(current_tools) + len(tools_to_add)
    
    if total_after_add > max_tools:
        # Calculate how many to remove
        excess = total_after_add - max_tools
        
        # Find removable tools (not protected, not in recommended)
        removable = [
            t['id'] for t in current_tools
            if t['id'] not in recommended_tools
            and t['name'] not in config.protected_tools
        ]
        
        tools_to_remove = removable[:excess]
        batch_detach_tools(agent_id, tools_to_remove)
    
    # 5. Attach new recommended tools
    if tools_to_add:
        batch_attach_tools(agent_id, tools_to_add)
    
    # 6. Return updated tool list
    return list_agent_tools(agent_id)
```

---

## Tool Pruning

### Prune Tools to Stay Under Limit

```python
from letta_toolkit import list_agent_tools, batch_detach_tools
from letta_toolkit.config import get_config
from letta_toolkit.protected import is_tool_protected

def prune_agent_tools(agent_id: str, target_count: int | None = None) -> int:
    """
    Remove excess tools from an agent to stay under the limit.
    
    Args:
        agent_id: The agent ID
        target_count: Target tool count (defaults to MAX_TOTAL_TOOLS)
        
    Returns:
        Number of tools removed
    """
    config = get_config()
    target = target_count or config.max_total_tools
    
    # Get current tools
    tools = list_agent_tools(agent_id)
    current_count = len(tools)
    
    if current_count <= target:
        return 0  # No pruning needed
    
    excess = current_count - target
    
    # Sort tools by priority (keep protected, remove older/less used first)
    removable = []
    for tool in tools:
        if not is_tool_protected(tool.get('name', '')):
            removable.append(tool)
    
    # Remove oldest first (or use your own criteria)
    tools_to_remove = [t['id'] for t in removable[:excess]]
    
    result = batch_detach_tools(agent_id, tools_to_remove)
    
    return len(result.successful)


# Usage
removed = prune_agent_tools("agent-123")
print(f"Removed {removed} excess tools")
```

### Scheduled Cleanup Job

```python
from letta_toolkit import list_agent_tools
from letta_toolkit.config import get_config

def cleanup_all_agents(agent_ids: list[str]) -> dict:
    """Run tool cleanup across multiple agents."""
    config = get_config()
    max_tools = config.max_total_tools
    
    report = {
        "agents_checked": len(agent_ids),
        "agents_over_limit": 0,
        "total_pruned": 0,
    }
    
    for agent_id in agent_ids:
        tools = list_agent_tools(agent_id)
        if len(tools) > max_tools:
            report["agents_over_limit"] += 1
            removed = prune_agent_tools(agent_id)
            report["total_pruned"] += removed
    
    return report
```

---

## Error Handling

### Handle API Errors Gracefully

```python
from letta_toolkit import list_agent_tools, attach_tool_to_agent
from letta_toolkit.client import LettaAPIError

def safe_list_tools(agent_id: str) -> list:
    """List tools with error handling."""
    try:
        return list_agent_tools(agent_id)
    except LettaAPIError as e:
        if e.status_code == 404:
            print(f"Agent {agent_id} not found")
        else:
            print(f"API error: {e}")
        return []

def safe_attach_tool(agent_id: str, tool_id: str) -> bool:
    """Attach tool with detailed error handling."""
    result = attach_tool_to_agent(agent_id, tool_id)
    
    if not result.success:
        if "not found" in (result.error or "").lower():
            print(f"Tool {tool_id} does not exist")
        elif "already" in (result.error or "").lower():
            print(f"Tool {tool_id} already attached")
        else:
            print(f"Unknown error: {result.error}")
        return False
    
    return True
```

### Retry Failed Operations

```python
import time
from letta_toolkit import batch_attach_tools

def attach_with_retry(agent_id: str, tool_ids: list[str], max_retries: int = 3):
    """Attach tools with retry for transient failures."""
    remaining = tool_ids.copy()
    
    for attempt in range(max_retries):
        if not remaining:
            break
            
        result = batch_attach_tools(agent_id, remaining)
        
        # Update remaining to just the failures
        remaining = [r.tool_id for r in result.failed]
        
        if remaining and attempt < max_retries - 1:
            print(f"Retrying {len(remaining)} failed tools...")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return {
        "success": len(tool_ids) - len(remaining),
        "failed": remaining,
    }
```

---

## Logging and Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logging for all toolkit modules
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("letta_toolkit").setLevel(logging.DEBUG)

# Or just for specific modules
logging.getLogger("letta_toolkit.client").setLevel(logging.DEBUG)
logging.getLogger("letta_toolkit.tools").setLevel(logging.DEBUG)
```

### Custom Logger Integration

```python
import logging
from letta_toolkit import list_agent_tools

# Create custom logger
logger = logging.getLogger("my_app.tools")

def get_agent_tools_with_logging(agent_id: str) -> list:
    """Get tools with custom logging."""
    logger.info(f"Fetching tools for agent {agent_id}")
    
    try:
        tools = list_agent_tools(agent_id)
        logger.info(f"Retrieved {len(tools)} tools")
        return tools
    except Exception as e:
        logger.error(f"Failed to fetch tools: {e}")
        raise
```

### Debug Tool Operations

```python
from letta_toolkit import attach_tool_to_agent
from letta_toolkit.client import LettaClient

def debug_attach(agent_id: str, tool_id: str):
    """Attach tool with verbose output for debugging."""
    import logging
    logging.getLogger("letta_toolkit").setLevel(logging.DEBUG)
    
    print(f"Attaching tool {tool_id} to agent {agent_id}...")
    result = attach_tool_to_agent(agent_id, tool_id)
    
    print(f"Result: success={result.success}")
    if result.error:
        print(f"Error: {result.error}")
    
    return result
```
