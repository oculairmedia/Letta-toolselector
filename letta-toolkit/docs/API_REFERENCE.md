# Letta Toolkit API Reference

Complete reference for all public functions and classes in the `letta-toolkit` package.

## Table of Contents

- [Configuration](#configuration)
  - [LettaConfig](#lettaconfig)
  - [get_config](#get_config)
  - [set_config](#set_config)
- [Client](#client)
  - [LettaClient](#lettaclient)
  - [LettaAPIError](#lettaapierror)
- [Tool Operations](#tool-operations)
  - [list_agent_tools](#list_agent_tools)
  - [get_tool_by_name](#get_tool_by_name)
  - [attach_tool_to_agent](#attach_tool_to_agent)
  - [detach_tool_from_agent](#detach_tool_from_agent)
  - [batch_attach_tools](#batch_attach_tools)
  - [batch_detach_tools](#batch_detach_tools)
- [Protected Tools](#protected-tools)
  - [ensure_protected_tools](#ensure_protected_tools)
  - [is_tool_protected](#is_tool_protected)
- [Result Types](#result-types)
  - [ToolOperationResult](#tooloperationresult)
  - [BatchOperationResult](#batchoperationresult)
  - [ProtectedToolsResult](#protectedtoolsresult)

---

## Configuration

### LettaConfig

Dataclass for Letta API configuration.

```python
from letta_toolkit.config import LettaConfig

@dataclass
class LettaConfig:
    base_url: str         # Letta API base URL
    api_key: str          # API key/password
    timeout: int          # Request timeout (seconds)
    max_retries: int      # Max retry attempts
    protected_tools: list[str]  # Tool names that cannot be detached
    max_total_tools: int  # Maximum tools per agent
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `api_url` | `str` | Full API URL with `/v1` suffix |
| `headers` | `dict` | Standard API headers with auth |

**Example:**

```python
config = LettaConfig(
    base_url="https://letta.example.com",
    api_key="secret-key",
    timeout=60,
    max_retries=5,
    protected_tools=["find_agents", "find_tools"],
    max_total_tools=25,
)

print(config.api_url)    # "https://letta.example.com/v1"
print(config.headers)    # {"Authorization": "Bearer secret-key", ...}
```

### get_config

Get the global configuration instance.

```python
def get_config() -> LettaConfig
```

**Returns:** Global `LettaConfig` instance (creates default if not set)

**Example:**

```python
from letta_toolkit.config import get_config

config = get_config()
print(config.base_url)
```

### set_config

Set the global configuration instance.

```python
def set_config(config: LettaConfig) -> None
```

**Parameters:**
- `config`: The configuration to use globally

**Example:**

```python
from letta_toolkit.config import LettaConfig, set_config

custom_config = LettaConfig(base_url="https://custom.letta.com")
set_config(custom_config)
```

---

## Client

### LettaClient

HTTP client for Letta API with automatic retry and error handling.

```python
from letta_toolkit.client import LettaClient

class LettaClient:
    def __init__(self, config: LettaConfig | None = None): ...
    def get(self, path: str, params: dict | None = None, **kwargs) -> Any: ...
    def post(self, path: str, json: dict | None = None, **kwargs) -> Any: ...
    def patch(self, path: str, json: dict | None = None, **kwargs) -> Any: ...
    def delete(self, path: str, **kwargs) -> Any: ...
    def close(self) -> None: ...
```

**Features:**
- Automatic retry on 5xx errors (3 attempts with exponential backoff)
- Session reuse for connection pooling
- Context manager support

**Example:**

```python
from letta_toolkit.client import LettaClient

# Basic usage
client = LettaClient()
tools = client.get(f"/agents/{agent_id}/tools")
client.close()

# Context manager (recommended)
with LettaClient() as client:
    tools = client.get(f"/agents/{agent_id}/tools")
    # Session automatically closed
```

### LettaAPIError

Exception raised for Letta API errors.

```python
class LettaAPIError(Exception):
    status_code: int | None   # HTTP status code
    response: dict | None     # Parsed error response
```

**Example:**

```python
from letta_toolkit.client import LettaClient, LettaAPIError

try:
    with LettaClient() as client:
        client.get("/agents/nonexistent")
except LettaAPIError as e:
    print(f"Status: {e.status_code}")
    print(f"Message: {e}")
```

---

## Tool Operations

### list_agent_tools

List all tools attached to an agent.

```python
def list_agent_tools(
    agent_id: str,
    *,
    include_all: bool = True,
    limit: int | None = None,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> list[dict[str, Any]]
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | The agent ID |
| `include_all` | `bool` | `True` | Fetch all tools (uses high limit) |
| `limit` | `int` | `None` | Custom limit override |
| `client` | `LettaClient` | `None` | Reuse existing client |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** List of tool dictionaries

**Example:**

```python
from letta_toolkit import list_agent_tools

tools = list_agent_tools("agent-123")
for tool in tools:
    print(f"{tool['name']}: {tool['id']}")
```

### get_tool_by_name

Find a tool by name from the global tools registry.

```python
def get_tool_by_name(
    tool_name: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> dict[str, Any] | None
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | `str` | required | Name of the tool |
| `client` | `LettaClient` | `None` | Reuse existing client |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** Tool dictionary if found, `None` otherwise

**Example:**

```python
from letta_toolkit.tools import get_tool_by_name

tool = get_tool_by_name("find_agents")
if tool:
    print(f"Tool ID: {tool['id']}")
```

### attach_tool_to_agent

Attach a single tool to an agent.

```python
def attach_tool_to_agent(
    agent_id: str,
    tool_id: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ToolOperationResult
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | The agent ID |
| `tool_id` | `str` | required | The tool ID to attach |
| `client` | `LettaClient` | `None` | Reuse existing client |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** `ToolOperationResult` indicating success/failure

**Notes:**
- Returns success if tool is already attached (409 status)
- Uses retry logic for transient failures

**Example:**

```python
from letta_toolkit import attach_tool_to_agent

result = attach_tool_to_agent("agent-123", "tool-456")
if result.success:
    print("Tool attached!")
else:
    print(f"Error: {result.error}")
```

### detach_tool_from_agent

Detach a single tool from an agent.

```python
def detach_tool_from_agent(
    agent_id: str,
    tool_id: str,
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ToolOperationResult
```

**Parameters:** Same as `attach_tool_to_agent`

**Returns:** `ToolOperationResult` indicating success/failure

**Notes:**
- Returns success if tool is not attached (404 status)

### batch_attach_tools

Attach multiple tools to an agent.

```python
def batch_attach_tools(
    agent_id: str,
    tool_ids: list[str],
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> BatchOperationResult
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | The agent ID |
| `tool_ids` | `list[str]` | required | List of tool IDs |
| `client` | `LettaClient` | `None` | Reuse existing client |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** `BatchOperationResult` with `successful` and `failed` lists

**Example:**

```python
from letta_toolkit import batch_attach_tools

result = batch_attach_tools("agent-123", ["tool-1", "tool-2", "tool-3"])
print(f"Success: {len(result.successful)}, Failed: {len(result.failed)}")

if not result.all_success:
    for failure in result.failed:
        print(f"Failed {failure.tool_id}: {failure.error}")
```

### batch_detach_tools

Detach multiple tools from an agent.

```python
def batch_detach_tools(
    agent_id: str,
    tool_ids: list[str],
    *,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> BatchOperationResult
```

**Parameters:** Same as `batch_attach_tools`

**Returns:** `BatchOperationResult` with `successful` and `failed` lists

---

## Protected Tools

### ensure_protected_tools

Ensure protected tools are attached to an agent.

```python
def ensure_protected_tools(
    agent_id: str,
    *,
    protected_tools: list[str] | None = None,
    client: LettaClient | None = None,
    config: LettaConfig | None = None,
) -> ProtectedToolsResult
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | The agent ID |
| `protected_tools` | `list[str]` | `None` | Tool names (uses config default) |
| `client` | `LettaClient` | `None` | Reuse existing client |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** `ProtectedToolsResult` with details

**Example:**

```python
from letta_toolkit import ensure_protected_tools

# Use default protected tools from config
result = ensure_protected_tools("agent-123")

# Or specify custom list
result = ensure_protected_tools(
    "agent-123",
    protected_tools=["find_agents", "find_tools", "search"]
)

print(f"Present: {result.already_present}")
print(f"Attached: {[t['name'] for t in result.attached]}")
print(f"Failed: {[t['name'] for t in result.failed]}")
```

### is_tool_protected

Check if a tool name is in the protected list.

```python
def is_tool_protected(
    tool_name: str,
    *,
    protected_tools: list[str] | None = None,
    config: LettaConfig | None = None,
) -> bool
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | `str` | required | Tool name to check |
| `protected_tools` | `list[str]` | `None` | Custom list (uses config default) |
| `config` | `LettaConfig` | `None` | Custom config |

**Returns:** `True` if the tool is protected

**Example:**

```python
from letta_toolkit.protected import is_tool_protected

if is_tool_protected("find_agents"):
    print("Cannot detach find_agents - it's protected!")
```

---

## Result Types

### ToolOperationResult

Result of a single tool operation (attach/detach).

```python
@dataclass
class ToolOperationResult:
    success: bool           # Whether operation succeeded
    tool_id: str           # ID of the tool
    tool_name: str | None  # Name (if known)
    error: str | None      # Error message (if failed)
```

### BatchOperationResult

Result of a batch tool operation.

```python
@dataclass
class BatchOperationResult:
    successful: list[ToolOperationResult]  # Successful operations
    failed: list[ToolOperationResult]      # Failed operations

    @property
    def all_success(self) -> bool: ...  # True if no failures

    @property
    def total(self) -> int: ...  # Total operation count
```

### ProtectedToolsResult

Result of ensure_protected_tools operation.

```python
@dataclass
class ProtectedToolsResult:
    success: bool                    # True if no failures
    attached: list[dict[str, Any]]   # Tools that were attached
    already_present: list[str]       # Tool names already attached
    failed: list[dict[str, Any]]     # Tools that failed to attach
    error: str | None               # Error message (if operation failed)
```

---

## Logging

The toolkit uses Python's standard logging module. Enable debug logging to see detailed API calls:

```python
import logging

# Enable debug logging for toolkit
logging.getLogger("letta_toolkit").setLevel(logging.DEBUG)

# Or enable for specific modules
logging.getLogger("letta_toolkit.client").setLevel(logging.DEBUG)
logging.getLogger("letta_toolkit.tools").setLevel(logging.DEBUG)
```
