# Letta Toolkit

Shared Python SDK for Letta agent tool management. This package provides a consistent interface for listing, attaching, and detaching tools from Letta agents.

## Features

- **Unified Configuration** - Single `LettaConfig` reads from environment variables
- **Automatic Retry** - Built-in retry logic for transient API failures (3 retries, exponential backoff)
- **Protected Tools** - Ensure critical tools are never accidentally detached
- **Batch Operations** - Efficient bulk attach/detach with detailed results
- **Session Reuse** - Connection pooling via persistent HTTP sessions
- **Type-Safe Results** - Dataclass results with success/failure details

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Installation and quick start
- [API Reference](docs/API_REFERENCE.md) - Complete function documentation
- [Migration Guide](docs/MIGRATION_GUIDE.md) - Migrate existing code to SDK
- [Examples](docs/EXAMPLES.md) - Common use cases and patterns

## Installation

```bash
# Install from local path (development)
pip install -e /opt/stacks/lettatoolsselector/letta-toolkit

# Or via git
pip install git+https://github.com/oculairmedia/lettatoolsselector.git#subdirectory=letta-toolkit
```

## Quick Start

```python
from letta_toolkit import list_agent_tools, ensure_protected_tools

# List all tools attached to an agent
tools = list_agent_tools("agent-123")
print(f"Agent has {len(tools)} tools")

# Ensure protected tools are attached
result = ensure_protected_tools("agent-123")
if result.attached:
    print(f"Attached missing tools: {[t['name'] for t in result.attached]}")
```

## Configuration

The toolkit reads configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTA_BASE_URL` | `https://letta.oculair.ca` | Letta API base URL |
| `LETTA_PASSWORD` | (empty) | API key for authentication |
| `LETTA_TIMEOUT` | `30` | Request timeout in seconds |
| `LETTA_MAX_RETRIES` | `3` | Max retry attempts |
| `PROTECTED_TOOLS` | `find_agents,find_tools` | Comma-separated protected tool names |
| `MAX_TOTAL_TOOLS` | `20` | Maximum tools per agent |

You can also configure programmatically:

```python
from letta_toolkit.config import LettaConfig, set_config

config = LettaConfig(
    base_url="https://my-letta.com",
    api_key="my-key",
    protected_tools=["find_agents", "search"],
)
set_config(config)
```

## API Reference

### Tool Listing

```python
from letta_toolkit import list_agent_tools

# Get all tools (handles pagination automatically)
tools = list_agent_tools("agent-123")

# Limit results
tools = list_agent_tools("agent-123", limit=10)
```

### Tool Attachment

```python
from letta_toolkit import attach_tool_to_agent, batch_attach_tools

# Attach single tool
result = attach_tool_to_agent("agent-123", "tool-456")
if result.success:
    print("Attached!")

# Attach multiple tools
result = batch_attach_tools("agent-123", ["tool-1", "tool-2", "tool-3"])
print(f"Attached: {len(result.successful)}, Failed: {len(result.failed)}")
```

### Tool Detachment

```python
from letta_toolkit import detach_tool_from_agent, batch_detach_tools

# Detach single tool
result = detach_tool_from_agent("agent-123", "tool-456")

# Detach multiple tools
result = batch_detach_tools("agent-123", ["tool-1", "tool-2"])
```

### Protected Tools

```python
from letta_toolkit import ensure_protected_tools

# Ensure protected tools are attached (uses config defaults)
result = ensure_protected_tools("agent-123")

# Custom protected tools list
result = ensure_protected_tools(
    "agent-123",
    protected_tools=["find_agents", "search", "send_message"]
)

# Check result
print(f"Already present: {result.already_present}")
print(f"Newly attached: {[t['name'] for t in result.attached]}")
print(f"Failed: {[t['name'] for t in result.failed]}")
```

## Development

```bash
# Install dev dependencies
cd letta-toolkit
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/

# Run type checker
mypy src/
```

## Architecture

```
letta-toolkit/
├── src/letta_toolkit/
│   ├── __init__.py      # Public API exports
│   ├── config.py        # LettaConfig dataclass, env var handling
│   ├── client.py        # LettaClient HTTP wrapper with retry
│   ├── tools.py         # Tool CRUD operations
│   └── protected.py     # Protected tools enforcement
├── tests/
│   ├── test_config.py   # Configuration tests
│   └── test_tools.py    # Tool operations tests
├── docs/
│   ├── GETTING_STARTED.md
│   ├── API_REFERENCE.md
│   ├── MIGRATION_GUIDE.md
│   └── EXAMPLES.md
└── pyproject.toml       # Package configuration
```

## Related Projects

- **letta-webhook-receiver** (`/opt/stacks/letta-webhook-receiver-new`) - Webhook handler for Letta agents
- **letta-toolselector** (`/opt/stacks/lettatoolsselector`) - Tool selection service

## License

MIT
