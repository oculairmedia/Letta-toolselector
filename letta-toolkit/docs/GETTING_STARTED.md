# Getting Started with Letta Toolkit

This guide walks you through setting up and using the `letta-toolkit` package for managing Letta agent tools.

## Prerequisites

- Python 3.10 or higher
- Access to a Letta API server
- API credentials (password/token)

## Installation

### From Local Path (Development)

```bash
# Clone or navigate to the repository
cd /opt/stacks/lettatoolsselector

# Install in editable mode with dev dependencies
pip install -e "./letta-toolkit[dev]"
```

### From Git

```bash
pip install git+https://github.com/oculairmedia/Letta-toolselector.git#subdirectory=letta-toolkit
```

### Verify Installation

```python
import letta_toolkit
print(letta_toolkit.__version__)  # Should print "0.1.0"
```

## Configuration

### Environment Variables (Recommended)

Set these environment variables before using the toolkit:

```bash
export LETTA_BASE_URL="https://letta.oculair.ca"
export LETTA_PASSWORD="your-api-key"
export PROTECTED_TOOLS="find_agents,find_tools,matrix_messaging"
export MAX_TOTAL_TOOLS="20"
```

### Programmatic Configuration

```python
from letta_toolkit.config import LettaConfig, set_config

# Create custom config
config = LettaConfig(
    base_url="https://my-letta-server.com",
    api_key="my-secret-key",
    timeout=60,
    max_retries=5,
    protected_tools=["find_agents", "find_tools", "matrix_messaging"],
    max_total_tools=25,
)

# Set as global default
set_config(config)
```

### Configuration Options

| Option | Env Variable | Default | Description |
|--------|--------------|---------|-------------|
| `base_url` | `LETTA_BASE_URL` | `https://letta.oculair.ca` | Letta API base URL |
| `api_key` | `LETTA_PASSWORD` | (empty) | API authentication key |
| `timeout` | `LETTA_TIMEOUT` | `30` | Request timeout in seconds |
| `max_retries` | `LETTA_MAX_RETRIES` | `3` | Max retry attempts |
| `protected_tools` | `PROTECTED_TOOLS` | `find_agents,find_tools,matrix_messaging` | Comma-separated protected tool names |
| `max_total_tools` | `MAX_TOTAL_TOOLS` | `20` | Maximum tools per agent |

## Quick Start Examples

### List Agent Tools

```python
from letta_toolkit import list_agent_tools

# Get all tools for an agent
agent_id = "agent-abc123"
tools = list_agent_tools(agent_id)

print(f"Agent has {len(tools)} tools:")
for tool in tools:
    print(f"  - {tool['name']} ({tool['id']})")
```

### Attach a Tool

```python
from letta_toolkit import attach_tool_to_agent

result = attach_tool_to_agent("agent-abc123", "tool-xyz789")

if result.success:
    print(f"Successfully attached tool {result.tool_id}")
else:
    print(f"Failed: {result.error}")
```

### Ensure Protected Tools

```python
from letta_toolkit import ensure_protected_tools

result = ensure_protected_tools("agent-abc123")

print(f"Already present: {result.already_present}")
print(f"Newly attached: {[t['name'] for t in result.attached]}")
print(f"Failed: {[t['name'] for t in result.failed]}")
```

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed function documentation
- See [Migration Guide](MIGRATION_GUIDE.md) for integrating with existing projects
- Check [Examples](EXAMPLES.md) for common use cases
