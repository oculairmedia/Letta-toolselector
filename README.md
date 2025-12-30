# Letta Tool Selector

An intelligent tool selection and management service for Letta AI agents. This service provides semantic search, automatic tool attachment/detachment, and intelligent pruning capabilities for managing tools across multiple Letta agents.

## Features

- **Semantic Tool Search**: Uses Weaviate vector database with OpenAI embeddings for intelligent tool discovery
- **Automatic Tool Management**: Automatically attaches relevant tools and detaches irrelevant ones based on context
- **MCP Tool Support**: Full support for Model Context Protocol (MCP) tools with automatic registration
- **Header-Aware MCP Requests**: Accepts `x-agent-id` headers so agents no longer need to send `agent_id` in tool payloads
- **Intelligent Pruning**: Configurable tool pruning to maintain optimal tool sets for agents
- **Tool Type Filtering**: Can be configured to manage only MCP tools, excluding Letta core tools
- **RESTful API**: Simple HTTP API for tool search, attachment, and management
- **Low-Latency MCP Worker**: Persistent FastAPI worker eliminates per-request Python process spawning for MCP calls

## Architecture

The system consists of four main services:

1. **API Server** (`api-server`): Main service handling tool search, attachment, and pruning
2. **Sync Service** (`sync-service`): Synchronizes tools between Letta and Weaviate
3. **Worker Service** (`worker-service`): FastAPI process that provides a persistent `find_tools` endpoint for the MCP server with HTTP connection pooling
4. **Time Service** (`time-service`): Manages time-based memory updates


## Further Documentation

- Embeddings usage and architecture: see [EMBEDDINGS_USAGE.md](./EMBEDDINGS_USAGE.md)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/oculairmedia/Letta-toolselector.git
cd Letta-toolselector
```

2. Copy the example environment file and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
docker-compose up -d
```

## Configuration

Key environment variables:

- `LETTA_API_URL`: URL of your Letta instance
- `LETTA_PASSWORD`: Authentication password for Letta
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `MANAGE_ONLY_MCP_TOOLS`: Set to `true` to only manage MCP tools
- `MAX_TOTAL_TOOLS`: Maximum total tools per agent (default: 30)
- `MAX_MCP_TOOLS`: Maximum MCP tools per agent (default: 20)
- `MIN_MCP_TOOLS`: Minimum MCP tools per agent (default: 7)
- `DEFAULT_DROP_RATE`: Tool pruning aggressiveness (0.0-1.0, default: 0.6)
- `WORKER_SERVICE_URL`: Base URL the MCP server uses to reach the worker service (default: `http://worker-service:3021`)
- `WORKER_REQUEST_TIMEOUT_MS`: Request timeout when the MCP server calls the worker (default: `15000`)
- `ENABLE_AGENT_ID_HEADER`: Enable `x-agent-id` header support (default: `true`)
- `REQUIRE_AGENT_ID`: Require an agent identifier from either header or payload (default: `true`)
- `STRICT_AGENT_ID_VALIDATION`: Reject headers that do not match the expected format (default: `false`)
- `DEBUG_AGENT_ID_SOURCE`: Emit log lines showing whether the header or payload provided the agent ID (default: `false`)

## Agent ID Handling

Both MCP entry points (`src/index.js` and `src/simple-server.js`) now resolve agent identity from the `x-agent-id` HTTP header. The header value takes precedence over the `agent_id` argument, letting Letta agents omit the field entirely. When both are supplied they must match; otherwise the request fails with a `-32602` JSON-RPC error.

Example header-only request:

```bash
curl -s http://localhost:3020/mcp \
  -H 'Content-Type: application/json' \
  -H 'x-agent-id: agent-1234' \
  -d '{
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "tools/call",
        "params": {
          "name": "find_tools",
          "arguments": {"query": "graphiti", "limit": 5}
        }
      }'
```

Set `ENABLE_AGENT_ID_HEADER=false` to disable header support or toggle the other environment variables listed above to adjust validation and logging behaviour.

## API Endpoints

### Search Tools
```http
POST /api/v1/tools/search
Content-Type: application/json

{
  "query": "search web content",
  "limit": 10
}
```

### Attach Tools
```http
POST /api/v1/tools/attach
Content-Type: application/json

{
  "agent_id": "agent-uuid",
  "query": "web scraping tools",
  "limit": 5,
  "keep_tools": ["tool-id-1", "tool-id-2"]
}
```

### Prune Tools
```http
POST /api/v1/tools/prune
Content-Type: application/json

{
  "agent_id": "agent-uuid",
  "user_prompt": "current task context",
  "drop_rate": 0.3,
  "keep_tool_ids": ["tool-id-1"]
}
```

### Health Check
```http
GET /api/health
```

## Tool Management Behavior

When `MANAGE_ONLY_MCP_TOOLS=true`:
- Only MCP tools (external_mcp type) are returned in search results
- Letta core tools (send_message, memory functions, etc.) are excluded
- Tool attachment only processes MCP tools
- Tool pruning preserves all Letta core tools

## Development

To mount source code for development:

1. Create a `docker-compose.override.yml`:
```yaml
services:
  api-server:
    volumes:
      - ./.env:/app/.env:ro
      - tool_cache_volume:/app/runtime_cache
      - ./tool-selector-api/api_server.py:/app/api_server.py:ro
      - ./tool-selector-api/weaviate_tool_search.py:/app/weaviate_tool_search.py:ro
```

2. Restart the service:
```bash
docker-compose restart api-server
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines]
