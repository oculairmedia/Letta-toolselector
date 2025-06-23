# Letta Tool Selector

An intelligent tool selection and management service for Letta AI agents. This service provides semantic search, automatic tool attachment/detachment, and intelligent pruning capabilities for managing tools across multiple Letta agents.

## Features

- **Semantic Tool Search**: Uses Weaviate vector database with OpenAI embeddings for intelligent tool discovery
- **Automatic Tool Management**: Automatically attaches relevant tools and detaches irrelevant ones based on context
- **MCP Tool Support**: Full support for Model Context Protocol (MCP) tools with automatic registration
- **Intelligent Pruning**: Configurable tool pruning to maintain optimal tool sets for agents
- **Tool Type Filtering**: Can be configured to manage only MCP tools, excluding Letta core tools
- **RESTful API**: Simple HTTP API for tool search, attachment, and management

## Architecture

The system consists of three main services:

1. **API Server** (`api-server`): Main service handling tool search, attachment, and pruning
2. **Sync Service** (`sync-service`): Synchronizes tools between Letta and Weaviate
3. **Time Service** (`time-service`): Manages time-based memory updates

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
- `DEFAULT_DROP_RATE`: Tool pruning aggressiveness (0.0-1.0, default: 0.6)

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
      - ./lettaaugment-source/api_server.py:/app/api_server.py:ro
      - ./lettaaugment-source/weaviate_tool_search.py:/app/weaviate_tool_search.py:ro
```

2. Restart the service:
```bash
docker-compose restart api-server
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines]