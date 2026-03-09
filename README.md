# Letta Tool Selector

Intelligent tool management for [Letta](https://github.com/letta-ai/letta) AI agents. Automatically discovers, attaches, and prunes tools using semantic search so agents always have the right tools for the task.

## How It Works

1. Agent sends a natural language query (e.g. "I need to search the web")
2. Query is expanded with synonyms and matched against tool embeddings in Weaviate
3. Most relevant tools are attached to the agent; irrelevant ones are pruned
4. Configurable limits and protection rules prevent over-pruning

## Architecture

Nine containerized services orchestrated via `compose.yaml`:

| Service | Stack | Port | Role |
|---------|-------|------|------|
| **MCP Server** | Node.js | 3020 | HTTP-based MCP server, exposes `find_tools` |
| **API Server** | Python/Quart | 8020 | Tool search, attach, prune REST API |
| **Worker Service** | Python/FastAPI | 3021 | Persistent `find_tools` endpoint for MCP |
| **Weaviate** | Go | 8091 | Vector DB for tool embeddings |
| **Embedding Proxy** | Python/FastAPI | 8450 | Rewrites OpenAI model names for vLLM |
| **Sync Service** | Python | — | Syncs tools between Letta API and Weaviate |
| **Time Service** | Python | — | Time-based memory block updates |
| **Dashboard API** | Python/FastAPI | 8025 | Dashboard backend |
| **Dashboard UI** | React | 3001 | Dashboard frontend |

## Quick Start

```bash
git clone https://github.com/oculairmedia/Letta-toolselector.git
cd Letta-toolselector
cp .env.example .env   # Edit with your configuration
docker compose up -d
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/tools/search` | Semantic tool search |
| `POST` | `/api/v1/tools/attach` | Attach tools with auto-detach + optional prune |
| `POST` | `/api/v1/tools/prune` | Intelligent relevance-based pruning |
| `POST` | `/api/v1/tools/sync` | Force Letta→Weaviate sync |
| `POST` | `/mcp` | MCP protocol endpoint |
| `GET`  | `/api/health` | Health check |

### Example: Search for tools

```bash
curl -s http://localhost:8020/api/v1/tools/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "search web content", "limit": 10}'
```

### Example: Attach tools to an agent

```bash
curl -s http://localhost:8020/api/v1/tools/attach \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": "agent-uuid",
    "query": "web scraping tools",
    "limit": 5
  }'
```

### Example: MCP call with agent ID header

```bash
curl -s http://localhost:3020/mcp \
  -H 'Content-Type: application/json' \
  -H 'x-agent-id: agent-1234' \
  -d '{
    "jsonrpc": "2.0", "id": "1",
    "method": "tools/call",
    "params": {"name": "find_tools", "arguments": {"query": "graphiti", "limit": 5}}
  }'
```

## Configuration

### Required

| Variable | Description |
|----------|-------------|
| `LETTA_API_URL` | Letta API endpoint |
| `LETTA_PASSWORD` | Letta authentication |
| `COHERE_API_KEY` | Cohere API key for embeddings and reranking |
| `WEAVIATE_URL` | Weaviate endpoint (default: `http://weaviate:8080/`) |

### Tool Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_MCP_TOOLS` | `7` | Minimum MCP tools per agent (prevents over-pruning) |
| `MAX_MCP_TOOLS` | `20` | Maximum MCP tools per agent |
| `MAX_TOTAL_TOOLS` | `30` | Maximum total tools including core Letta tools |
| `DEFAULT_DROP_RATE` | `0.6` | Pruning aggressiveness (0.0–1.0) |
| `NEVER_DETACH_TOOLS` | `find_tools` | Comma-separated tools protected from removal |
| `MANAGE_ONLY_MCP_TOOLS` | `true` | Only manage MCP tools, ignore Letta core tools |

### Agent ID Handling

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AGENT_ID_HEADER` | `true` | Accept `x-agent-id` HTTP header |
| `REQUIRE_AGENT_ID` | `true` | Require agent ID from header or payload |
| `WORKER_SERVICE_URL` | `http://worker-service:3021` | Worker service URL |
| `WORKER_REQUEST_TIMEOUT_MS` | `15000` | Worker request timeout |

## Development

Mount source code for live reloading:

```yaml
# docker-compose.override.yml
services:
  api-server:
    volumes:
      - ./.env:/app/.env:ro
      - tool_cache_volume:/app/runtime_cache
      - ./tool-selector-api/api_server.py:/app/api_server.py:ro
```

```bash
docker compose restart api-server
```

## Project Structure

```
├── compose.yaml              # Main service orchestration
├── tool-selector-api/        # Python API server + tool management
├── worker-service/           # FastAPI worker for MCP
├── src/                      # Node.js MCP server
├── embedding-proxy/          # vLLM embedding proxy
├── dashboard-ui/             # React dashboard frontend
├── dashboard-backend/        # FastAPI dashboard backend
├── docs/                     # Active reference documentation
│   └── archive/              # Historical analysis & status reports
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── letta_tool_utils.py       # Shared: dynamic tool ID lookup
├── tool_selector_client.py   # Shared: tool selector API client
├── qwen3_reranker_utils.py   # Shared: Qwen3 reranker formatting
└── ollama_reranker_adapter.py # Shared: Ollama reranker adapter
```

## Documentation

- **[API Contract](docs/API_CONTRACT.md)** — Request/response schemas
- **[Deployment Guide](docs/DEPLOYMENT.md)** — Production deployment
- **[Compose Setup](docs/COMPOSE_SETUP.md)** — Docker Compose configuration
- **[Embeddings](docs/EMBEDDINGS_USAGE.md)** — Embedding providers and architecture
- **[Tool Selector Guide](docs/TOOL_SELECTOR_GUIDE.md)** — User guide
