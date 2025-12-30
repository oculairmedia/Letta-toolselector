# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

The Letta Tool Selector is a multi-service intelligent tool management system for Letta AI agents. It uses semantic search via Weaviate vector database to automatically discover, attach, and prune tools based on conversational context.

### Core Services Architecture

The system consists of 5 containerized services working together:

1. **MCP Server** (Node.js, port 3020) - HTTP-based MCP server exposing `find_tools` function
2. **API Server** (Python/Quart, port 8020) - Main tool management with RESTful endpoints  
3. **Weaviate** (port 8080) - Vector database storing tool embeddings for semantic search
4. **Sync Service** (Python background) - Synchronizes tools between Letta API and Weaviate every 5 minutes
5. **Time Service** (Python background) - Manages time-based memory updates for agents

### Key Data Flow

1. Natural language queries → Query expansion with synonyms
2. Weaviate hybrid search (75% vector, 25% keyword) across tool descriptions
3. Unregistered MCP tools are automatically registered with Letta
4. Existing irrelevant tools are detached based on embedding similarity
5. Most relevant tools are attached up to configured limits
6. Optional auto-pruning maintains optimal tool sets

## Common Development Commands

### Starting Services

```bash
# Start full stack
docker-compose -f compose.yaml up -d

# Start individual services
docker-compose up -d weaviate        # Vector database first
docker-compose up -d api-server      # Main API server
docker-compose up -d sync-service    # Background sync
docker-compose up -d mcp-server      # MCP protocol server

# Monitor logs
docker-compose logs -f
docker-compose logs -f api-server    # Specific service
```

### Node.js MCP Server Development

```bash
npm install
npm start                            # or npm run dev
```

### Python API Server Development

```bash
# Setup (Windows)
setup_venv.bat && start_server.bat

# Setup (Linux/macOS)
python -m venv venv && source venv/bin/activate
pip install -r tool-selector-api/requirements.txt
cd tool-selector-api && python api_server.py
```

### Testing

```bash
# Core functionality tests
python test_api_connection.py        # Test Letta API connection
python test_weaviate_search.py       # Test vector search
python test_attach_and_prune.py      # Test tool attachment/pruning
python test_pruning_endpoint.py      # Test pruning logic

# Tool management tests  
python find_attach_tools.py          # Interactive tool attachment
python test_minimum_tools.py         # Test minimum tool enforcement
python interactive_search.py         # Interactive search testing
```

### Health Monitoring

```bash
curl http://localhost:3020/health     # MCP server health
curl http://localhost:8020/health     # API server health
```

## Tool Management Workflow

### Intelligent Pruning System

The system enforces configurable tool limits with intelligent relevance-based pruning:

- **MIN_MCP_TOOLS=7** - Agents always keep at least 7 MCP tools (pruning disabled below this threshold)
- **MAX_MCP_TOOLS=20** - Maximum MCP tools per agent  
- **MAX_TOTAL_TOOLS=30** - Maximum total tools including core Letta tools
- **DEFAULT_DROP_RATE=0.6** - Aggressiveness of pruning (60% drop rate)

### Never-Detach Protection

Critical tools are protected from removal via `NEVER_DETACH_TOOLS` environment variable (default: `find_tools`). The system also protects:
- Core Letta tools (when `EXCLUDE_LETTA_CORE_TOOLS=true`)
- Tools in explicit keep lists
- Newly attached tools

### Tool Type Management

Configure tool management scope:
- `MANAGE_ONLY_MCP_TOOLS=true` - Only manage external MCP tools, ignore Letta core tools
- `EXCLUDE_LETTA_CORE_TOOLS=true` - Don't touch official Letta tools
- `EXCLUDE_OFFICIAL_TOOLS=true` - Exclude official tools from management

## Key API Endpoints

### Core Tool Management
- `POST /api/v1/tools/search` - Semantic tool search with query expansion
- `POST /api/v1/tools/attach` - Attach tools with auto-detachment and optional pruning
- `POST /api/v1/tools/prune` - Manual intelligent pruning based on relevance
- `POST /api/v1/tools/sync` - Force sync between Letta and Weaviate

### MCP Integration
- `POST /mcp` - MCP protocol endpoint for tool discovery

## Critical Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=sk-...                 # Required for embeddings
LETTA_API_URL=https://letta.example.com/v1
LETTA_PASSWORD=your_password
WEAVIATE_URL=http://weaviate:8080/
```

### Tool Limits and Behavior
```bash
MIN_MCP_TOOLS=7                       # Minimum MCP tools (prevents over-pruning)
MAX_MCP_TOOLS=20                      # Maximum MCP tools per agent
MAX_TOTAL_TOOLS=30                    # Maximum total tools per agent
DEFAULT_DROP_RATE=0.6                 # Pruning aggressiveness (0.0-1.0)
NEVER_DETACH_TOOLS=find_tools         # Comma-separated protected tools
```

### Query Expansion for Multifunctional Tools
```bash
ENABLE_QUERY_EXPANSION=true           # Enable automatic query expansion (default: true)
USE_UNIVERSAL_EXPANSION=true          # Use schema-based universal expansion (default: true)
```

The query expansion system improves discovery of tools by dynamically analyzing tool schemas:

**Universal Expansion (Recommended)**
The universal expander (`universal_query_expansion.py`) works by:
1. Analyzing tool JSON schemas to detect operation parameters
2. Building a "tool family" index from tool names (e.g., `create_book`, `delete_book` -> "book" family)
3. Mapping entities to their MCP servers (e.g., "book" -> "bookstack")
4. Detecting operation intent from natural language queries
5. Injecting relevant keywords based on detected patterns

Example: Query "create a book" automatically:
- Detects intent: CREATE
- Detects entity: "book"  
- Finds related tools: create_book, delete_book, list_books, etc.
- Adds MCP server: "bookstack"
- Expands to: "create a book bookstack books crud manage delete list..."

**Legacy Expansion (Fallback)**
If universal expansion is unavailable, falls back to hardcoded mappings in `query_expansion.py`.

## Weaviate Integration

The system uses Weaviate as a vector database for semantic tool search:

- **Schema**: "Tool" collection with vectorized descriptions
- **Embedding Model**: Ollama Qwen3-Embedding-4B (or OpenAI text-embedding-3-small)
- **Search**: Hybrid search (configurable alpha, default 0.75)
- **Cache**: File-based caching at `/app/runtime_cache/tool_cache.json`

### Manual Schema Setup
```bash
python init_weaviate_schema.py        # Initialize Weaviate schema
python upload_tools_to_weaviate.py    # Upload tools with embeddings
```

### Syncing Tools to Weaviate with Ollama Embeddings

When Weaviate is empty (after container restart or data loss), use this command to re-sync all tools with Ollama embeddings:

```bash
cd /opt/stacks/lettatoolsselector/tool-selector-api

# Export all required environment variables and run the upload script
export EMBEDDING_PROVIDER=ollama && \
export USE_OLLAMA_EMBEDDINGS=true && \
export OLLAMA_EMBEDDING_HOST=192.168.50.80 && \
export OLLAMA_EMBEDDING_MODEL='dengcao/Qwen3-Embedding-4B:Q4_K_M' && \
export OPENAI_API_KEY='${OPENAI_API_KEY}' && \
export LETTA_API_URL='https://letta2.oculair.ca/v1' && \
export LETTA_PASSWORD='lettaSecurePass123' && \
python3 upload_tools_to_weaviate.py

# This will:
# 1. Fetch all tools from Letta API (typically ~179 tools)
# 2. Generate embeddings using Ollama's Qwen3-Embedding-4B model
# 3. Upload tools with embeddings to Weaviate for semantic search
# 4. Show progress during upload and report success/skip counts
```

## Development Patterns

### Letta SDK Migration

The API server supports two modes for communicating with the Letta API:

1. **aiohttp mode** (default): Direct HTTP calls using aiohttp
2. **SDK mode**: Uses the official `letta-client` SDK (v1.3.1+)

Enable SDK mode with `USE_LETTA_SDK=true`. The SDK client wrapper is in `tool-selector-api/letta_sdk_client.py`.

Migrated functions (support both modes):
- `detach_tool()` - Detach tools from agents
- `attach_tool()` - Attach tools to agents  
- `fetch_agent_info()` - Get agent name
- `fetch_agent_tools()` - List agent's tools
- `register_tool()` - Register MCP tools

The SDK wrapper uses a thread pool executor to run synchronous SDK calls in async context, maintaining compatibility with the Quart async architecture.

### Async Architecture
The API server uses async/await throughout with Quart (async Flask) and aiohttp for high concurrency tool operations.

### Error Handling
Services implement graceful degradation with retry logic and comprehensive health monitoring. Failed operations don't cascade to break the entire workflow.

### Tool Registration Flow
1. MCP tools are discovered from configured MCP servers
2. Unregistered tools are automatically registered with Letta API
3. Tools are indexed in Weaviate with embeddings for search
4. Cache is updated for fast subsequent operations

### Tool Protection Hierarchy
1. **Core Letta tools** - Never touched when `EXCLUDE_LETTA_CORE_TOOLS=true`
2. **Never-detach list** - Protected by `NEVER_DETACH_TOOLS` environment variable
3. **Explicit keep lists** - Protected in API calls via `keep_tool_ids` parameter
4. **Newly attached tools** - Protected during same operation via `newly_matched_tool_ids`
5. **Minimum threshold** - At least `MIN_MCP_TOOLS` MCP tools always preserved

## Utility Scripts

### Tool Operations
```bash
python find_tools_enhanced.py         # Advanced tool discovery with rules
python detach_mcp_tools.py            # Bulk detach MCP tools from agent
python delete_tools.py                # Permanently delete tools from system
python remove_obsolete_tools.py       # Clean up obsolete tools between Letta/Weaviate
```

### Analysis and Debugging
```bash
python letta_tool_utils.py            # Dynamic tool ID lookup utilities
python check_mcp_tools.py             # Check MCP tool registration status
python compare_tools.py               # Compare tool sets between agents
```