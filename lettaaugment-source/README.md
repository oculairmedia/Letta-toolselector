# LettaSearch MCP Tool

A system designed to enhance Letta agents with dynamic tool management capabilities. It allows agents to search for, attach, and detach tools based on natural language queries.

## Overview

The LettaSearch MCP Tool is a system designed to enhance Letta agents with dynamic tool management capabilities. It allows agents to search for, attach, and detach tools based on natural language queries, making them more versatile and capable of handling a wider range of tasks.

The system consists of several components that work together to provide seamless tool management:

1. **API Server**: A Flask-based server that handles requests for tool search and attachment
2. **Weaviate Integration**: Vector database for semantic search of tools
3. **MCP Tool Management**: Handles registration and management of MCP (Model Context Protocol) tools
4. **Docker Containerization**: Ensures consistent deployment across environments

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Letta Agent   │◄────┤  LettaSearch    │◄────┤    Weaviate     │
│                 │     │   MCP Tool      │     │  Vector Search  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               ▲
                               │
                               ▼
                        ┌─────────────────┐
                        │                 │
                        │   MCP Servers   │
                        │                 │
                        └─────────────────┘
```

## Components

### API Server (`api_server.py`)

The core of the system, handling HTTP requests for:
- Tool search
- Tool attachment/detachment
- Tool synchronization

The server provides endpoints that allow agents to find and attach tools based on natural language queries, automatically managing the lifecycle of tools attached to an agent.

### Weaviate Tool Search (`weaviate_tool_search.py`)

Provides semantic search capabilities using the Weaviate vector database:
- Performs hybrid search (vector + keyword)
- Expands queries with synonyms for better matching
- Returns ranked results with relevance scores

### Tool Management Scripts

- `find_attach_tools_v3.py`: Client script for finding and attaching tools
- `upload_tools_to_weaviate.py`: Synchronizes tools with Weaviate
- `detach_mcp_tools.py`: Handles detachment of MCP tools

## Docker Setup

The system is containerized using Docker for easy deployment and scaling.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY *.json .

ENV WEAVIATE_URL=${WEAVIATE_URL}
ENV WEAVIATE_API_KEY=${WEAVIATE_API_KEY}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

EXPOSE 3001

CMD ["python", "api_server.py"]
```

### Docker Compose

The `docker-compose.yml` file defines the services. Note that there are two main configurations:

1.  **Local Development:** Uses `build: .` to build the image directly from local source code. This is useful for testing changes locally.
2.  **Remote Deployment:** Uses `image: oculair/lettaaugment:latest` to pull the pre-built image from a registry. This is the standard configuration for deploying to a server.

**Example `docker-compose.yml` for Remote Deployment:**

```yaml
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.0
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "text2vec-openai"
      ENABLE_MODULES: "text2vec-openai"
      OPENAI_APIKEY: ${OPENAI_API_KEY}
      CLUSTER_HOSTNAME: "node1"
      READINESS_MAX_WAIT_SECS: 300
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - letta-tools

  api-server:
    image: oculair/lettaaugment:latest # Use pre-built image
    container_name: weaviate-tools-api
    ports:
      - "8020:3001"
    environment:
      - WEAVIATE_URL=${WEAVIATE_URL}
      - WEAVIATE_API_KEY=${WEAVIATE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=3001
    restart: unless-stopped
    volumes:
      - ./.env:/app/.env:ro
      - tool_cache_volume:/app/cache # Add shared volume for cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    networks:
      - letta-tools

  sync-service:
    image: oculair/lettaaugment:latest # Use pre-built image
    container_name: weaviate-tools-sync
    command: python sync_service.py
    environment:
      - WEAVIATE_URL=${WEAVIATE_URL}
      - WEAVIATE_API_KEY=${WEAVIATE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - API_URL=http://api-server:3001
      - SYNC_INTERVAL=300
    volumes:
      - ./.env:/app/.env:ro
      - tool_cache_volume:/app/cache # Add shared volume for cache
    restart: unless-stopped
    depends_on:
      api-server:
        condition: service_healthy
    networks:
      - letta-tools

  time-service:
    image: oculair/lettaaugment:latest # Use pre-built image
    container_name: weaviate-tools-time
    command: python time_memory_service.py
    environment:
      - WEAVIATE_URL=${WEAVIATE_URL}
      - WEAVIATE_API_KEY=${WEAVIATE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LETTA_API_URL=https://letta2.oculair.ca/v1
      - UPDATE_INTERVAL=60
    volumes:
      - ./.env:/app/.env:ro
    restart: unless-stopped
    networks:
      - letta-tools

networks:
  letta-tools:
    driver: bridge

volumes:
  weaviate_data:
  tool_cache_volume: # Define the shared volume
```

### Environment Variables

Required environment variables in `.env`:
- `WEAVIATE_URL`: URL of the Weaviate instance
- `WEAVIATE_API_KEY`: API key for Weaviate
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `LETTA_URL`: URL of the Letta API (default: https://letta2.oculair.ca/v1)

## API Endpoints

### 1. `/api/v1/tools/search` (POST)

Search for tools based on a query.

**Request:**
```json
{
  "query": "github repository",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "name": "github-mcp-server__create_repository",
    "description": "Create a new GitHub repository",
    "distance": 0.15,
    "tags": ["github", "repository", "create"]
  },
  ...
]
```

### 2. `/api/v1/tools/attach` (POST)

Find and attach tools to an agent.

**Request:**
```json
{
  "query": "github repository",
  "agent_id": "agent-33718e73-f85a-4cfd-a42b-d0a6feeaf5a5",
  "limit": 2,
  "keep_tools": []
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 2 candidates, attached 2 tool(s) to agent agent-33718e73-f85a-4cfd-a42b-d0a6feeaf5a5",
  "details": {
    "detached_tools": ["tool-d9c0e4d1-13f5-46b2-acb7-963771693185", "tool-95decb78-5c96-4229-b262-a7757382dcdf"],
    "failed_detachments": [],
    "processed_count": 2,
    "passed_filter_count": 2,
    "success_count": 2,
    "failure_count": 0,
    "successful_attachments": [...],
    "failed_attachments": [],
    "preserved_tools": [],
    "target_agent": "agent-33718e73-f85a-4cfd-a42b-d0a6feeaf5a5"
  }
}
```

### 3. `/api/v1/tools/sync` (POST)

Synchronize tools with Weaviate.

**Response:**
```json
{
  "success": true,
  "message": "Tool synchronization completed",
  "details": {
    "uploaded": 0,
    "skipped": 0,
    "failed": 0
  }
}
```

### 4. `/api/health` (GET)

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Tool Detachment Process

The system manages tool detachment to ensure that agents don't have too many tools attached at once, which could cause context window issues.

### How Tool Detachment Works:

1. When new tools are attached, the system identifies existing MCP tools that need to be detached
2. Tools specified in the `keep_tools` list are preserved
3. Detachment is performed sequentially with retry logic for reliability
4. The system logs detailed information about the detachment process

### Recent Fix: Variable Name Conflict

A critical issue was fixed where a variable name conflict was causing tool detachment to fail:

```python
# Before (problematic code)
mcp_tools_response = requests.get(f"{LETTA_URL}/tools/mcp/servers/{server_name}/tools", headers=HEADERS)
mcp_tools_response.raise_for_status()
mcp_tools = mcp_tools_response.json()  # This was overwriting the mcp_tools list!

# After (fixed code)
mcp_tools_response = requests.get(f"{LETTA_URL}/tools/mcp/servers/{server_name}/tools", headers=HEADERS)
mcp_tools_response.raise_for_status()
server_mcp_tools = mcp_tools_response.json()  # Renamed to avoid conflict
```

This fix ensures that the list of MCP tools from the agent is not overwritten during the tool registration process, allowing proper identification of tools that need to be detached.

## Deployment

### Building and Pushing the Docker Image (for Remote Deployment)

To prepare the image for deployment on a remote server, build it locally and push it to a Docker registry (like Docker Hub).

```bash
# Build the image using buildx (recommended)
docker buildx build -t oculair/lettaaugment:latest .

# Push the image to the registry
docker push oculair/lettaaugment:latest
```

### Running the Application

**Local Development:**

For local development and testing, you can build and run the services directly using Docker Compose. Ensure your `docker-compose.yml` uses the `build: .` directive for the relevant services.

```bash
# Build (if needed) and start all services
docker-compose up --build -d
```

**Remote Deployment:**

On the remote server, ensure you have the `docker-compose.yml` file configured to use the `image: oculair/lettaaugment:latest` directive (as shown in the Docker Setup section). You also need the `.env` file with the required environment variables.

```bash
# Pull the latest image if necessary
docker pull oculair/lettaaugment:latest

# Start all services using the pre-built image
docker-compose up -d
```

### Managing the Services (using Docker Compose)

```bash
# Stop all services
docker-compose down

# View logs for all services
docker-compose logs -f

# View logs for a specific service (e.g., api-server)
docker-compose logs -f api-server

# Restart all services
docker-compose restart
```

## Troubleshooting

### Common Issues

1. **Tool Detachment Failures**
   - Check if the Letta API is accessible
   - Verify that the agent ID is correct
   - Look for error messages in the logs

2. **Search Returns No Results**
   - Ensure Weaviate is properly configured
   - Check if tools have been synchronized with Weaviate
   - Try broadening your search query

3. **Docker Container Crashes**
   - Check environment variables
   - Verify network connectivity to Weaviate and Letta API
   - Inspect logs for error messages

### Debugging

For detailed debugging, use:

```bash
docker logs lettaaugment-prod
```

## Client Usage

The `find_attach_tools_v3.py` script can be used as a client to interact with the API:

```bash
python find_attach_tools_v3.py --query "github repository" --agent_id "agent-33718e73-f85a-4cfd-a42b-d0a6feeaf5a5" --limit 5
```

## Documentation

For more detailed documentation, please see:

- [Full Documentation](https://knowledge.oculair.ca/books/dynamic-tool-loading-for-letta/page/lettasearch-mcp-tool-documentation)
- [MCP Tool Integration Technical Documentation](MCP_TOOL_INTEGRATION.md)

## Project Management

This project is managed using Plane. You can find the project at:

- [LettaSearch MCP Tool Project](https://plane.oculair.ca/workspace/e7e61bde-61d2-4489-b1fd-1a162ceb833f/projects/a90ccb76-3ba3-41d5-a21e-1434618c8d73/issues)

## Conclusion

The LettaSearch MCP Tool provides a powerful way to enhance Letta agents with dynamic tool management. By leveraging semantic search and automated tool lifecycle management, it enables agents to adapt to different tasks and user needs.

The system is designed to be reliable, with features like sequential detachment with retry logic, and efficient, using hybrid search to find the most relevant tools for a given query.
