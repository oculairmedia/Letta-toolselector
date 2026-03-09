# Letta Tool Selector - Retrieval Pipeline Testing Guide

## Overview

This document provides comprehensive guidance for testing the retrieval pipeline of the Letta Tool Selector system. The pipeline uses semantic search via Weaviate vector database to automatically discover, attach, and prune tools based on conversational context.

## Architecture Components

### Core Services

1. **API Server** (Python/Quart, port 8020) - Main tool management with RESTful endpoints
2. **MCP Server** (Node.js, port 3020) - HTTP-based MCP server exposing `find_tools` function
3. **Weaviate** (port 8080) - Vector database storing tool embeddings for semantic search
4. **Sync Service** (Python background) - Synchronizes tools between Letta API and Weaviate every 5 minutes
5. **Dashboard Backend** (Python, port 8030) - Backend API for the management dashboard
6. **Frontend** (React, port 8406) - Web-based management interface

### Data Flow

1. **Query Processing**: Natural language queries → Query expansion with synonyms
2. **Search**: Weaviate hybrid search (75% vector, 25% keyword) across tool descriptions
3. **Registration**: Unregistered MCP tools are automatically registered with Letta
4. **Pruning**: Existing irrelevant tools are detached based on embedding similarity
5. **Attachment**: Most relevant tools are attached up to configured limits

## Testing Environment Setup

### Prerequisites

```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Navigate to the project directory
cd /opt/stacks/lettatoolsselector
```

### Starting the Full Stack

```bash
# Start all services
docker-compose -f compose.yaml up -d

# Verify services are running
docker-compose ps

# Check service health
curl http://localhost:8020/api/v1/health  # API Server
curl http://localhost:3020/health         # MCP Server
curl http://localhost:8030/api/v1/health  # Dashboard Backend
curl http://localhost:8080/v1/meta        # Weaviate
```

### Service URLs

- **API Server**: http://localhost:8020
- **MCP Server**: http://localhost:3020
- **Dashboard Backend**: http://localhost:8030
- **Frontend Dashboard**: http://localhost:8406
- **Weaviate**: http://localhost:8080

## Key Testing Endpoints

### 1. Tool Search and Discovery

#### Basic Tool Search
```bash
curl -X POST http://localhost:8020/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "file management and text processing",
    "limit": 10,
    "min_score": 30.0
  }'
```

#### Advanced Search with Reranking
```bash
curl -X POST http://localhost:8020/api/v1/tools/search/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data analysis and visualization",
    "limit": 15,
    "min_score": 25.0,
    "rerank_top_k": 10,
    "include_details": true
  }'
```

### 2. Tool Attachment and Management

#### Attach Tools to Agent
```bash
curl -X POST http://localhost:8020/api/v1/tools/attach \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-12345",
    "query": "I need tools for web scraping and data extraction",
    "limit": 5,
    "min_score": 40.0,
    "auto_prune": true,
    "drop_rate": 0.6
  }'
```

#### Manual Tool Pruning
```bash
curl -X POST http://localhost:8020/api/v1/tools/prune \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-12345",
    "user_prompt": "Remove tools not related to data processing",
    "drop_rate": 0.5,
    "keep_tool_ids": ["tool-important-1", "tool-critical-2"]
  }'
```

### 3. MCP Protocol Testing

#### MCP Find Tools
```bash
curl -X POST http://localhost:3020/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: test-session-123" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "find_tools",
      "arguments": {
        "query": "email automation and notification systems",
        "agent_id": "agent-test-456",
        "limit": 8,
        "min_score": 35.0,
        "detailed_response": true
      }
    }
  }'
```

### 4. System Configuration and Health

#### Get System Configuration
```bash
curl -X GET http://localhost:8020/api/v1/config/tool-selector
```

#### Test Weaviate Connection
```bash
curl -X POST http://localhost:8020/api/v1/config/weaviate/test \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://localhost:8080",
    "api_key": null
  }'
```

#### Get Embedding Configuration
```bash
curl -X GET http://localhost:8020/api/v1/config/embedding
```

### 5. Benchmarking and Performance Testing

#### Create Benchmark Query Set
```bash
curl -X POST http://localhost:8020/api/v1/benchmark/query-sets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-retrieval-benchmark",
    "description": "Testing retrieval pipeline performance",
    "queries": [
      {
        "text": "file operations and text processing",
        "expected_tool_categories": ["filesystem", "text_processing"]
      },
      {
        "text": "web scraping and data extraction",
        "expected_tool_categories": ["web_scraping", "data_extraction"]
      }
    ]
  }'
```

#### Run Benchmark
```bash
curl -X POST http://localhost:8020/api/v1/benchmark/query-sets/{query_set_id}/run \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "min_score": 30.0,
    "use_reranking": true
  }'
```

## Testing Scenarios

### Scenario 1: Basic Retrieval Testing

```bash
#!/bin/bash
# Test basic retrieval functionality

echo "Testing basic tool search..."
curl -X POST http://localhost:8020/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database operations and SQL queries",
    "limit": 5,
    "min_score": 30.0
  }' | jq '.results[] | {name: .name, score: .score, description: .description}'

echo "Testing MCP protocol search..."
curl -X POST http://localhost:3020/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: test-basic" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "find_tools",
      "arguments": {
        "query": "image processing and computer vision",
        "limit": 3,
        "min_score": 40.0
      }
    }
  }' | jq '.result.tools[]'
```

### Scenario 2: End-to-End Tool Attachment

```bash
#!/bin/bash
# Test complete tool attachment workflow

AGENT_ID="test-agent-$(date +%s)"
echo "Testing with agent ID: $AGENT_ID"

echo "1. Attaching tools..."
ATTACH_RESULT=$(curl -s -X POST http://localhost:8020/api/v1/tools/attach \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_id\": \"$AGENT_ID\",
    \"query\": \"machine learning and data science workflows\",
    \"limit\": 8,
    \"min_score\": 35.0,
    \"auto_prune\": false
  }")

echo "Attachment result:"
echo $ATTACH_RESULT | jq '.attached_tools[] | {name: .name, score: .score}'

echo "2. Testing pruning..."
curl -s -X POST http://localhost:8020/api/v1/tools/prune \
  -H "Content-Type: application/json" \
  -d "{
    \"agent_id\": \"$AGENT_ID\",
    \"user_prompt\": \"Keep only tools for data analysis, remove visualization tools\",
    \"drop_rate\": 0.4
  }" | jq '.pruning_summary'
```

### Scenario 3: Performance and Load Testing

```bash
#!/bin/bash
# Test system performance under load

echo "Testing concurrent searches..."
for i in {1..10}; do
  (
    curl -s -X POST http://localhost:8020/api/v1/tools/search \
      -H "Content-Type: application/json" \
      -d "{
        \"query\": \"test query $i for concurrent testing\",
        \"limit\": 5
      }" > /dev/null
    echo "Request $i completed"
  ) &
done
wait

echo "All concurrent requests completed"
```

## Monitoring and Debugging

### Service Logs

```bash
# View API server logs
docker-compose logs -f api-server

# View MCP server logs
docker-compose logs -f mcp-server

# View Weaviate logs
docker-compose logs -f weaviate

# View all logs
docker-compose logs -f
```

### Health Monitoring

```bash
# Check all service health
curl http://localhost:8020/api/v1/health
curl http://localhost:3020/health
curl http://localhost:8030/api/v1/health

# Check Weaviate schema
curl http://localhost:8080/v1/schema

# Get system status
curl http://localhost:8020/api/v1/maintenance/status
```

### Performance Metrics

```bash
# Get analytics data
curl http://localhost:8020/api/v1/analytics

# Check Weaviate connection pool stats
curl http://localhost:8020/api/v1/weaviate/pool-stats

# Get cost control status (if enabled)
curl http://localhost:8020/api/v1/cost-control/status
```

## Common Testing Utilities

### Python Testing Scripts

The system includes several Python testing utilities located in `/opt/stacks/lettatoolsselector/tool-selector-api/`:

```bash
# Test API connection
python test_api_connection.py

# Test Weaviate search
python test_weaviate_search.py

# Test tool attachment and pruning
python test_attach_and_prune.py

# Interactive search testing
python interactive_search.py

# Test search accuracy
python test_search_accuracy.py
```

### Dashboard Testing

Access the web dashboard at http://localhost:8406 to:

- View real-time system status
- Test search queries interactively
- Monitor tool attachment operations
- Configure system parameters
- View performance analytics

## Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker logs and ensure ports are available
2. **Search returning no results**: Verify Weaviate has tools indexed
3. **MCP protocol errors**: Check session ID headers and JSON format
4. **Performance issues**: Monitor resource usage and connection pools

### Resetting the System

```bash
# Stop all services
docker-compose down

# Clear volumes and restart
docker-compose down -v
docker-compose up -d

# Re-sync tools to Weaviate
curl -X POST http://localhost:8020/api/v1/tools/sync
```

### Environment Variables

Key configuration variables for testing:

```bash
# Tool limits
MAX_TOTAL_TOOLS=30
MAX_MCP_TOOLS=20
MIN_MCP_TOOLS=7

# Search parameters
DEFAULT_MIN_SCORE=35.0
DEFAULT_DROP_RATE=0.6

# Embedding configuration
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_HOST=192.168.50.80
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M

# Reranking
ENABLE_RERANKING=true
RERANK_TOP_K=10
```

## API Documentation

For complete API documentation, visit:
- Main API: http://localhost:8020/docs (if Swagger is enabled)
- Dashboard API: http://localhost:8030/api/v1/docs

This testing guide provides comprehensive coverage for validating the retrieval pipeline functionality, performance, and reliability of the Letta Tool Selector system.