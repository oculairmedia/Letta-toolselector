# Test Organization

This directory contains all test files organized by functionality. Each subdirectory focuses on specific aspects of the Letta Tool Selector system.

## Directory Structure

### 📡 `/api/` - API & Server Tests
Integration and functionality tests for the API server and search endpoints.
- `test_api_connection.py` - Basic API connectivity tests
- `test_api_direct.py` - Direct API endpoint testing
- `test_cost_control_api.py` - Cost control API endpoint tests
- `test_reranker_api_contract.py` - Reranker API contract validation
- `test_advanced_search.py` - Advanced search functionality tests
- `test_comprehensive_search.py` - Comprehensive search scenario tests
- `test_search.py` - Basic search functionality tests
- `test_strategies.py` - Search strategy tests
- `test_remote_server.py` - Remote server integration tests

### 🗄️ `/weaviate/` - Vector Database Tests
Tests for Weaviate vector database integration and operations.
- `test_weaviate_integration.py` - Core Weaviate integration tests
- `test_weaviate_attach.py` - Tool attachment via Weaviate
- `test_weaviate_embedding.py` - Embedding storage and retrieval
- `test_weaviate_search.py` - Vector search functionality
- `test_weaviate_tools.py` - Tool management in Weaviate

### 🧠 `/embedding/` - Embedding & Vector Tests
Tests for embedding generation, storage, and retrieval across different providers.
- `test_embedding_debug.py` - Embedding generation debugging
- `test_embedding_fixed.py` - Fixed embedding functionality tests
- `test_embedding_integration.py` - End-to-end embedding integration
- `test_embedding_local.py` - Local embedding provider tests
- `test_remote_embedding.py` - Remote embedding provider tests
- `test_specialized_embedding.py` - Specialized embedding configurations
- `test_graphiti_embedding_config.py` - Graphiti embedding configuration tests

### 🔄 `/reranking/` - Reranking & Search Optimization Tests
Tests for result reranking and search result optimization.
- `test_improved_reranking.py` - Enhanced reranking functionality
- `test_reranker_integration.py` - Reranker service integration
- `test_reranking_integration.py` - End-to-end reranking tests
- `test_search_comparison.py` - Original vs reranked search comparison
- `debug_reranker.py` - Reranker debugging utilities

### 🛠️ `/tools/` - Tool Management Tests
Tests for tool discovery, attachment, pruning, and lifecycle management.
- `test_agent_lookup.py` - Agent discovery and lookup
- `test_attach_and_prune.py` - Tool attachment and pruning logic
- `test_detach_tools.py` - Tool detachment functionality
- `test_dynamic_lookup.py` - Dynamic tool discovery
- `test_tool_lookup.py` - Tool lookup utilities
- `test_minimum_tools.py` - Minimum tool enforcement
- `test_min_mcp_tools.py` - MCP tool minimum requirements
- `test_never_detach_config.py` - Protected tool configuration
- `test_pruning_endpoint.py` - Pruning endpoint tests
- `test_remote_prune_detailed.py` - Detailed remote pruning tests
- `test_simple_prune.py` - Basic pruning functionality
- `test_performance_validation.py` - Performance validation tests
- `test_postizz_mcp_compliance.py` - MCP compliance validation
- `test_cost_control.py` - Cost control and limits
- `test_env_parsing.py` - Environment configuration parsing
- `test_prompts_only.py` - Prompt-only tool operations

### 📊 `/dashboard/` - Dashboard & Frontend Tests
Tests for the web dashboard and frontend functionality.
- `ab_testing_framework.py` - A/B testing framework for frontend

### 🔍 `/debug/` - Debug & Development Tools
Debug utilities and development testing tools.
- `debug_existing_embeddings.py` - Debug existing embedding data
- `test_check_logs.py` - Log checking and validation utilities

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/api/ -v
python -m pytest tests/weaviate/ -v
python -m pytest tests/embedding/ -v
```

### Run Individual Tests
```bash
# Run specific test file
python tests/api/test_api_connection.py
python tests/weaviate/test_weaviate_integration.py
```

### Test Environment Setup
Most tests require:
1. Running Weaviate instance (port 8080)
2. API server running (port 8020)  
3. Environment variables configured in `.env`
4. MCP server running (port 3020)

```bash
# Start services
docker-compose -f compose.yaml up -d

# Run tests after services are ready
python -m pytest tests/ -v
```

## Test Categories by Complexity

### 🟢 Unit Tests (Fast, Isolated)
- Most files in `/tools/` for configuration parsing
- Basic API connectivity tests
- Embedding provider unit tests

### 🟡 Integration Tests (Medium, Service Dependencies) 
- Weaviate integration tests
- API endpoint tests with database
- Tool attachment/detachment tests

### 🔴 End-to-End Tests (Slow, Full System)
- Comprehensive search tests
- Reranking integration tests
- Performance validation tests

## Adding New Tests

When adding new test files, place them in the appropriate category:

1. **API/Server functionality** → `/api/`
2. **Vector database operations** → `/weaviate/`
3. **Embedding generation/retrieval** → `/embedding/`
4. **Result reranking** → `/reranking/`
5. **Tool management** → `/tools/`
6. **Web interface** → `/dashboard/`
7. **Debug utilities** → `/debug/`

Follow the naming convention: `test_<functionality>.py` for test files, `debug_<component>.py` for debug utilities.