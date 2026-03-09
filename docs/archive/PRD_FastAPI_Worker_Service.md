# Product Requirements Document: FastAPI Worker Service for MCP Performance Optimization

## Executive Summary

The current MCP server implementation spawns a new Python process for every `find_tools` call, resulting in 50-150ms performance penalty per request. This PRD outlines the implementation of a persistent FastAPI worker service to eliminate process spawning overhead and implement HTTP connection pooling for significant performance improvements.

**Business Impact:**
- Eliminate 50-150ms process startup penalty per MCP call
- Reduce network overhead through HTTP connection pooling
- Improve system reliability and resource efficiency
- Enable better monitoring and scaling capabilities

## Problem Statement

### Current Performance Issues

Based on code analysis of the MCP server implementation:

#### 1. Process Spawning Overhead
**File:** `src/index.js` (Lines 149-225)
**Issue:** Every `find_tools` call spawns a new Python process via `child_process.spawn()`

```javascript
// Current (INEFFICIENT) - Lines 149-150
const child = spawn(pythonArgs[0], pythonArgs.slice(1));
```

**Performance Impact:**
- 50-150ms Python interpreter startup time per call
- Module import overhead (`requests`, `json`, `letta_tool_utils`)
- Process creation/destruction resource waste
- No connection reuse to API server

#### 2. No HTTP Connection Pooling
**File:** `find_tools.py` (Lines 135-145)
**Issue:** Each Python process creates new HTTP connections to API server

```python
# Current (INEFFICIENT) - New connection every call
response = requests.post(ATTACH_ENDPOINT, json=payload, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
```

#### 3. Resource Inefficiency
- Memory allocation/deallocation for each process
- TCP connection establishment overhead
- No session reuse between calls
- Unbounded process creation under load

## Requirements

### Functional Requirements

#### FR1: Persistent Worker Service
- **Priority:** Critical
- **Description:** Create FastAPI service that replaces Python process spawning
- **Acceptance Criteria:**
  - FastAPI service runs persistently on dedicated port (3021)
  - Implements `/find_tools` endpoint with same functionality as current script
  - Maintains all existing authentication and configuration logic
  - Provides health check endpoint for monitoring

#### FR2: HTTP Connection Pooling
- **Priority:** Critical
- **Description:** Implement persistent HTTP session for API server communication
- **Acceptance Criteria:**
  - Use `requests.Session()` for connection pooling
  - Maintain persistent connections to API server
  - Reuse authentication headers across requests
  - Handle connection failures gracefully

#### FR3: MCP Server Integration
- **Priority:** High
- **Description:** Update Node.js MCP server to use worker service instead of process spawning
- **Acceptance Criteria:**
  - Replace `spawn()` calls with HTTP requests to worker service
  - Maintain existing error handling and timeout behavior
  - Implement fallback to process spawning if worker unavailable
  - Preserve all existing MCP protocol compliance

#### FR4: Docker Integration
- **Priority:** High
- **Description:** Integrate worker service into existing Docker architecture
- **Acceptance Criteria:**
  - Add worker service to `docker-compose.yaml`
  - Configure proper service dependencies and health checks
  - Use consistent environment variable patterns
  - Maintain existing deployment workflows

### Technical Requirements

#### TR1: Performance Optimization
- **Priority:** Critical
- **Description:** Achieve significant performance improvements over current implementation
- **Acceptance Criteria:**
  - Eliminate 50-150ms process startup penalty
  - Reduce network overhead by 20-30ms through connection pooling
  - Support concurrent requests without performance degradation
  - Memory usage remains within acceptable limits

#### TR2: Reliability and Error Handling
- **Priority:** High
- **Description:** Maintain or improve system reliability
- **Acceptance Criteria:**
  - Graceful handling of worker service failures
  - Proper error propagation to MCP clients
  - Health monitoring and automatic recovery
  - Comprehensive logging for debugging

#### TR3: Backward Compatibility
- **Priority:** Medium
- **Description:** Maintain compatibility with existing MCP clients and workflows
- **Acceptance Criteria:**
  - No changes to MCP protocol interface
  - Identical response formats and error messages
  - Seamless migration without client updates
  - Rollback capability to previous implementation

## Implementation Plan

### Phase 1: FastAPI Worker Service Development

#### 1.1 Create Worker Service Structure
**New Directory:** `worker-service/`
**Files to Create:**
- `worker-service/main.py` - FastAPI application
- `worker-service/models.py` - Request/response models
- `worker-service/config.py` - Configuration management
- `worker-service/requirements.txt` - Python dependencies
- `worker-service/Dockerfile` - Container configuration

#### 1.2 Implement Core FastAPI Application
**File:** `worker-service/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from letta_tool_utils import (
    get_find_tools_id_with_fallback,
    get_tool_selector_base_url,
    build_tool_selector_headers,
    get_tool_selector_timeout,
)

app = FastAPI(title="Letta Tools Worker Service", version="1.0.0")

# Global HTTP session for connection pooling
session = requests.Session()
session.headers.update(build_tool_selector_headers())

class FindToolsRequest(BaseModel):
    query: Optional[str] = None
    agent_id: Optional[str] = None
    keep_tools: Optional[str] = None
    limit: int = 10
    min_score: float = 50.0
    request_heartbeat: bool = False

@app.post("/find_tools")
async def find_tools_endpoint(request: FindToolsRequest):
    """Find and attach tools for agent - persistent worker implementation."""
    # Implementation mirrors existing find_tools.py logic
    # but uses persistent session for HTTP calls
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "worker-service"}
```

#### 1.3 Implement Request Processing Logic
**File:** `worker-service/main.py` (continued)

```python
@app.post("/find_tools")
async def find_tools_endpoint(request: FindToolsRequest):
    try:
        # Sanitize inputs (reuse existing logic)
        sanitized_limit = _sanitize_limit(request.limit)
        sanitized_min_score = _sanitize_min_score(request.min_score)
        keep_tool_ids = _prepare_keep_tools(request.keep_tools, request.agent_id)
        
        # Build payload
        payload = {
            "limit": sanitized_limit,
            "min_score": sanitized_min_score,
            "keep_tools": keep_tool_ids,
            "request_heartbeat": bool(request.request_heartbeat),
        }
        
        if request.query:
            payload["query"] = request.query
        if request.agent_id:
            payload["agent_id"] = request.agent_id
        
        # Use persistent session for API call
        api_url = f"{get_tool_selector_base_url()}/api/v1/tools/attach"
        response = session.post(
            api_url,
            json=payload,
            timeout=get_tool_selector_timeout()
        )
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Phase 2: MCP Server Integration

#### 2.1 Update Node.js MCP Server
**File:** `src/index.js`
**Changes Required:**
- Replace `spawn()` calls with HTTP requests to worker service
- Add worker service health checking
- Implement fallback mechanism

```javascript
// Add axios import
import axios from 'axios';

// Configuration
const WORKER_SERVICE_URL = process.env.WORKER_SERVICE_URL || 'http://worker-service:3021';
const WORKER_TIMEOUT_MS = 30000;

class ToolSelectorServer {
    constructor() {
        // ... existing code
        this.workerServiceAvailable = true;
        this.checkWorkerHealth();
    }
    
    async checkWorkerHealth() {
        try {
            await axios.get(`${WORKER_SERVICE_URL}/health`, { timeout: 5000 });
            this.workerServiceAvailable = true;
        } catch (error) {
            this.workerServiceAvailable = false;
            console.warn('Worker service unavailable, falling back to process spawning');
        }
    }
    
    async handleFindTools(args) {
        if (this.workerServiceAvailable) {
            return await this.handleFindToolsViaWorker(args);
        } else {
            return await this.handleFindToolsViaProcess(args);
        }
    }
    
    async handleFindToolsViaWorker(args) {
        try {
            const response = await axios.post(
                `${WORKER_SERVICE_URL}/find_tools`,
                args,
                { timeout: WORKER_TIMEOUT_MS }
            );
            
            return {
                content: [{
                    type: 'text',
                    text: JSON.stringify(response.data)
                }]
            };
        } catch (error) {
            // Fallback to process spawning on worker failure
            this.workerServiceAvailable = false;
            return await this.handleFindToolsViaProcess(args);
        }
    }
    
    async handleFindToolsViaProcess(args) {
        // Keep existing process spawning logic as fallback
        // ... existing implementation
    }
}
```

### Phase 3: Docker Integration and Deployment

#### 3.1 Create Worker Service Dockerfile
**File:** `worker-service/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy shared utilities from parent directory
COPY ../letta_tool_utils.py .

EXPOSE 3021

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3021/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3021"]
```

#### 3.2 Update Docker Compose Configuration
**File:** `compose.yaml`
**Add new service:**

```yaml
  worker-service:
    build:
      context: ./worker-service
      dockerfile: Dockerfile
    container_name: letta-toolselector-worker
    ports:
      - "3021:3021"
    environment:
      - LETTA_API_URL=${LETTA_API_URL}
      - LETTA_PASSWORD=${LETTA_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - WORKER_SERVICE_PORT=3021
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3021/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    depends_on:
      api-server:
        condition: service_healthy
    networks:
      - letta-tools
    restart: unless-stopped

  mcp-server:
    # ... existing configuration
    environment:
      # ... existing environment variables
      - WORKER_SERVICE_URL=http://worker-service:3021
    depends_on:
      # ... existing dependencies
      worker-service:
        condition: service_healthy
```

## Success Criteria

### Quantitative Metrics
1. **Response Time Improvement:** 50-150ms reduction in average response time
2. **Throughput Increase:** Support for higher concurrent request volume
3. **Resource Efficiency:** Reduced memory and CPU usage per request
4. **Connection Reuse:** Measurable reduction in TCP connection establishment

### Qualitative Metrics
1. **System Reliability:** No increase in error rates or failures
2. **Monitoring Capability:** Better observability through dedicated health endpoints
3. **Scalability:** Ability to scale worker service independently
4. **Maintainability:** Cleaner separation of concerns and easier debugging

### Performance Benchmarks
- **Before:** 200-300ms average response time with process spawning
- **After:** 50-100ms average response time with worker service
- **Concurrency:** Support 10+ concurrent requests without degradation
- **Memory:** Stable memory usage under load

## Risk Assessment

### High Risk
1. **Service Dependencies:** Risk of worker service becoming single point of failure
   - **Mitigation:** Implement fallback to process spawning, health monitoring
2. **Network Latency:** Risk of network overhead between MCP server and worker
   - **Mitigation:** Co-locate services, optimize request/response payloads

### Medium Risk
1. **Configuration Complexity:** Risk of increased deployment complexity
   - **Mitigation:** Clear documentation, automated health checks
2. **Memory Leaks:** Risk of persistent service developing memory leaks
   - **Mitigation:** Monitoring, automatic restarts, resource limits

### Low Risk
1. **Compatibility Issues:** Risk of breaking existing MCP functionality
   - **Mitigation:** Comprehensive testing, gradual rollout with fallback

## Timeline

### Week 1: Worker Service Development
- Create FastAPI application structure
- Implement core find_tools endpoint
- Add health monitoring and error handling

### Week 2: Integration and Testing
- Update MCP server to use worker service
- Implement fallback mechanisms
- Create comprehensive test suite

### Week 3: Docker Integration and Deployment
- Create Docker configuration
- Update docker-compose.yaml
- Deploy to staging environment

### Week 4: Performance Testing and Optimization
- Conduct performance benchmarks
- Optimize connection pooling and resource usage
- Production deployment and monitoring

## Dependencies

1. **FastAPI Framework:** Modern async Python web framework
2. **Docker Infrastructure:** Existing containerized deployment
3. **Network Configuration:** Service-to-service communication setup
4. **Monitoring Tools:** Health check and performance monitoring capabilities

## Appendix

### File References
- `src/index.js` - Current MCP server implementation
- `find_tools.py` - Current Python script logic
- `letta_tool_utils.py` - Shared authentication and configuration utilities
- `compose.yaml` - Docker compose configuration

### Technical References
- FastAPI Documentation
- Docker Compose Service Dependencies
- HTTP Connection Pooling Best Practices
- MCP Protocol Specification
