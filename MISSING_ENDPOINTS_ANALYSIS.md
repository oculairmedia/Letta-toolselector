# Missing Endpoints Analysis

## Overview
This document outlines the API endpoints that the frontend expects but are not currently implemented in the dashboard backend.

## Frontend Expected Endpoints vs Backend Implementation

### ✅ Implemented Endpoints
- `GET /api/v1/health` - Health check (implemented in health.py)
- `GET /api/v1/search` - Search tools (implemented in search.py)
- `POST /api/v1/search` - Search tools (implemented in search.py)
- `POST /api/v1/rerank` - Rerank tools (implemented in rerank.py)
- `GET /api/v1/rerank/models` - Available reranking models (implemented in rerank.py)
- `GET /api/v1/rerank/config` - Reranking configuration (implemented in rerank.py)
- `GET /api/v1/config` - Backend configuration (implemented in config.py)
- `GET /api/v1/config/limits` - API limits (implemented in config.py)

### ❌ Missing Critical Endpoints

#### 1. Tools Management
**Frontend expects:**
- `POST /api/v1/tools/search` - Standard tool search
- `POST /api/v1/tools/search/rerank` - Search with reranking
- `GET /api/v1/tools` - Get all tools
- `POST /api/v1/tools/refresh` - Refresh tool index

**Current status:** 
- These exist in `tool-selector-api/api_server.py` but not in dashboard backend
- Dashboard backend has `/search` and `/rerank` but frontend calls `/tools/search`

#### 2. Configuration Presets
**Frontend expects:**
- `GET /api/v1/config/presets` - List configuration presets
- `POST /api/v1/config/presets` - Create configuration preset
- `PUT /api/v1/config/presets/{id}` - Update configuration preset
- `DELETE /api/v1/config/presets/{id}` - Delete configuration preset

**Current status:** 
- PresetManager class exists in `preset_manager.py`
- No API endpoints exposed in routers

#### 3. Reranker Configuration Management
**Frontend expects:**
- `GET /api/v1/config/reranker` - Get reranker configuration
- `PUT /api/v1/config/reranker` - Update reranker configuration
- `POST /api/v1/config/reranker/test` - Test reranker connection

**Current status:** 
- Only `GET /api/v1/rerank/config` exists (different path)
- No PUT or test endpoints

#### 4. Ollama Integration
**Frontend expects:**
- `GET /api/v1/ollama/models` - Get available Ollama models

**Current status:** 
- Not implemented in dashboard backend

#### 5. Evaluation System
**Frontend expects:**
- `POST /api/v1/evaluations` - Submit evaluation
- `GET /api/v1/evaluations` - Get evaluations with query/limit

**Current status:** 
- Not implemented in dashboard backend
- Evaluation framework exists but no API endpoints

#### 6. Analytics
**Frontend expects:**
- `GET /api/v1/analytics` - Get analytics with optional date range

**Current status:** 
- Not implemented in dashboard backend

## Path Mismatches

### Search Endpoints
- **Frontend calls:** `/api/v1/tools/search`
- **Backend implements:** `/api/v1/search`
- **Issue:** Path mismatch - frontend expects tools prefix

### Reranker Config
- **Frontend calls:** `/api/v1/config/reranker`
- **Backend implements:** `/api/v1/rerank/config`
- **Issue:** Path structure mismatch

## Implementation Gaps

### 1. Missing Router Files
Need to create:
- `app/routers/presets.py` - Configuration presets CRUD
- `app/routers/evaluations.py` - Evaluation submission and retrieval
- `app/routers/analytics.py` - Analytics and metrics
- `app/routers/ollama.py` - Ollama model management
- `app/routers/tools.py` - Tool management (or update search.py paths)

### 2. Missing Models
Need to create:
- `app/models/presets.py` - Preset request/response models
- `app/models/evaluations.py` - Evaluation models
- `app/models/analytics.py` - Analytics models
- `app/models/ollama.py` - Ollama models

### 3. Missing Services Integration
Need to wire up:
- PresetManager to API endpoints
- Evaluation framework to API endpoints
- Analytics/metrics calculation to API endpoints
- Ollama client to API endpoints

## Placeholder Implementations

### Safety Module Placeholders
In `dashboard-backend/safety.py`:
```python
def check_letta_api_isolation() -> bool:
    return True  # Placeholder - would check actual API configuration

def check_agent_modification_blocked() -> bool:
    return True  # Placeholder

def check_tool_attachment_blocked() -> bool:
    return True  # Placeholder

def check_database_read_only() -> bool:
    return True  # Placeholder
```

These need actual implementation to verify safety constraints.

## Priority for Implementation

### High Priority (Frontend Broken Without These)
1. **Configuration Presets API** - Frontend preset manager won't work
2. **Tools Search Path Fix** - Search functionality broken due to path mismatch
3. **Reranker Config API** - Configuration panel won't work

### Medium Priority (Features Won't Work)
4. **Evaluations API** - Evaluation features won't work
5. **Ollama Models API** - Model selection won't work
6. **Analytics API** - Analytics dashboard won't work

### Low Priority (Safety/Monitoring)
7. **Safety Placeholder Implementation** - Security verification
8. **Rate Limiting Verification** - Ensure middleware is active

## Recommended Next Steps

1. **Fix path mismatches** - Update search endpoints to match frontend expectations
2. **Implement presets API** - Wire up existing PresetManager to FastAPI endpoints
3. **Add missing router files** - Create the missing endpoint implementations
4. **Update main.py** - Include new routers in the FastAPI app
5. **Test all endpoints** - Verify frontend can successfully call all expected endpoints

## Notes

- The backend has most of the business logic implemented (PresetManager, safety, audit, etc.)
- The main gap is exposing this functionality through the correct API endpoints
- Path mismatches are causing the frontend to fail even for implemented functionality
