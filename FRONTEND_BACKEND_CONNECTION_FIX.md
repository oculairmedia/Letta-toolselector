# Frontend-Backend Connection Fix Guide

## Current Status: Connection Refused Errors

The frontend is getting `ERR_CONNECTION_REFUSED` when trying to connect to `http://localhost:8030/api/v1/*` endpoints. Based on Context7 best practices for debugging connection issues, here are the required changes:

## Root Cause Analysis

✅ **Frontend Configuration**: Correctly pointing to `localhost:8030`
❌ **Backend Server**: Not running on port 8030
✅ **CORS Configuration**: Properly configured in backend
❌ **Dependencies**: Backend dependencies may not be installed
❌ **CRITICAL: Endpoint Mismatches**: Frontend and backend have misaligned API endpoints

## Endpoint Alignment Issues Found

### 🚨 Critical Mismatches

| Frontend Calls | Backend Implements | Status |
|----------------|-------------------|---------|
| `GET /config/embedding` | ❌ **MISSING** | Not implemented |
| `GET /models/embedding` | ❌ **MISSING** | Not implemented |
| `GET /models/reranker` | ❌ **MISSING** | Not implemented |
| `GET /config/reranker` | `GET /rerank/config` | ❌ **PATH MISMATCH** |
| `POST /tools/search` | `POST /search` | ❌ **PATH MISMATCH** |
| `GET /tools` | ✅ `GET /tools` | ✅ **ALIGNED** |
| `POST /tools/refresh` | ✅ `POST /tools/refresh` | ✅ **ALIGNED** |
| `GET /health` | ✅ `GET /health` | ✅ **ALIGNED** |

### 🔍 Detailed Analysis

**1. Missing Model Endpoints**
- Frontend expects: `/models/embedding` and `/models/reranker`
- Backend has: No model discovery endpoints
- Impact: Model selection dropdowns will fail

**2. Configuration Endpoint Mismatch**
- Frontend expects: `/config/reranker` and `/config/embedding`
- Backend has: `/rerank/config` only
- Impact: Configuration panels won't load

**3. Search Endpoint Path Mismatch**
- Frontend expects: `/tools/search`
- Backend has: `/search` (under different router)
- Impact: Search functionality completely broken

**4. Missing Endpoints**
- `/config/presets` - Configuration presets CRUD
- `/evaluations` - Evaluation submission/retrieval
- `/analytics` - Usage analytics and metrics
- `/config/embedding` - Embedding configuration

## Required Changes

### 1. Fix Critical Endpoint Mismatches (PRIORITY 1)

**A. Add Missing Router Include**
The `tools` router is not included in `app/main.py`. Add this line:

```python
# In dashboard-backend/app/main.py, line 126:
app.include_router(tools.router, prefix=f"{settings.API_V1_STR}")
```

**B. Create Missing Model Endpoints**
Create `dashboard-backend/app/routers/models.py`:

```python
from fastapi import APIRouter
from typing import Dict, Any, List

router = APIRouter(tags=["models"])

@router.get("/models/embedding")
async def get_embedding_models():
    return {
        "models": [
            {"id": "text-embedding-ada-002", "name": "OpenAI Ada 002", "provider": "openai", "dimensions": 1536, "max_tokens": 8191, "cost_per_1k": 0.0001, "recommended": True},
            {"id": "text-embedding-3-small", "name": "OpenAI Embedding 3 Small", "provider": "openai", "dimensions": 1536, "max_tokens": 8191, "cost_per_1k": 0.00002, "recommended": False}
        ],
        "total": 2,
        "providers": ["openai"]
    }

@router.get("/models/reranker")
async def get_reranker_models():
    return {
        "models": [
            {"id": "bge-reranker-large", "name": "BGE Reranker Large", "provider": "huggingface", "type": "cross-encoder", "cost_per_1k": 0.0, "recommended": True},
            {"id": "bge-reranker-base", "name": "BGE Reranker Base", "provider": "huggingface", "type": "cross-encoder", "cost_per_1k": 0.0, "recommended": False}
        ],
        "total": 2,
        "providers": ["huggingface"],
        "types": ["cross-encoder"]
    }
```

**C. Fix Configuration Endpoint Paths**
Add to `dashboard-backend/app/routers/config.py`:

```python
@router.get("/config/reranker")
async def get_reranker_config():
    """Get reranker configuration (frontend-compatible endpoint)."""
    return {
        "provider": "huggingface",
        "model": settings.RERANKER_MODEL,
        "parameters": {
            "top_k": 10,
            "threshold": 0.5
        }
    }

@router.get("/config/embedding")
async def get_embedding_config():
    """Get embedding configuration."""
    return {
        "model": "text-embedding-ada-002",
        "provider": "openai",
        "parameters": {
            "dimensions": 1536
        }
    }
```

### 2. Install Backend Dependencies

```bash
cd dashboard-backend
pip install -r requirements.txt
```

### 3. Start the Dashboard Backend Server

```bash
cd dashboard-backend
python start.py
```

**Expected Output**:
```
🚀 Starting LDTS Reranker Testing Dashboard Backend v1.0.0
📡 Server: 0.0.0.0:8030
🔒 Read-only mode: True
⚡ Rate limiting: True
🤖 Reranking: True
🌐 Environment: development
📚 API docs: http://0.0.0.0:8030/api/v1/docs
```

### 4. Verify Backend is Running

```bash
# Test health endpoint
curl http://localhost:8030/api/v1/health

# Test the corrected endpoints
curl http://localhost:8030/api/v1/config/reranker
curl http://localhost:8030/api/v1/models/embedding
curl http://localhost:8030/api/v1/tools
```

**Expected Response**: JSON responses instead of connection errors.

### 4. Fix Frontend Proxy Configuration (Optional)

**File**: `dashboard-ui/package.json`

**Current**:
```json
"proxy": "http://192.168.50.90:8020"
```

**Change to**:
```json
"proxy": "http://localhost:8030"
```

**Why**: Ensures proxy requests go to the correct backend port.

### 5. Restart Frontend with Correct Environment

```bash
cd dashboard-ui
PORT=8406 REACT_APP_API_BASE_URL=http://localhost:8030 npm start
```

## Debugging Steps (Context7 Best Practices)

### Axios Error Handling
The frontend already implements proper error handling:

```typescript
// Check error types as per Axios documentation
if (error.response) {
  // Server responded with error status
  console.log(error.response.status);
} else if (error.request) {
  // Request made but no response (current issue)
  console.log('No response received');
} else {
  // Request setup error
  console.log('Request setup error');
}
```

### FastAPI CORS Configuration
Backend CORS is properly configured:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Network Verification Commands

```bash
# Check what's listening on port 8030
netstat -tlnp | grep :8030

# Check if backend process is running
ps aux | grep "python.*start.py"

# Test direct API call
curl -v http://localhost:8030/api/v1/health
```

## Expected Results After Fix

1. ✅ Backend server running on `http://localhost:8030`
2. ✅ Frontend successfully connecting to backend
3. ✅ API calls returning data instead of `ERR_CONNECTION_REFUSED`
4. ✅ Dashboard components loading properly
5. ✅ Configuration data being fetched successfully

## Service Architecture (After Fix)

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| Frontend Dashboard | 8406 | ✅ Running | React development server |
| Dashboard Backend | 8030 | ❌ **NEEDS TO START** | FastAPI backend |
| LDTS API Server | 8020 | ✅ Available | Main tool management API |
| MCP Server | 3020 | ✅ Available | MCP protocol server |

## Troubleshooting

### If Backend Won't Start

1. **Check Python Path**:
   ```bash
   cd dashboard-backend
   python -c "from app.main import app; print('Import successful')"
   ```

2. **Check Dependencies**:
   ```bash
   python -c "import fastapi, uvicorn; print('Dependencies OK')"
   ```

3. **Check Port Availability**:
   ```bash
   netstat -tlnp | grep :8030
   ```

### If Frontend Still Can't Connect

1. **Verify Backend Health**:
   ```bash
   curl http://localhost:8030/api/v1/health
   ```

2. **Check Browser Network Tab**: Look for actual request URLs and response codes

3. **Verify Environment Variables**:
   ```bash
   echo $REACT_APP_API_BASE_URL
   ```

## Next Steps After Connection Fix

1. **Test All API Endpoints**: Verify all dashboard functionality works
2. **Implement Missing Endpoints**: Add any missing tool browser APIs
3. **Performance Testing**: Ensure connection stability under load
4. **Error Monitoring**: Set up proper error tracking for production

## Quick Fix Summary

**Immediate Action Required**:

### Step 1: Fix Missing Router Include
```bash
# Edit dashboard-backend/app/main.py and add this line after line 126:
app.include_router(tools.router, prefix=f"{settings.API_V1_STR}")
```

### Step 2: Create Missing Endpoints
```bash
# Create dashboard-backend/app/routers/models.py with the model endpoints shown above
# Add the config endpoints to dashboard-backend/app/routers/config.py
```

### Step 3: Include New Router
```bash
# In dashboard-backend/app/main.py, add:
from app.routers import models
app.include_router(models.router, prefix=f"{settings.API_V1_STR}")
```

### Step 4: Start Backend
```bash
cd dashboard-backend
pip install -r requirements.txt
python start.py
```

### Step 5: Verify All Endpoints
```bash
# Test the endpoints that were failing:
curl http://localhost:8030/api/v1/health
curl http://localhost:8030/api/v1/config/embedding
curl http://localhost:8030/api/v1/models/embedding
curl http://localhost:8030/api/v1/models/reranker
curl http://localhost:8030/api/v1/tools
```

## Root Cause Summary

The connection issues were caused by **endpoint mismatches**, not just the backend not running:

1. **Missing Tools Router**: The tools router wasn't included in main.py
2. **Missing Model Endpoints**: `/models/embedding` and `/models/reranker` didn't exist
3. **Missing Config Endpoints**: `/config/embedding` didn't exist
4. **Path Mismatches**: Some endpoints had different paths than expected

Once these endpoint alignment issues are fixed AND the backend is running, the frontend should successfully connect and display data instead of connection errors.
