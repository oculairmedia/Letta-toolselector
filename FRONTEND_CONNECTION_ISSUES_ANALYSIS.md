# Frontend Connection Issues Analysis & Resolution

## Problem Summary

The frontend dashboard is throwing `ERR_CONNECTION_REFUSED` errors when trying to connect to the backend API. The errors show attempts to connect to `http://localhost:8020/api/v1/...` endpoints.

## Root Cause Analysis

### 1. Configuration Mismatch
- **Frontend API Service**: Defaults to `http://localhost:8030` (line 19 in `dashboard-ui/src/services/api.ts`)
- **Frontend Proxy**: Configured to `http://192.168.50.90:8020` (line 52 in `dashboard-ui/package.json`)
- **Backend Configuration**: Configured to run on port `8030` (line 14 in `dashboard-backend/config/settings.py`)
- **Actual Error**: Frontend trying to connect to `localhost:8020`

### 2. Backend Server Not Running
- No processes currently running (confirmed via `list-processes`)
- Dashboard backend needs to be started on port 8030

### 3. Port Assignment Confusion
- **Port 8020**: Reserved for main LDTS API server (`tool-selector-api/api_server.py`)
- **Port 8030**: Reserved for dashboard backend
- **Port 3000**: Frontend development server
- **Port 3020**: MCP server

## Current Configuration Files

### Frontend API Configuration
**File**: `dashboard-ui/src/services/api.ts` (lines 18-26)
```typescript
const baseURL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8030';
this.client = axios.create({
  baseURL: `${baseURL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});
```

### Frontend Proxy Configuration
**File**: `dashboard-ui/package.json` (line 52)
```json
"proxy": "http://192.168.50.90:8020"
```

### Backend Configuration
**File**: `dashboard-backend/config/settings.py` (lines 13-14)
```python
HOST: str = "0.0.0.0"
PORT: int = 8030
```

## Resolution Steps

### Step 1: Install Backend Dependencies
```bash
cd dashboard-backend
pip install -r requirements.txt
```

### Step 2: Start Dashboard Backend
```bash
cd dashboard-backend
python start.py
```
This should start the server on `http://localhost:8030`

### Step 3: Fix Frontend Configuration
Choose one of these approaches:

#### Option A: Update Frontend Proxy (Recommended)
**File**: `dashboard-ui/package.json`
```json
"proxy": "http://localhost:8030"
```

#### Option B: Set Environment Variable
Create `dashboard-ui/.env`:
```bash
REACT_APP_API_BASE_URL=http://localhost:8030
```

#### Option C: Update Backend Port (Not Recommended)
Change `dashboard-backend/config/settings.py`:
```python
PORT: int = 8020
```

### Step 4: Restart Frontend
After making configuration changes:
```bash
cd dashboard-ui
npm start
```

## Expected Behavior After Fix

1. Backend server running on `http://localhost:8030`
2. Frontend connecting to correct backend port
3. API calls succeeding instead of `ERR_CONNECTION_REFUSED`
4. Dashboard components loading properly

## Service Architecture Overview

| Service | Port | Purpose |
|---------|------|---------|
| Frontend Dashboard | 3000 | React development server |
| Dashboard Backend | 8030 | FastAPI backend for dashboard |
| LDTS API Server | 8020 | Main tool management API |
| MCP Server | 3020 | MCP protocol server |
| Weaviate | 8080 | Vector database |

## Missing Backend Endpoints

Note: The dashboard backend is missing the tool browser API endpoints that the frontend expects. These need to be implemented:

- `GET /api/v1/tools/browse` - Browse tools with pagination
- `GET /api/v1/tools/{id}/detail` - Get tool details
- `GET /api/v1/tools/categories` - Get categories
- `GET /api/v1/tools/sources` - Get sources
- `GET /api/v1/tools/export` - Export tools
- `POST /api/v1/tools/refresh` - Refresh tool index

## Verification Commands

After implementing the fix:

```bash
# Check if backend is running
curl http://localhost:8030/api/v1/health

# Check if frontend can reach backend
curl http://localhost:8030/api/v1/config/reranker
```

## Next Steps

1. Install backend dependencies
2. Start dashboard backend server
3. Fix frontend proxy configuration
4. Restart frontend
5. Implement missing tool browser API endpoints (separate task)
