# Docker Compose Configuration Guide

## Active Configuration Files

### Primary: `compose.yaml`
**Purpose**: Production deployment configuration  
**Status**: ✅ Active (default)

This is the **single source of truth** for the deployment. It defines:
- All services (api-server, sync-service, weaviate, mcp-server, etc.)
- Default environment variables with fallbacks
- Volume mounts and networking
- Health checks and restart policies

### Development: `compose.dev.yaml`
**Purpose**: Development overrides  
**Status**: ⚠️ Optional (use with `-f` flag)

Development-specific settings:
- Hot-reload configurations
- Development logging levels
- Debug ports exposed

**Usage**:
```bash
docker compose -f compose.yaml -f compose.dev.yaml up
```

### Reranker Variant: `compose-with-reranker.yaml`
**Purpose**: Alternative configuration with reranking features  
**Status**: ⚠️ Optional variant

Includes additional reranking capabilities for tool search.

**Usage**:
```bash
docker compose -f compose-with-reranker.yaml up
```

---

## Environment Configuration

### `.env.example` (Version Controlled)
- Complete list of all environment variables
- Documentation for each variable
- Safe default values (no secrets)
- **Always kept in sync** with actual requirements

### `.env` (Git Ignored)
- Your local/production overrides
- Contains actual secrets and API keys
- Not committed to version control
- Copy from `.env.example` and customize

**Setup**:
```bash
cp .env.example .env
# Edit .env with your actual values
```

---

## Deprecated Files

The following files have been deprecated and should not be used:

| File | Status | Reason |
|------|--------|--------|
| `docker-compose.yml.deprecated` | ❌ Deprecated | Replaced by `compose.yaml` |
| `docker-compose.override.yml.deprecated` | ❌ Deprecated | Caused config conflicts |
| `docker-compose.update.yml.deprecated` | ❌ Deprecated | Functionality merged |

These files are kept with `.deprecated` suffix for reference only.

---

## Service Architecture

### Core Services

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| **api-server** | weaviate-tools-api | 8020 | Main tool management API |
| **weaviate** | lettatoolsselector-weaviate-1 | 8080 | Vector database for tool search |
| **sync-service** | weaviate-tools-sync | - | Background tool sync (every 5min) |
| **mcp-server** | toolselector-mcp | 3020 | MCP protocol server |
| **worker-service** | letta-toolselector-worker | 3021 | Background task processing |

### Optional Services

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| **reranker-ollama-adapter** | ollama-reranker-adapter | 8091 | Ollama-based reranking |
| **dashboard-frontend** | letta-toolselector-frontend | - | Admin UI (if enabled) |
| **dashboard-backend** | letta-toolselector-dashboard-backend | - | Dashboard API (if enabled) |
| **time-service** | weaviate-tools-time | - | Time-based memory updates |

---

## Required Environment Variables

### Critical (Must Set)
```bash
LETTA_API_URL=https://your-letta-instance.com/v1
LETTA_PASSWORD=your-letta-password
OPENAI_API_KEY=sk-your-openai-api-key  # Or use Ollama
```

### Optional (Have Defaults)
```bash
# Tool Management
MAX_TOTAL_TOOLS=30
MAX_MCP_TOOLS=20
MIN_MCP_TOOLS=7
DEFAULT_DROP_RATE=0.6
NEVER_DETACH_TOOLS=find_tools

# Embedding Provider
EMBEDDING_PROVIDER=ollama  # or 'openai'
OLLAMA_EMBEDDING_HOST=192.168.50.80
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M
```

See `.env.example` for complete list with documentation.

---

## Common Operations

### Start Services
```bash
# Production
docker compose up -d

# Development with overrides
docker compose -f compose.yaml -f compose.dev.yaml up -d

# View logs
docker compose logs -f api-server
```

### Update Configuration
1. Edit `.env` with new values
2. Restart affected services:
   ```bash
   docker compose restart api-server
   ```

### Validate Configuration
```bash
# Check which compose file is being used
docker compose config | head -20

# Validate environment variables
docker compose config | grep -A 50 "environment:"
```

### Health Checks
```bash
# API server
curl http://localhost:8020/api/v1/health

# Weaviate
curl http://localhost:8080/v1/.well-known/ready
```

---

## Migration from Old Setup

If you were using `docker-compose.yml`:

1. **Backup** your current setup
2. **Copy** configuration to `compose.yaml` format
3. **Move** `docker-compose.yml` to `docker-compose.yml.deprecated`
4. **Test** with `docker compose up -d`
5. **Verify** services are running: `docker compose ps`

---

## Troubleshooting

### "Found multiple config files" Warning
This means deprecated files still exist. Rename them:
```bash
mv docker-compose.yml docker-compose.yml.deprecated
```

### Environment Variable Not Set
Check if the variable exists in `.env`:
```bash
grep VARIABLE_NAME .env
```

If missing, add it from `.env.example`.

### Service Won't Start
1. Check logs: `docker compose logs service-name`
2. Verify environment: `docker compose config`
3. Check health: `docker compose ps`

---

## Best Practices

1. **Never commit `.env`** - It contains secrets
2. **Always update `.env.example`** - When adding new variables
3. **Use `compose.yaml`** - Single source of truth
4. **Override with `-f`** - Don't modify main compose file
5. **Document changes** - Update this file when architecture changes

---

## Version History

### 2025-11-29
- Consolidated to `compose.yaml` as primary file
- Deprecated old `docker-compose.yml` files
- Enhanced `.env.example` with PROTECTED_TOOLS
- Created this documentation

---

## References

- **Main Config**: `compose.yaml`
- **Env Template**: `.env.example`
- **API Contract**: `API_CONTRACT.md`
- **Main Docs**: `README.md`
