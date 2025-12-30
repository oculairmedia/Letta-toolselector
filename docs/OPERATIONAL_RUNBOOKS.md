# Operational Runbooks

Quick reference guides for common operational tasks with the Letta Tool Selector.

---

## Table of Contents

1. [Change Tool Limits](#change-tool-limits)
2. [Add/Remove Protected Tools](#addremove-protected-tools)
3. [Manually Prune Agent Tools](#manually-prune-agent-tools)
4. [Debug "Tool Not Attaching" Issues](#debug-tool-not-attaching-issues)
5. [Check Tool Selector Health](#check-tool-selector-health)
6. [View Audit Logs](#view-audit-logs)
7. [Force Tool Sync](#force-tool-sync)
8. [Restart Services Safely](#restart-services-safely)

---

## Change Tool Limits

**When**: You need to adjust MAX_TOTAL_TOOLS, MAX_MCP_TOOLS, or MIN_MCP_TOOLS

### Steps

1. **Edit `.env` file**:
   ```bash
   cd /opt/stacks/lettatoolsselector
   nano .env
   ```

2. **Update the limits**:
   ```bash
   MAX_TOTAL_TOOLS=30      # Maximum total tools per agent
   MAX_MCP_TOOLS=20        # Maximum external MCP tools
   MIN_MCP_TOOLS=7         # Minimum MCP tools (won't prune below this)
   ```

3. **Restart the API server**:
   ```bash
   docker compose restart api-server
   ```

4. **Verify the changes**:
   ```bash
   curl -s http://localhost:8020/api/v1/health | jq '.config'
   ```

   **Expected output**:
   ```json
   {
     "MAX_TOTAL_TOOLS": 30,
     "MAX_MCP_TOOLS": 20,
     "MIN_MCP_TOOLS": "7",
     ...
   }
   ```

### Notes
- Changes take effect immediately after restart
- Existing agents are not automatically re-pruned
- Use the pruning API to enforce new limits on existing agents

---

## Add/Remove Protected Tools

**When**: You need to protect specific tools from auto-detachment

### Steps

1. **Edit `.env` file**:
   ```bash
   cd /opt/stacks/lettatoolsselector
   nano .env
   ```

2. **Update protected tools** (comma-separated tool names):
   ```bash
   NEVER_DETACH_TOOLS=find_tools,send_message,search_documents
   # Or use the alias
   PROTECTED_TOOLS=find_tools,send_message,search_documents
   ```

3. **Restart the API server**:
   ```bash
   docker compose restart api-server
   ```

4. **Verify protection is active**:
   ```bash
   curl -s http://localhost:8020/api/v1/health | jq '.config.PROTECTED_TOOLS'
   ```

   **Expected output**:
   ```json
   ["find_tools", "send_message", "search_documents"]
   ```

### Notes
- Protected tools are **never detached** during pruning
- Matching is done by tool name (case-sensitive substring match)
- Tool IDs can also be protected via the `keep_tools` API parameter

---

## Manually Prune Agent Tools

**When**: An agent has too many tools and you need to manually prune

### Option 1: Via API (Recommended)

```bash
curl -X POST http://localhost:8020/api/v1/tools/prune \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-uuid-here",
    "user_prompt": "search and analyze documents",
    "drop_rate": 0.6,
    "keep_tool_ids": ["tool-uuid-to-keep"]
  }'
```

**Parameters**:
- `agent_id`: **Required** - Agent to prune
- `user_prompt`: **Required** - Context for relevance scoring
- `drop_rate`: Optional - How aggressive (0.0-1.0, default 0.6)
- `keep_tool_ids`: Optional - Specific tools to preserve

**Expected response**:
```json
{
  "success": true,
  "message": "Pruning completed...",
  "details": {
    "mcp_tools_detached_count": 5,
    "final_mcp_tool_ids_kept_on_agent": ["tool-1", "tool-2"],
    ...
  }
}
```

### Option 2: Via Python Script

```bash
cd /opt/stacks/lettatoolsselector
python tool-selector-api/detach_mcp_tools.py \
  --agent-id agent-uuid-here \
  --keep-count 10
```

### Notes
- Pruning respects MIN_MCP_TOOLS (won't go below minimum)
- Protected tools are never removed
- Use lower `drop_rate` (e.g., 0.3) for gentler pruning

---

## Debug "Tool Not Attaching" Issues

**When**: Tools aren't being attached to agents as expected

### Diagnostic Steps

**1. Check API Server Health**:
```bash
curl http://localhost:8020/api/v1/health | jq
```

Verify:
- `status: "OK"`
- `weaviate.status: "OK"`
- `tool_cache_in_memory.status: "OK"`

**2. Test Tool Search**:
```bash
curl -X POST http://localhost:8020/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search documents",
    "limit": 10
  }' | jq
```

**Expected**: List of matching tools with scores

**3. Check Min Score Threshold**:
```bash
grep DEFAULT_MIN_SCORE .env
```

If score is too high (e.g., > 50), lower it:
```bash
DEFAULT_MIN_SCORE=35.0
```

**4. View API Logs**:
```bash
docker compose logs api-server --tail 100 --follow
```

Look for:
- `"Failed to attach tool"` - Permission/ID issues
- `"Weaviate client not ready"` - Database connection issues
- `"Tool cache not loaded"` - Cache initialization problems

**5. Verify Tool Exists in Weaviate**:
```bash
curl -s http://localhost:8020/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{"query": "EXACT_TOOL_NAME", "limit": 1}' | jq
```

If tool not found, trigger a sync:
```bash
curl -X POST http://localhost:8020/api/v1/tools/sync
```

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| No tools match | Weaviate empty | Run sync endpoint |
| Tools match but don't attach | Agent already at MAX_TOTAL_TOOLS | Increase limit or prune |
| Specific tool never attaches | Tool is not registered | Check Letta API `/tools` endpoint |
| Score always 0 | Embedding provider issue | Check `EMBEDDING_PROVIDER` in .env |

---

## Check Tool Selector Health

**When**: Verifying system is working correctly

### Quick Health Check

```bash
curl http://localhost:8020/api/v1/health
```

**Expected Response**:
```json
{
  "status": "OK",
  "version": "1.0.0",
  "config": {
    "MAX_TOTAL_TOOLS": 30,
    "PROTECTED_TOOLS": ["find_tools"]
  },
  "details": {
    "weaviate": {"status": "OK"},
    "tool_cache_in_memory": {"status": "OK", "size": 219},
    "mcp_servers_cache_file": {"status": "OK"}
  }
}
```

### Detailed Health Check

**1. Check All Services**:
```bash
docker compose ps
```

All services should show "Up" and "healthy"

**2. Check Weaviate**:
```bash
curl http://localhost:8080/v1/.well-known/ready
```

**3. Check Tool Cache**:
```bash
docker compose exec api-server ls -lh /app/runtime_cache/
```

Verify `tool_cache.json` exists and has recent timestamp

**4. Check Logs for Errors**:
```bash
docker compose logs --tail 50 | grep -i error
```

### Health Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `OK` | All systems healthy | None |
| `DEGRADED` | Weaviate OK but cache issues | Check cache logs |
| `ERROR` | Weaviate or critical service down | Restart Weaviate |

---

## View Audit Logs

**When**: Investigating tool changes or debugging operations

### View Real-Time Audit Events

```bash
docker compose logs api-server --follow | grep '"event_type"'
```

### Filter by Event Type

**Tool Attachments**:
```bash
docker compose logs api-server | grep '"action":"attach"' | jq
```

**Tool Detachments**:
```bash
docker compose logs api-server | grep '"action":"detach"' | jq
```

**Pruning Operations**:
```bash
docker compose logs api-server | grep '"event_type":"tool_pruning"' | jq
```

### Search by Agent ID

```bash
docker compose logs api-server | \
  grep '"agent_id":"agent-uuid-here"' | jq
```

### Example Audit Event

```json
{
  "event_type": "tool_management",
  "timestamp": "2025-11-29T12:00:00.000000+00:00",
  "action": "attach",
  "agent_id": "agent-123",
  "tool_id": "tool-456",
  "tool_name": "document_search",
  "source": "api_attach_endpoint",
  "success": true,
  "reason": "Matched user query",
  "correlation_id": "req-789",
  "metadata": {"match_score": 87.5}
}
```

---

## Force Tool Sync

**When**: Tools missing from Weaviate or cache is stale

### Trigger Sync

```bash
curl -X POST http://localhost:8020/api/v1/tools/sync
```

**Expected Response**:
```json
{
  "success": true,
  "message": "Tool sync initiated",
  "tools_synced": 219
}
```

### Verify Sync Completed

```bash
curl http://localhost:8020/api/v1/health | \
  jq '.details.tool_cache_in_memory'
```

Check that `last_loaded` timestamp is recent

### Manual Cache Refresh

If sync doesn't work:

```bash
# Restart sync service
docker compose restart sync-service

# Wait 30 seconds, then check
docker compose logs sync-service --tail 20
```

---

## Restart Services Safely

**When**: Applying config changes or resolving issues

### Restart Individual Service

```bash
# API Server (tool management)
docker compose restart api-server

# Sync Service (background sync)
docker compose restart sync-service

# Weaviate (vector database) - CAUTION: May cause downtime
docker compose restart weaviate
```

### Restart All Services

```bash
docker compose restart
```

### Full Restart (Nuclear Option)

```bash
# Stop all
docker compose down

# Start all (waits for health checks)
docker compose up -d

# Monitor startup
docker compose logs --follow
```

### Verify After Restart

```bash
# Check all services are healthy
docker compose ps

# Check API health
curl http://localhost:8020/api/v1/health

# Check Weaviate
curl http://localhost:8080/v1/.well-known/ready
```

### Safe Restart Order

If you need to restart multiple services:

1. **sync-service** (low impact)
2. **api-server** (brief API unavailability)
3. **weaviate** (LAST - causes search unavailability)

---

## Emergency Procedures

### API Server Not Responding

```bash
# Check if running
docker compose ps api-server

# View recent logs
docker compose logs api-server --tail 100

# Restart
docker compose restart api-server

# If still failing, check .env
grep -E "LETTA_API_URL|LETTA_PASSWORD|WEAVIATE_URL" .env
```

### Weaviate Connection Failures

```bash
# Check Weaviate status
curl http://localhost:8080/v1/.well-known/ready

# Restart Weaviate
docker compose restart weaviate

# Wait for health check (30-60 seconds)
watch -n 2 'curl -s http://localhost:8080/v1/.well-known/ready'

# Restart API server to reconnect
docker compose restart api-server
```

### Tool Cache Corruption

```bash
# Remove corrupted cache
docker compose exec api-server rm /app/runtime_cache/tool_cache.json

# Restart to rebuild
docker compose restart api-server sync-service

# Verify rebuild
curl http://localhost:8020/api/v1/health | jq '.details.tool_cache_in_memory'
```

---

## Additional Resources

- **Architecture Overview**: `README.md`
- **API Contract**: `API_CONTRACT.md`
- **Compose Setup**: `COMPOSE_SETUP.md`
- **Configuration Guide**: `.env.example`

**Last Updated**: 2025-11-29
