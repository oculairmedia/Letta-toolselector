<!-- VIBESYNC:project-info:START -->
# Agent Instructions

## Huly Integration

- **Project Code**: `LTSEL`
- **Project Name**: Letta Tools Selector
- **Letta Agent ID**: `agent-5a145be1-3c15-491e-b5a8-e9f3fe1e716e`

## Workflow Instructions

1. **Before starting work**: Search Huly for related issues using `huly-mcp` with project code `LTSEL`
2. **Issue references**: All issues for this project use the format `LTSEL-XXX` (e.g., `LTSEL-123`)
3. **On task completion**: Report to this project's Letta agent via `matrix-identity-bridge` using `talk_to_agent`
4. **Memory**: Store important discoveries in Graphiti with `graphiti-mcp_add_memory`
<!-- VIBESYNC:project-info:END -->

<!-- VIBESYNC:reporting-hierarchy:START -->
## PM Agent Communication

**Project PM Agent:** `Huly - Letta Tools Selector` (agent-5a145be1-3c15-491e-b5a8-e9f3fe1e716e)

### Reporting Hierarchy

```
Emmanuel (Stakeholder)
    ↓
Meridian (Director of Engineering)
    ↓
PM Agent (Technical Product Owner - mega-experienced)
    ↓ communicates with
You (Developer Agent - experienced)
```

### MANDATORY: Report to PM Agent

**BEFORE reporting outcomes to the user**, send a report to the PM agent via Matrix:

```json
{
  "operation": "talk_to_agent",
  "agent": "Huly - Letta Tools Selector",
  "message": "<your report>",
  "caller_directory": "/opt/stacks/lettatoolsselector"
}
```

### When to Contact PM Agent

| Situation             | Action                                                              |
| --------------------- | ------------------------------------------------------------------- |
| Task completed        | Report outcome to PM before responding to user                      |
| Blocking question     | Forward to PM - they know user's wishes and will escalate if needed |
| Architecture decision | Consult PM for guidance                                             |
| Unclear requirements  | PM can clarify or contact user                                      |

### Report Format

```
**Status**: [Completed/Blocked/In Progress]
**Task**: [Brief description]
**Outcome**: [What was done/What's blocking]
**Files Changed**: [List if applicable]
**Next Steps**: [If any]
```
<!-- VIBESYNC:reporting-hierarchy:END -->

<!-- VIBESYNC:beads-instructions:START -->
## Beads Issue Tracking

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

### Beads Sync Flow (Hybrid System)

Beads uses a **hybrid sync** approach for reliability:

#### Automatic Sync (Real-time)

- `bd create`, `bd update`, `bd close` write to SQLite DB
- File watcher detects DB changes automatically
- Syncs to Huly within ~30-60 seconds

#### Git Persistence (`bd sync`)

- `bd sync` exports to JSONL and commits to git
- Required for cross-machine persistence
- Run before ending session to ensure changes are saved

### Best Practice

```bash
bd create "New task"   # Auto-syncs to Huly
bd close some-issue    # Auto-syncs to Huly
bd sync                # Git backup (recommended before session end)
```
<!-- VIBESYNC:beads-instructions:END -->

<!-- VIBESYNC:bookstack-docs:START -->
## BookStack Documentation

- **Source of truth**: [BookStack](https://knowledge.oculair.ca)
- **Local sync**: `docs/bookstack/` (read-only mirror, syncs hourly)
- **To read docs**: Check `docs/bookstack/{book-slug}/` in your project directory
- **To create/edit docs**: Use `bookstack-mcp` tools to write directly to BookStack
- **Never edit** files in `docs/bookstack/` locally — they will be overwritten on next sync
- **PRDs and design docs** must be stored in BookStack, not local markdown files
<!-- VIBESYNC:bookstack-docs:END -->

<!-- VIBESYNC:session-completion:START -->
## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**

- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- VIBESYNC:session-completion:END -->

<!-- VIBESYNC:codebase-context:START -->
## Codebase Context

**Project**: Letta Tools Selector (`LTSEL`)
**Path**: `/opt/stacks/lettatoolsselector`

This project's PM agent has a `codebase_ast` memory block with live structural data including:

- File counts and function counts per directory
- Key modules and their roles
- Quality signals (doc gaps, untested modules, complexity hotspots)
- Recent file changes

Ask the PM agent for architectural guidance before making significant changes.
<!-- VIBESYNC:codebase-context:END -->

## Architecture Overview

Multi-service tool management system for Letta AI agents. Uses Weaviate vector DB for semantic search to auto-discover, attach, and prune tools.

### Services (compose.yaml)

| Service | Stack | Port | Role |
|---------|-------|------|------|
| MCP Server | Node.js | 3020 | HTTP-based MCP server, exposes `find_tools` |
| API Server | Python/Quart | 8020 | Tool search, attach, prune REST API |
| Worker Service | Python/FastAPI | 3021 | Persistent `find_tools` endpoint for MCP |
| Weaviate | Go | 8080 | Vector DB for tool embeddings |
| Embedding Proxy | Python/FastAPI | 8450 | Rewrites OpenAI model names for vLLM |
| Sync Service | Python | — | Syncs tools between Letta API and Weaviate |
| Time Service | Python | — | Time-based memory block updates |
| Dashboard API | Python/FastAPI | 8025 | Dashboard backend |
| Dashboard UI | React | 3001 | Dashboard frontend |

### Key API Endpoints

- `POST /api/v1/tools/search` — Semantic tool search
- `POST /api/v1/tools/attach` — Attach tools with auto-detach + optional prune
- `POST /api/v1/tools/prune` — Intelligent relevance-based pruning
- `POST /api/v1/tools/sync` — Force Letta→Weaviate sync
- `POST /mcp` — MCP protocol endpoint
- `GET /api/health` — Health check

### Common Commands

```bash
docker compose up -d                 # Start all services
docker compose logs -f api-server    # Follow specific service logs
docker compose restart api-server    # Restart after code change
```

### Key Configuration

```bash
LETTA_API_URL=http://192.168.50.90:8289/v1
LETTA_PASSWORD=<from .env>
WEAVIATE_URL=http://weaviate:8080/
MIN_MCP_TOOLS=7        # Minimum MCP tools per agent
MAX_MCP_TOOLS=20       # Maximum MCP tools per agent
MAX_TOTAL_TOOLS=30     # Maximum total tools per agent
DEFAULT_DROP_RATE=0.6  # Pruning aggressiveness (0.0-1.0)
NEVER_DETACH_TOOLS=find_tools  # Protected from removal
MANAGE_ONLY_MCP_TOOLS=true     # Only manage MCP tools
```

### Root-level Python modules (imported by services, DO NOT move)

- `letta_tool_utils.py` — Dynamic tool ID lookup utilities
- `tool_selector_client.py` — Client library for tool selector API
- `qwen3_reranker_utils.py` — Qwen3 reranker instruction formatting
- `ollama_reranker_adapter.py` — Ollama reranker adapter

See `docs/` for detailed reference docs, `docs/archive/` for historical analysis.
