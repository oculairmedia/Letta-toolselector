# Semantic Enrichment System Design

## Problem Statement

Current tool search relies on original tool descriptions which are often:
- Too brief ("Create a new issue")
- Missing context about the MCP server's domain
- Lacking natural language terms users actually search for
- Not decomposed into searchable action-entity pairs

## Proposed Solution: Multi-Layer Semantic Enrichment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Enrichment Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ MCP Server   │───►│ Tool         │───►│ Weaviate     │       │
│  │ Profiler     │    │ Enricher     │    │ Indexer      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Server       │    │ Enhanced     │    │ Tool         │       │
│  │ Profiles     │    │ Descriptions │    │ Collection   │       │
│  │ (cached)     │    │ (cached)     │    │ (indexed)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: MCP Server Profiler

**Purpose**: Create semantic profiles for each MCP server that provide context for all its tools.

**Process**:
1. Fetch all tools from an MCP server
2. Analyze tool names, descriptions, and parameters
3. Use Claude Sonnet (via Anthropic proxy) to generate a server profile

**Server Profile Schema**:
```python
@dataclass
class MCPServerProfile:
    server_name: str
    domain: str  # e.g., "project management", "file operations"
    primary_capabilities: List[str]  # e.g., ["issue tracking", "sprint planning"]
    entity_types: List[str]  # e.g., ["issue", "project", "sprint", "milestone"]
    action_verbs: List[str]  # e.g., ["create", "query", "update", "close"]
    integration_context: str  # How it fits in workflows
    semantic_keywords: List[str]  # Related terms users might search
    profile_hash: str  # Hash of tool list for change detection
    last_updated: datetime
```

**Example Profile for Huly**:
```json
{
  "server_name": "huly",
  "domain": "project management and issue tracking",
  "primary_capabilities": [
    "issue lifecycle management",
    "project organization",
    "sprint planning",
    "workflow automation"
  ],
  "entity_types": ["issue", "project", "sprint", "milestone", "label", "assignee"],
  "action_verbs": ["create", "update", "query", "assign", "close", "transition"],
  "integration_context": "Huly is a project management platform similar to Jira or Linear. Use these tools when users need to track tasks, manage projects, or coordinate team work.",
  "semantic_keywords": [
    "task", "ticket", "bug", "feature request", "todo",
    "backlog", "kanban", "agile", "scrum",
    "project management", "issue tracker", "work item"
  ],
  "profile_hash": "sha256:abc123...",
  "last_updated": "2025-12-28T22:00:00Z"
}
```

### Layer 2: Tool Enricher

**Purpose**: Generate semantically rich descriptions for each tool using server context.

**Process**:
1. Load MCP server profile
2. Analyze tool's JSON schema (parameters, operations)
3. Use Claude Sonnet to generate enhanced description
4. Extract action-entity pairs for indexing

**Enhanced Tool Schema**:
```python
@dataclass
class EnrichedTool:
    tool_id: str
    name: str
    original_description: str
    
    # Enriched fields
    enhanced_description: str  # 200-400 word rich description
    action_entities: List[str]  # e.g., ["create issue", "update issue", "query issues"]
    semantic_keywords: List[str]  # Merged from server + tool-specific
    use_cases: List[str]  # Natural language scenarios
    
    # Metadata
    mcp_server_name: str
    server_domain: str  # From server profile
    enrichment_hash: str  # Hash for change detection
    last_enriched: datetime
```

**Enrichment Prompt Template**:
```
You are enhancing tool descriptions for a semantic search system.

## MCP Server Context
Server: {server_name}
Domain: {server_domain}
Capabilities: {server_capabilities}
Entity Types: {entity_types}

## Tool to Enhance
Name: {tool_name}
Current Description: {original_description}
Parameters: {json_schema}

## Your Task
Generate an enhanced description that:
1. Explains what the tool does in the context of {server_domain}
2. Lists 3-5 specific use cases with natural language
3. Includes semantic keywords users would search for
4. Mentions related entity types ({entity_types})
5. Describes how this fits in common workflows

## Action-Entity Extraction
Also extract all action-entity pairs this tool supports.
For example, if the tool can "create issue" and "update issue", list both.

Output JSON:
{
  "enhanced_description": "...",
  "action_entities": ["create issue", "update issue"],
  "use_cases": ["When a user reports a bug...", "..."],
  "semantic_keywords": ["bug report", "task creation", ...]
}
```

### Layer 3: Change Detection & Incremental Updates

**Purpose**: Only re-enrich when MCP servers or tools actually change.

**Change Detection Strategy**:
```python
def compute_server_hash(tools: List[dict]) -> str:
    """Hash of tool names + descriptions + schemas."""
    content = json.dumps(
        [(t['name'], t['description'], t.get('json_schema', '')) 
         for t in sorted(tools, key=lambda x: x['name'])],
        sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()

def needs_re_enrichment(server_name: str, current_tools: List[dict]) -> bool:
    """Check if server profile needs updating."""
    current_hash = compute_server_hash(current_tools)
    cached_profile = load_cached_profile(server_name)
    
    if not cached_profile:
        return True
    if cached_profile.profile_hash != current_hash:
        return True
    if cached_profile.last_updated < datetime.now() - timedelta(days=7):
        return True  # Force refresh weekly
    return False
```

### Layer 4: Weaviate Indexing

**Enhanced Schema**:
```python
properties = [
    # Existing fields
    Property(name="tool_id", data_type=DataType.TEXT),
    Property(name="name", data_type=DataType.TEXT),
    Property(name="description", data_type=DataType.TEXT),
    Property(name="enhanced_description", data_type=DataType.TEXT),
    Property(name="mcp_server_name", data_type=DataType.TEXT),
    
    # NEW: Enrichment fields
    Property(
        name="action_entities", 
        data_type=DataType.TEXT_ARRAY,
        description="Action-entity pairs like 'create issue', 'delete file'"
    ),
    Property(
        name="semantic_keywords",
        data_type=DataType.TEXT_ARRAY, 
        description="Search terms from server + tool enrichment"
    ),
    Property(
        name="server_domain",
        data_type=DataType.TEXT,
        description="MCP server's domain (e.g., 'project management')"
    ),
    Property(
        name="use_cases",
        data_type=DataType.TEXT_ARRAY,
        description="Natural language use case scenarios"
    ),
]
```

**Hybrid Search Enhancement**:
```python
# Updated query with new fields
result = collection.query.hybrid(
    query=query,
    alpha=0.75,
    limit=30,
    query_properties=[
        "name^3",                    # Tool name most important
        "action_entities^2.5",       # NEW: "create issue" matches directly
        "enhanced_description^2",    
        "semantic_keywords^2",       # NEW: Domain keywords
        "use_cases^1.5",             # NEW: Scenario matching
        "server_domain^1.5",         # NEW: "project management" matches
        "description^1",
        "tags"
    ],
    return_metadata=MetadataQuery(score=True)
)
```

## Implementation Plan

### Phase 1: MCP Server Profiler (Week 1)
- [ ] Create `semantic_enrichment/server_profiler.py`
- [ ] Implement Anthropic proxy client for Claude Sonnet
- [ ] Build server profile generation with caching
- [ ] Add hash-based change detection

### Phase 2: Tool Enricher (Week 1-2)
- [ ] Create `semantic_enrichment/tool_enricher.py`
- [ ] Implement enrichment prompt with server context
- [ ] Extract action-entity pairs from JSON schemas
- [ ] Build enrichment cache with hash validation

### Phase 3: Weaviate Schema Update (Week 2)
- [ ] Add new properties to Tool collection
- [ ] Update sync service to populate new fields
- [ ] Modify hybrid search query properties

### Phase 4: Incremental Sync (Week 2-3)
- [ ] Integrate with existing sync_service.py
- [ ] Only re-enrich changed tools
- [ ] Add enrichment status to health check

### Phase 5: Evaluation (Week 3)
- [ ] Compare search accuracy before/after
- [ ] Test edge cases (CRUD tools, unified tools)
- [ ] Measure latency impact

## API Design

### Enrichment Endpoints

```
POST /api/v1/enrichment/server/{server_name}/profile
  - Force re-profile an MCP server

GET /api/v1/enrichment/server/{server_name}/status
  - Get profile status and hash

POST /api/v1/enrichment/sync
  - Run full enrichment sync (detect changes, update as needed)

GET /api/v1/enrichment/stats
  - Get enrichment statistics
```

## Cost Considerations

**Claude Sonnet API Usage**:
- Server profiles: ~500 tokens output x 26 servers = 13K tokens
- Tool enrichment: ~300 tokens output x 181 MCP tools = 54K tokens
- Total initial: ~70K tokens (~$0.21 at Sonnet pricing)
- Incremental: Only changed tools, typically <5K tokens/day

**Caching Strategy**:
- Server profiles: Cache indefinitely, refresh on tool changes
- Tool enrichments: Cache indefinitely, refresh on description/schema changes
- Weekly forced refresh to catch LLM improvements

## Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| "create issue" → huly_issue_ops | Sometimes misses | Direct match |
| "project management" → huly_* | Low recall | High recall |
| "track bugs" → huly_issue_ops | Misses | Direct match |
| CRUD tool discovery | Inconsistent | Reliable |
| Cross-server workflows | No context | Workflow-aware |

## Files to Create

```
tool-selector-api/
├── semantic_enrichment/
│   ├── __init__.py
│   ├── server_profiler.py      # MCP server profiling
│   ├── tool_enricher.py        # Tool description enrichment
│   ├── anthropic_client.py     # Claude Sonnet via proxy
│   ├── enrichment_cache.py     # Hash-based caching
│   ├── change_detector.py      # Detect what needs updating
│   └── models.py               # Pydantic models for profiles
├── routes/
│   └── enrichment.py           # API endpoints
└── services/
    └── enrichment_service.py   # Business logic
```
