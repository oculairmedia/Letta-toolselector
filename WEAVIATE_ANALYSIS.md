# Weaviate Tool Database Analysis

## Overview

The Letta Tool Selector uses Weaviate as a vector database to store and search through tools using semantic embeddings. This document provides a comprehensive breakdown of the data structure and frontend capabilities.

## Database Statistics

- **Total Tools**: 180 tools stored in Weaviate
- **Vector Model**: Ollama Qwen3-Embedding-4B (hosted at 192.168.50.80:11434)
- **Search Method**: Hybrid search (75% vector, 25% keyword/BM25)
- **Distance Metric**: Cosine similarity

## Tool Data Structure

### Core Properties

| Property | Type | Description | Vectorized |
|----------|------|-------------|------------|
| `tool_id` | text | Unique identifier (e.g., "tool-6e14b724-f646-4b53-82e9-6fcd6928a10a") | Yes |
| `name` | text | Tool function name (e.g., "read_bookshelf", "huly_delete_component") | Yes |
| `description` | text | Original tool description from Letta API | Yes |
| `enhanced_description` | text | AI-enhanced description for better embeddings (nullable) | Yes |
| `source_type` | text | Source language/runtime (all currently "python") | Yes |
| `tool_type` | text | Tool category classification | Yes |
| `tags` | text[] | Array of categorization tags | Yes |
| `json_schema` | text | Complete JSON schema for tool parameters and interface | Yes |
| `mcp_server_name` | text | Source MCP server name | Yes |

### Tool Type Distribution

| Tool Type | Count | Description |
|-----------|-------|-------------|
| `external_mcp` | 156 | Tools from external MCP servers |
| `letta_core` | 4 | Core Letta functionality tools |
| `letta_sleeptime_core` | 4 | Sleep/timing related core tools |
| `letta_voice_sleeptime_core` | 4 | Voice + sleep core tools |
| `letta_builtin` | 3 | Built-in Letta tools |

### MCP Server Distribution (External MCP Tools)

| MCP Server | Tool Count | Description |
|------------|------------|-------------|
| `huly` | 35 | Project management and issue tracking |
| `letta` | 31 | Agent and memory management |
| `bookstack` | 20 | Documentation and knowledge management |
| `resumerx` | 18 | Resume/CV processing tools |
| `filesystem` | 14 | File system operations |
| `ghost` | 10 | CMS and content management |
| `hayhooks` | 8 | Haystack AI pipeline tools |
| `graphiti` | 7 | Graph memory and knowledge management |
| `matrix` | 5 | Matrix messaging protocol |
| `postiz` | 3 | Social media management |
| `claude-code-mcp` | 2 | Claude Code integration |
| `context7` | 2 | Context/documentation tools |
| `lettatoolsselector` | 1 | Tool selection system |

## Sample Data Output

### Tool Record Example (Huly Project Management Tool)

```json
{
  "tool_id": "tool-b6519afe-8e05-4dca-9338-43899eca10a3",
  "name": "huly_delete_component",
  "description": "Delete a component from a project",
  "enhanced_description": "Given an MCP tool description, encode its external service capabilities This is an MCP (Model Context Protocol) tool that provides external service integration. Tool: huly_delete_component Content: Delete a component from a project Focus on the external service functionality and integration capabilities it provides.",
  "source_type": "python",
  "tool_type": "external_mcp",
  "tags": ["mcp:huly"],
  "mcp_server_name": "huly",
  "json_schema": "{\"name\": \"huly_delete_component\", \"description\": \"Delete a component from a project\", \"parameters\": {\"type\": \"object\", \"properties\": {\"project_identifier\": {\"type\": \"string\", \"description\": \"Project identifier (e.g., \\\"PROJ\\\")\"}, \"component_label\": {\"type\": \"string\", \"description\": \"Component label to delete\"}, \"force\": {\"type\": \"boolean\", \"description\": \"Force deletion even if issues use this component (default: false)\", \"default\": false}, \"dry_run\": {\"type\": \"boolean\", \"description\": \"Preview deletion impact without actually deleting (default: false)\", \"default\": false}, \"request_heartbeat\": {\"type\": \"boolean\", \"description\": \"Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.\"}}, \"required\": [\"project_identifier\", \"component_label\", \"request_heartbeat\"]}}"
}
```

### Tool Record Example (BookStack Documentation Tool)

```json
{
  "tool_id": "tool-6e14b724-f646-4b53-82e9-6fcd6928a10a",
  "name": "read_bookshelf",
  "description": "Retrieves details of a specific bookshelf in Bookstack",
  "enhanced_description": "Given an MCP tool description, encode its external service capabilities This is an MCP (Model Context Protocol) tool that provides external service integration. Tool: read_bookshelf Content: Retrieves details of a specific bookshelf in Bookstack Focus on the external service functionality and integration capabilities it provides.",
  "source_type": "python",
  "tool_type": "external_mcp",
  "tags": ["mcp:bookstack"],
  "mcp_server_name": "bookstack",
  "json_schema": "{\"name\": \"read_bookshelf\", \"description\": \"Retrieves details of a specific bookshelf in Bookstack\", \"parameters\": {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"number\", \"description\": \"The ID of the bookshelf to retrieve\"}, \"request_heartbeat\": {\"type\": \"boolean\", \"description\": \"Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.\"}}, \"required\": [\"id\", \"request_heartbeat\"]}}"
}
```

### Tool Record Example (Graphiti Graph Memory Tool)

```json
{
  "tool_id": "tool-cc71965a-dbed-42f5-8bf6-7227252725a5",
  "name": "delete_entity_edge",
  "description": "Delete an entity edge from the graph memory via FastAPI server.\n\n    Args:\n        uuid: UUID of the entity edge to delete\n    ",
  "enhanced_description": null,
  "source_type": "python",
  "tool_type": "external_mcp",
  "tags": ["mcp:graphiti"],
  "mcp_server_name": "graphiti",
  "json_schema": "{\"name\": \"delete_entity_edge\", \"description\": \"Delete an entity edge from the graph memory via FastAPI server.\\n\\n    Args:\\n        uuid: UUID of the entity edge to delete\\n    \", \"parameters\": {\"properties\": {\"uuid\": {\"title\": \"Uuid\", \"type\": \"string\"}, \"request_heartbeat\": {\"type\": \"boolean\", \"description\": \"Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.\"}}, \"required\": [\"uuid\", \"request_heartbeat\"], \"title\": \"delete_entity_edgeArguments\", \"type\": \"object\"}}"
}
```

## Frontend Display Capabilities

### 1. Tool Browser/Explorer

**Grid/List View Features:**
- Tool name, description, and MCP server source
- Visual indicators for tool types (external_mcp, letta_core, etc.)
- Server-based color coding or icons
- Quick action buttons for common operations

**Server Grouping:**
- Collapsible sections organized by MCP server
- Server statistics (tool count, last sync time)
- Server health indicators

**Advanced Filtering:**
- Tool type filter (external_mcp, letta_core, letta_builtin, etc.)
- MCP server filter with multi-select
- Tag-based filtering
- Search across names, descriptions, and enhanced descriptions

### 2. Tool Detail Views

**Schema Viewer:**
- Parse JSON schema into user-friendly parameter tables
- Required vs optional parameter indicators
- Parameter type information with examples
- Default value display

**Documentation Display:**
- Primary description with formatting
- Enhanced description when available
- Usage examples generated from schema
- Related tools suggestions

### 3. Analytics Dashboard

**Distribution Visualizations:**
- Pie chart: Tools per MCP server
- Bar chart: Tool type breakdown
- Timeline: Tool registration history
- Heatmap: Tool usage patterns (if tracking data available)

**Search Analytics:**
- Most searched tools ranking
- Popular categories trending
- Search success/failure rates
- Query performance metrics

### 4. Management Interface

**Bulk Operations:**
- Multi-select tool checkboxes
- Batch enable/disable tools
- Bulk tag management
- Mass tool updates

**Status Monitoring:**
- Tool availability indicators
- Last sync timestamps
- Error states and diagnostics
- Health check results

**Data Management:**
- Force sync with Letta API
- Clear/rebuild tool cache
- Export tool configurations
- Import/restore tool sets

## Technical Implementation Notes

### Weaviate Configuration
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Vector Cache**: 1TB limit
- **Cleanup Interval**: 300 seconds
- **BM25 Settings**: b=0.75, k1=1.2
- **Dynamic EF**: 100-500 range

### API Access Patterns
- **GraphQL Endpoint**: `http://localhost:8080/v1/graphql`
- **REST API**: `http://localhost:8080/v1/objects/Tool`
- **Batch Operations**: Available for bulk updates
- **Real-time**: WebSocket support for live updates

### Performance Considerations
- Pagination recommended for large result sets
- Hybrid search provides best balance of semantic and keyword matching
- Consider caching frequently accessed tool schemas
- Implement lazy loading for tool details

## Query Examples

### Get All Tools with Pagination
```graphql
{
  Get {
    Tool(limit: 20, offset: 0) {
      tool_id
      name
      description
      mcp_server_name
      tool_type
    }
  }
}
```

### Filter by MCP Server
```graphql
{
  Get {
    Tool(where: {
      path: ["mcp_server_name"]
      operator: Equal
      valueText: "huly"
    }) {
      tool_id
      name
      description
    }
  }
}
```

### Semantic Search
```graphql
{
  Get {
    Tool(
      hybrid: {
        query: "project management tasks"
        alpha: 0.75
      }
      limit: 10
    ) {
      tool_id
      name
      description
      mcp_server_name
      _additional {
        score
      }
    }
  }
}
```

This data structure provides a rich foundation for building comprehensive tool management and discovery interfaces with advanced search, filtering, and visualization capabilities.