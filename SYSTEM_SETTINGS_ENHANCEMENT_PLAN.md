# System Settings Enhancement Plan
## Full Tool Selector Configuration Control

### Overview

The current SystemSettings page functions as a read-only dashboard but needs to be transformed into a comprehensive admin control panel for the tool selector system. This document outlines the required enhancements to provide full configuration control over embedding models, tool management, and system behavior.

## Current State Analysis

### What Works
- ✅ System health monitoring (API server, database, search engine status)
- ✅ Tool inventory display (total tools, sources, categories)
- ✅ Tool refresh functionality
- ✅ Performance metrics display
- ✅ Source breakdown visualization

### Critical Issues
- ❌ Hardcoded embedding model display ("OpenAI text-embedding-3-small")
- ❌ No embedding model configuration controls
- ❌ No reembedding functionality
- ❌ No tool selector behavior configuration
- ❌ No system parameter controls
- ❌ Read-only interface with no admin controls

## Priority 1: Essential Missing Features

### 1. Embedding Model Management Section

**Current Problem**: Shows hardcoded "OpenAI text-embedding-3-small" text
**Required Solution**: Dynamic embedding model configuration with reembedding controls

#### Features Needed:
- **Current Model Display**: Show actual current embedding model from API
- **Provider Selection**: Dropdown for OpenAI vs Ollama providers
- **Model Selection**: Dynamic model list based on selected provider
- **Reembedding Controls**: "Start Reembedding" button with progress tracking
- **Progress Integration**: Embed ReembeddingProgress component
- **Status Indicators**: Show embedding model health and last update time

#### UI Components:
```tsx
<Accordion>
  <AccordionSummary>Embedding Model Configuration</AccordionSummary>
  <AccordionDetails>
    <FormControl>Provider Selection</FormControl>
    <FormControl>Model Selection</FormControl>
    <Button>Start Reembedding</Button>
    <ReembeddingProgress />
  </AccordionDetails>
</Accordion>
```

### 2. Tool Selector Configuration Section

**Current Problem**: No way to configure tool selector behavior
**Required Solution**: Interactive controls for all tool selector parameters

#### Configuration Parameters:
- **MAX_TOTAL_TOOLS**: Slider (10-100, default: 30)
- **MAX_MCP_TOOLS**: Slider (5-50, default: 20) 
- **MIN_MCP_TOOLS**: Slider (1-20, default: 7)
- **DEFAULT_DROP_RATE**: Slider (0.1-0.9, default: 0.6)
- **EXCLUDE_LETTA_CORE_TOOLS**: Toggle switch
- **EXCLUDE_OFFICIAL_TOOLS**: Toggle switch
- **MANAGE_ONLY_MCP_TOOLS**: Toggle switch

#### UI Layout:
```tsx
<Card>
  <CardHeader>Tool Selector Configuration</CardHeader>
  <CardContent>
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Typography>Tool Limits</Typography>
        <Slider label="Max Total Tools" />
        <Slider label="Max MCP Tools" />
        <Slider label="Min MCP Tools" />
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography>Behavior Settings</Typography>
        <Slider label="Drop Rate" />
        <Switch label="Exclude Letta Core Tools" />
        <Switch label="Exclude Official Tools" />
        <Switch label="Manage Only MCP Tools" />
      </Grid>
    </Grid>
  </CardContent>
</Card>
```

### 3. Weaviate & Search Engine Configuration

**Current Problem**: No way to configure Weaviate or search parameters
**Required Solution**: Comprehensive search engine parameter controls

#### Weaviate Configuration:
- **Connection Settings**: URL, API key status, connection test
- **Vector Index Parameters**:
  - HNSW ef parameter (search accuracy vs speed)
  - Dynamic ef settings (min: 100, max: 500, factor: 8)
  - Vector cache size and cleanup intervals
- **Hybrid Search Parameters**:
  - Alpha parameter (0.0-1.0, default: 0.75) - controls vector vs keyword balance
  - Fusion type: rankedFusion vs relativeScoreFusion
  - Max vector distance threshold
- **Performance Settings**:
  - Candidate pool size before reranking
  - Batch processing sizes
  - Timeout settings for search and reranking operations

#### Search Quality Controls:
- **Semantic Search Weight**: Vector search influence (0.0-1.0)
- **Keyword Search Weight**: BM25 search influence (0.0-1.0)
- **Score Thresholds**: Minimum relevance scores for results
- **Result Limits**: Max results per query, pagination settings

## Priority 2: Advanced Features

### 4. System Maintenance Tools

#### Features:
- **Cache Management**: Clear search cache, reset indexes
- **Database Cleanup**: Remove orphaned records, optimize storage
- **Configuration Export/Import**: Backup and restore settings
- **Force Sync**: Manual tool synchronization (enhance existing)

### 5. Environment & API Configuration

#### API Endpoint Management:
- **LDTS Integration**:
  - LDTS_API_URL (default: http://localhost:8020)
  - LDTS_MCP_URL (default: http://localhost:3020)
  - Connection status and health checks
- **Weaviate Configuration**:
  - WEAVIATE_URL (default: http://localhost:8080)
  - WEAVIATE_API_KEY status (masked display)
  - Connection testing and validation
- **Ollama Configuration**:
  - OLLAMA_BASE_URL (default: http://192.168.50.80:11434)
  - Available models discovery via /api/tags
  - Model status and health monitoring

#### Provider API Key Management:
- **OpenAI Configuration**:
  - OPENAI_API_KEY status (masked: sk-...abc123)
  - Usage limits and rate limiting status
  - Cost tracking and budget alerts
- **Security Features**:
  - Masked display of sensitive keys
  - Key validation and testing
  - Rotation reminders and expiry tracking

#### Performance & Rate Limiting:
- **Request Limits**: Configure per-minute/hour limits
- **Timeout Configuration**: Set timeouts for various operations
- **Retry Logic**: Configure retry attempts and backoff strategies
- **Circuit Breakers**: Automatic failover and recovery settings

### 6. Enhanced Monitoring & Diagnostics

#### Features:
- **Real-time Metrics**: Live performance data
- **Error Log Display**: Recent errors and warnings
- **Health Check Details**: Detailed component status
- **Usage Analytics**: Tool usage patterns and statistics

## Implementation Plan

### Phase 1: Core Functionality (Week 1)
1. **Embedding Model Management**:
   - Dynamic embedding model display from API
   - Provider selection (OpenAI vs Ollama) with model discovery
   - "Start Reembedding" button with progress tracking
   - Integration of ReembeddingProgress component

2. **Tool Selector Configuration**:
   - Tool limits sliders (MAX_TOTAL_TOOLS, MAX_MCP_TOOLS, MIN_MCP_TOOLS)
   - Drop rate slider with visual feedback
   - Policy toggles (EXCLUDE_LETTA_CORE_TOOLS, etc.)
   - Real-time configuration validation

### Phase 2: Advanced Controls (Week 2)
1. **Weaviate Configuration**:
   - Hybrid search alpha parameter control (0.0-1.0)
   - HNSW index parameters (ef, dynamic ef settings)
   - Connection testing and health monitoring
   - Vector cache management controls

2. **Ollama Integration**:
   - Model discovery via /api/tags endpoint
   - Running models display via /api/ps
   - Model pull/delete operations
   - Embedding model switching with automatic reembedding

3. **Environment Management**:
   - API endpoint configuration (LDTS, Weaviate, Ollama URLs)
   - API key status display (masked)
   - Connection testing for all services
   - Environment variable validation

### Phase 3: System Maintenance & Polish (Week 3)
1. **Advanced Features**:
   - System cache clearing and database cleanup
   - Configuration export/import functionality
   - Audit logging for all configuration changes
   - Performance metrics and diagnostics

2. **Safety & Validation**:
   - Confirmation dialogs for dangerous operations
   - Configuration rollback capabilities
   - Real-time validation with error/warning messages
   - Impact assessment for major changes

3. **Testing & Documentation**:
   - Comprehensive testing of all configuration changes
   - User documentation and help tooltips
   - Error handling and recovery procedures
   - Performance optimization

## Backend API Requirements

### New Endpoints Needed:
```
GET /api/v1/config/tool-selector     # Get tool selector configuration
PUT /api/v1/config/tool-selector     # Update tool selector configuration
GET /api/v1/config/weaviate          # Get Weaviate configuration
PUT /api/v1/config/weaviate          # Update Weaviate configuration
GET /api/v1/config/ollama            # Get Ollama configuration
PUT /api/v1/config/ollama            # Update Ollama configuration
POST /api/v1/system/cache/clear      # Clear system caches
POST /api/v1/system/database/cleanup # Database maintenance
GET /api/v1/system/diagnostics       # Detailed system diagnostics
GET /api/v1/system/environment       # Get environment variables status
PUT /api/v1/system/environment       # Update environment variables
```

### Enhanced Endpoints:
```
GET /api/v1/config/embedding         # Already exists, needs enhancement
PUT /api/v1/config/embedding         # Already exists, needs enhancement
POST /api/v1/reembedding/start       # Already exists
GET /api/v1/reembedding/progress     # Already exists
GET /api/v1/config/presets           # Already exists
POST /api/v1/config/presets          # Already exists
PUT /api/v1/config/presets/<id>      # Already exists
```

### Tool Selector Configuration Schema:
```json
{
  "tool_limits": {
    "max_total_tools": 30,
    "max_mcp_tools": 20,
    "min_mcp_tools": 7
  },
  "behavior": {
    "default_drop_rate": 0.6,
    "exclude_letta_core_tools": true,
    "exclude_official_tools": true,
    "manage_only_mcp_tools": true
  },
  "scoring": {
    "min_score_default": 70.0,
    "semantic_weight": 0.7,
    "keyword_weight": 0.3
  }
}
```

## Safety Considerations

### Validation & Confirmation:
- Validate all configuration changes before applying
- Show impact warnings for major changes (reembedding, tool limits)
- Require confirmation for dangerous operations
- Provide clear rollback capabilities

### Logging & Audit:
- Log all configuration changes with timestamps
- Track who made changes (if authentication added)
- Maintain configuration history
- Alert on critical system changes

## UI/UX Guidelines

### Organization:
- Use expandable Accordion components for logical grouping
- Separate read-only status from interactive controls
- Clear visual hierarchy with proper spacing
- Consistent button and control styling

### Feedback:
- Real-time validation feedback
- Progress indicators for long operations
- Success/error notifications
- Clear status indicators

### Accessibility:
- Proper ARIA labels for all controls
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support

## Success Metrics

### Functionality:
- ✅ Admin can change embedding models and trigger reembedding
- ✅ Admin can configure all tool selector parameters
- ✅ Admin can monitor and control system behavior
- ✅ All changes are validated and safely applied

### User Experience:
- ✅ Intuitive interface for non-technical administrators
- ✅ Clear feedback for all operations
- ✅ No accidental system disruption
- ✅ Easy rollback of problematic changes

## Advanced Configuration Details

### Weaviate Hybrid Search Alpha Parameter Deep Dive

Based on Context7 research, the alpha parameter is critical for search quality:

```javascript
// Alpha Parameter Effects:
// α = 0.0: Pure keyword search (BM25)
// α = 0.5: Equal weighting of vector and keyword
// α = 0.75: Default - favors vector search (75% vector, 25% keyword)
// α = 1.0: Pure vector search

const hybridSearchConfig = {
  alpha: 0.75,           // Default Weaviate setting
  fusionType: "rankedFusion", // or "relativeScoreFusion"
  maxVectorDistance: 0.6  // Quality threshold
}
```

### Ollama Model Management Integration

The system should integrate with Ollama's API for dynamic model discovery:

```bash
# Available Ollama API endpoints for model management:
GET /api/tags              # List available models
POST /api/pull             # Download new models
DELETE /api/delete         # Remove models
POST /api/show             # Get model details
GET /api/ps               # List running models
POST /api/embeddings      # Generate embeddings
```

### Tool Selector Environment Variables

Current environment variables that need UI controls:

```bash
# Tool Management Limits
MAX_TOTAL_TOOLS=30         # Maximum tools per agent
MAX_MCP_TOOLS=20          # Maximum MCP tools
MIN_MCP_TOOLS=7           # Minimum MCP tools required
DEFAULT_DROP_RATE=0.6     # Tool pruning aggressiveness (0.1-0.9)

# Tool Behavior Policies
EXCLUDE_LETTA_CORE_TOOLS=true    # Skip Letta built-in tools
EXCLUDE_OFFICIAL_TOOLS=true      # Skip official Letta tools
MANAGE_ONLY_MCP_TOOLS=true       # Only manage MCP tools

# Search Quality Settings
MIN_SCORE_DEFAULT=70.0           # Minimum relevance score
SEMANTIC_WEIGHT=0.7              # Vector search weight
KEYWORD_WEIGHT=0.3               # BM25 search weight

# Provider Configuration
EMBEDDING_PROVIDER=ollama         # openai or ollama
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M
EMBEDDING_DIMENSION=2560          # Model-specific dimension
```

### Real-time Configuration Validation

The system should validate configurations before applying:

```typescript
interface ConfigValidation {
  field: string;
  isValid: boolean;
  errorMessage?: string;
  warningMessage?: string;
  suggestedValue?: any;
}

// Example validations:
const validations = [
  {
    field: "alpha",
    isValid: value >= 0.0 && value <= 1.0,
    errorMessage: "Alpha must be between 0.0 and 1.0"
  },
  {
    field: "max_total_tools",
    isValid: value >= 1 && value <= 100,
    warningMessage: value > 50 ? "High tool limits may impact performance" : undefined
  }
];
```

This enhancement plan transforms the SystemSettings from a passive dashboard into an active administrative control center for the tool selector system.
