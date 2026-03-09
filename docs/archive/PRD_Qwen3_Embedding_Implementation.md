# Product Requirements Document: Qwen3-Embedding-4B Proper Implementation

## Executive Summary

The Letta Tool Selector currently uses Qwen3-Embedding-4B model but does not implement the proper instruction formatting and query handling required by this model. This results in significantly degraded search performance and poor tool discovery accuracy. This PRD outlines the requirements to implement proper Qwen3 formatting throughout the system to achieve optimal embedding performance.

**Business Impact:**
- Improved tool search accuracy (estimated 15-30% improvement)
- Better semantic matching between user queries and tool capabilities
- Reduced false positives in tool recommendations
- Enhanced user experience through more relevant tool suggestions

## Problem Statement

### Current Implementation Issues

Based on code analysis, the following critical issues were identified:

#### 1. Incorrect Instruction Format
**File:** `tool-selector-api/specialized_embedding.py`
**Issue:** Uses generic instruction templates instead of Qwen3's required "Instruct:" and "Query:" format.

```python
# Current (INCORRECT)
PromptType.SEARCH_QUERY: PromptTemplate(
    instruction="Given a tool search request, find tools that match the user's intent",
    context="The user is looking for tools to help accomplish a specific task or goal.",
    suffix="Focus on understanding the user's underlying need..."
)
```

**Should be:**
```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
```

#### 2. Wrong Embedding API Usage
**File:** `tool-selector-api/embedding_providers.py`
**Issue:** Uses standard OpenAI-compatible API calls that don't implement Qwen3's last token pooling.

```python
# Current (INCORRECT) - Lines 201-204
response = await self.client.embeddings.create(
    model=self.model,
    input=texts
)
```

#### 3. Query Contamination
**File:** `tool-selector-api/weaviate_tool_search.py`
**Issue:** Adds filler text that hurts recall with last token pooling embedders.

```python
# Current (INCORRECT) - Line 117
enhanced_query = enhance_query_for_embedding(expanded_query)
```

#### 4. Improper Tool Description Enhancement
**File:** `tool-selector-api/upload_tools_to_weaviate.py`
**Issue:** Uses wrong enhancement format for tool descriptions.

```python
# Current (INCORRECT) - Lines 225-230
enhanced_description = enhance_tool_for_embedding(
    tool_description=raw_description,
    tool_name=tool_name,
    tool_type=tool.get("tool_type", "general"),
    tool_source=tool.get("source_type", "python")
)
```

## Requirements

### Functional Requirements

#### FR1: Implement Proper Qwen3 Instruction Format
- **Priority:** Critical
- **Description:** Replace current instruction templates with Qwen3's "Instruct:" and "Query:" format
- **Acceptance Criteria:**
  - All search queries use format: `Instruct: {task_description}\nQuery: {query}`
  - Tool descriptions use raw text without instruction prefixes
  - Task-specific instructions are properly implemented

#### FR2: Remove Query Enhancement/Contamination
- **Priority:** Critical
- **Description:** Remove all filler text additions that degrade last token pooling performance
- **Acceptance Criteria:**
  - `enhance_query_for_embedding()` functions are removed or refactored
  - Queries end with the most relevant noun
  - No unnecessary descriptive text is added to queries

#### FR3: Implement Last Token Pooling
- **Priority:** High
- **Description:** Ensure proper last token pooling for Qwen3 embeddings
- **Acceptance Criteria:**
  - Either implement direct Transformers usage or configure Ollama correctly
  - Verify embeddings use last token pooling methodology
  - Maintain compatibility with existing Weaviate integration

#### FR4: Task-Specific Instruction Implementation
- **Priority:** High
- **Description:** Use appropriate instructions for different embedding contexts
- **Acceptance Criteria:**
  - Search queries: "Given a web search query, retrieve relevant passages that answer the query"
  - Tool descriptions: No instruction prefix (raw text)
  - Consistent application across all embedding operations

### Technical Requirements

#### TR1: Backward Compatibility
- **Priority:** High
- **Description:** Maintain compatibility with existing tool embeddings during transition
- **Acceptance Criteria:**
  - Existing embeddings continue to work during migration
  - Gradual migration strategy implemented
  - Rollback capability maintained

#### TR2: Performance Optimization
- **Priority:** Medium
- **Description:** Optimize embedding generation and retrieval performance
- **Acceptance Criteria:**
  - No significant performance degradation during transition
  - Embedding generation time remains acceptable
  - Memory usage stays within current limits

#### TR3: Configuration Management
- **Priority:** Medium
- **Description:** Add configuration options for Qwen3-specific settings
- **Acceptance Criteria:**
  - Environment variables for Qwen3 configuration
  - Ability to toggle between old and new formatting
  - Clear documentation of configuration options

## Implementation Plan

### Phase 1: Core Infrastructure Changes

#### 1.1 Update Specialized Embedding Module
**File:** `tool-selector-api/specialized_embedding.py`
**Changes Required:**
- Replace `PromptTemplate` system with Qwen3 format functions
- Implement `get_detailed_instruct(task_description: str, query: str) -> str`
- Add task-specific instruction constants
- Remove context and suffix additions

**New Functions to Add:**
```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    """Generate Qwen3-compatible instruction format."""
    return f'Instruct: {task_description}\nQuery: {query}'

def get_search_instruction() -> str:
    """Get standard search instruction for Qwen3."""
    return "Given a web search query, retrieve relevant passages that answer the query"

def format_query_for_qwen3(query: str) -> str:
    """Format query for Qwen3 without contamination."""
    # Clean query and ensure it ends with most relevant noun
    cleaned_query = query.strip()
    return cleaned_query
```

#### 1.2 Update Embedding Providers
**File:** `tool-selector-api/embedding_providers.py`
**Changes Required:**
- Add Qwen3-specific embedding provider class
- Implement proper last token pooling
- Add configuration for instruction formatting
- Maintain backward compatibility

**New Class to Add:**
```python
class Qwen3EmbeddingProvider(OllamaEmbeddingProvider):
    """Qwen3-specific embedding provider with proper instruction formatting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_instruction_format = True
        self.use_last_token_pooling = True
    
    async def get_embeddings_with_instructions(
        self, 
        texts: List[str], 
        task_description: str = None
    ) -> EmbeddingResult:
        """Get embeddings using Qwen3 instruction format."""
        # Implementation details...
```

#### 1.3 Update Search Functions
**File:** `tool-selector-api/weaviate_tool_search.py`
**Changes Required:**
- Remove `enhance_query_for_embedding()` calls
- Implement proper Qwen3 query formatting
- Add configuration toggle for new vs old behavior

**Changes:**
- Line 8: Remove import of `enhance_query_for_embedding`
- Line 117: Replace with proper Qwen3 formatting
- Add new function for Qwen3 query preparation

### Phase 2: Tool Upload and Enhancement

#### 2.1 Update Tool Upload Process
**File:** `tool-selector-api/upload_tools_to_weaviate.py`
**Changes Required:**
- Modify tool description enhancement to use raw text
- Remove instruction prefixes for tool descriptions
- Update batch upload process

**Changes:**
- Lines 225-230: Replace `enhance_tool_for_embedding()` with raw description
- Add Qwen3-specific tool preparation function
- Update schema to support new embedding format

#### 2.2 Update API Server
**File:** `tool-selector-api/api_server.py`
**Changes Required:**
- Update search endpoints to use new formatting
- Add configuration endpoints for Qwen3 settings
- Maintain backward compatibility

### Phase 3: Configuration and Testing

#### 3.1 Environment Configuration
**New Environment Variables:**
```bash
# Qwen3-specific configuration
USE_QWEN3_FORMAT=true
QWEN3_INSTRUCTION_MODE=search  # search, document, auto
QWEN3_LAST_TOKEN_POOLING=true
QWEN3_MIGRATION_MODE=gradual   # immediate, gradual, disabled
```

#### 3.2 Migration Strategy
- **Gradual Migration:** Support both old and new formats during transition
- **A/B Testing:** Compare performance between old and new implementations
- **Rollback Plan:** Ability to revert to previous implementation

## Success Criteria

### Quantitative Metrics
1. **Search Accuracy Improvement:** 15-30% improvement in relevant tool discovery
2. **Precision/Recall:** Measurable improvement in search precision and recall
3. **Response Time:** No degradation in search response times
4. **Error Rate:** No increase in embedding generation errors

### Qualitative Metrics
1. **User Satisfaction:** Improved relevance of tool recommendations
2. **System Stability:** No regressions in system stability
3. **Maintainability:** Cleaner, more maintainable codebase

### Testing Requirements
1. **Unit Tests:** All new functions have comprehensive unit tests
2. **Integration Tests:** End-to-end search functionality testing
3. **Performance Tests:** Benchmark comparison before/after implementation
4. **A/B Testing:** Side-by-side comparison of old vs new implementations

## Risk Assessment

### High Risk
1. **Breaking Changes:** Risk of breaking existing functionality
   - **Mitigation:** Gradual migration with feature flags
2. **Performance Degradation:** Risk of slower embedding generation
   - **Mitigation:** Performance testing and optimization

### Medium Risk
1. **Compatibility Issues:** Risk of incompatibility with existing embeddings
   - **Mitigation:** Backward compatibility layer
2. **Configuration Complexity:** Risk of complex configuration management
   - **Mitigation:** Clear documentation and sensible defaults

### Low Risk
1. **User Adoption:** Risk of user resistance to changes
   - **Mitigation:** Transparent communication and gradual rollout

## Timeline

### Week 1-2: Infrastructure Changes
- Update specialized embedding module
- Implement Qwen3 embedding provider
- Create configuration framework

### Week 3-4: Integration and Testing
- Update search functions
- Modify tool upload process
- Implement comprehensive testing

### Week 5-6: Migration and Validation
- Deploy gradual migration
- Conduct A/B testing
- Performance validation and optimization

### Week 7: Documentation and Rollout
- Complete documentation
- Full production rollout
- Monitor and optimize

## Dependencies

1. **Ollama Configuration:** Ensure Ollama supports Qwen3 last token pooling
2. **Weaviate Compatibility:** Verify Weaviate handles new embedding format
3. **Testing Infrastructure:** Adequate testing environment for validation

## Appendix

### File References
- `tool-selector-api/specialized_embedding.py` - Core embedding enhancement logic
- `tool-selector-api/embedding_providers.py` - Embedding provider implementations
- `tool-selector-api/weaviate_tool_search.py` - Search functionality
- `tool-selector-api/upload_tools_to_weaviate.py` - Tool upload and enhancement
- `tool-selector-api/api_server.py` - API endpoints and configuration

### Technical References
- Qwen3-Embedding-4B Documentation
- Weaviate Vector Database Documentation
- Ollama API Documentation
- Last Token Pooling Implementation Guidelines
