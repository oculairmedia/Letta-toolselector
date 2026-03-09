# Weaviate + Ollama Reranker Integration: Detailed Implementation Plan

## Executive Summary

This document outlines the implementation of a two-stage retrieval system to improve tool selection accuracy by 1-5% while maintaining backward compatibility with the existing 175 tools. The solution integrates Qwen3-Reranker-4B hosted on Ollama with Weaviate's native reranking capabilities through an adapter service.

**Key Benefits:**
- Improved tool selection accuracy (target: 1-5% improvement)
- Backward compatible implementation
- Leverages existing Ollama infrastructure
- Maintains Weaviate-native query ergonomics
- Incremental rollout with feature flags

## Architecture Overview

### Current State
```
User Query → Weaviate (Vector/BM25/Hybrid Search) → Top-N Results → User
```

### Proposed State
```
User Query → Weaviate (Initial Retrieval) → Reranker (Top-k) → Reordered Results → User
                ↓
        Ollama Adapter ← Qwen3-Reranker-4B
```

### Integration Patterns

**Pattern A: Native Transformers Reranker (Baseline)**
- Use Weaviate's built-in `reranker-transformers` module
- Quick validation of two-stage pipeline
- Standard cross-encoder model (ms-marco-MiniLM-L-6-v2)

**Pattern B: Ollama Adapter (Target)**
- Custom adapter service implementing Weaviate's reranker inference API
- Calls Ollama-hosted Qwen3-Reranker-4B
- Instruction-aware prompting for 1-5% accuracy gains

## Detailed Implementation

### 1. Docker Compose Configuration

#### Baseline Configuration (Pattern A)
```yaml
version: '3.8'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.25.0
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      # Existing modules
      ENABLE_API_BASED_MODULES: 'true'
      ENABLE_MODULES: 'text2vec-ollama,generative-ollama,reranker-transformers'
      
      # Reranker configuration
      RERANKER_INFERENCE_API: 'http://reranker-transformers:8080'
      
      # Standard Weaviate config
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      CLUSTER_HOSTNAME: 'node1'
      ASYNC_INDEXING: 'true'  # For better ingestion performance
    volumes:
      - weaviate_data:/var/lib/weaviate

  reranker-transformers:
    image: cr.weaviate.io/semitechnologies/reranker-transformers:cross-encoder-ms-marco-MiniLM-L-6-v2
    environment:
      ENABLE_CUDA: '0'  # Set to '1' if GPU available
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  # Existing Ollama service (unchanged)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h

volumes:
  weaviate_data:
  ollama_data:
```

#### Target Configuration (Pattern B)
```yaml
version: '3.8'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.25.0
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      # Existing modules
      ENABLE_API_BASED_MODULES: 'true'
      ENABLE_MODULES: 'text2vec-ollama,generative-ollama,reranker-transformers'
      
      # Ollama adapter configuration
      RERANKER_INFERENCE_API: 'http://reranker-ollama-adapter:8080'
      
      # Standard Weaviate config
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      CLUSTER_HOSTNAME: 'node1'
      ASYNC_INDEXING: 'true'
    volumes:
      - weaviate_data:/var/lib/weaviate

  reranker-ollama-adapter:
    image: lettatoolsselector/reranker-ollama-adapter:latest
    ports:
      - "8081:8080"  # For debugging/monitoring
    environment:
      OLLAMA_BASE_URL: 'http://ollama:11434'
      OLLAMA_MODEL: 'qwen3-reranker-4b'
      LOG_LEVEL: 'INFO'
      BATCH_SIZE: '10'
      TIMEOUT_SECONDS: '30'
      ENABLE_FEATURE_FLAG: 'true'
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Existing Ollama service (unchanged)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h

volumes:
  weaviate_data:
  ollama_data:
```

### 2. Schema Configuration (Backward Compatible)

#### Collection Creation with Reranker
```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.Client("http://localhost:8080")

# Create Tools collection with reranker configuration
client.collections.create(
    name="Tools",
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="tags", data_type=DataType.TEXT_ARRAY),
        Property(name="usage_examples", data_type=DataType.TEXT),
        Property(name="documentation_url", data_type=DataType.TEXT),
    ],
    # Optional: Set default reranker for this collection
    reranker_config=Configure.reranker(
        model_name="reranker-transformers",
        model_version="1.0.0",
        parameters={"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    ),
    # Existing vectorizer configuration (unchanged)
    vectorizer_config=Configure.Vectorizer.text2vec_ollama(
        model="nomic-embed-text"
    )
)
```

#### TypeScript Configuration
```typescript
import weaviate from 'weaviate';

const client = new weaviate.Client({
    scheme: 'http',
    host: 'localhost:8080',
});

await client.collections.create({
    name: 'Tools',
    properties: [
        { name: 'name', dataType: 'text' },
        { name: 'description', dataType: 'text' },
        { name: 'category', dataType: 'text' },
        { name: 'tags', dataType: 'text[]' },
        { name: 'usage_examples', dataType: 'text' },
        { name: 'documentation_url', dataType: 'text' },
    ],
    rerankerConfig: {
        modelName: 'reranker-transformers',
        modelVersion: '1.0.0',
        parameters: {
            model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        },
    },
    vectorizerConfig: {
        text2vecOllama: {
            model: 'nomic-embed-text',
        },
    },
});
```

### 3. Query Implementation Patterns

#### Python Client Examples
```python
import weaviate
from weaviate.classes.query import Query

# CORRECTED: Python v4 client syntax for reranking
def search_tools_hybrid_rerank(query: str, k: int = 10) -> dict:
    """
    Two-stage search: hybrid retrieval + reranking
    """
    response = (
        client.collections.get("Tools")
        .query.hybrid(
            query=query,
            alpha=0.7,  # Favor vector search slightly
            limit=50    # Initial candidate pool
        )
        .with_rerank(
            query=query,
            k=k  # Rerank top-k results
        )
        .objects
    )
    return response

# Pattern 2: Vector search with reranking
def search_tools_vector_rerank(query: str, k: int = 10) -> dict:
    """
    Two-stage search: vector retrieval + reranking
    """
    response = (
        client.collections.get("Tools")
        .query.near_text(
            query=query,
            limit=50
        )
        .with_rerank(
            query=query,
            k=k
        )
        .objects
    )
    return response

# Pattern 3: Feature flag controlled reranking
def search_tools_with_flag(query: str, enable_rerank: bool = False, k: int = 10) -> dict:
    """
    Backward compatible search with optional reranking
    """
    collection = client.collections.get("Tools")

    if enable_rerank:
        response = (
            collection.query.hybrid(query=query, limit=50)
            .with_rerank(query=query, k=k)
            .objects
        )
    else:
        response = collection.query.hybrid(query=query, limit=50).objects

    return response
```

#### TypeScript Client Examples
```typescript
import weaviate from 'weaviate';

// Pattern 1: Hybrid search with reranking
async function searchToolsHybridRerank(query: string, k: number = 10): Promise<any> {
    const response = await client.query
        .get('Tools', ['name', 'description', 'category', 'tags'])
        .withHybrid({
            query: query,
            alpha: 0.7,
            properties: ['name^2', 'description', 'tags']
        })
        .withRerank({
            query: query,
            k: k
        })
        .withLimit(50)
        .do();

    return response;
}

// Pattern 2: Feature flag controlled
async function searchToolsWithFlag(
    query: string,
    enableRerank: boolean = false,
    k: number = 10
): Promise<any> {
    let queryBuilder = client.query
        .get('Tools', ['name', 'description', 'category'])
        .withHybrid({ query: query })
        .withLimit(50);

    if (enableRerank) {
        queryBuilder = queryBuilder.withRerank({ query: query, k: k });
    }

    return await queryBuilder.do();
}
```

#### GraphQL Query Examples
```graphql
# CORRECTED: Basic reranked search
query SearchToolsWithRerank($query: String!, $k: Int!) {
  Get {
    Tools(
      hybrid: {
        query: $query
        alpha: 0.7
      }
      limit: 50
      rerank: {
        query: $query
        k: $k
      }
    ) {
      name
      description
      category
      _additional {
        score
        rerank {
          score
        }
      }
    }
  }
}

# Vector search with reranking
query SearchToolsVectorRerank($query: String!, $k: Int!) {
  Get {
    Tools(
      nearText: {
        concepts: [$query]
      }
      limit: 50
      rerank: {
        query: $query
        k: $k
      }
    ) {
      name
      description
      _additional {
        distance
        rerank {
          score
        }
      }
    }
  }
}
```

### 4. Ollama Adapter Service Implementation

#### Service Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Weaviate     │───▶│  Ollama Adapter  │───▶│     Ollama      │
│                 │    │                  │    │ (Qwen3-Reranker)│
│ reranker module │◀───│  HTTP Service    │◀───│                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### API Contract (Weaviate → Adapter)
**CORRECTED: Based on Weaviate reranker-transformers module specification**

```json
{
  "query": "best tool for parsing CSV files",
  "documents": [
    "CSVParser - A powerful tool for parsing CSV files with advanced filtering",
    "DataProcessor - General purpose data processing tool",
    "TextAnalyzer - Advanced text processing and analysis tool"
  ]
}
```

#### Expected Response (Adapter → Weaviate)
```json
{
  "scores": [0.95, 0.23, 0.15]
}
```

**Note**: The adapter must implement the exact API contract that Weaviate's reranker-transformers module expects. The request contains a query string and an array of document strings. The response is an array of scores in the same order as the input documents.

#### Adapter Implementation (Python/FastAPI)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx
import asyncio
import logging
from contextlib import asynccontextmanager

# Configuration
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "qwen/qwen3-reranker-4b"  # CORRECTED: Proper Ollama model name
BATCH_SIZE = 10
TIMEOUT_SECONDS = 30

# CORRECTED: Pydantic models matching Weaviate's API contract
class RerankRequest(BaseModel):
    query: str
    documents: List[str]  # Array of document strings, not objects

class RerankResponse(BaseModel):
    scores: List[float]  # Array of scores in same order as documents

# Instruction template for Qwen3 (instruction-aware)
RERANK_INSTRUCTION = """Given a search query and a document, determine how relevant the document is to the query.
Output only a single number between 0.0 and 1.0, where:
- 0.0 means completely irrelevant
- 1.0 means perfectly relevant
- Consider semantic meaning, not just keyword matching

Query: {query}
Document: {document}

Relevance score:"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warm up the model
    await warmup_model()
    yield
    # Shutdown: cleanup if needed

app = FastAPI(
    title="Ollama Reranker Adapter",
    description="Adapter service for Weaviate reranker integration with Ollama",
    version="1.0.0",
    lifespan=lifespan
)

async def warmup_model():
    """Warm up the Ollama model to reduce first-request latency"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            warmup_prompt = RERANK_INSTRUCTION.format(
                query="test query",
                document="test document"
            )

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "num_predict": 10
                    }
                }
            )

            if response.status_code == 200:
                logging.info("Model warmed up successfully")
            else:
                logging.warning(f"Model warmup failed: {response.status_code}")

    except Exception as e:
        logging.error(f"Model warmup error: {e}")

async def score_document_pair(query: str, document: str) -> float:
    """Score a single query-document pair using Ollama"""
    try:
        prompt = RERANK_INSTRUCTION.format(query=query, document=document)

        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic scoring
                        "top_p": 1.0,
                        "num_predict": 10,   # Short response expected
                        "stop": ["\n", " "]  # Stop at first number
                    }
                }
            )

            if response.status_code != 200:
                logging.error(f"Ollama request failed: {response.status_code}")
                return 0.0

            result = response.json()
            score_text = result.get("response", "0.0").strip()

            # Parse and validate score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp to [0,1]
            except ValueError:
                logging.warning(f"Invalid score format: {score_text}")
                return 0.0

    except Exception as e:
        logging.error(f"Scoring error: {e}")
        return 0.0

async def batch_score_documents(query: str, documents: List[str]) -> List[float]:
    """Score multiple documents in batches"""
    scores = []

    # Process in batches to avoid overwhelming Ollama
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]

        # Score batch concurrently
        tasks = [
            score_document_pair(query, document)
            for document in batch
        ]

        batch_scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results and exceptions
        for document, score in zip(batch, batch_scores):
            if isinstance(score, Exception):
                logging.error(f"Scoring failed for document: {score}")
                scores.append(0.0)
            else:
                scores.append(score)

    return scores

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Main reranking endpoint - CORRECTED to match Weaviate API contract"""
    try:
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")

        # Score all documents (return scores in same order as input)
        scores = await batch_score_documents(request.query, request.documents)

        return RerankResponse(scores=scores)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Quick test of Ollama connectivity
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")

            if response.status_code == 200:
                return {"status": "healthy", "ollama": "connected"}
            else:
                return {"status": "degraded", "ollama": "disconnected"}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring"""
    # In production, integrate with Prometheus/monitoring system
    return {
        "model": OLLAMA_MODEL,
        "batch_size": BATCH_SIZE,
        "timeout": TIMEOUT_SECONDS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

#### Dockerfile for Adapter
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
pydantic==2.5.0
```

### 5. Performance Tuning and Monitoring

#### Key Performance Metrics
```python
# Metrics to track
METRICS = {
    "retrieval_latency_ms": "Time for initial Weaviate search",
    "rerank_latency_ms": "Time for reranking operation",
    "total_latency_ms": "End-to-end query time",
    "rerank_hit_rate": "Percentage of queries using rerank",
    "accuracy_improvement": "MRR@k improvement vs baseline",
    "error_rate": "Percentage of failed rerank requests",
    "fallback_rate": "Percentage using fallback (no rerank)"
}
```

#### Tuning Parameters
```yaml
# Performance tuning configuration
performance:
  # Retrieval stage
  initial_limit: 50          # Candidate pool size
  hybrid_alpha: 0.7          # Vector vs keyword balance

  # Reranking stage
  rerank_k: 10               # Number of results to rerank
  batch_size: 10             # Ollama batch processing
  timeout_seconds: 30        # Request timeout

  # Caching
  enable_cache: true         # Cache rerank results
  cache_ttl_seconds: 300     # Cache expiration

  # Feature flags
  enable_rerank: true        # Global rerank toggle
  rerank_percentage: 100     # Gradual rollout percentage
```

#### Monitoring Dashboard Queries
```promql
# Prometheus queries for monitoring

# Average rerank latency
histogram_quantile(0.95, rate(rerank_duration_seconds_bucket[5m]))

# Rerank success rate
rate(rerank_requests_total{status="success"}[5m]) / rate(rerank_requests_total[5m])

# Accuracy improvement
increase(accuracy_improvement_total[1h]) / increase(queries_total[1h])

# Error rate
rate(rerank_requests_total{status="error"}[5m])
```

### 6. Rollout Strategy (Aligned with Huly Issues)

#### Phase 1: LMP-146 - Prototype and Baseline
**Objective**: Establish baseline with native transformers reranker

**Tasks**:
- [ ] Deploy Pattern A (native reranker-transformers)
- [ ] Implement basic withRerank() queries
- [ ] Measure baseline accuracy and latency
- [ ] Create monitoring dashboard

**Success Criteria**:
- Two-stage pipeline functional
- Baseline metrics established
- No regression in existing functionality

**Timeline**: 1 week

#### Phase 2: LMP-147 - Schema and Compose Updates
**Objective**: Prepare infrastructure for Ollama adapter

**Tasks**:
- [ ] Update docker-compose.yml with adapter service
- [ ] Add reranker configuration to Tools collection
- [ ] Implement feature flags for gradual rollout
- [ ] Add health checks and monitoring

**Success Criteria**:
- Infrastructure ready for adapter deployment
- Feature flags operational
- Backward compatibility maintained

**Timeline**: 1 week

#### Phase 3: LMP-148 - Ollama Adapter Implementation
**Objective**: Deploy and integrate Ollama adapter with Qwen3

**Tasks**:
- [ ] Build and deploy reranker-ollama-adapter service
- [ ] Pull and configure Qwen3-Reranker-4B in Ollama: `ollama pull qwen/qwen3-reranker-4b`
- [ ] Implement instruction-aware prompting
- [ ] Add error handling and fallback logic
- [ ] Verify API contract compatibility with Weaviate reranker-transformers module

**Success Criteria**:
- Adapter service operational
- Qwen3 reranker functional
- Error handling robust

**Timeline**: 2 weeks

#### Phase 4: LMP-149 - Performance Optimization
**Objective**: Tune performance parameters for production

**Tasks**:
- [ ] Optimize k, batch_size, and timeout parameters
- [ ] Implement caching layer
- [ ] Add GPU acceleration if available
- [ ] Load testing and capacity planning

**Success Criteria**:
- Latency within acceptable bounds (<500ms p95)
- Throughput meets requirements
- Resource utilization optimized

**Timeline**: 1 week

#### Phase 5: LMP-150 - Monitoring and Observability
**Objective**: Comprehensive monitoring and alerting

**Tasks**:
- [ ] Deploy Prometheus metrics collection
- [ ] Create Grafana dashboards
- [ ] Set up alerting for errors and performance
- [ ] Implement A/B testing framework

**Success Criteria**:
- Full observability stack operational
- Alerts configured and tested
- A/B testing ready

**Timeline**: 1 week

#### Phase 6: LMP-151 - Gradual Rollout and Validation
**Objective**: Safe production rollout with validation

**Tasks**:
- [ ] Start with 10% traffic using reranking
- [ ] Monitor accuracy improvements and stability
- [ ] Gradually increase to 50%, then 100%
- [ ] Validate 1-5% accuracy improvement target

**Success Criteria**:
- 1-5% accuracy improvement achieved
- No stability issues
- Full production deployment

**Timeline**: 2 weeks

### 7. Risk Mitigation

#### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Ollama adapter latency | High | Medium | Implement caching, batching, timeouts |
| Qwen3 model compatibility | High | Low | Thorough testing, fallback to transformers |
| Weaviate API changes | Medium | Low | Pin versions, test upgrades |
| Resource exhaustion | Medium | Medium | Resource limits, monitoring, auto-scaling |

#### Operational Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Service dependencies | High | Medium | Health checks, circuit breakers, fallbacks |
| Data quality issues | Medium | Medium | Input validation, error logging |
| Rollback complexity | Medium | Low | Feature flags, blue-green deployment |
| Monitoring gaps | Low | Medium | Comprehensive metrics, alerting |

#### Fallback Strategies
```python
# Fallback logic in query implementation
def search_tools_with_fallback(query: str, enable_rerank: bool = True):
    try:
        if enable_rerank:
            # Attempt reranked search
            return search_tools_hybrid_rerank(query)
    except Exception as e:
        logging.warning(f"Rerank failed, falling back: {e}")

    # Fallback to standard search
    return search_tools_standard(query)
```

### 8. Success Metrics and Validation

#### Accuracy Metrics
- **MRR@10** (Mean Reciprocal Rank): Target 1-5% improvement
- **NDCG@10** (Normalized Discounted Cumulative Gain): Secondary metric
- **Click-through Rate**: User engagement with top results
- **Task Completion Rate**: Successful tool selection

#### Performance Metrics
- **Latency P95**: <500ms end-to-end
- **Throughput**: Handle existing query volume
- **Error Rate**: <1% rerank failures
- **Availability**: >99.9% uptime

#### Operational Metrics
- **Resource Utilization**: CPU, memory, disk within limits
- **Cost Impact**: Minimal infrastructure cost increase
- **Deployment Success**: Zero-downtime deployments
- **Rollback Time**: <5 minutes if needed

### 9. Future Enhancements

#### Short-term (3-6 months)
- [ ] Multi-model reranker ensemble
- [ ] Personalized reranking based on user context
- [ ] Advanced caching strategies
- [ ] GPU acceleration optimization

#### Long-term (6-12 months)
- [ ] Custom fine-tuned reranker for tool domain
- [ ] Real-time learning from user feedback
- [ ] Cross-lingual tool search support
- [ ] Integration with tool usage analytics

### 10. Conclusion

This implementation plan provides a comprehensive approach to integrating Ollama-hosted Qwen3-Reranker-4B with Weaviate while maintaining backward compatibility and enabling gradual rollout. The two-stage retrieval system should achieve the target 1-5% accuracy improvement while preserving system stability and performance.

The modular design allows for easy rollback and provides multiple fallback mechanisms to ensure reliability. The detailed monitoring and observability framework will enable data-driven optimization and validation of the improvements.

**Next Steps**:
1. **CRITICAL**: Verify Weaviate reranker-transformers API contract through testing
2. Review and approve corrected implementation plan
3. Set up development environment with proper Ollama model: `ollama pull qwen/qwen3-reranker-4b`
4. Begin Phase 1 (LMP-146) implementation with native reranker
5. Establish baseline metrics and monitoring

## Implementation Corrections Made:

1. **Fixed Python Client API**: Updated to use correct Weaviate v4 client syntax
2. **Corrected API Contract**: Simplified to match reranker-transformers module expectations
3. **Fixed Ollama Model Name**: Changed to proper format `qwen/qwen3-reranker-4b`
4. **Updated GraphQL Syntax**: Corrected reranking query structure
5. **Simplified Adapter Logic**: Removed unnecessary complexity in request/response handling
