# Search System Testing Dashboard - Product Requirements Document

## Executive Summary

The Search System Testing Dashboard is a comprehensive web-based interface designed to enable thorough testing, evaluation, and configuration of all search system components without affecting production environments. This dashboard will provide a safe, isolated environment for evaluating search performance, configuring embedding models, reranker models, Weaviate hyperparameters, and all other search system parameters while managing data re-embedding processes.

## Problem Statement

### Current Challenges
- No comprehensive testing interface for the entire search system stack
- Limited ability to evaluate search performance across all configurable parameters
- Difficulty in comparing different embedding models, reranker configurations, and Weaviate settings
- Manual processes for testing search queries and evaluating hyperparameter impacts
- Risk of affecting production systems during testing and configuration
- No systematic way to test Weaviate-specific parameters (alpha, fusion types, query properties, etc.)
- Lack of visibility into how embedding model changes affect search quality

### Goals
- **Primary**: Create a safe, comprehensive testing environment for all search system components
- **Secondary**: Enable data-driven decision making for embedding models, rerankers, and Weaviate configurations
- **Tertiary**: Establish best practices for holistic search system evaluation and optimization

## User Personas & Use Cases

### Primary Users
1. **ML Engineers**: Testing and optimizing reranker configurations
2. **Data Scientists**: Evaluating search performance and model effectiveness
3. **System Administrators**: Managing embedding models and data re-processing
4. **Product Managers**: Understanding search quality and performance metrics

### Key Use Cases
1. **Holistic Search Quality Evaluation**: Test queries across all system parameters and manually evaluate result relevance
2. **Multi-Component Model Comparison**: A/B test different embedding models, reranker models, and Weaviate configurations
3. **Comprehensive Parameter Optimization**: Tune embedding dimensions, Weaviate alpha values, fusion types, query properties, reranker settings, and all other exposed parameters
4. **Embedding Model Testing**: Compare OpenAI vs Ollama vs custom embedding models with different dimensions and configurations
5. **Weaviate Configuration Testing**: Test hybrid search parameters, vector/keyword balance, query property weights, and fusion algorithms
6. **Data Management**: Monitor embedding status across different models and trigger re-processing with new configurations
7. **Performance Benchmarking**: Establish baseline metrics across all parameter combinations and track improvements

## Functional Requirements

### Core Testing Features
- **Real-time Search Interface**: Enter queries and view results instantly with any parameter combination
- **Multi-Component Comparison**: Side-by-side comparison across embedding models, Weaviate settings, and reranker configurations
- **Parameter Impact Analysis**: Visualize how individual parameter changes affect search results
- **Manual Evaluation Tools**: Rate result relevance and save evaluations with parameter context
- **Query Management**: Save, organize, and replay test queries with specific parameter sets
- **Export Capabilities**: Export test results with full parameter configurations for external analysis

### Comprehensive Configuration Management
- **Embedding Model Selection**: Switch between OpenAI (text-embedding-3-small/large, ada-002), Ollama models, and custom providers
- **Embedding Dimension Testing**: Test different embedding dimensions and their impact on search quality
- **Weaviate Hyperparameter Control**:
  - Alpha values (vector vs keyword search balance)
  - Fusion types (RELATIVE_SCORE, RANKED)
  - Query properties and their weights (name^2, description^1.5, tags)
  - Search limits and candidate pool sizes
  - Distance metrics and similarity thresholds
- **Reranker Model Configuration**: Configure cross-encoder, bi-encoder, LambdaMART, and custom reranker models
- **Advanced Search Parameters**:
  - Hybrid search configurations
  - Vector search parameters
  - Keyword search parameters
  - BM25 parameters
  - Custom scoring functions
- **Configuration Presets**: Save and load parameter combinations for different use cases
- **YAML Configuration Editor**: Direct editing of comprehensive configuration files
- **Parameter Validation**: Real-time validation of parameter combinations and constraints

### Data Management and Embedding Operations

#### Tool Inventory and Metadata
- **Tool Browser**: Searchable, filterable view of all tools with metadata
- **Source Tracking**: MCP server origin, registration timestamp, update history
- **Usage Analytics**: Query frequency, attachment patterns, performance metrics
- **Dependency Mapping**: Tool relationships and co-occurrence patterns

#### Comprehensive Embedding Versioning
- **Version Metadata Schema**:
  ```yaml
  embedding_version:
    provider: "openai"              # openai, ollama, custom
    model: "text-embedding-3-small" # specific model identifier
    dimension: 1536                 # vector dimension
    pooling: "mean"                 # mean, cls, max, weighted
    normalize_l2: true              # L2 normalization applied
    max_tokens: 8192                # truncation limit
    precision: "fp32"               # fp32, fp16, int8
    created_at: "2024-03-15T10:30:00Z"
    schema_version: "v2.1"          # configuration schema version
  ```
- **Per-Tool Vector Status**: 
  - Current embedding version per tool
  - Migration status (pending, in_progress, completed, failed)
  - Quality metrics (norm, similarity to previous version)
  - Validation flags (dimension mismatch, corrupted vector)

#### Dual-Index Strategy for Safe Migration
- **Parallel Index Management**:
  - **Current Index**: production embeddings serving live queries
  - **Candidate Index**: new embedding version being built/tested
  - **Shadow Testing**: route percentage of queries to candidate for comparison
- **Migration Workflow**:
  1. **Build Phase**: Create candidate index with new embedding configuration
  2. **Validation Phase**: Run quality checks and performance benchmarks
  3. **Testing Phase**: A/B test candidate vs current with statistical monitoring
  4. **Cutover Phase**: Atomic swap from current to candidate index
  5. **Rollback Capability**: Instant revert to previous index if issues detected
- **Safety Mechanisms**:
  - Automated quality gates (minimum similarity thresholds, performance SLAs)
  - Manual approval required for cutover
  - Monitoring alerts for embedding drift or performance degradation

#### Re-embedding Operations
- **Batch Processing**:
  - Resumable batches with checkpoint/restart capability
  - Configurable batch sizes (10-1000 tools) based on provider limits
  - Progress tracking with ETA and throughput metrics
  - Failed item retry with exponential backoff
- **Incremental Updates**:
  - Delta detection: identify tools with content/metadata changes
  - Selective re-embedding of modified tools only
  - Version-aware updates preserving unchanged embeddings
- **Quality Assurance**:
  - Pre-flight validation of new embedding configurations
  - Post-embedding quality checks (dimension, norm distribution)
  - Similarity drift detection vs previous versions

#### Advanced Analytics and Quality Monitoring
- **Vector Quality Metrics**:
  - **Norm Distribution Histograms**: identify outliers and scaling issues
  - **Dimensionality Analysis**: effective dimension usage, sparsity patterns
  - **Cluster Analysis**: detect duplicates, identify embedding quality issues
  - **Similarity Drift Monitoring**: track embedding stability across versions
- **Content Quality Assessment**:
  - **OOV Detection**: out-of-vocabulary terms affecting embedding quality
  - **Short Content Alerts**: tools with insufficient text for quality embeddings
  - **Language Detection**: multilingual content requiring specialized models
  - **PII Detection**: identify potential privacy concerns in tool content
- **Performance Monitoring**:
  - **Embedding Latency**: track provider response times and batching efficiency
  - **Storage Utilization**: vector database size and memory usage
  - **Search Performance**: impact of embedding changes on query latency
  - **Cost Tracking**: embedding generation costs per provider and model

#### Schema and Migration Management  
- **Schema Evolution**:
  - Version-controlled embedding schemas with backward compatibility
  - Migration scripts for schema upgrades
  - Rollback procedures for failed migrations
- **Mixed-Version Detection**:
  - Automatic detection of tools with mismatched embedding versions
  - Warnings for queries spanning multiple embedding versions
  - Cleanup procedures for orphaned or outdated embeddings
- **Data Integrity**:
  - Vector-metadata consistency validation
  - Corruption detection and repair procedures
  - Backup and restore capabilities for embedding data

### Evaluation Framework
- **Relevance Scoring Interface**: Manual scoring with Context7 best practices
- **Batch Evaluation Tools**: Process multiple queries automatically
- **A/B Testing Framework**: Compare model configurations systematically
- **Performance Metrics**: Track precision, recall, NDCG, and custom metrics
- **Benchmark Query Sets**: Predefined queries for consistent evaluation


## Hyperparameter Coverage Matrix

### Embedder Parameters
- **Provider**: OpenAI, Ollama, custom HTTP endpoints
- **Model**: text-embedding-3-small/large, ada-002, Qwen3-Embedding-4B, custom variants
- **Dimension**: 384, 768, 1024, 1536, 2560, 4096, custom dimensions
- **Pooling Strategy**: CLS token, mean pooling, max pooling, weighted average
- **Normalization**: L2 normalization on/off, unit vector scaling
- **Truncation**: max_tokens (512, 2048, 8192), truncation strategy (head/tail/balanced)
- **Precision/Quantization**: fp32, fp16, int8/int4 (if supported by provider)
- **Batching**: batch_size (1-64), concurrency limits, request queuing
- **Rate Limiting**: requests_per_minute, tokens_per_minute, backoff strategy
- **Retry Logic**: max_retries, exponential_backoff, timeout_seconds

### Weaviate Retrieval Parameters
- **Vector Search**:
  - distance_metric: cosine, dot product, l2/euclidean
  - limit: result count (1-1000)
  - certainty/score thresholds: minimum similarity scores
  - target_vector_length: validation for embedding dimensions
  - schema_vectorizer_overrides: per-class embedding settings
- **Hybrid Search**:
  - alpha: vector vs keyword balance (0.0-1.0, typical 0.25-0.95)
  - fusion_type: RELATIVE_SCORE, RANKED (RRF if available)
  - candidate_pool_size: initial retrieval before fusion (20-200)
  - query_properties: per-field search with weights
    - name^2.0: exact name matches heavily weighted
    - description^1.5: description text moderately weighted  
    - tags^1.0: tag matches baseline weight
    - enhanced_description^1.8: enriched descriptions
- **Keyword/BM25 Search**:
  - k1: term frequency saturation (0.5-3.0, typical 1.2)
  - b: field length normalization (0.0-1.0, typical 0.75)
  - stopwords: language-specific stop word lists
  - min_char_length: minimum term length (1-4)
  - per_field_boosts: individual field importance multipliers
- **Operational Settings**:
  - shard_count/replication_factor: if applicable for staging
  - timeout_seconds: query timeout limits
  - retry_attempts: failed query retry logic
  - consistency_level: eventual/strong consistency (if used)
  - enabled_modules: text2vec-openai, text2vec-ollama, reranker-transformers

### Reranker Parameters  
- **Model Configuration**:
  - model_type: cross-encoder, bi-encoder, LambdaMART, neural_ranking
  - model_id: specific model identifier/version
  - baseline_modes: noop (passthrough), shuffle, random for A/B testing
- **Scoring Parameters**:
  - top_k: number of results to rerank (1-100)
  - score_threshold: minimum relevance threshold (0.0-1.0)
  - score_scaling: linear, sigmoid, or custom scaling functions
- **Processing Parameters**:
  - batch_size: documents per inference batch (1-64)
  - max_document_length: character/token limits (512-4096)
  - truncation_policy: head, tail, balanced, or smart truncation
  - query_expansion: enable/disable query augmentation
- **Performance Parameters**:
  - latency_budget_ms: maximum processing time (100-2000ms)  
  - fallback_mode: fail_open (return original), fail_closed (return empty)
  - cache_ttl: result caching duration (0-3600 seconds)
  - parallel_processing: enable concurrent document processing

## Experiments and Grid Search Framework

### Experiment Design
- **Experiment Types**:
  - Grid Search: systematic exploration of parameter combinations
  - Random Search: stochastic sampling for large parameter spaces
  - Manual Configuration: specific parameter sets for targeted testing
  - Bayesian Optimization: intelligent parameter space exploration
- **Parameter Sweep Strategies**:
  - Full factorial: test all parameter combinations (small spaces)
  - Fractional factorial: statistically designed subsets (large spaces)  
  - Latin hypercube sampling: space-filling designs for continuous parameters
  - Multi-armed bandit: adaptive exploration with early stopping

### Benchmark Query Management
- **Query Set Categories**:
  - Tool discovery: "find tools for X", "I need to Y"
  - Domain-specific: scheduling, social posting, data analysis, content creation
  - Difficulty levels: simple (exact matches), moderate (semantic), complex (multi-intent)
- **Ground Truth Management**:
  - Expert annotations with relevance scores (0.0-1.0 scale)
  - Inter-annotator agreement tracking (Cohen's kappa)
  - Relevance guidelines per domain and query type
- **Dataset Versioning**:
  - Query set hashing for reproducibility
  - Annotation version tracking
  - Benchmark evolution and backward compatibility

### Statistical Analysis Framework
- **Evaluation Metrics**:
  - Precision@K (K=1,3,5,10): percentage of relevant results in top-K
  - NDCG@K: discounted cumulative gain with position weighting
  - MRR (Mean Reciprocal Rank): average of 1/rank of first relevant result
  - MAP (Mean Average Precision): mean of precision at each relevant result
  - Custom metrics: domain-specific relevance measures
- **Significance Testing**:
  - Paired t-tests for configuration comparisons
  - Bootstrap confidence intervals (95%, 99%) for metric estimates
  - Multiple comparison correction (Bonferroni, FDR)
  - Effect size reporting (Cohen's d, practical significance)
- **Experimental Design**:
  - Randomization of query order and configuration testing
  - Cross-validation for robust metric estimation  
  - Power analysis for sample size determination
  - A/B/n testing with statistical stopping rules

### Cost and Resource Management
- **Budget Controls**:
  - Daily API token limits per provider (OpenAI, Ollama)
  - Per-experiment cost caps with automatic termination
  - Compute budget tracking for reranker inference
  - Cost-per-query estimation and forecasting
- **Rate Limiting and Backoff**:
  - Respectful API usage with provider-specific limits
  - Exponential backoff for rate limit violations  
  - Queue management for high-volume experiments
  - Parallel experiment execution with resource allocation
- **Resource Optimization**:
  - Caching of expensive operations (embeddings, reranking)
  - Batch processing for efficiency gains
  - Early stopping for clearly inferior configurations
  - Resource usage monitoring and alerting

### Experiment Artifact Management
- **Configuration Tracking**:
  - Complete parameter snapshots with git-style hashing
  - Configuration diff visualization between experiments
  - Parameter importance analysis via sensitivity testing
  - Configuration reproducibility verification
- **Result Storage**:
  - Full result matrices with metric distributions
  - Statistical test results and confidence intervals
  - Performance timeline tracking (latency, throughput)
  - Error analysis and failure mode documentation
- **Visualization and Reporting**:
  - Parameter vs performance scatter plots
  - Heatmaps for grid search results
  - Statistical significance matrices
  - Automated experiment summary reports
## Technical Architecture

### System Integration
- **Existing Infrastructure**: Leverage current Weaviate, API server, and embedding providers
- **Read-Only Mode**: No live tool attachments or production system modifications
- **API Integration**: Connect to existing search endpoints and embedding services
- **Configuration System**: Extend current embedding provider configuration

### Technology Stack
- **Backend**: FastAPI with async support for ML workloads
- **Frontend**: React with Material-UI for responsive interface
- **Database**: SQLite for test results, configurations, and evaluation data
- **Configuration**: YAML files following Metarank patterns
- **Visualization**: Chart.js for metrics and performance dashboards

### Architecture Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI       │    │   Weaviate      │
│   Dashboard     │◄──►│   Backend       │◄──►│   Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Reranker      │
                       │   Services      │
                       └─────────────────┘
```

### Safety Measures
- **Isolated Environment**: Separate from production tool attachment systems
- **Read-Only Operations**: No modifications to agent configurations
- **Configuration Validation**: Prevent invalid or dangerous configurations
- **Audit Logging**: Track all configuration changes and test activities

## User Interface Design

### Main Dashboard
- **Search Interface**: Prominent search bar with real-time results
- **Configuration Panel**: Model selection and parameter controls
- **Results Comparison**: Split view for original vs reranked results
- **Evaluation Tools**: Rating interface and evaluation history

### Configuration Management
- **Model Selection**: Dropdown menus for embedding and reranker models
- **Parameter Sliders**: Visual controls for alpha, thresholds, and limits
- **Configuration Editor**: YAML editor with syntax highlighting
- **Preset Manager**: Save, load, and share configuration presets

### Data Management
- **Tool Browser**: Searchable table of all tools with metadata
- **Embedding Status**: Visual indicators for embedding coverage
- **Re-embedding Queue**: Progress tracking for batch operations
- **Analytics Dashboard**: Charts and metrics for data quality

### Evaluation Interface
- **Query Builder**: Create and organize test query sets
- **Relevance Rating**: Star ratings or numerical scores for results
- **Comparison View**: Side-by-side model performance comparison
- **Report Generator**: Export evaluation results and metrics

## Context7 Best Practices Integration

### Metarank Configuration Patterns
- **YAML Structure**: Follow Metarank configuration file patterns
- **Model Definitions**: Support for LambdaMART, cross-encoder, and bi-encoder models
- **Feature Extractors**: Configure relevancy, position, and diversity features
- **Evaluation Metrics**: Implement standard ranking evaluation metrics

### LangSearch Semantic Reranking
- **Score Interpretation**: Implement 0-1 scoring scale with defined ranges
  - 0.75-1.0: Highly relevant (fully answers question)
  - 0.5-0.75: Relevant (lacks completeness)
  - 0.2-0.5: Somewhat relevant (partially answers)
  - 0.1-0.2: Related (answers small part)
  - 0.0-0.1: Not significantly relevant
- **API Integration**: Support for LangSearch rerank API patterns
- **Performance Optimization**: Efficient batch processing and caching

### Evaluation Standards
- **Relevance Guidelines**: Clear criteria for manual evaluation
- **Benchmark Queries**: Standard test sets for consistent evaluation
- **Metric Definitions**: Precision@K, NDCG, MRR, and custom metrics
- **A/B Testing**: Statistical significance testing for model comparisons

## Implementation Plan

### Phase 1: MVP (4-6 weeks)
- **Core Search Interface**: Basic query input and result display
- **Simple Reranking**: Integration with one reranker model
- **Manual Evaluation**: Basic relevance rating interface
- **Configuration Management**: Essential model selection
- **Safety Measures**: Read-only mode implementation

### Phase 2: Advanced Features (6-8 weeks)
- **Multiple Reranker Models**: Support for various model types
- **A/B Testing Framework**: Systematic model comparison
- **Batch Evaluation**: Automated testing of query sets
- **Advanced Configuration**: Full parameter control and presets
- **Data Management**: Embedding status and re-processing tools

### Phase 3: Production Integration (4-6 weeks)
- **Performance Analytics**: Comprehensive metrics and reporting
- **Integration Testing**: Validation with existing workflows
- **Documentation**: User guides and best practices
- **Optimization**: Performance tuning and scalability improvements

## Success Metrics

### Primary Metrics
- **Testing Coverage**: Number of queries tested and evaluated across all parameter combinations
- **Configuration Accuracy**: Successful model configurations without errors or invalid parameter combinations  
- **Evaluation Quality**: Inter-annotator agreement (Cohen's kappa > 0.7) and evaluation consistency
- **Safety Record**: Zero production system impacts, 100% read-only mode compliance

### Performance and Latency Targets
- **Search Performance**:
  - Embedding generation: P95 < 500ms per batch
  - Weaviate retrieval: P95 < 200ms per query
  - Reranking: P95 < 1000ms for top-20 results
  - End-to-end query: P95 < 2000ms total
- **System Reliability**:
  - Dashboard uptime: 99.9% availability
  - Experiment success rate: >95% completion without errors
  - Configuration validation: 100% accuracy preventing invalid configs

### Search Quality and Confidence Metrics  
- **Benchmark Performance**:
  - Coverage across domains: tool discovery (>90% queries), social posting (>85% queries), data analysis (>80% queries)
  - Difficulty level coverage: simple (>95% success), moderate (>85% success), complex (>70% success)
- **Configuration Confidence**:
  - Statistical significance: >80% of experiments show configs that beat baseline with p<0.05
  - Effect size validation: >60% of improvements show practically significant effect (Cohen's d > 0.2)
  - Reproducibility: >90% of experiment results reproducible within 5% variance

### User Adoption and Efficiency
- **Active Usage**: 
  - Daily active users: >10 users across ML engineers, data scientists, system admins
  - Session frequency: >3 sessions per user per week  
  - Query volume: >100 test queries per day across all users
- **Time and Cost Efficiency**:
  - Configuration testing time: 75% reduction vs manual testing
  - Experiment iteration cycles: <4 hours from hypothesis to results
  - Cost optimization: API costs per query within 90% of baseline after optimization
- **Decision Support**: 
  - Data-driven configuration changes: >80% of production changes informed by dashboard insights
  - Parameter optimization success: >3 significant improvements per month in key metrics

### Business Impact Metrics
- **Search Quality Improvements**:
  - Precision@5 improvement: >10% relative improvement over baseline  
  - NDCG@10 improvement: >8% relative improvement over baseline
  - User satisfaction: >85% positive feedback on search result quality
- **Risk Reduction**:
  - Production incident prevention: 0 search-related outages due to configuration changes
  - Configuration validation: 100% prevention of invalid production configurations
  - Rollback capability: <5 minute recovery time for embedding migration issues

## Risk Assessment

### Technical Risks
- **Integration Complexity**: Challenges integrating with existing systems
- **Performance Impact**: Dashboard affecting search system performance
- **Configuration Errors**: Invalid configurations causing system issues
- **Data Consistency**: Synchronization issues with live data

### Mitigation Strategies
- **Isolated Environment**: Complete separation from production systems
- **Comprehensive Testing**: Thorough validation before deployment
- **Configuration Validation**: Automated checks for configuration validity
- **Monitoring and Alerting**: Real-time monitoring of system health

## Future Considerations

### Potential Enhancements
- **Multi-User Support**: Role-based access and collaboration features
- **Automated Testing**: CI/CD integration for continuous evaluation
- **Advanced Analytics**: Machine learning for configuration optimization
- **API Extensions**: External integrations and webhook support

### Scalability Planning
- **Distributed Processing**: Support for large-scale batch operations
- **Cloud Integration**: Deployment options for cloud environments
- **Performance Optimization**: Caching and optimization strategies
- **Data Archival**: Long-term storage for evaluation history

## Detailed Technical Specifications

### API Endpoints
```
GET  /api/v1/search/test          - Test search with configuration
POST /api/v1/rerank/compare       - Compare reranking models
GET  /api/v1/models/embedding     - List available embedding models
GET  /api/v1/models/reranker      - List available reranker models
POST /api/v1/config/validate      - Validate configuration
GET  /api/v1/tools/inventory      - Get tool inventory and status
POST /api/v1/embeddings/reprocess - Trigger re-embedding
GET  /api/v1/evaluations/history  - Get evaluation history
POST /api/v1/evaluations/batch    - Run batch evaluation
```

### Configuration Schema
```yaml
# Dashboard Configuration
dashboard:
  mode: testing  # testing, evaluation, production
  safety:
    read_only: true
    prevent_attachments: true
    audit_logging: true
    no_attach_mode_banner: true

# Search Configuration
search:
  embedding:
    provider: openai  # openai, ollama, custom
    model: text-embedding-3-small
    dimension: 1536
    pooling: mean           # mean | cls | max | weighted
    normalize_l2: true
    max_tokens: 8192
    precision: fp32         # fp32 | fp16 | int8 | int4
    batch_size: 32          # 1-64
    rate_limit:
      requests_per_minute: 3000
      tokens_per_minute: 1000000
      backoff_strategy: exponential
    retry:
      max_attempts: 3
      timeout_seconds: 30
    versioning:
      embedding_version: "v2.1"
      schema_version: "2024-03"

  weaviate:
    distance_metric: cosine   # cosine | dot | l2
    vector_search:
      limit: 100
      certainty_threshold: 0.7
      target_vector_validation: true
    bm25:
      k1: 1.2               # 0.5-3.0
      b: 0.75               # 0.0-1.0  
      stopwords: "english"
      min_char_length: 2
      field_boosts:
        name: 2.0
        description: 1.5
        tags: 1.0
    hybrid:
      alpha: 0.75               # vector vs keyword balance (0.0-1.0)
      fusion_type: relative_score  # relative_score | ranked
      candidate_pool_size: 50   # 20-200
      query_properties:
        - "name^2.0"
        - "description^1.5" 
        - "enhanced_description^1.8"
        - "tags^1.0"
    operational:
      timeout_seconds: 30
      retry_attempts: 2
      consistency_level: eventual  # eventual | strong
      enabled_modules:
        - text2vec-openai
        - text2vec-ollama
        - reranker-transformers

# Reranker Configuration  
reranker:
  enabled: true
  type: cross-encoder            # cross-encoder | bi-encoder | lambdamart | noop | shuffle
  model: metarank/ce-msmarco-MiniLM-L6-v2
  scoring:
    top_k: 10                   # 1-100
    threshold: 0.5              # 0.0-1.0
    scaling: linear             # linear | sigmoid | custom
  processing:
    batch_size: 16              # 1-64
    max_doc_length: 2048        # 512-4096
    truncation: head            # head | tail | balanced | smart
    query_expansion: false
  performance:
    latency_budget_ms: 500      # 100-2000
    fallback_mode: fail_open    # fail_open | fail_closed
    cache_ttl: 300              # 0-3600 seconds
    parallel_processing: true

  # Provider-specific configurations
  ollama:
    base_url: "http://192.168.50.80:11434"
    model: "dengcao/Qwen3-Reranker-4B:Q5_K_M"
    timeout: 30
  
  langsearch:
    api_key: ${LANGSEARCH_API_KEY}
    model: langsearch-reranker-v1
    return_documents: true

# Experiment Configuration
experiments:
  cost_controls:
    daily_budget_usd: 50.0
    per_experiment_cap_usd: 10.0
    token_budget_per_provider:
      openai: 100000
      ollama: unlimited
  statistical:
    confidence_level: 0.95      # 0.90, 0.95, 0.99
    min_sample_size: 30
    effect_size_threshold: 0.05  # minimum meaningful improvement
    multiple_comparison_correction: "fdr"  # bonferroni | fdr | none
  grid_search:
    max_combinations: 1000
    early_stopping: true
    parallel_jobs: 4

# Evaluation Configuration
evaluation:
  metrics:
    - precision_at_k: [1, 3, 5, 10]
    - ndcg_at_k: [1, 3, 5, 10] 
    - mrr: true
    - map: true
    - latency_p95: true
    - cost_per_query: true

  relevance_scale:
    highly_relevant: [0.75, 1.0]    # fully answers question
    relevant: [0.5, 0.75]           # lacks completeness  
    somewhat_relevant: [0.2, 0.5]   # partially answers
    related: [0.1, 0.2]             # answers small part
    not_relevant: [0.0, 0.1]        # not significantly relevant

  benchmark_sets:
    - name: "tool_discovery_basic"
      queries: ["find tools for X", "I need to Y"]  
      difficulty: 1
    - name: "domain_specific"
      queries: ["schedule meeting", "post to social media"]
      difficulty: 2
    - name: "multi_intent_complex"
      queries: ["analyze data and create report"]
      difficulty: 3

# UI/UX Configuration
ui:
  presets:
    default:
      description: "Balanced performance and recall"
      embedding: {provider: "openai", model: "text-embedding-3-small"}
      weaviate: {alpha: 0.75}
      reranker: {enabled: true, top_k: 10}
    recall_heavy:
      description: "Maximize recall, accept lower precision"
      embedding: {dimension: 1536}
      weaviate: {alpha: 0.5, candidate_pool_size: 100}
      reranker: {top_k: 20, threshold: 0.2}
    precision_heavy:
      description: "High precision, stricter filtering"
      weaviate: {alpha: 0.9, candidate_pool_size: 20}
      reranker: {top_k: 5, threshold: 0.8}
    low_latency:
      description: "Optimize for speed"
      embedding: {batch_size: 1}
      weaviate: {candidate_pool_size: 20}
      reranker: {enabled: false}

  comparison:
    show_rank_changes: true
    show_score_deltas: true
    highlight_significant_changes: true
    explain_parameter_impact: true

# Operational Safeguards
operations:
  rate_limiting:
    global_queries_per_second: 10
    per_user_queries_per_minute: 100
  safety:
    max_payload_size_mb: 10
    pii_redaction: true
    dangerous_action_confirmation: true
  monitoring:
    latency_sla_ms: 2000
    error_rate_threshold: 0.05
    cost_alert_threshold_usd: 40.0
```

### Database Schema
```sql
-- Test Results
CREATE TABLE test_results (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    results JSON NOT NULL,
    evaluation_scores JSON,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Configurations
CREATE TABLE configurations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config_yaml TEXT NOT NULL,
    hash TEXT UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Evaluations
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY,
    test_result_id INTEGER REFERENCES test_results(id),
    evaluator TEXT NOT NULL,
    relevance_scores JSON NOT NULL,
    notes TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Benchmark Queries
CREATE TABLE benchmark_queries (
    id INTEGER PRIMARY KEY,
    query_set_name TEXT NOT NULL,
    query TEXT NOT NULL,
    expected_results JSON,
    difficulty_level INTEGER DEFAULT 1
);
```

### Component Architecture

#### Backend Components
```python
# Core Services
class SearchService:
    """Handles search operations with different configurations"""

class RerankerService:
    """Manages reranking operations and model switching"""

class ConfigurationManager:
    """Manages configuration validation and switching"""

class EvaluationEngine:
    """Handles evaluation metrics and batch processing"""

class DataManager:
    """Manages tool inventory and embedding operations"""

# API Controllers
class SearchController:
    """REST endpoints for search testing"""

class ConfigurationController:
    """REST endpoints for configuration management"""

class EvaluationController:
    """REST endpoints for evaluation operations"""
```

#### Frontend Components
```javascript
// Main Dashboard Components
const SearchInterface = () => {
  // Real-time search with configuration controls
};

const ResultsComparison = () => {
  // Side-by-side comparison of original vs reranked results
};

const ConfigurationPanel = () => {
  // Model selection and parameter controls


};

const EvaluationInterface = () => {
  // Manual evaluation and rating interface
};

// Management Components
const ModelManager = () => {
  // Embedding and reranker model management
};

const DataDashboard = () => {
  // Tool inventory and embedding status
};

const AnalyticsDashboard = () => {
  // Performance metrics and evaluation history
};
```

### Integration Points

#### Existing System Integration
- **Weaviate Connection**: Reuse existing client initialization and connection management
- **Embedding Providers**: Leverage current multi-provider system (OpenAI, Ollama)
- **API Server**: Extend existing FastAPI server with new testing endpoints
- **Configuration System**: Build on existing embedding configuration patterns

#### New Service Integration
- **Reranker Services**: New microservices for different reranker types
- **Evaluation Database**: SQLite database for test results and evaluations
- **Configuration Validation**: New service for validating complex configurations
- **Batch Processing**: Queue system for large-scale operations

### Security and Safety

#### Safety Measures
```python
class SafetyValidator:
    """Ensures all operations are safe for testing environment"""

    def validate_read_only_mode(self):
        """Prevent any write operations to production systems"""

    def validate_configuration(self, config):
        """Ensure configuration won't harm existing systems"""

    def audit_log_operation(self, operation, user, config):
        """Log all operations for audit trail"""
```

#### Access Control
- **Role-Based Access**: Different permissions for different user types
- **Configuration Approval**: Require approval for certain configuration changes
- **Audit Logging**: Complete audit trail of all operations
- **Rate Limiting**: Prevent abuse of expensive operations

## Implementation Roadmap

### Week 1-2: Foundation
- [ ] Set up FastAPI backend with basic structure
- [ ] Create React frontend with Material-UI
- [ ] Implement basic search interface
- [ ] Set up SQLite database and basic models

### Week 3-4: Core Search Features
- [ ] Integrate with existing Weaviate search
- [ ] Implement configuration management
- [ ] Add basic reranker integration
- [ ] Create results comparison interface

### Week 5-6: Evaluation Framework
- [ ] Build manual evaluation interface
- [ ] Implement evaluation metrics calculation
- [ ] Add batch evaluation capabilities
- [ ] Create evaluation history tracking

### Week 7-8: Advanced Features
- [ ] Add A/B testing framework
- [ ] Implement data management interface
- [ ] Create analytics dashboard
- [ ] Add configuration presets

### Week 9-10: Polish and Testing
- [ ] Comprehensive testing and bug fixes
- [ ] Performance optimization
- [ ] Documentation and user guides
- [ ] Deployment preparation

## Success Criteria

### Technical Success
- [ ] Zero production system impacts
- [ ] Sub-second search response times
- [ ] 99.9% uptime for testing operations
- [ ] Successful integration with all existing services

### User Success
- [ ] 90% user satisfaction with interface usability
- [ ] 50% reduction in time to test new configurations
- [ ] 100% of test queries properly evaluated
- [ ] Successful adoption by all target user personas

### Business Success
- [ ] Improved search quality metrics
- [ ] Faster model iteration cycles
- [ ] Data-driven configuration decisions
- [ ] Reduced risk of production issues

## Conclusion

The Reranker Testing Dashboard will provide a comprehensive, safe environment for testing and optimizing search and reranking systems. By integrating Context7 best practices and maintaining strict safety measures, this dashboard will enable data-driven decision making while protecting production systems from any potential impacts.

The phased implementation approach ensures early value delivery while building toward a comprehensive solution that meets all testing and evaluation needs.
