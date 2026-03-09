# LDTS Search/Reranker Testing Dashboard – Status and Gap Analysis

Date: 2025-09-09
Scope: Map key LDTS issues to current code evidence and identify gaps blocking the testing dashboard effort.

## Executive Summary
- Foundations exist for multi-provider embeddings and Weaviate hybrid search.
- Client-side reranking module exists but is not exposed via dedicated testing endpoints.
- No frontend dashboard yet; no evaluation metrics/A-B testing framework implemented.
- Safety/audit controls referenced in planning but not enforced in the API layer.
- Embedding versioning, dual-index, and re-embedding orchestration not yet present.

## Implemented Foundations (evidence in repo)
- Embedding provider factory (OpenAI + Ollama)
  - File: tool-selector-api/embedding_providers.py
  - Highlights: `EmbeddingProviderFactory.create()` / `create_from_env()`, async provider interface, batching, env overrides.
- Weaviate search with optional client-side reranking
  - File: tool-selector-api/weaviate_tool_search_with_reranking.py
  - Two-stage retrieval + HTTP reranker adapter; flags: ENABLE_RERANKING, RERANK_INITIAL_LIMIT, RERANK_TOP_K.
- Base Weaviate hybrid search (non-reranking)
  - File: tool-selector-api/weaviate_tool_search.py
  - Uses v4 client, HybridFusion, query expansion.
- API server (Quart/Hypercorn) search and attach flows
  - File: tool-selector-api/api_server.py
  - Provides /api/v1/tools/search; recognizes `rerank_score` when present in results.
- Compose/env wiring for providers and reranking
  - Files: compose.yaml, compose-with-reranker.yaml, .env.example
  - Vars: EMBEDDING_PROVIDER, OLLAMA_*, ENABLE_RERANKING, RERANK_*.
- Documentation alignment
  - Files: EMBEDDINGS_USAGE.md, GRAPHITI_EMBEDDING_COMPATIBILITY.md

## Missing or Partial vs PRD and Issues
- Testing API surface (PRD):
  - NOT FOUND: GET /api/v1/search/test, POST /api/v1/rerank/compare, models listing, config validate, eval history, batch eval.
- Reranker wiring in API:
  - API currently imports the standard search; reranking module exists but not exposed via testing endpoints.
- Safety and audit enforcement:
  - No explicit read-only guard/audit middleware observed in api_server.py.
- Evaluation framework:
  - No metrics engine (Precision@K, NDCG, MRR, MAP), no A/B testing, no batch evaluation.
- UI/Frontend dashboard:
  - No React/Material-UI code present.
- Embedding lifecycle management:
  - No embedding version metadata, dual-index orchestration, or re-embedding job controller.

## Issue-by-Issue Mapping (selected)

- LDTS-26 FastAPI server with async support – Status: done (Huly)
  - Repo: API server uses Quart/Hypercorn async; functional.
  - Gap: Testing-specific endpoints absent.

- LDTS-27 Search testing API endpoints – Status: done (Huly)
  - Repo: Only /api/v1/tools/search exists; no dedicated testing endpoints per PRD.
  - Gap: Implement /api/v1/search/test and /api/v1/rerank/compare with param overrides.

- LDTS-28 Configuration validation service – Status: done (Huly)
  - Repo: No standalone validator service found; some validation inline.
  - Gap: Add config validation endpoint and schema-based checks (embedding/weaviate/reranker params).

- LDTS-29 Integration layer with Weaviate/embedding – Status: done (Huly)
  - Repo: weaviate_tool_search.py and embedding_providers.py present.
  - Gap: Expose param overrides via API for experiments.

- LDTS-30 Safety measures/read-only enforcement – Status: done (Huly)
  - Repo: No read-only guard in api_server.py.
  - Gap: Middleware/guard + config flags; ensure testing endpoints have no side effects.

- LDTS-31 Audit logging for testing – Status: done (Huly)
  - Repo: No dedicated audit trail in api_server.py.
  - Gap: Central audit logger for all testing actions with config snapshot.

- LDTS-32 Rate limiting/resource mgmt – Status: done (Huly)
  - Repo: No explicit per-endpoint rate limiting seen.
  - Gap: Lightweight rate limiter for testing endpoints; provider-level backoff is handled in providers.

- LDTS-33 YAML configuration schema – Status: done (Huly)
  - Repo: PRD has detailed schema; code not yet loading/validating YAML.
  - Gap: Add load/validate endpoint and server-side schema.

- LDTS-34 Multi-provider embedding config – Status: done (Huly)
  - Repo: embedding_providers.py implemented; env-based switching works.
  - Gap: UI/API to switch per-test.

- LDTS-35 Weaviate hyperparameter controls – Status: done (Huly)
  - Repo: Hybrid search with defaults; no API to override alpha/fusion/candidate_pool.
  - Gap: Add override support on testing endpoints.

- LDTS-36 Reranker model configuration system – Status: backlog
  - Repo: Client-side reranking module exists; model selection not exposed via API.
  - Gap: API config for reranker type/model, k, batch size, truncation, latency budget.

- LDTS-39 Manual evaluation interface – Status: backlog
  - Repo: None.
  - Gap: UI + endpoints to record human relevance ratings.

- LDTS-41 Metrics engine (P@K, NDCG, MRR, MAP) – Status: backlog
  - Repo: None.
  - Gap: Implement metric calculations and persistence.

- LDTS-42 A/B testing with significance – Status: backlog
  - Repo: None.
  - Gap: Compare configs, compute deltas, significance tests.

- LDTS-43 Benchmark query management – Status: backlog
  - Repo: None.
  - Gap: CRUD for benchmark sets; replay runner.

- LDTS-44 Evaluation history/reporting – Status: backlog
  - Repo: None.
  - Gap: DB schema + endpoints; PRD includes example tables.

- LDTS-48 Embedding versioning – Status: backlog
  - Repo: None.
  - Gap: Store per-vector version metadata (provider/model/dimension/pooling/etc.).

- LDTS-49 Dual-index strategy – Status: backlog
  - Repo: None.
  - Gap: Build candidate index in parallel; switch/cutover with rollback.

- LDTS-50 Re-embedding batches/resume – Status: backlog
  - Repo: None.
  - Gap: Job controller + progress tracking; resumable batches.

- LDTS-56 Grid search framework – Status: backlog
  - Repo: None.
  - Gap: Parameter sweeps over embedding/weaviate/reranker configs.

- LDTS-57 Statistical analysis engine – Status: backlog
  - Repo: None.
  - Gap: Confidence intervals, multiple-comparison correction.

- LDTS-58 Cost controls/budgets – Status: backlog
  - Repo: None.
  - Gap: Track token/requests per provider; budget limits and alerts.

- LDTS-62/63 Integration polish and client management – Status: backlog
  - Repo: Base integration exists.
  - Gap: Connection pooling, health checks, retry policies across services.

- LDTS-18/19/20/21/22/23/25 UI/UX components – Status: backlog
  - Repo: None.
  - Gap: React/Material-UI dashboard and subviews (search, config panel, comparison, manual eval, data mgmt, analytics, responsive layout).

## Quick Close Opportunities (low risk, high impact)
1) Add testing endpoints (no side effects):
   - GET /api/v1/search/test (accepts overrides for embedding/weaviate/reranker; returns detailed ranks/scores)
   - POST /api/v1/rerank/compare (two configs; returns side-by-side with delta stats)
2) Add minimal safety/audit wrappers on those endpoints:
   - Read-only guard (deny any write ops)
   - Audit log with user, payload, and config hash
3) Expose model lists and config validation:
   - GET /api/v1/models/embedding, /api/v1/models/reranker
   - POST /api/v1/config/validate (schema-based)

## Evidence Pointers (files)
- Embedding providers: tool-selector-api/embedding_providers.py
- Weaviate search (base): tool-selector-api/weaviate_tool_search.py
- Weaviate search (rerank): tool-selector-api/weaviate_tool_search_with_reranking.py
- API server: tool-selector-api/api_server.py
- Compose/env: compose.yaml, compose-with-reranker.yaml, .env.example
- Docs: EMBEDDINGS_USAGE.md, GRAPHITI_EMBEDDING_COMPATIBILITY.md, RERANKER_TESTING_DASHBOARD_PRD.md

## Notes
- api_server.py currently recognizes rerank_score if present but is wired to the non-reranking search by default; testing endpoints should explicitly call the reranking path with flags.
- Safety/audit items marked as “done” in issues are not visible in this service; they may be planned elsewhere—this doc reflects only this repo.

