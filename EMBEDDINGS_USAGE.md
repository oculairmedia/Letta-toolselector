# Embeddings Usage in Letta Tools Selector

This document explains how embeddings are created, stored, queried, and consumed across this project. It focuses on the Python "lettaaugment-source" service that indexes and searches tools using Weaviate, plus fallbacks and related configs used by the agent management flows.

## High-level overview
- Vector DB: Weaviate (v4 client) hosts a Tool collection with text fields vectorized by OpenAI’s text embedding models.
- Ingestion: Tools are inserted into Weaviate; the configured vectorizer automatically generates and stores embeddings for selected properties.
- Query: Searches use Weaviate’s hybrid search (vector + keyword) with query expansion. Scores are returned per result.
- Embedding retrieval: Utility functions fetch stored vectors or compute ad-hoc embeddings for a text via Weaviate or via a direct OpenAI fallback.
- API surface: The Quart API exposes the search and attachment workflow; it imports embedding helpers and uses similarity utilities.
- Agent-level config: Separate from Weaviate usage, when creating agents, the project configures the agent’s own embedding provider/model (Google AI text-embedding-004) — this is for the agent platform and is independent of the Weaviate store.

## Key files and responsibilities
- lettaaugment-source/init_weaviate_schema.py
  - Defines a Tool collection with OpenAI vectorizer text2vec-openai set to model "text-embedding-3-small".
- lettaaugment-source/upload_tools_to_weaviate.py
  - Creates/ensures the Tool collection and uploads tool objects (batch). Uses text2vec-openai via Configure.Vectorizer.
- lettaaugment-source/weaviate_tool_search.py
  - Initializes Weaviate client.
  - search_tools(): performs hybrid search over the Tool collection with query expansion, returning results and scores.
  - get_embedding_for_text(): tries to obtain a vector for arbitrary text via a Weaviate GraphQL nearText trick; falls back to direct OpenAI embeddings API.
  - get_tool_embedding_by_id(): fetches the stored vector for a tool (include_vector=True) with GraphQL fallback.
- lettaaugment-source/fallback_embedding.py
  - Minimal direct call to OpenAI Embeddings API (text-embedding-3-small) as a fallback utility.
- lettaaugment-source/api_server.py
  - Quart API that calls search_tools() in /api/v1/tools/attach and manages tool attaching/pruning flows.
  - Contains cosine_similarity() helper suitable for vector comparisons.
- lettaaugment-source/tool_finder_agent.py
  - When creating an agent, sets embedding_config to Google AI text-embedding-004 (separate from Weaviate/openai vector store usage).
- lettaaugment-source/test_*.py (embedding-related tests)
  - test_embedding_local.py, debug_existing_embeddings.py, etc. provide diagnostics for GraphQL nearText and direct OpenAI fallbacks.

## Data model and vectorization

### Tool collection schema
Two places define how the Tool collection and its vectorizer are configured:

1) init_weaviate_schema.py
- Uses custom connection and creates Tool with:
  - vectorizer: text2vec-openai
  - moduleConfig.text2vec-openai.model: text-embedding-3-small
  - Properties vectorized: name (TEXT), description (TEXT), tags (string[]), json_schema (text)
  - Some properties have vectorizePropertyName=False

2) upload_tools_to_weaviate.py → get_or_create_tool_schema()
- Creates Tool with:
  - vectorizer_config = Configure.Vectorizer.text2vec_openai(model="ada", model_version="002")
  - Properties similar to above; description and json_schema set vectorize_property_name=False

Important: There is a model mismatch between these two schema creators.
- init_weaviate_schema.py: text-embedding-3-small
- upload_tools_to_weaviate.py: text-embedding-ada-002 (model="ada", version="002")

Implications:
- If both scripts are used at different times, the collection may be re-created with different vectorizer settings. Ensure your environment/path uses only one setup path consistently to avoid mixed embeddings.
- The upload script also deletes and recreates the collection each run (to force schema), while init_weaviate_schema deletes if exists then creates.

### Environment requirements
- OPENAI_API_KEY must be set (used by Weaviate vectorizer via header X-OpenAI-Api-Key and by direct OpenAI fallback calls).
- WEAVIATE_HTTP_HOST/PORT and WEAVIATE_GRPC_HOST/PORT can override defaults (defaults often set to "weaviate" inside Docker; localhost for local tests).

## Ingestion workflow (embeddings at write time)
File: lettaaugment-source/upload_tools_to_weaviate.py

- Connects to Weaviate with OpenAI API key in headers.
- Ensures Tool collection (see mismatch note above) and uses client.collections.create(... vectorizer_config=Configure.Vectorizer.text2vec_openai(...)).
- Fetches the full set of tools (fetch_all_tools_async()).
- Adds objects via a dynamic batch with properties:
  - tool_id, name, description, source_type, tool_type, tags, json_schema
- Embeddings are not computed client-side; the Weaviate vectorizer module computes/stores vectors for the configured properties automatically upon insert.

## Query workflow (embeddings at read time)
File: lettaaugment-source/weaviate_tool_search.py

- preprocess_query(): expands the user query with domain synonyms to improve recall.
- search_tools(query, limit):
  - client.collections.get("Tool").query.hybrid(
    - query: expanded query
    - alpha: 0.75 (75% vector, 25% keyword)
    - query_properties: ["name^2", "description^1.5", "tags"]
    - return_metadata: MetadataQuery(score=True)
  )
  - Post-processes results to attach distance = 1 - score for compatibility.

Why hybrid search?
- Hybrid search blends vector similarity with keyword matching; this is often helpful when data contains structured language (names, tags) but still benefits from semantic retrieval.

### Embeddings for arbitrary text (ad-hoc vectorization)
- get_embedding_for_text(text):
  - Attempts a Weaviate GraphQL nearText query that returns _additional.vector for a stub result, using the collection’s vectorizer to embed the input concept.
  - If any part fails (common in some Weaviate versions/configs), it falls back to direct OpenAI embeddings API (text-embedding-3-small) to compute the vector client-side.

### Fetching stored embeddings for objects
- get_tool_embedding_by_id(tool_id):
  - First tries collection.query.fetch_objects(filters=Filter.by_id().equal(tool_id), include_vector=True)
  - If not present or invalid, it falls back to a GraphQL query requesting _additional.vector for the Tool with that UUID.

## API server usage of embeddings
File: lettaaugment-source/api_server.py

- /api/v1/tools/attach uses search_tools() under the hood to find relevant tools for an agent and then manages attachments/detachments via Letta API.
- cosine_similarity(vec1, vec2) is provided for vector comparisons; while not heavily used in the shown sections, it supports any future logic needing explicit similarity on vectors retrieved by the helpers above.
- The server imports get_embedding_for_text and get_tool_embedding_by_id; these are available for diagnostic or potential scoring functions (some logic is elided in the clipped file).

## Fallbacks and diagnostics
- Direct OpenAI fallback:
  - In weaviate_tool_search.py and fallback_embedding.py the fallback hits https://api.openai.com/v1/embeddings with model text-embedding-3-small.
  - Requires OPENAI_API_KEY and internet connectivity.
- Diagnostic scripts:
  - debug_existing_embeddings.py: experiments with fetching vectors via Weaviate and via nearText; prints shapes/lengths.
  - test_embedding_local.py: tests local Weaviate access (localhost), checks GraphQL nearText vector extraction, and direct OpenAI embedding.

## Agent-level embeddings (separate from Weaviate)
File: lettaaugment-source/tool_finder_agent.py

- When creating an agent via make_api_request("POST", "agents", data=agent_data), the payload includes:
  - embedding_config.embedding_endpoint_type: "google_ai"
  - embedding_model: "text-embedding-004"
  - embedding_dim: 768
- This configuration controls how the agent platform embeds its own internal content/memories. It does not change the Weaviate store’s vectorization, which is configured independently (OpenAI text2vec-openai).

## End-to-end flow summary
1) Upload/refresh tool index
   - upload_tools_to_weaviate.py creates the Tool collection (vectorizer: text2vec-openai) and uploads tool documents. Weaviate computes embeddings automatically per configured fields.
2) Search from user/agent request
   - The MCP server (src/index.js) invokes find_tools.py, which calls the API server /api/v1/tools/attach.
   - The API server calls search_tools() (Weaviate hybrid search) to retrieve top candidate tools.
   - Results are filtered/processed and attached to the agent via Letta API.
3) Optional ad-hoc vector ops
   - Embedding helpers can fetch a tool’s stored vector or compute a vector for arbitrary text (Weaviate nearText -> _additional.vector or OpenAI fallback).

## Known pitfalls and recommendations
- Ensure a single schema path: Pick either init_weaviate_schema.py (text-embedding-3-small) or upload_tools_to_weaviate.py (ada-002) as your canonical schema creator. Mixed usage will cause re-indexing with different embedding models.
- Environment-targeted hosts: Local tests use localhost; Docker may use host "weaviate". Override via WEAVIATE_* env vars as needed.
- OpenAI API key: Required for text2vec-openai vectorizer and fallbacks. Missing key will degrade or disable vectorization.
- Hybrid parameters: alpha=0.75 and property boosts (name^2, description^1.5) tune the ranking. Adjust if you need more keyword precision or semantic recall.
- GraphQL nearText for ad-hoc embedding is brittle across versions; rely on it opportunistically and keep the direct OpenAI fallback available.

## Quick references (functions and where)
- Initialize client (Weaviate): lettaaugment-source/weaviate_tool_search.py:init_client()
- Hybrid search: lettaaugment-source/weaviate_tool_search.py:search_tools()
- Ad-hoc text embedding: lettaaugment-source/weaviate_tool_search.py:get_embedding_for_text(); lettaaugment-source/fallback_embedding.py:get_embedding_for_text_direct()
- Object vector fetch: lettaaugment-source/weaviate_tool_search.py:get_tool_embedding_by_id()
- Schema creation (OpenAI text-embedding-3-small): lettaaugment-source/init_weaviate_schema.py
- Schema creation (ada-002): lettaaugment-source/upload_tools_to_weaviate.py:get_or_create_tool_schema()
- API attach/search entrypoint: lettaaugment-source/api_server.py:/api/v1/tools/attach
- Node MCP launcher: src/index.js (delegates to find_tools.py -> API server)

## Minimal setup checklist
- Set OPENAI_API_KEY in environment (.env)
- Configure WEAVIATE_* host/ports if not running in Docker alongside Weaviate
- Choose and standardize the schema creation path (text-embedding-3-small vs ada-002)
- Run upload_tools_to_weaviate.py (or init_weaviate_schema.py + a data loader) to build the index
- Use /api/v1/tools/attach with a query to validate hybrid search returns expected tools

