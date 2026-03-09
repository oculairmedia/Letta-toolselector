# Graphiti Embedding Compatibility

The Letta Tool Selector now fully supports Graphiti's embedding configuration for seamless compatibility between the two systems.

## Configuration

### Graphiti-Compatible Models

**OpenAI Provider:**
- **Model**: `text-embedding-3-small`
- **Dimensions**: `1536`

**Ollama Provider (Graphiti Setup):**
- **Model**: `dengcao/Qwen3-Embedding-4B:Q4_K_M`
- **Dimensions**: `2560`
- **Host**: `192.168.50.80:11434`

## Environment Variables

### Basic Provider Selection
```bash
# Choose embedding provider
EMBEDDING_PROVIDER=openai  # or ollama

# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama configuration (Graphiti-compatible)
OLLAMA_EMBEDDING_HOST=192.168.50.80
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M
EMBEDDING_DIMENSION=2560
USE_OLLAMA_EMBEDDINGS=true
```

### Alternative Ollama Configuration
```bash
# Alternative: Use full URL instead of host
OLLAMA_BASE_URL=http://192.168.50.80:11434
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M
OLLAMA_EMBEDDING_DIMENSIONS=2560
```

## Usage Examples

### Default (Environment-based)
```python
from embedding_providers import EmbeddingProviderFactory

# Uses EMBEDDING_PROVIDER environment variable
provider = EmbeddingProviderFactory.create_from_env()
embeddings = await provider.get_embeddings(['test text'])
```

### Explicit Provider Selection
```python
# Force OpenAI
provider = EmbeddingProviderFactory.create('openai')

# Force Ollama with Graphiti settings
provider = EmbeddingProviderFactory.create('ollama')
```

### Context Manager (Recommended)
```python
from embedding_providers import EmbeddingProviderContext

async with EmbeddingProviderContext() as provider:
    embedding = await provider.get_single_embedding('search query')
    print(f"Provider: {provider.provider_name}")
    print(f"Dimensions: {len(embedding)}")
```

## Compatibility Testing

Run the compatibility test to verify your configuration:

```bash
python test_graphiti_embedding_config.py
```

This will test:
- ✅ Configuration validation
- ✅ Provider availability  
- ✅ OpenAI provider functionality
- ✅ Ollama configuration (Graphiti-compatible)
- ✅ Environment variable support

## Integration with Existing Code

All existing embedding functions automatically use the new provider system:

- `fallback_embedding.py` - Fallback embedding generation
- `weaviate_tool_search.py` - Tool search embeddings
- `api_server.py` - API endpoint embeddings

## Benefits

1. **Graphiti Compatibility**: Same models and dimensions as your Graphiti setup
2. **Flexibility**: Switch between OpenAI (easy) and Ollama (self-hosted) 
3. **High Dimensions**: 2560-dimensional embeddings with Qwen3-Embedding-4B
4. **Consistent**: Same embedding space across Letta Tool Selector and Graphiti
5. **Environment-driven**: Configure via environment variables

## Migration from Previous Setup

If you were using the old embedding configuration:

1. **OpenAI users**: No changes needed - same `text-embedding-3-small` model
2. **Adding Ollama**: Set `EMBEDDING_PROVIDER=ollama` and Ollama environment variables
3. **Graphiti users**: Your existing environment variables work automatically

The system maintains backward compatibility while adding Graphiti support.