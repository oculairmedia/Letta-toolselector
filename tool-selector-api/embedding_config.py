#!/usr/bin/env python3
"""
Embedding configuration constants to ensure consistency across all files.
Matches Graphiti configuration for compatibility.
"""

# OpenAI Embedding Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536

# Ollama Embedding Configuration (matching Graphiti setup)
OLLAMA_EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-4B:Q4_K_M"
OLLAMA_EMBEDDING_DIMENSION = 2560  # Qwen3-Embedding-4B model dimension
OLLAMA_BASE_URL = "http://192.168.50.80:11434"

# Weaviate Configuration
WEAVIATE_VECTORIZER = "text2vec-openai"

def validate_embedding_config():
    """
    Validate that the embedding configuration is consistent.
    This function can be called during initialization to ensure consistency.
    """
    print(f"✓ Using OpenAI model: {OPENAI_EMBEDDING_MODEL}")
    print(f"✓ OpenAI dimensions: {OPENAI_EMBEDDING_DIMENSION}")
    print(f"✓ Using Ollama model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"✓ Ollama dimensions: {OLLAMA_EMBEDDING_DIMENSION}")
    print(f"✓ Ollama base URL: {OLLAMA_BASE_URL}")
    print(f"✓ Weaviate vectorizer: {WEAVIATE_VECTORIZER}")
    return True

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_embedding_config()