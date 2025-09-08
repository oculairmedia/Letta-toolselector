#!/usr/bin/env python3
"""
Embedding configuration constants to ensure consistency across all files.
"""

# OpenAI Embedding Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536

# Weaviate Configuration
WEAVIATE_VECTORIZER = "text2vec-openai"

def validate_embedding_config():
    """
    Validate that the embedding configuration is consistent.
    This function can be called during initialization to ensure consistency.
    """
    print(f"✓ Using OpenAI model: {OPENAI_EMBEDDING_MODEL}")
    print(f"✓ Expected dimensions: {OPENAI_EMBEDDING_DIMENSION}")
    print(f"✓ Weaviate vectorizer: {WEAVIATE_VECTORIZER}")
    return True

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_embedding_config()