#!/usr/bin/env python3
"""
Fallback embedding function that uses the new unified embedding provider system
"""

import asyncio
from typing import List
from embedding_providers import EmbeddingProviderFactory

def get_embedding_for_text_direct(text: str) -> List[float]:
    """
    Get embedding using the unified embedding provider system as a fallback when Weaviate vectorizer fails
    """
    try:
        # Use the unified embedding provider system
        return asyncio.run(_get_embedding_async(text))
        
    except Exception as e:
        print(f"Error getting embedding using provider system: {e}")
        return []

async def _get_embedding_async(text: str) -> List[float]:
    """
    Async helper function to get embeddings using the provider system
    """
    # Create provider based on environment
    provider = EmbeddingProviderFactory.create_from_env()
    
    try:
        embedding = await provider.get_single_embedding(text)
        return embedding
    finally:
        await provider.close()

if __name__ == "__main__":
    # Test the function
    test_text = "I need to search for remote software engineering jobs"
    result = get_embedding_for_text_direct(test_text)
    
    if result:
        print(f"✅ Successfully generated embedding for: '{test_text}'")
        print(f"Embedding length: {len(result)}")
        print(f"First 5 values: {result[:5]}")
    else:
        print("❌ Failed to generate embedding")