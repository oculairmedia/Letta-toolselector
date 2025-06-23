#!/usr/bin/env python3
"""
Fallback embedding function that uses OpenAI directly instead of Weaviate's vectorizer
"""

import os
import openai
from typing import List

def get_embedding_for_text_direct(text: str) -> List[float]:
    """
    Get embedding directly from OpenAI API as a fallback when Weaviate vectorizer fails
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("OpenAI API key not found in environment")
            return []
            
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Get embedding using the same model as Weaviate
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        # Extract the embedding vector
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        print(f"Error getting embedding from OpenAI directly: {e}")
        return []

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