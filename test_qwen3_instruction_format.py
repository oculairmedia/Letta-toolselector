#!/usr/bin/env python3
"""
Test script to verify if Ollama properly handles Qwen3 instruction formatting.
This script compares embeddings between raw queries and instruction-formatted queries.
"""

import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any
import json
import os

# Configuration
OLLAMA_HOST = "192.168.50.80"
OLLAMA_PORT = 11434
MODEL = "dengcao/Qwen3-Embedding-4B:Q4_K_M"

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Generate Qwen3-compatible instruction format."""
    return f'Instruct: {task_description}\nQuery: {query}'

async def get_embedding(session: aiohttp.ClientSession, prompt: str) -> List[float]:
    """Get embedding from Ollama for a given prompt."""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    payload = {
        "model": MODEL,
        "prompt": prompt
    }

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            result = await response.json()
            return result.get("embedding", [])
        else:
            raise Exception(f"Failed to get embedding: {response.status}")

async def test_instruction_format():
    """Test if Ollama properly processes Qwen3 instruction format."""

    # Test cases
    test_cases = [
        {
            "name": "Web Search Tools",
            "raw_query": "web search tools",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query"
        },
        {
            "name": "File Management",
            "raw_query": "file management and text processing",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query"
        },
        {
            "name": "Database Operations",
            "raw_query": "database operations and SQL queries",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query"
        }
    ]

    results = []

    async with aiohttp.ClientSession() as session:
        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")

            # Get raw query embedding
            raw_query = test_case["raw_query"]
            print(f"Raw query: '{raw_query}'")
            raw_embedding = await get_embedding(session, raw_query)

            # Get instruction-formatted embedding
            instruction_query = get_detailed_instruct(test_case["task_description"], raw_query)
            print(f"Instruction query: '{instruction_query[:100]}...'")
            instruction_embedding = await get_embedding(session, instruction_query)

            # Calculate similarity
            similarity = cosine_similarity(raw_embedding, instruction_embedding)
            print(f"Cosine similarity: {similarity:.4f}")

            # Determine if instruction format makes a difference
            # High similarity (>0.95) suggests instructions aren't being processed differently
            # Low similarity (<0.95) suggests instructions are being processed
            makes_difference = similarity < 0.95
            print(f"Instruction format processed differently: {makes_difference}")

            results.append({
                "name": test_case["name"],
                "raw_query": raw_query,
                "instruction_query": instruction_query,
                "similarity": similarity,
                "instruction_processed": makes_difference,
                "raw_embedding_dim": len(raw_embedding),
                "instruction_embedding_dim": len(instruction_embedding)
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    processed_count = sum(1 for r in results if r["instruction_processed"])
    total_tests = len(results)

    print(f"Tests run: {total_tests}")
    print(f"Instruction format processed differently: {processed_count}/{total_tests}")

    if processed_count == total_tests:
        print("✅ GOOD: Ollama properly processes Qwen3 instruction format!")
        print("   The embedding model is differentiating between raw and instruction-formatted queries.")
    elif processed_count == 0:
        print("❌ ISSUE: Ollama does NOT process Qwen3 instruction format!")
        print("   All embeddings are too similar - instructions may be ignored.")
    else:
        print("⚠️  MIXED: Some instruction formats processed, others not.")
        print("   This suggests inconsistent instruction processing.")

    # Detailed results
    print(f"\nDetailed Results:")
    for result in results:
        print(f"  {result['name']}: similarity={result['similarity']:.4f}, processed={result['instruction_processed']}")

    # Recommendations
    print(f"\nRecommendations:")
    if processed_count == total_tests:
        print("  ✅ Proceed with implementing Qwen3 instruction format in the search pipeline")
        print("  ✅ Update query enhancement to use proper 'Instruct:' and 'Query:' format")
        print("  ✅ Remove generic query enhancement that adds filler text")
    elif processed_count == 0:
        print("  ❌ Current Ollama setup does not support Qwen3 instruction format")
        print("  ❌ Consider implementing direct Transformers usage instead")
        print("  ❌ Or investigate Ollama configuration for proper instruction support")
    else:
        print("  ⚠️  Investigate why instruction processing is inconsistent")
        print("  ⚠️  Test with different instruction formats or Ollama configurations")

    return results

if __name__ == "__main__":
    print("Testing Qwen3 Instruction Format Support in Ollama")
    print("="*60)
    print(f"Ollama Host: {OLLAMA_HOST}:{OLLAMA_PORT}")
    print(f"Model: {MODEL}")
    print()

    try:
        results = asyncio.run(test_instruction_format())

        # Save results to file for reference
        with open("qwen3_instruction_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: qwen3_instruction_test_results.json")

    except Exception as e:
        print(f"Error running test: {e}")
        exit(1)