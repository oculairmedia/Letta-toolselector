#!/usr/bin/env python3
"""Debug script to test the reranker model directly"""
import asyncio
import httpx
import json

OLLAMA_BASE_URL = "http://192.168.50.80:11434"
OLLAMA_MODEL = "dengcao/Qwen3-Reranker-4B:Q5_K_M"

RERANK_INSTRUCTION_TEMPLATE = """Question: How relevant is this document to the query? Answer with only a number from 0.0 to 1.0.

Query: {query}
Document: {document}

Relevance score:"""

async def test_ollama_response():
    """Test what Ollama actually returns"""
    
    query = "tool for creating blog posts"
    document = "Ghost CMS - A powerful blogging platform with API for creating and managing blog posts"
    
    prompt = RERANK_INSTRUCTION_TEMPLATE.format(query=query, document=document)
    
    print(f"Testing Ollama model: {OLLAMA_MODEL}")
    print(f"Prompt:\n{prompt}\n")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10  # Limit to short response
                    }
                }
            )
            
            print(f"HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                print(f"Raw response: '{raw_response}'")
                print(f"Response length: {len(raw_response)}")
                
                # Try to parse like our adapter does
                score_text = raw_response.strip()
                score_clean = score_text.split()[0] if score_text else "0.0"
                score_clean = ''.join(c for c in score_clean if c.isdigit() or c == '.')
                
                try:
                    score = float(score_clean) if score_clean else 0.0
                    score = max(0.0, min(1.0, score))
                    print(f"Parsed score: {score}")
                except ValueError as e:
                    print(f"Failed to parse score: {e}")
                    print(f"score_clean: '{score_clean}'")
                
            else:
                print(f"Error response: {response.text}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_response())