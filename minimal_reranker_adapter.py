#!/usr/bin/env python3
"""
Minimal Ollama Reranker Adapter for Weaviate Integration Testing
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal Ollama Reranker Test")

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]

OLLAMA_BASE_URL = "http://192.168.50.80:11434"
OLLAMA_MODEL = "dengcao/Qwen3-Reranker-4B:Q5_K_M"

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Minimal reranker implementation for testing"""
    logger.info(f"Reranking {len(request.documents)} documents for query: {request.query[:50]}...")
    
    scores = []
    
    for i, document in enumerate(request.documents):
        logger.info(f"Processing document {i+1}/{len(request.documents)}")
        
        # Test different prompt formats
        prompts = [
            # Format 1: Simple
            f"Query: {request.query}\nDocument: {document}\nRelevance score (0.0-1.0):",
            
            # Format 2: Detailed instruction
            f"""Given a search query and a document, determine how relevant the document is to the query.
Output only a single number between 0.0 and 1.0, where:
- 0.0 means completely irrelevant  
- 1.0 means perfectly relevant

Query: {request.query}
Document: {document}

Relevance score:"""
        ]
        
        # Try format 2 (more detailed)
        prompt = prompts[1]
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                logger.info(f"Calling Ollama with prompt length: {len(prompt)}")
                
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 10,
                            "stop": ["\n", " ", "\t"]
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    score_text = result.get("response", "0.0").strip()
                    logger.info(f"Ollama response: '{score_text}'")
                    
                    try:
                        # Extract just the number
                        score_clean = score_text.split()[0] if score_text else "0.0"
                        score = float(score_clean)
                        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                        scores.append(score)
                        logger.info(f"Parsed score: {score}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse score '{score_text}': {e}")
                        scores.append(0.0)
                else:
                    logger.error(f"Ollama request failed: {response.status_code}")
                    scores.append(0.0)
                    
        except Exception as e:
            logger.error(f"Error scoring document: {e}")
            scores.append(0.0)
    
    logger.info(f"Final scores: {scores}")
    return RerankResponse(scores=scores)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            
            if response.status_code == 200:
                return {"status": "healthy", "ollama": "connected", "model": OLLAMA_MODEL}
            else:
                return {"status": "degraded", "ollama": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    return {"message": "Minimal Ollama Reranker Adapter", "model": OLLAMA_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")