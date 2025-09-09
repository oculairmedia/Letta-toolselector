#!/usr/bin/env python3
"""
Production Ollama Reranker Adapter for Weaviate Integration
Implements Weaviate's reranker-transformers API using Ollama-hosted Qwen3-Reranker-4B
"""
import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.50.80:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dengcao/Qwen3-Reranker-4B:Q5_K_M")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response Models
class RerankRequest(BaseModel):
    """Weaviate reranker API request format"""
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    k: Optional[int] = Field(None, description="Number of top results to return")

class RerankResponse(BaseModel):
    """Weaviate reranker API response format"""
    scores: List[float] = Field(..., description="Relevance scores in document order")

# Cache implementation
class SimpleCache:
    """Simple in-memory cache with TTL"""
    def __init__(self, ttl: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

# Initialize cache
cache = SimpleCache(ttl=CACHE_TTL) if ENABLE_CACHE else None

# Metrics tracking
class Metrics:
    """Simple metrics tracking"""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency = 0.0
        self.model_load_time = None
    
    def to_dict(self) -> Dict:
        avg_latency = self.total_latency / max(self.successful_requests, 1)
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "average_latency_ms": avg_latency * 1000,
            "model_load_time_ms": self.model_load_time * 1000 if self.model_load_time else None
        }

metrics = Metrics()

# Instruction template for Qwen3-Reranker
RERANK_INSTRUCTION_TEMPLATE = """Question: How relevant is this document to the query? Answer with only a number from 0.0 to 1.0.

Query: {query}
Document: {document}

Relevance score:"""

async def warmup_model():
    """Warm up the Ollama model to reduce first-request latency"""
    logger.info(f"Warming up model {OLLAMA_MODEL}...")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            warmup_prompt = RERANK_INSTRUCTION_TEMPLATE.format(
                query="test query",
                document="test document"
            )
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "num_predict": 10
                    }
                }
            )
            
            if response.status_code == 200:
                elapsed = time.time() - start_time
                metrics.model_load_time = elapsed
                logger.info(f"Model warmed up successfully in {elapsed:.2f}s")
            else:
                logger.warning(f"Model warmup failed: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Model warmup error: {e}")

async def score_document_pair(
    query: str, 
    document: str,
    session: httpx.AsyncClient
) -> float:
    """Score a single query-document pair using Ollama"""
    
    # Generate prompt
    prompt = RERANK_INSTRUCTION_TEMPLATE.format(
        query=query,
        document=document[:1000]  # Truncate very long documents
    )
    
    # Check cache if enabled
    cache_key = f"{query}::{document[:100]}" if cache else None
    if cache_key and cache:
        cached_score = cache.get(cache_key)
        if cached_score is not None:
            metrics.cache_hits += 1
            return cached_score
        metrics.cache_misses += 1
    
    try:
        response = await session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "num_predict": 10
                }
            },
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama request failed: {response.status_code}")
            return 0.0
        
        result = response.json()
        score_text = result.get("response", "0.0").strip()
        
        # Parse score
        try:
            # Clean the response (sometimes contains extra text)
            score_clean = score_text.split()[0] if score_text else "0.0"
            # Remove any non-numeric characters except decimal point
            score_clean = ''.join(c for c in score_clean if c.isdigit() or c == '.')
            score = float(score_clean)
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
            
            # Cache the result
            if cache_key and cache:
                cache.set(cache_key, score)
            
            return score
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse score '{score_text}': {e}")
            return 0.0
    
    except httpx.TimeoutException:
        logger.warning(f"Timeout scoring document")
        return 0.0
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return 0.0

async def batch_score_documents(
    query: str,
    documents: List[str]
) -> List[float]:
    """Score multiple documents in batches for efficiency"""
    scores = []
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as session:
        # Process in batches
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            
            # Score batch concurrently
            tasks = [
                score_document_pair(query, doc, session)
                for doc in batch
            ]
            
            batch_scores = await asyncio.gather(*tasks)
            scores.extend(batch_scores)
    
    return scores

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await warmup_model()
    yield
    # Shutdown
    if cache:
        cache.clear()

# FastAPI Application
app = FastAPI(
    title="Ollama Reranker Adapter",
    description="Weaviate reranker adapter using Ollama-hosted Qwen3-Reranker-4B",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Main reranking endpoint matching Weaviate's reranker-transformers API
    """
    start_time = time.time()
    metrics.total_requests += 1
    
    try:
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")
        
        # Limit k to reasonable bounds
        k = min(request.k, len(request.documents)) if request.k else len(request.documents)
        k = min(k, 100)  # Maximum 100 results
        
        logger.info(f"Reranking {len(request.documents)} documents for query: '{request.query[:50]}...'")
        
        # Score all documents
        scores = await batch_score_documents(request.query, request.documents)
        
        # If k is specified and less than total documents, set lower scores to 0
        if request.k and request.k < len(scores):
            # Get indices of top-k scores
            indexed_scores = [(score, idx) for idx, score in enumerate(scores)]
            indexed_scores.sort(reverse=True, key=lambda x: x[0])
            top_k_indices = set(idx for _, idx in indexed_scores[:k])
            
            # Set non-top-k scores to 0
            scores = [
                score if idx in top_k_indices else 0.0
                for idx, score in enumerate(scores)
            ]
        
        # Track metrics
        elapsed = time.time() - start_time
        metrics.successful_requests += 1
        metrics.total_latency += elapsed
        
        logger.info(f"Reranking completed in {elapsed:.2f}s")
        
        return RerankResponse(scores=scores)
    
    except HTTPException:
        metrics.failed_requests += 1
        raise
    except Exception as e:
        metrics.failed_requests += 1
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check Ollama connectivity
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            
            ollama_status = "connected" if response.status_code == 200 else "disconnected"
            
            return {
                "status": "healthy" if ollama_status == "connected" else "degraded",
                "ollama": ollama_status,
                "model": OLLAMA_MODEL,
                "cache_enabled": ENABLE_CACHE,
                "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
            }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model": OLLAMA_MODEL
        }

@app.get("/.well-known/ready")
async def ready():
    """Weaviate readiness check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return {"ready": True}
    except Exception:
        pass
    
    # Return 503 if not ready
    return JSONResponse(
        status_code=503,
        content={"ready": False, "error": "Ollama backend not available"}
    )

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring and optimization"""
    return {
        "model": OLLAMA_MODEL,
        "batch_size": BATCH_SIZE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "cache_enabled": ENABLE_CACHE,
        "cache_ttl_seconds": CACHE_TTL if ENABLE_CACHE else None,
        **metrics.to_dict()
    }

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "Ollama Reranker Adapter",
        "version": "1.0.0",
        "model": OLLAMA_MODEL,
        "api_endpoints": {
            "rerank": "/rerank",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

# Middleware to track start time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    if not hasattr(app.state, 'start_time'):
        app.state.start_time = time.time()
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level=LOG_LEVEL.lower()
    )