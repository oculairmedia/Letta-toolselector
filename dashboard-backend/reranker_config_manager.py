"""
LDTS-36: Reranker model configuration system

Provides comprehensive configuration and management for different reranker models
including Ollama, Cohere, and future providers for the LDTS dashboard.
"""

import asyncio
import aiohttp
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from pathlib import Path
import os

from qwen3_reranker_utils import (
    DEFAULT_RERANK_INSTRUCTION,
    build_prompt,
    extract_yes_probability,
)

logger = logging.getLogger(__name__)

class RerankerStatus(Enum):
    """Reranker model status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    UNKNOWN = "unknown"

class RerankerType(Enum):
    """Types of reranker models"""
    CROSS_ENCODER = "cross_encoder"
    BI_ENCODER = "bi_encoder"
    NEURAL_RERANKER = "neural_reranker"
    TRADITIONAL_ML = "traditional_ml"

@dataclass
class RerankResult:
    """Result from reranking operation"""
    reranked_indices: List[int]           # Original indices in new order
    reranked_scores: List[float]          # Reranking scores
    original_scores: List[float]          # Original scores
    provider: str                         # Reranker provider
    model: str                           # Reranker model name
    processing_time: float               # Processing time in seconds
    query: str                          # Original query
    document_count: int                 # Number of documents reranked
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RerankerMetrics:
    """Reranker performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    total_documents_processed: int = 0
    average_latency: float = 0.0
    average_documents_per_request: float = 0.0
    last_request_time: Optional[float] = None
    status: RerankerStatus = RerankerStatus.UNKNOWN
    error_rate: float = 0.0
    throughput_docs_per_second: float = 0.0

@dataclass
class RerankerConfig:
    """Base reranker configuration"""
    provider: str
    model: str
    enabled: bool = True
    max_batch_size: int = 16
    timeout_seconds: int = 30
    temperature: float = 0.0
    top_k: int = 100
    threshold: float = 0.0
    retry_attempts: int = 2
    custom_params: Dict[str, Any] = field(default_factory=dict)

class RerankerProvider(ABC):
    """Abstract base class for reranker providers"""
    
    def __init__(self, provider_name: str, config: RerankerConfig):
        self.provider_name = provider_name
        self.config = config
        self.metrics = RerankerMetrics()
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the reranker provider"""
        pass
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[str], 
                    original_scores: Optional[List[float]] = None) -> RerankResult:
        """Rerank documents based on query relevance"""
        pass
    
    @abstractmethod
    async def health_check(self) -> RerankerStatus:
        """Check reranker health"""
        pass
    
    def _update_metrics(self, success: bool, processing_time: float, 
                       document_count: int):
        """Update reranker metrics"""
        self.metrics.total_requests += 1
        self.metrics.total_processing_time += processing_time
        self.metrics.total_documents_processed += document_count
        self.metrics.last_request_time = time.time()
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update derived metrics
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            self.metrics.average_latency = self.metrics.total_processing_time / self.metrics.total_requests
            self.metrics.average_documents_per_request = self.metrics.total_documents_processed / self.metrics.total_requests
        
        if self.metrics.total_processing_time > 0:
            self.metrics.throughput_docs_per_second = self.metrics.total_documents_processed / self.metrics.total_processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get reranker metrics as dictionary"""
        return {
            "provider_name": self.provider_name,
            "model": self.config.model,
            "enabled": self.config.enabled,
            "status": self.metrics.status.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "error_rate": self.metrics.error_rate,
            "average_latency_ms": self.metrics.average_latency * 1000,
            "average_documents_per_request": self.metrics.average_documents_per_request,
            "throughput_docs_per_second": self.metrics.throughput_docs_per_second,
            "total_documents_processed": self.metrics.total_documents_processed,
            "last_request_time": self.metrics.last_request_time
        }

class OllamaRerankerProvider(RerankerProvider):
    """Ollama reranker provider"""
    
    def __init__(self, config: Dict[str, Any]):
        reranker_config = RerankerConfig(
            provider="ollama",
            model=config.get('model', 'qwen3-reranker-4b'),
            enabled=config.get('enabled', True),
            max_batch_size=config.get('max_batch_size', 16),
            timeout_seconds=config.get('timeout_seconds', 30),
            temperature=config.get('temperature', 0.1),
            top_k=config.get('top_k', 100),
            threshold=config.get('threshold', 0.0)
        )
        
        super().__init__("ollama", reranker_config)
        
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.keep_alive = config.get('keep_alive', '5m')
        self.num_predict = config.get('num_predict', 1)
        self.session = None
    
    async def initialize(self) -> bool:
        """Initialize Ollama reranker"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
            
            # Test the connection and model availability
            await self.health_check()
            
            self._initialized = True
            logger.info(f"Ollama reranker initialized: {self.config.model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama reranker: {e}")
            if self.session:
                await self.session.close()
            return False
    
    async def rerank(self, query: str, documents: List[str], 
                    original_scores: Optional[List[float]] = None) -> RerankResult:
        """Rerank documents using Ollama reranker"""
        if not self._initialized:
            raise RuntimeError("Reranker not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare rerank scores for all documents
            rerank_scores = []
            
            # Process documents in batches
            batch_size = min(self.config.max_batch_size, len(documents))
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Process each document in the batch
                batch_scores = []
                for doc in batch_docs:
                    score = await self._score_document(query, doc)
                    batch_scores.append(score)
                
                rerank_scores.extend(batch_scores)
            
            # Create ranked indices based on scores
            indexed_scores = list(enumerate(rerank_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
            
            reranked_indices = [idx for idx, _ in indexed_scores]
            sorted_scores = [score for _, score in indexed_scores]
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(True, processing_time, len(documents))
            self.metrics.status = RerankerStatus.HEALTHY
            
            return RerankResult(
                reranked_indices=reranked_indices,
                reranked_scores=sorted_scores,
                original_scores=original_scores or [0.0] * len(documents),
                provider=self.provider_name,
                model=self.config.model,
                processing_time=processing_time,
                query=query,
                document_count=len(documents),
                metadata={
                    "base_url": self.base_url,
                    "batch_size": batch_size,
                    "temperature": self.config.temperature,
                    "keep_alive": self.keep_alive
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time, len(documents))
            self.metrics.status = RerankerStatus.UNAVAILABLE
            logger.error(f"Ollama reranking failed: {e}")
            raise
    
    async def _score_document(self, query: str, document: str) -> float:
        """Score a single document against the query"""
        try:
            # Construct reranker prompt using Qwen3 format
            prompt = build_prompt(
                query=query,
                document=document,
                instruction=DEFAULT_RERANK_INSTRUCTION,
            )

            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.1,
                        "num_predict": 4,
                        "stop": ["\n", "<|im_end|>", "<|im_start|>"],
                        "logprobs": 5,
                    },
                    "keep_alive": self.keep_alive
                }
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Extract score from response
                score = extract_yes_probability(result)

                return score

        except Exception as e:
            logger.warning(f"Failed to score document: {e}")
            return 0.0
    
    async def health_check(self) -> RerankerStatus:
        """Check Ollama reranker health"""
        try:
            if not self.session:
                return RerankerStatus.UNAVAILABLE
            
            # Check if Ollama is running
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                tags_data = await response.json()
                
                # Check if our reranker model is available
                models = [model["name"] for model in tags_data.get("models", [])]
                model_available = any(self.config.model in model for model in models)
                
                if not model_available:
                    logger.warning(f"Reranker model '{self.config.model}' not found. Available: {models}")
                    self.metrics.status = RerankerStatus.UNAVAILABLE
                    return RerankerStatus.UNAVAILABLE
            
            # Test reranking with a simple example
            try:
                await self._score_document("test query", "test document")
                self.metrics.status = RerankerStatus.HEALTHY
                return RerankerStatus.HEALTHY
            except Exception as e:
                logger.warning(f"Reranker test failed: {e}")
                self.metrics.status = RerankerStatus.DEGRADED
                return RerankerStatus.DEGRADED
            
        except Exception as e:
            logger.warning(f"Ollama reranker health check failed: {e}")
            self.metrics.status = RerankerStatus.UNAVAILABLE
            return RerankerStatus.UNAVAILABLE
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

class CohereRerankerProvider(RerankerProvider):
    """Cohere reranker provider (placeholder for future implementation)"""
    
    def __init__(self, config: Dict[str, Any]):
        reranker_config = RerankerConfig(
            provider="cohere",
            model=config.get('model', 'rerank-english-v2.0'),
            enabled=config.get('enabled', False),
            max_batch_size=config.get('max_batch_size', 96),
            timeout_seconds=config.get('timeout_seconds', 30)
        )
        
        super().__init__("cohere", reranker_config)
        self.api_key = config.get('api_key')
    
    async def initialize(self) -> bool:
        """Initialize Cohere reranker"""
        # Placeholder - future implementation
        logger.info("Cohere reranker not yet implemented")
        return False
    
    async def rerank(self, query: str, documents: List[str], 
                    original_scores: Optional[List[float]] = None) -> RerankResult:
        """Rerank using Cohere API"""
        raise NotImplementedError("Cohere reranker not yet implemented")
    
    async def health_check(self) -> RerankerStatus:
        """Check Cohere API health"""
        return RerankerStatus.UNAVAILABLE

class MultiRerankerManager:
    """Multi-provider reranker manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, RerankerProvider] = {}
        self.default_reranker = config.get('default_reranker', 'ollama_reranker')
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all enabled reranker providers"""
        try:
            logger.info("Initializing multi-reranker manager...")
            
            # Initialize Ollama reranker
            if 'ollama_reranker' in self.config and self.config['ollama_reranker'].get('enabled', False):
                ollama_reranker = OllamaRerankerProvider(self.config['ollama_reranker'])
                if await ollama_reranker.initialize():
                    self.providers['ollama_reranker'] = ollama_reranker
                    logger.info("Ollama reranker initialized")
                else:
                    logger.warning("Ollama reranker initialization failed")
            
            # Initialize Cohere reranker (future)
            if 'cohere_reranker' in self.config and self.config['cohere_reranker'].get('enabled', False):
                cohere_reranker = CohereRerankerProvider(self.config['cohere_reranker'])
                if await cohere_reranker.initialize():
                    self.providers['cohere_reranker'] = cohere_reranker
                    logger.info("Cohere reranker initialized")
            
            if not self.providers:
                logger.warning("No reranker providers could be initialized")
                return False
            
            self._initialized = True
            logger.info(f"Reranker manager initialized with providers: {list(self.providers.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize reranker manager: {e}")
            return False
    
    async def rerank(self, query: str, documents: List[str], 
                    original_scores: Optional[List[float]] = None,
                    reranker_name: Optional[str] = None) -> RerankResult:
        """Rerank documents using specified or default reranker"""
        if not self._initialized:
            raise RuntimeError("Reranker manager not initialized")
        
        # Determine reranker to use
        if reranker_name is None:
            reranker_name = self.default_reranker
        
        if reranker_name not in self.providers:
            raise ValueError(f"Reranker '{reranker_name}' not available. Available: {list(self.providers.keys())}")
        
        provider = self.providers[reranker_name]
        
        try:
            result = await provider.rerank(query, documents, original_scores)
            logger.debug(f"Successfully reranked {len(documents)} documents using {reranker_name}")
            return result
        except Exception as e:
            logger.error(f"Reranking failed with {reranker_name}: {e}")
            raise
    
    async def health_check_all(self) -> Dict[str, RerankerStatus]:
        """Check health of all rerankers"""
        results = {}
        
        for reranker_name, provider in self.providers.items():
            try:
                status = await provider.health_check()
                results[reranker_name] = status
            except Exception as e:
                logger.error(f"Health check failed for {reranker_name}: {e}")
                results[reranker_name] = RerankerStatus.UNAVAILABLE
        
        return results
    
    def get_provider_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all reranker providers"""
        return {name: provider.get_metrics() for name, provider in self.providers.items()}
    
    def get_available_rerankers(self) -> List[str]:
        """Get list of available reranker names"""
        return list(self.providers.keys())
    
    def get_reranker_config(self, reranker_name: str) -> Dict[str, Any]:
        """Get configuration for a specific reranker"""
        if reranker_name in self.providers:
            provider = self.providers[reranker_name]
            return {
                "provider": provider.provider_name,
                "model": provider.config.model,
                "enabled": provider.config.enabled,
                "max_batch_size": provider.config.max_batch_size,
                "timeout_seconds": provider.config.timeout_seconds,
                "temperature": provider.config.temperature,
                "top_k": provider.config.top_k,
                "threshold": provider.config.threshold
            }
        return {}
    
    async def shutdown(self):
        """Shutdown all reranker providers"""
        logger.info("Shutting down reranker providers...")
        
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    logger.error(f"Error closing reranker provider: {e}")
        
        self.providers.clear()
        self._initialized = False

# Global reranker manager instance
reranker_manager: Optional[MultiRerankerManager] = None

async def initialize_reranker_manager(config: Dict[str, Any]) -> bool:
    """Initialize global reranker manager"""
    global reranker_manager
    
    reranker_manager = MultiRerankerManager(config)
    return await reranker_manager.initialize()

def get_reranker_manager() -> MultiRerankerManager:
    """Get global reranker manager instance"""
    if reranker_manager is None:
        raise RuntimeError("Reranker manager not initialized")
    return reranker_manager

# Convenience functions
async def rerank_documents(query: str, documents: List[str], 
                          original_scores: Optional[List[float]] = None,
                          reranker_name: Optional[str] = None) -> RerankResult:
    """Rerank documents using global manager"""
    return await get_reranker_manager().rerank(query, documents, original_scores, reranker_name)
