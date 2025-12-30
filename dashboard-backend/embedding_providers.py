"""
LDTS-34: Multi-provider embedding configuration system

Implements a flexible, configurable embedding provider system that supports
multiple providers (OpenAI, Ollama, HuggingFace) with automatic fallbacks,
cost tracking, and performance monitoring.
"""

import asyncio
import aiohttp
import openai
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

logger = logging.getLogger(__name__)

class ProviderStatus(Enum):
    """Embedding provider status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

@dataclass
class EmbeddingResult:
    """Result from embedding operation"""
    embeddings: List[List[float]]
    provider: str
    model: str
    dimensions: int
    processing_time: float
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderMetrics:
    """Provider performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    average_latency: float = 0.0
    last_request_time: Optional[float] = None
    status: ProviderStatus = ProviderStatus.UNKNOWN
    error_rate: float = 0.0

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.enabled = config.get('enabled', False)
        self.metrics = ProviderMetrics()
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed a list of texts"""
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Check provider health"""
        pass
    
    async def embed_single_text(self, text: str) -> EmbeddingResult:
        """Embed a single text (convenience method)"""
        result = await self.embed_texts([text])
        return EmbeddingResult(
            embeddings=result.embeddings[:1],
            provider=result.provider,
            model=result.model,
            dimensions=result.dimensions,
            processing_time=result.processing_time,
            token_count=result.token_count,
            cost_usd=result.cost_usd,
            metadata=result.metadata
        )
    
    def _update_metrics(self, success: bool, processing_time: float, 
                       token_count: Optional[int] = None, cost: Optional[float] = None):
        """Update provider metrics"""
        self.metrics.total_requests += 1
        self.metrics.total_processing_time += processing_time
        self.metrics.last_request_time = time.time()
        
        if success:
            self.metrics.successful_requests += 1
            if token_count:
                self.metrics.total_tokens += token_count
            if cost:
                self.metrics.total_cost_usd += cost
        else:
            self.metrics.failed_requests += 1
        
        # Update derived metrics
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            self.metrics.average_latency = self.metrics.total_processing_time / self.metrics.total_requests
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics as dictionary"""
        return {
            "provider_name": self.provider_name,
            "enabled": self.enabled,
            "status": self.metrics.status.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "error_rate": self.metrics.error_rate,
            "average_latency_ms": self.metrics.average_latency * 1000,
            "total_tokens": self.metrics.total_tokens,
            "total_cost_usd": self.metrics.total_cost_usd,
            "last_request_time": self.metrics.last_request_time
        }

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        self.client = None
        self.model = config.get('model', 'text-embedding-3-small')
        self.dimensions = config.get('dimensions', 1536)
        self.api_key = config.get('api_key')
        self.batch_size = config.get('batch_size', 100)
        self.timeout = config.get('timeout_seconds', 30)
        self.cost_per_1k_tokens = config.get('cost_tracking', {}).get('cost_per_1k_tokens', 0.00002)
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        try:
            if not self.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            # Initialize the OpenAI client
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Test the connection
            await self.health_check()
            
            self._initialized = True
            logger.info("OpenAI embedding provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            return False
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using OpenAI API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        start_time = time.time()
        
        try:
            # Process in batches if needed
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                total_tokens += response.usage.total_tokens
            
            processing_time = time.time() - start_time
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            
            # Update metrics
            self._update_metrics(True, processing_time, total_tokens, cost)
            self.metrics.status = ProviderStatus.HEALTHY
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                processing_time=processing_time,
                token_count=total_tokens,
                cost_usd=cost,
                metadata={
                    "batch_size": self.batch_size,
                    "batches_processed": (len(texts) + self.batch_size - 1) // self.batch_size
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time)
            self.metrics.status = ProviderStatus.UNAVAILABLE
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def health_check(self) -> ProviderStatus:
        """Check OpenAI API health"""
        try:
            if not self.client:
                return ProviderStatus.UNAVAILABLE
            
            # Test with a simple embedding
            await self.client.embeddings.create(
                model=self.model,
                input=["test"],
                dimensions=min(self.dimensions, 512)  # Use smaller dimensions for health check
            )
            
            self.metrics.status = ProviderStatus.HEALTHY
            return ProviderStatus.HEALTHY
            
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            self.metrics.status = ProviderStatus.UNAVAILABLE
            return ProviderStatus.UNAVAILABLE

class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ollama", config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'nomic-embed-text')
        self.dimensions = config.get('dimensions', 768)
        self.batch_size = config.get('batch_size', 50)
        self.timeout = config.get('timeout_seconds', 60)
        self.keep_alive = config.get('keep_alive', '10m')
        self.session = None
    
    async def initialize(self) -> bool:
        """Initialize Ollama client"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test the connection
            await self.health_check()
            
            self._initialized = True
            logger.info("Ollama embedding provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            if self.session:
                await self.session.close()
            return False
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using Ollama API"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        
        start_time = time.time()
        
        try:
            all_embeddings = []
            
            # Process texts in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                for text in batch:
                    async with self.session.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model,
                            "prompt": text,
                            "keep_alive": self.keep_alive
                        }
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        all_embeddings.append(result["embedding"])
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(True, processing_time)
            self.metrics.status = ProviderStatus.HEALTHY
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                processing_time=processing_time,
                metadata={
                    "base_url": self.base_url,
                    "batch_size": self.batch_size,
                    "keep_alive": self.keep_alive
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time)
            self.metrics.status = ProviderStatus.UNAVAILABLE
            logger.error(f"Ollama embedding failed: {e}")
            raise
    
    async def health_check(self) -> ProviderStatus:
        """Check Ollama API health"""
        try:
            if not self.session:
                return ProviderStatus.UNAVAILABLE
            
            # Check if Ollama is running
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                tags_data = await response.json()
                
                # Check if our model is available
                models = [model["name"] for model in tags_data.get("models", [])]
                if self.model not in models:
                    logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {models}")
                    self.metrics.status = ProviderStatus.DEGRADED
                    return ProviderStatus.DEGRADED
            
            # Test embedding
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": "test",
                    "keep_alive": "1m"
                }
            ) as response:
                response.raise_for_status()
            
            self.metrics.status = ProviderStatus.HEALTHY
            return ProviderStatus.HEALTHY
            
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self.metrics.status = ProviderStatus.UNAVAILABLE
            return ProviderStatus.UNAVAILABLE
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider (placeholder for future implementation)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("huggingface", config)
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.dimensions = config.get('dimensions', 384)
    
    async def initialize(self) -> bool:
        """Initialize HuggingFace provider"""
        # Placeholder - future implementation
        logger.info("HuggingFace provider not yet implemented")
        return False
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts using HuggingFace API"""
        raise NotImplementedError("HuggingFace provider not yet implemented")
    
    async def health_check(self) -> ProviderStatus:
        """Check HuggingFace API health"""
        return ProviderStatus.UNAVAILABLE

class MultiProviderEmbeddingManager:
    """Multi-provider embedding manager with automatic fallbacks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.default_provider = config.get('default_provider', 'openai')
        self.fallback_provider = config.get('fallback_provider', 'ollama')
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all enabled providers"""
        try:
            logger.info("Initializing multi-provider embedding manager...")
            
            # Initialize OpenAI provider
            if self.config.get('openai', {}).get('enabled', False):
                openai_provider = OpenAIEmbeddingProvider(self.config['openai'])
                if await openai_provider.initialize():
                    self.providers['openai'] = openai_provider
                    logger.info("OpenAI provider initialized")
                else:
                    logger.warning("OpenAI provider initialization failed")
            
            # Initialize Ollama provider
            if self.config.get('ollama', {}).get('enabled', False):
                ollama_provider = OllamaEmbeddingProvider(self.config['ollama'])
                if await ollama_provider.initialize():
                    self.providers['ollama'] = ollama_provider
                    logger.info("Ollama provider initialized")
                else:
                    logger.warning("Ollama provider initialization failed")
            
            # Initialize HuggingFace provider (future)
            if self.config.get('huggingface', {}).get('enabled', False):
                hf_provider = HuggingFaceEmbeddingProvider(self.config['huggingface'])
                if await hf_provider.initialize():
                    self.providers['huggingface'] = hf_provider
                    logger.info("HuggingFace provider initialized")
            
            if not self.providers:
                logger.error("No embedding providers could be initialized")
                return False
            
            self._initialized = True
            logger.info(f"Embedding manager initialized with providers: {list(self.providers.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            return False
    
    async def embed_texts(self, texts: List[str], provider_name: Optional[str] = None,
                         use_fallback: bool = True) -> EmbeddingResult:
        """Embed texts using specified or default provider"""
        if not self._initialized:
            raise RuntimeError("Manager not initialized")
        
        # Determine provider to use
        if provider_name is None:
            provider_name = self.default_provider
        
        # Try primary provider
        if provider_name in self.providers:
            try:
                result = await self.providers[provider_name].embed_texts(texts)
                logger.debug(f"Successfully embedded {len(texts)} texts using {provider_name}")
                return result
            except Exception as e:
                logger.warning(f"Primary provider {provider_name} failed: {e}")
                
                # Try fallback if enabled
                if use_fallback and self.fallback_provider and self.fallback_provider != provider_name:
                    if self.fallback_provider in self.providers:
                        try:
                            logger.info(f"Trying fallback provider: {self.fallback_provider}")
                            result = await self.providers[self.fallback_provider].embed_texts(texts)
                            logger.info(f"Fallback successful using {self.fallback_provider}")
                            return result
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider {self.fallback_provider} also failed: {fallback_error}")
                
                raise e
        else:
            raise ValueError(f"Provider '{provider_name}' not available. Available: {list(self.providers.keys())}")
    
    async def embed_single_text(self, text: str, provider_name: Optional[str] = None,
                              use_fallback: bool = True) -> EmbeddingResult:
        """Embed single text (convenience method)"""
        result = await self.embed_texts([text], provider_name, use_fallback)
        return EmbeddingResult(
            embeddings=result.embeddings[:1],
            provider=result.provider,
            model=result.model,
            dimensions=result.dimensions,
            processing_time=result.processing_time,
            token_count=result.token_count,
            cost_usd=result.cost_usd,
            metadata=result.metadata
        )
    
    async def health_check_all(self) -> Dict[str, ProviderStatus]:
        """Check health of all providers"""
        results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                status = await provider.health_check()
                results[provider_name] = status
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                results[provider_name] = ProviderStatus.UNAVAILABLE
        
        return results
    
    def get_provider_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all providers"""
        return {name: provider.get_metrics() for name, provider in self.providers.items()}
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].config
        return {}
    
    async def shutdown(self):
        """Shutdown all providers"""
        logger.info("Shutting down embedding providers...")
        
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    logger.error(f"Error closing provider: {e}")
        
        self.providers.clear()
        self._initialized = False

# Global embedding manager instance
embedding_manager: Optional[MultiProviderEmbeddingManager] = None

async def initialize_embedding_providers(config: Dict[str, Any]) -> bool:
    """Initialize global embedding manager"""
    global embedding_manager
    
    embedding_manager = MultiProviderEmbeddingManager(config)
    return await embedding_manager.initialize()

def get_embedding_manager() -> MultiProviderEmbeddingManager:
    """Get global embedding manager instance"""
    if embedding_manager is None:
        raise RuntimeError("Embedding manager not initialized")
    return embedding_manager

# Convenience functions
async def embed_texts(texts: List[str], provider_name: Optional[str] = None) -> EmbeddingResult:
    """Embed texts using global manager"""
    return await get_embedding_manager().embed_texts(texts, provider_name)

async def embed_text(text: str, provider_name: Optional[str] = None) -> EmbeddingResult:
    """Embed single text using global manager"""
    return await get_embedding_manager().embed_single_text(text, provider_name)
# --- Shim added to provide EmbeddingProviderFactory compatibility with tool-selector-api ---
class EmbeddingProviderFactory:
    """
    Compatibility shim so that 'from embedding_providers import EmbeddingProviderFactory'
    works for modules (e.g. upload_tools_to_weaviate.py) that expect the factory API
    from tool-selector-api.

    This shim delegates actual embedding work to the already-initialized global
    MultiProviderEmbeddingManager (if available). It intentionally provides only
    the minimal surface needed: create(), create_from_env(), list_providers().
    Returned provider instances expose:
        - get_embeddings(List[str]) -> EmbeddingResult
        - get_single_embedding(str) -> List[float]

    If the embedding manager has not been initialized yet, calling embedding
    methods will raise a RuntimeError, which mirrors the existing manager guard.
    """
    class _AdapterProvider:
        def __init__(self, provider_name: str):
            self._provider_name = provider_name
            self.model = None
            self.dimensions = None  # Populated lazily after first call

        @property
        def provider_name(self) -> str:
            return self._provider_name

        async def get_embeddings(self, texts):
            manager = get_embedding_manager()
            result = await manager.embed_texts(texts, provider_name=self._provider_name)
            # Cache model/dimension for potential introspection
            self.model = result.model
            self.dimensions = result.dimensions
            return result

        async def get_single_embedding(self, text: str):
            manager = get_embedding_manager()
            result = await manager.embed_single_text(text, provider_name=self._provider_name)
            # result.embeddings is a list with one embedding
            if result.embeddings:
                self.model = result.model
                self.dimensions = result.dimensions
                return result.embeddings[0]
            return []

        async def close(self):
            # Compatibility no-op (original providers expose close())
            return

    @classmethod
    def create(cls, provider: str, **kwargs):
        # Validate provider availability if manager initialized
        try:
            manager = get_embedding_manager()
            available = manager.get_available_providers()
            if provider not in available:
                raise ValueError(f"Provider '{provider}' not available. Available: {available}")
        except Exception:
            # Manager not initialized yet; defer validation until first call
            pass
        return cls._AdapterProvider(provider)

    @classmethod
    def create_from_env(cls):
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        return cls.create(provider)

    @classmethod
    def list_providers(cls):
        try:
            return get_embedding_manager().get_available_providers()
        except Exception:
            # Manager not initialized; return common defaults
            return ["openai", "ollama", "huggingface"]

# Convenience functions mirroring tool-selector-api API expectations
async def get_embedding_for_text(text: str, provider: Optional[str] = None) -> List[float]:
    p = EmbeddingProviderFactory.create(provider) if provider else EmbeddingProviderFactory.create_from_env()
    try:
        return await p.get_single_embedding(text)
    finally:
        await p.close()

async def get_embeddings_for_texts(texts: List[str], provider: Optional[str] = None):
    p = EmbeddingProviderFactory.create(provider) if provider else EmbeddingProviderFactory.create_from_env()
    try:
        return await p.get_embeddings(texts)
    finally:
        await p.close()

class EmbeddingProviderContext:
    """
    Minimal async context manager for compatibility.
    Usage:
        async with EmbeddingProviderContext("openai") as provider:
            emb = await provider.get_single_embedding("text")
    """
    def __init__(self, provider: Optional[str] = None, **kwargs):
        self._provider_name = provider
        self._provider = None
        self._kwargs = kwargs

    async def __aenter__(self):
        if self._provider_name:
            self._provider = EmbeddingProviderFactory.create(self._provider_name, **self._kwargs)
        else:
            self._provider = EmbeddingProviderFactory.create_from_env()
        return self._provider

    async def __aexit__(self, exc_type, exc, tb):
        if self._provider:
            await self._provider.close()
# --- End EmbeddingProviderFactory shim ---