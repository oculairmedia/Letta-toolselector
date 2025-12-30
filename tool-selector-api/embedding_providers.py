"""
Unified Embedding Provider System for Letta Tool Selector

This module implements a factory pattern for embedding providers, supporting both
OpenAI and Ollama endpoints with consistent interfaces for seamless switching.

Usage:
    # Use OpenAI (default)
    provider = EmbeddingProviderFactory.create('openai')
    embeddings = await provider.get_embeddings(['text1', 'text2'])
    
    # Use Ollama
    provider = EmbeddingProviderFactory.create('ollama')
    embeddings = await provider.get_embeddings(['text1', 'text2'])
    
    # Auto-detect from environment
    provider = EmbeddingProviderFactory.create_from_env()
    embeddings = await provider.get_embeddings(['text1', 'text2'])
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import aiohttp
from openai import AsyncOpenAI
from embedding_config import (
    OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIMENSION,
    OLLAMA_EMBEDDING_MODEL, OLLAMA_EMBEDDING_DIMENSION, OLLAMA_BASE_URL
)

from specialized_embedding import (
    is_qwen3_format_enabled,
    get_search_instruction,
    format_query_for_qwen3,
    get_detailed_instruct,
    QWEN3_LAST_TOKEN_POOLING,
)

logger = logging.getLogger(__name__)


def _bool_from_env(value: Optional[str], default: bool) -> bool:
    """Convert environment string flags to boolean values."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class EmbeddingResult:
    """Standardized embedding result structure."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    usage: Dict[str, Any]
    provider: str


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""
    pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model: str, dimensions: int):
        self.model = model
        self.dimensions = dimensions
        self._client = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Get embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass
    
    async def close(self):
        """Close the provider and cleanup resources."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(
        self, 
        model: str = OPENAI_EMBEDDING_MODEL,
        dimensions: int = OPENAI_EMBEDDING_DIMENSION,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__(model, dimensions)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url
        
        if not self.api_key:
            raise EmbeddingProviderError("OpenAI API key not provided")
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            client_kwargs = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            self._client = AsyncOpenAI(**client_kwargs)
        return self._client
    
    async def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Get embeddings using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                provider=self.provider_name
            )
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingProviderError(f"OpenAI embedding failed: {e}")
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        result = await self.get_embeddings([text])
        return result.embeddings[0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider implementation with OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str = OLLAMA_EMBEDDING_MODEL,
        dimensions: int = OLLAMA_EMBEDDING_DIMENSION,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 120
    ):
        super().__init__(model, dimensions)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Use AsyncOpenAI client pointing to Ollama's OpenAI-compatible endpoint
        self._openai_client = None
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of Ollama OpenAI-compatible client."""
        if self._openai_client is None:
            # Ollama's OpenAI-compatible endpoint
            openai_url = f"{self.base_url}/v1"
            self._openai_client = AsyncOpenAI(
                base_url=openai_url,
                api_key="ollama",  # Ollama doesn't require real API key but AsyncOpenAI needs one
                timeout=self.timeout
            )
        return self._openai_client
    
    async def _check_model_availability(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        return self.model in models
            return False
        except Exception as e:
            logger.warning(f"Could not check Ollama model availability: {e}")
            return True  # Assume available and let embedding call fail if not

    def _build_extra_body(self) -> Optional[Dict[str, Any]]:
        """Hook for subclasses to provide provider-specific request options."""
        return None

    async def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Get embeddings using Ollama's OpenAI-compatible API."""
        try:
            # Check if model is available
            if not await self._check_model_availability():
                logger.warning(f"Model {self.model} might not be available in Ollama")
            
            # Use OpenAI-compatible client
            extra_body = self._build_extra_body()
            request_kwargs = {}
            if extra_body:
                request_kwargs['extra_body'] = extra_body
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                **request_kwargs
            )

            embeddings = [item.embedding for item in response.data]
            
            # Validate dimensions match expectation
            if embeddings and len(embeddings[0]) != self.dimensions:
                actual_dims = len(embeddings[0])
                logger.warning(f"Expected {self.dimensions} dimensions, got {actual_dims}")
                self.dimensions = actual_dims
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                usage={
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', len(' '.join(texts).split())),
                    'total_tokens': getattr(response.usage, 'total_tokens', len(' '.join(texts).split()))
                },
                provider=self.provider_name
            )
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise EmbeddingProviderError(f"Ollama embedding failed: {e}")
    
    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        result = await self.get_embeddings([text])
        return result.embeddings[0]
    
    async def close(self):
        """Close the provider and cleanup resources."""
        if self._openai_client and hasattr(self._openai_client, 'close'):
            await self._openai_client.close()
        await super().close()


class Qwen3EmbeddingProvider(OllamaEmbeddingProvider):
    """Qwen3-specific embedding provider with proper instruction formatting."""

    def __init__(
        self,
        model: str = OLLAMA_EMBEDDING_MODEL,
        dimensions: int = OLLAMA_EMBEDDING_DIMENSION,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 120,
        pooling_strategy: Optional[str] = None,
        use_instruction_format: bool = True,
        use_last_token_pooling: bool = QWEN3_LAST_TOKEN_POOLING,
    ):
        super().__init__(model=model, dimensions=dimensions, base_url=base_url, timeout=timeout)
        env_pooling = os.getenv('QWEN3_POOLING_STRATEGY')
        self.pooling_strategy = pooling_strategy or env_pooling or 'last_token'
        self.use_instruction_format = _bool_from_env(os.getenv('QWEN3_USE_INSTRUCTION_FORMAT'), use_instruction_format)
        self.use_last_token_pooling = _bool_from_env(os.getenv('QWEN3_LAST_TOKEN_POOLING'), use_last_token_pooling)
        self.default_task_description = get_search_instruction()

    @property
    def provider_name(self) -> str:
        return 'qwen3'

    def _build_extra_body(self) -> Optional[Dict[str, Any]]:
        if not self.use_last_token_pooling or not self.pooling_strategy:
            return None
        return {
            'options': {'pooling': self.pooling_strategy},
            'pooling': self.pooling_strategy
        }

    def _format_with_instruction(self, text: str, task_description: str) -> str:
        if text and text.lstrip().lower().startswith('instruct:'):
            return text
        cleaned = format_query_for_qwen3(text)
        return get_detailed_instruct(task_description, cleaned)

    async def get_embeddings_with_instructions(
        self,
        texts: List[str],
        task_description: Optional[str] = None
    ) -> EmbeddingResult:
        if not self.use_instruction_format:
            return await self.get_embeddings(texts)
        instruction = task_description or self.default_task_description
        formatted_texts = [self._format_with_instruction(text, instruction) for text in texts]
        return await super().get_embeddings(formatted_texts)

    async def get_single_embedding(
        self,
        text: str,
        task_description: Optional[str] = None
    ) -> List[float]:
        if self.use_instruction_format and task_description is not None:
            result = await self.get_embeddings_with_instructions([text], task_description=task_description)
            return result.embeddings[0] if result.embeddings else []
        result = await super().get_embeddings([text])
        return result.embeddings[0] if result.embeddings else []


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    # Registry of available providers
    _providers = {
        'openai': OpenAIEmbeddingProvider,
        'ollama': OllamaEmbeddingProvider,
        'qwen3': Qwen3EmbeddingProvider,
    }
    
    # Default configurations for each provider
    _default_configs = {
        'openai': {
            'model': OPENAI_EMBEDDING_MODEL,
            'dimensions': OPENAI_EMBEDDING_DIMENSION
        },
        'ollama': {
            'model': OLLAMA_EMBEDDING_MODEL,  # Graphiti-compatible Qwen3-Embedding-4B model
            'dimensions': OLLAMA_EMBEDDING_DIMENSION,  # 2560 dimensions for Qwen3-Embedding-4B
            'base_url': OLLAMA_BASE_URL
        },
        'qwen3': {
            'model': OLLAMA_EMBEDDING_MODEL,
            'dimensions': OLLAMA_EMBEDDING_DIMENSION,
            'base_url': OLLAMA_BASE_URL,
            'pooling_strategy': os.getenv('QWEN3_POOLING_STRATEGY', 'last_token'),
            'use_instruction_format': True,
            'use_last_token_pooling': QWEN3_LAST_TOKEN_POOLING,
        }
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new embedding provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create(
        cls, 
        provider: str, 
        **kwargs
    ) -> EmbeddingProvider:
        """Create an embedding provider by name."""
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise EmbeddingProviderError(
                f"Unknown provider '{provider}'. Available: {available}"
            )
        
        provider_class = cls._providers[provider]
        
        # Merge default config with provided kwargs
        config = cls._default_configs.get(provider, {}).copy()
        config.update(kwargs)
        
        return provider_class(**config)
    
    @classmethod
    def create_from_env(cls) -> EmbeddingProvider:
        """Create embedding provider based on environment variables."""
        provider_name = os.getenv('EMBEDDING_PROVIDER', 'openai').lower()

        if provider_name == 'ollama' and is_qwen3_format_enabled():
            provider_name = 'qwen3'

        config_overrides = {}

        if provider_name == 'openai':
            if os.getenv('OPENAI_API_KEY'):
                config_overrides['api_key'] = os.getenv('OPENAI_API_KEY')
            if os.getenv('OPENAI_BASE_URL'):
                config_overrides['base_url'] = os.getenv('OPENAI_BASE_URL')
            if os.getenv('OPENAI_EMBEDDING_MODEL'):
                config_overrides['model'] = os.getenv('OPENAI_EMBEDDING_MODEL')

        elif provider_name in {'ollama', 'qwen3'}:
            if os.getenv('OLLAMA_EMBEDDING_HOST'):
                # Support Graphiti's OLLAMA_EMBEDDING_HOST variable
                host = os.getenv('OLLAMA_EMBEDDING_HOST')
                port = os.getenv('OLLAMA_PORT', '11434')
                config_overrides['base_url'] = f"http://{host}:{port}"
            elif os.getenv('OLLAMA_BASE_URL'):
                config_overrides['base_url'] = os.getenv('OLLAMA_BASE_URL')
            if os.getenv('OLLAMA_EMBEDDING_MODEL'):
                config_overrides['model'] = os.getenv('OLLAMA_EMBEDDING_MODEL')
            if os.getenv('EMBEDDING_DIMENSION'):
                # Support Graphiti's EMBEDDING_DIMENSION variable
                config_overrides['dimensions'] = int(os.getenv('EMBEDDING_DIMENSION'))
            elif os.getenv('OLLAMA_EMBEDDING_DIMENSIONS'):
                config_overrides['dimensions'] = int(os.getenv('OLLAMA_EMBEDDING_DIMENSIONS'))
            if provider_name == 'qwen3':
                if os.getenv('QWEN3_POOLING_STRATEGY'):
                    config_overrides['pooling_strategy'] = os.getenv('QWEN3_POOLING_STRATEGY')
                if os.getenv('QWEN3_USE_INSTRUCTION_FORMAT'):
                    config_overrides['use_instruction_format'] = _bool_from_env(
                        os.getenv('QWEN3_USE_INSTRUCTION_FORMAT'), True
                    )
                if os.getenv('QWEN3_LAST_TOKEN_POOLING'):
                    config_overrides['use_last_token_pooling'] = _bool_from_env(
                        os.getenv('QWEN3_LAST_TOKEN_POOLING'), QWEN3_LAST_TOKEN_POOLING
                    )
        
        return cls.create(provider_name, **config_overrides)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """Get default configuration for a provider."""
        return cls._default_configs.get(provider, {}).copy()


# Convenience functions for backward compatibility and ease of use

async def get_embedding_for_text(
    text: str, 
    provider: Optional[str] = None
) -> List[float]:
    """Get embedding for a single text using the specified or default provider."""
    if provider is None:
        embedding_provider = EmbeddingProviderFactory.create_from_env()
    else:
        embedding_provider = EmbeddingProviderFactory.create(provider)
    
    try:
        return await embedding_provider.get_single_embedding(text)
    finally:
        await embedding_provider.close()


async def get_embeddings_for_texts(
    texts: List[str], 
    provider: Optional[str] = None
) -> EmbeddingResult:
    """Get embeddings for multiple texts using the specified or default provider."""
    if provider is None:
        embedding_provider = EmbeddingProviderFactory.create_from_env()
    else:
        embedding_provider = EmbeddingProviderFactory.create(provider)
    
    try:
        return await embedding_provider.get_embeddings(texts)
    finally:
        await embedding_provider.close()


# Context manager for embedding provider management
class EmbeddingProviderContext:
    """Context manager for embedding provider lifecycle management."""
    
    def __init__(self, provider: Optional[str] = None, **kwargs):
        self.provider_name = provider
        self.kwargs = kwargs
        self.provider = None
    
    async def __aenter__(self) -> EmbeddingProvider:
        if self.provider_name is None:
            self.provider = EmbeddingProviderFactory.create_from_env()
        else:
            self.provider = EmbeddingProviderFactory.create(self.provider_name, **self.kwargs)
        return self.provider
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.provider:
            await self.provider.close()


# Example usage and testing functions
async def test_providers():
    """Test function to demonstrate provider usage."""
    test_texts = [
        "This is a test sentence for embeddings.",
        "Another test sentence with different content."
    ]
    
    for provider_name in EmbeddingProviderFactory.list_providers():
        try:
            print(f"\n--- Testing {provider_name} provider ---")
            
            async with EmbeddingProviderContext(provider_name) as provider:
                result = await provider.get_embeddings(test_texts)
                print(f"Model: {result.model}")
                print(f"Dimensions: {result.dimensions}")
                print(f"Embeddings count: {len(result.embeddings)}")
                print(f"Provider: {result.provider}")
                print(f"Usage: {result.usage}")
                
        except Exception as e:
            print(f"Error testing {provider_name}: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    asyncio.run(test_providers())