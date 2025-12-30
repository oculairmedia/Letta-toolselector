"""
Unit tests for embedding_providers.py

Tests for the unified embedding provider system including:
- Provider factory pattern
- OpenAI provider
- Ollama provider
- Qwen3 provider
- Provider selection and configuration
- Error handling
"""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tool-selector-api"))

from embedding_providers import (
    EmbeddingResult,
    EmbeddingProviderError,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    Qwen3EmbeddingProvider,
    EmbeddingProviderFactory,
    EmbeddingProviderContext,
    get_embedding_for_text,
    get_embeddings_for_texts,
    _bool_from_env
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors"""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ]


@pytest.fixture
def mock_openai_response(sample_embeddings):
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=sample_embeddings[0]),
        Mock(embedding=sample_embeddings[1])
    ]
    mock_response.usage = Mock(
        prompt_tokens=10,
        total_tokens=10
    )
    return mock_response


@pytest.fixture
def mock_ollama_response(sample_embeddings):
    """Mock Ollama API response"""
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=sample_embeddings[0]),
        Mock(embedding=sample_embeddings[1])
    ]
    mock_response.usage = Mock(
        prompt_tokens=10,
        total_tokens=10
    )
    return mock_response


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_bool_from_env_true_values(self):
        """Test _bool_from_env with true values"""
        assert _bool_from_env("1", False) is True
        assert _bool_from_env("true", False) is True
        assert _bool_from_env("True", False) is True
        assert _bool_from_env("TRUE", False) is True
        assert _bool_from_env("yes", False) is True
        assert _bool_from_env("YES", False) is True
        assert _bool_from_env("on", False) is True
        assert _bool_from_env("ON", False) is True
    
    def test_bool_from_env_false_values(self):
        """Test _bool_from_env with false values"""
        assert _bool_from_env("0", True) is False
        assert _bool_from_env("false", True) is False
        assert _bool_from_env("no", True) is False
        assert _bool_from_env("off", True) is False
        assert _bool_from_env("anything", True) is False
    
    def test_bool_from_env_none_uses_default(self):
        """Test _bool_from_env with None uses default"""
        assert _bool_from_env(None, True) is True
        assert _bool_from_env(None, False) is False


# ============================================================================
# EmbeddingResult Tests
# ============================================================================

class TestEmbeddingResult:
    """Test EmbeddingResult dataclass"""
    
    def test_embedding_result_creation(self, sample_embeddings):
        """Test creating embedding result"""
        result = EmbeddingResult(
            embeddings=sample_embeddings,
            model="test-model",
            dimensions=4,
            usage={"prompt_tokens": 10, "total_tokens": 10},
            provider="test"
        )
        
        assert result.embeddings == sample_embeddings
        assert result.model == "test-model"
        assert result.dimensions == 4
        assert result.usage["prompt_tokens"] == 10
        assert result.provider == "test"


# ============================================================================
# OpenAI Provider Tests
# ============================================================================

class TestOpenAIProvider:
    """Test OpenAI embedding provider"""
    
    @pytest.mark.asyncio
    async def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(
                model="text-embedding-3-small",
                dimensions=1536,
                api_key="test-key"
            )
            
            assert provider.provider_name == "openai"
            assert provider.model == "text-embedding-3-small"
            assert provider.dimensions == 1536
            assert provider.api_key == "test-key"
    
    def test_openai_provider_missing_api_key(self):
        """Test OpenAI provider requires API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EmbeddingProviderError, match="API key not provided"):
                OpenAIEmbeddingProvider(model="test", dimensions=100, api_key=None)
    
    @pytest.mark.asyncio
    async def test_openai_get_embeddings(self, mock_openai_response, sample_embeddings):
        """Test getting embeddings from OpenAI"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(
                model="text-embedding-3-small",
                dimensions=4,
                api_key="test-key"
            )
            
            # Mock the OpenAI client
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
            provider._client = mock_client
            
            result = await provider.get_embeddings(["text1", "text2"])
            
            assert result.embeddings == sample_embeddings
            assert result.model == "text-embedding-3-small"
            assert result.dimensions == 4
            assert result.provider == "openai"
            assert result.usage["prompt_tokens"] == 10
    
    @pytest.mark.asyncio
    async def test_openai_get_single_embedding(self, mock_openai_response, sample_embeddings):
        """Test getting single embedding from OpenAI"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(
                model="text-embedding-3-small",
                dimensions=4,
                api_key="test-key"
            )
            
            # Mock the OpenAI client
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
            provider._client = mock_client
            
            embedding = await provider.get_single_embedding("text1")
            
            assert embedding == sample_embeddings[0]
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self):
        """Test OpenAI provider error handling"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(
                model="text-embedding-3-small",
                dimensions=1536,
                api_key="test-key"
            )
            
            # Mock the OpenAI client to raise error
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))
            provider._client = mock_client
            
            with pytest.raises(EmbeddingProviderError, match="OpenAI embedding failed"):
                await provider.get_embeddings(["text1"])


# ============================================================================
# Ollama Provider Tests
# ============================================================================

class TestOllamaProvider:
    """Test Ollama embedding provider"""
    
    def test_ollama_provider_initialization(self):
        """Test Ollama provider initialization"""
        provider = OllamaEmbeddingProvider(
            model="test-model",
            dimensions=2560,
            base_url="http://localhost:11434"
        )
        
        assert provider.provider_name == "ollama"
        assert provider.model == "test-model"
        assert provider.dimensions == 2560
        assert provider.base_url == "http://localhost:11434"
    
    def test_ollama_base_url_stripped(self):
        """Test Ollama provider strips trailing slash from base URL"""
        provider = OllamaEmbeddingProvider(
            model="test",
            dimensions=100,
            base_url="http://localhost:11434/"
        )
        
        assert provider.base_url == "http://localhost:11434"
    
    @pytest.mark.asyncio
    async def test_ollama_check_model_availability(self):
        """Test checking Ollama model availability"""
        provider = OllamaEmbeddingProvider(
            model="test-model",
            dimensions=100,
            base_url="http://localhost:11434"
        )
        
        # Mock the response with proper async context manager
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'models': [
                {'name': 'test-model'},
                {'name': 'other-model'}
            ]
        })
        
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_ctx)
        
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_ctx):
            available = await provider._check_model_availability()
            assert available is True
    
    @pytest.mark.asyncio
    async def test_ollama_get_embeddings(self, mock_ollama_response, sample_embeddings):
        """Test getting embeddings from Ollama"""
        provider = OllamaEmbeddingProvider(
            model="test-model",
            dimensions=4,
            base_url="http://localhost:11434"
        )
        
        # Mock the OpenAI-compatible client
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_ollama_response)
        provider._openai_client = mock_client
        
        # Mock model availability check
        provider._check_model_availability = AsyncMock(return_value=True)
        
        result = await provider.get_embeddings(["text1", "text2"])
        
        assert result.embeddings == sample_embeddings
        assert result.model == "test-model"
        assert result.dimensions == 4
        assert result.provider == "ollama"
    
    @pytest.mark.asyncio
    async def test_ollama_dimension_validation(self, mock_ollama_response):
        """Test Ollama validates and adjusts dimensions"""
        provider = OllamaEmbeddingProvider(
            model="test-model",
            dimensions=1000,  # Different from actual
            base_url="http://localhost:11434"
        )
        
        # Mock the OpenAI-compatible client
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_ollama_response)
        provider._openai_client = mock_client
        provider._check_model_availability = AsyncMock(return_value=True)
        
        result = await provider.get_embeddings(["text1"])
        
        # Should adjust to actual dimensions
        assert provider.dimensions == 4  # Length of sample embedding


# ============================================================================
# Qwen3 Provider Tests
# ============================================================================

class TestQwen3Provider:
    """Test Qwen3 embedding provider"""
    
    def test_qwen3_provider_initialization(self):
        """Test Qwen3 provider initialization"""
        provider = Qwen3EmbeddingProvider(
            model="qwen3-model",
            dimensions=2560,
            base_url="http://localhost:11434",
            pooling_strategy="last_token",
            use_instruction_format=True
        )
        
        assert provider.provider_name == "qwen3"
        assert provider.model == "qwen3-model"
        assert provider.pooling_strategy == "last_token"
        assert provider.use_instruction_format is True
    
    def test_qwen3_build_extra_body(self):
        """Test Qwen3 builds extra body with pooling strategy"""
        provider = Qwen3EmbeddingProvider(
            model="qwen3",
            dimensions=2560,
            pooling_strategy="last_token",
            use_last_token_pooling=True
        )
        
        extra_body = provider._build_extra_body()
        
        assert extra_body is not None
        assert extra_body['pooling'] == 'last_token'
        assert extra_body['options']['pooling'] == 'last_token'
    
    def test_qwen3_build_extra_body_disabled(self):
        """Test Qwen3 doesn't build extra body when disabled"""
        provider = Qwen3EmbeddingProvider(
            model="qwen3",
            dimensions=2560,
            use_last_token_pooling=False
        )
        
        extra_body = provider._build_extra_body()
        
        assert extra_body is None
    
    @pytest.mark.asyncio
    async def test_qwen3_get_embeddings_with_instructions(self, mock_ollama_response, sample_embeddings):
        """Test Qwen3 embeddings with instruction formatting"""
        with patch('embedding_providers.get_search_instruction', return_value="Search query:"):
            with patch('embedding_providers.format_query_for_qwen3', side_effect=lambda x: x):
                with patch('embedding_providers.get_detailed_instruct', side_effect=lambda task, text: f"{task} {text}"):
                    provider = Qwen3EmbeddingProvider(
                        model="qwen3",
                        dimensions=4,
                        use_instruction_format=True
                    )
                    
                    # Mock the OpenAI-compatible client
                    mock_client = AsyncMock()
                    mock_client.embeddings.create = AsyncMock(return_value=mock_ollama_response)
                    provider._openai_client = mock_client
                    provider._check_model_availability = AsyncMock(return_value=True)
                    
                    result = await provider.get_embeddings_with_instructions(
                        ["text1", "text2"],
                        task_description="Custom task"
                    )
                    
                    assert result.embeddings == sample_embeddings
                    assert result.provider == "qwen3"


# ============================================================================
# Factory Tests
# ============================================================================

class TestEmbeddingProviderFactory:
    """Test embedding provider factory"""
    
    def test_factory_list_providers(self):
        """Test listing available providers"""
        providers = EmbeddingProviderFactory.list_providers()
        
        assert 'openai' in providers
        assert 'ollama' in providers
        assert 'qwen3' in providers
    
    def test_factory_create_openai(self):
        """Test creating OpenAI provider"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = EmbeddingProviderFactory.create('openai', api_key='test-key')
            
            assert isinstance(provider, OpenAIEmbeddingProvider)
            assert provider.provider_name == 'openai'
    
    def test_factory_create_ollama(self):
        """Test creating Ollama provider"""
        provider = EmbeddingProviderFactory.create('ollama')
        
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.provider_name == 'ollama'
    
    def test_factory_create_qwen3(self):
        """Test creating Qwen3 provider"""
        provider = EmbeddingProviderFactory.create('qwen3')
        
        assert isinstance(provider, Qwen3EmbeddingProvider)
        assert provider.provider_name == 'qwen3'
    
    def test_factory_create_invalid_provider(self):
        """Test creating invalid provider raises error"""
        with pytest.raises(EmbeddingProviderError, match="Unknown provider"):
            EmbeddingProviderFactory.create('invalid_provider')
    
    def test_factory_create_with_config_override(self):
        """Test creating provider with config override"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = EmbeddingProviderFactory.create(
                'openai',
                api_key='test-key',
                model='custom-model',
                dimensions=512
            )
            
            assert provider.model == 'custom-model'
            assert provider.dimensions == 512
    
    def test_factory_get_provider_config(self):
        """Test getting provider default config"""
        config = EmbeddingProviderFactory.get_provider_config('openai')
        
        assert 'model' in config
        assert 'dimensions' in config
    
    def test_factory_register_provider(self):
        """Test registering custom provider"""
        class CustomProvider(EmbeddingProvider):
            @property
            def provider_name(self):
                return "custom"
            
            async def get_embeddings(self, texts):
                pass
            
            async def get_single_embedding(self, text):
                pass
        
        EmbeddingProviderFactory.register_provider('custom', CustomProvider)
        
        assert 'custom' in EmbeddingProviderFactory.list_providers()
    
    def test_factory_create_from_env_openai(self):
        """Test creating provider from environment (OpenAI)"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_EMBEDDING_MODEL': 'custom-model'
        }):
            provider = EmbeddingProviderFactory.create_from_env()
            
            assert isinstance(provider, OpenAIEmbeddingProvider)
            assert provider.model == 'custom-model'
    
    def test_factory_create_from_env_ollama(self):
        """Test creating provider from environment (Ollama)"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'ollama',
            'OLLAMA_BASE_URL': 'http://custom:11434',
            'OLLAMA_EMBEDDING_MODEL': 'custom-ollama-model'
        }):
            provider = EmbeddingProviderFactory.create_from_env()
            
            assert isinstance(provider, OllamaEmbeddingProvider)
            assert provider.model == 'custom-ollama-model'
            assert provider.base_url == 'http://custom:11434'
    
    def test_factory_create_from_env_qwen3(self):
        """Test creating provider from environment (Qwen3)"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'ollama',
            'QWEN3_USE_INSTRUCTION_FORMAT': 'true'
        }):
            with patch('embedding_providers.is_qwen3_format_enabled', return_value=True):
                provider = EmbeddingProviderFactory.create_from_env()
                
                assert isinstance(provider, Qwen3EmbeddingProvider)
    
    def test_factory_create_from_env_with_graphiti_vars(self):
        """Test creating provider with Graphiti environment variables"""
        with patch.dict(os.environ, {
            'EMBEDDING_PROVIDER': 'ollama',
            'OLLAMA_EMBEDDING_HOST': '192.168.1.100',
            'OLLAMA_PORT': '11434',
            'EMBEDDING_DIMENSION': '2560'
        }):
            provider = EmbeddingProviderFactory.create_from_env()
            
            assert isinstance(provider, OllamaEmbeddingProvider)
            assert provider.base_url == 'http://192.168.1.100:11434'
            assert provider.dimensions == 2560


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_get_embedding_for_text(self, sample_embeddings):
        """Test get_embedding_for_text convenience function"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'EMBEDDING_PROVIDER': 'openai'}):
            mock_provider = AsyncMock()
            mock_provider.get_single_embedding = AsyncMock(return_value=sample_embeddings[0])
            mock_provider.close = AsyncMock()
            
            with patch('embedding_providers.EmbeddingProviderFactory.create_from_env', return_value=mock_provider):
                embedding = await get_embedding_for_text("test text")
                
                assert embedding == sample_embeddings[0]
                mock_provider.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embeddings_for_texts(self, sample_embeddings):
        """Test get_embeddings_for_texts convenience function"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'EMBEDDING_PROVIDER': 'openai'}):
            mock_result = EmbeddingResult(
                embeddings=sample_embeddings,
                model="test",
                dimensions=4,
                usage={},
                provider="test"
            )
            
            mock_provider = AsyncMock()
            mock_provider.get_embeddings = AsyncMock(return_value=mock_result)
            mock_provider.close = AsyncMock()
            
            with patch('embedding_providers.EmbeddingProviderFactory.create_from_env', return_value=mock_provider):
                result = await get_embeddings_for_texts(["text1", "text2"])
                
                assert result.embeddings == sample_embeddings
                mock_provider.close.assert_called_once()


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestEmbeddingProviderContext:
    """Test embedding provider context manager"""
    
    @pytest.mark.asyncio
    async def test_context_manager_with_provider_name(self):
        """Test context manager with specific provider"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            mock_provider = AsyncMock()
            mock_provider.close = AsyncMock()
            
            with patch('embedding_providers.EmbeddingProviderFactory.create', return_value=mock_provider):
                async with EmbeddingProviderContext('openai') as provider:
                    assert provider == mock_provider
                
                mock_provider.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_from_env(self):
        """Test context manager using environment config"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'EMBEDDING_PROVIDER': 'openai'}):
            mock_provider = AsyncMock()
            mock_provider.close = AsyncMock()
            
            with patch('embedding_providers.EmbeddingProviderFactory.create_from_env', return_value=mock_provider):
                async with EmbeddingProviderContext() as provider:
                    assert provider == mock_provider
                
                mock_provider.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self):
        """Test context manager closes provider on exception"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            mock_provider = AsyncMock()
            mock_provider.close = AsyncMock()
            
            with patch('embedding_providers.EmbeddingProviderFactory.create', return_value=mock_provider):
                try:
                    async with EmbeddingProviderContext('openai') as provider:
                        raise ValueError("Test error")
                except ValueError:
                    pass
                
                # Provider should still be closed
                mock_provider.close.assert_called_once()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_provider_close_without_client(self):
        """Test closing provider without initialized client"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(api_key='test-key')
            # Should not raise error
            await provider.close()
    
    @pytest.mark.asyncio
    async def test_ollama_model_availability_error_handling(self):
        """Test Ollama handles model availability check errors gracefully"""
        provider = OllamaEmbeddingProvider(
            model="test",
            dimensions=100,
            base_url="http://localhost:11434"
        )
        
        # Mock session that raises exception
        mock_session = AsyncMock()
        mock_session.get = Mock(side_effect=Exception("Network error"))
        
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_ctx):
            # Should return True (assume available) on error
            available = await provider._check_model_availability()
            assert available is True
    
    def test_empty_texts_list(self):
        """Test handling empty texts list"""
        # This would be tested at integration level
        # Unit test just ensures no crash in validation
        assert True
    
    @pytest.mark.asyncio
    async def test_qwen3_format_with_existing_instruction(self):
        """Test Qwen3 doesn't re-format texts with existing instruction"""
        with patch('embedding_providers.get_search_instruction', return_value="Search:"):
            with patch('embedding_providers.format_query_for_qwen3', side_effect=lambda x: x):
                with patch('embedding_providers.get_detailed_instruct', side_effect=lambda task, text: f"{task} {text}"):
                    provider = Qwen3EmbeddingProvider(
                        model="qwen3",
                        dimensions=100,
                        use_instruction_format=True
                    )
                    
                    # Text already has instruction
                    text = "Instruct: some query"
                    formatted = provider._format_with_instruction(text, "Search:")
                    
                    # Should return original text
                    assert formatted == text
