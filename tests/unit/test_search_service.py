"""
Unit tests for the search_service module.

Tests cover:
- Configuration (SearchConfig, RerankerConfig, QueryExpansionConfig)
- Reranker implementations (VLLMReranker, OllamaReranker)
- Search functions (search, search_with_reranking)
- Query expansion
- Utility functions
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tool-selector-api"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_search_service():
    """Reset search_service module state before each test."""
    import search_service
    search_service._weaviate_client = None
    search_service._letta_sdk_client_func = None
    search_service._config = search_service.SearchConfig()
    search_service._reranker = None
    search_service._expand_query_func = None
    search_service._expand_query_with_analysis_func = None
    yield
    # Clean up after test
    search_service._weaviate_client = None
    search_service._letta_sdk_client_func = None
    search_service._config = search_service.SearchConfig()
    search_service._reranker = None


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client."""
    client = MagicMock()
    client.is_ready.return_value = True
    return client


@pytest.fixture
def mock_letta_sdk_client_func():
    """Create a mock Letta SDK client function."""
    mock_client = Mock()
    mock_client.search_tools_with_scores = AsyncMock(return_value=[
        {"name": "tool1", "description": "A tool", "score": 0.9}
    ])
    return Mock(return_value=mock_client)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestSearchConfig:
    """Tests for SearchConfig dataclass."""
    
    def test_default_config(self):
        """Should have sensible defaults."""
        from search_service import SearchConfig
        
        config = SearchConfig()
        
        assert config.provider == "weaviate"
        assert config.reranker.enabled is True
        assert config.expansion.enabled is True
    
    def test_custom_config(self):
        """Should accept custom values."""
        from search_service import SearchConfig, RerankerConfig, QueryExpansionConfig
        
        config = SearchConfig(
            provider="hybrid",
            reranker=RerankerConfig(enabled=False, provider="ollama"),
            expansion=QueryExpansionConfig(enabled=False)
        )
        
        assert config.provider == "hybrid"
        assert config.reranker.enabled is False
        assert config.reranker.provider == "ollama"
        assert config.expansion.enabled is False


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""
    
    def test_default_reranker_config(self):
        """Should have vLLM defaults."""
        from search_service import RerankerConfig
        
        config = RerankerConfig()
        
        assert config.enabled is True
        assert config.provider == "vllm"
        assert "11435" in config.url
        assert config.model == "qwen3-reranker-4b"
        assert config.timeout == 30.0
    
    def test_ollama_config(self):
        """Should support Ollama configuration."""
        from search_service import RerankerConfig
        
        config = RerankerConfig(
            provider="ollama",
            url="http://ollama-adapter:8080/rerank"
        )
        
        assert config.provider == "ollama"
        assert "ollama" in config.url


class TestServiceConfiguration:
    """Tests for search_service.configure function."""
    
    def test_configure_basic(self, mock_weaviate_client):
        """Should set basic configuration."""
        import search_service
        
        search_service.configure(
            weaviate_client=mock_weaviate_client
        )
        
        assert search_service._weaviate_client is mock_weaviate_client
        assert search_service.is_configured() is True
    
    def test_configure_with_letta_sdk(self, mock_letta_sdk_client_func):
        """Should configure with Letta SDK."""
        import search_service
        
        search_service.configure(
            letta_sdk_client_func=mock_letta_sdk_client_func
        )
        
        assert search_service._letta_sdk_client_func is mock_letta_sdk_client_func
        assert search_service.is_configured() is True
    
    def test_configure_with_custom_config(self, mock_weaviate_client):
        """Should accept custom SearchConfig."""
        import search_service
        from search_service import SearchConfig, RerankerConfig
        
        config = SearchConfig(
            provider="hybrid",
            reranker=RerankerConfig(enabled=False)
        )
        
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        
        assert search_service.get_config().provider == "hybrid"
        assert search_service.get_config().reranker.enabled is False
    
    def test_configure_creates_reranker(self, mock_weaviate_client):
        """Should create reranker when enabled."""
        import search_service
        from search_service import SearchConfig, RerankerConfig
        
        config = SearchConfig(
            reranker=RerankerConfig(enabled=True, provider="vllm")
        )
        
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        
        # Reranker should be created
        assert search_service.get_reranker() is not None


# ============================================================================
# Reranker Tests
# ============================================================================

class TestVLLMReranker:
    """Tests for VLLMReranker implementation."""
    
    @pytest.mark.asyncio
    async def test_rerank_success(self):
        """Should rerank documents successfully."""
        from search_service import VLLMReranker
        
        reranker = VLLMReranker(url="http://test:8080/rerank")
        
        # Mock the httpx response
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.7}
                ]
            }
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            results = await reranker.rerank(
                query="test query",
                documents=["doc1", "doc2"],
                top_k=2
            )
        
        assert len(results) == 2
        assert results[0].index == 1
        assert results[0].relevance_score == 0.9
    
    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Should return empty list for empty documents."""
        from search_service import VLLMReranker
        
        reranker = VLLMReranker()
        
        results = await reranker.rerank("query", [], 10)
        
        assert results == []


class TestOllamaReranker:
    """Tests for OllamaReranker implementation."""
    
    @pytest.mark.asyncio
    async def test_rerank_success(self):
        """Should rerank documents successfully."""
        from search_service import OllamaReranker
        
        reranker = OllamaReranker(url="http://test:8080/rerank")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"index": 0, "relevance_score": 0.85}
                ]
            }
            mock_response.raise_for_status = Mock()
            
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            results = await reranker.rerank(
                query="test",
                documents=["single doc"],
                top_k=1
            )
        
        assert len(results) == 1
        assert results[0].relevance_score == 0.85


class TestCreateReranker:
    """Tests for create_reranker factory function."""
    
    def test_create_vllm_reranker(self):
        """Should create VLLMReranker for vllm provider."""
        from search_service import create_reranker, RerankerConfig, VLLMReranker
        
        config = RerankerConfig(provider="vllm")
        reranker = create_reranker(config)
        
        assert isinstance(reranker, VLLMReranker)
    
    def test_create_ollama_reranker(self):
        """Should create OllamaReranker for ollama provider."""
        from search_service import create_reranker, RerankerConfig, OllamaReranker
        
        config = RerankerConfig(provider="ollama")
        reranker = create_reranker(config)
        
        assert isinstance(reranker, OllamaReranker)
    
    def test_create_unknown_provider(self):
        """Should raise for unknown provider."""
        from search_service import create_reranker, RerankerConfig
        
        config = RerankerConfig(provider="unknown")
        
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            create_reranker(config)


# ============================================================================
# Search Function Tests
# ============================================================================

class TestSearch:
    """Tests for search_service.search function."""
    
    @pytest.mark.asyncio
    async def test_search_weaviate_provider(self, mock_weaviate_client):
        """Should search via Weaviate when configured."""
        import search_service
        from search_service import SearchConfig
        
        config = SearchConfig(provider="weaviate")
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        
        # Mock the Weaviate search
        with patch('search_service.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = [{"name": "tool1", "score": 0.9}]
            
            results = await search_service.search("test query", limit=5)
        
        assert len(results) == 1
        assert results[0]["name"] == "tool1"
    
    @pytest.mark.asyncio
    async def test_search_letta_provider(self, mock_letta_sdk_client_func):
        """Should search via Letta SDK when configured."""
        import search_service
        from search_service import SearchConfig
        
        config = SearchConfig(provider="letta")
        search_service.configure(
            letta_sdk_client_func=mock_letta_sdk_client_func,
            config=config
        )
        
        results = await search_service.search("test query", limit=5)
        
        assert len(results) == 1
        assert results[0]["name"] == "tool1"
    
    @pytest.mark.asyncio
    async def test_search_hybrid_fallback(self, mock_weaviate_client):
        """Should fallback to Weaviate when Letta fails in hybrid mode."""
        import search_service
        from search_service import SearchConfig
        
        # Create a failing Letta SDK
        failing_sdk_client = Mock()
        failing_sdk_client.search_tools_with_scores = AsyncMock(
            side_effect=RuntimeError("Letta failed")
        )
        failing_sdk_func = Mock(return_value=failing_sdk_client)
        
        config = SearchConfig(provider="hybrid")
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            letta_sdk_client_func=failing_sdk_func,
            config=config
        )
        
        with patch('search_service.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = [{"name": "weaviate_tool", "score": 0.8}]
            
            results = await search_service.search("test query")
        
        # Should get Weaviate results after Letta failure
        assert len(results) == 1
        assert results[0]["name"] == "weaviate_tool"


class TestQueryExpansion:
    """Tests for query expansion functionality."""
    
    def test_expand_query_disabled(self, mock_weaviate_client):
        """Should return original query when expansion disabled."""
        import search_service
        from search_service import SearchConfig, QueryExpansionConfig
        
        config = SearchConfig(
            expansion=QueryExpansionConfig(enabled=False)
        )
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        
        result = search_service.expand_query("create book")
        
        assert result == "create book"
    
    def test_expand_query_no_function(self, mock_weaviate_client):
        """Should return original query when no expansion function."""
        import search_service
        from search_service import SearchConfig, QueryExpansionConfig
        
        config = SearchConfig(
            expansion=QueryExpansionConfig(enabled=True)
        )
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        # Expansion function not loaded
        search_service._expand_query_func = None
        
        result = search_service.expand_query("create book")
        
        assert result == "create book"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestFormatToolForReranking:
    """Tests for format_tool_for_reranking utility."""
    
    def test_basic_format(self):
        """Should format tool with basic fields."""
        from search_service import format_tool_for_reranking
        
        tool = {
            "name": "create_document",
            "description": "Creates a new document in the system"
        }
        
        result = format_tool_for_reranking(tool)
        
        assert "Tool Name: create_document" in result
        assert "Description: Creates a new document" in result
        assert "Actions: create" in result
    
    def test_format_with_mcp_server(self):
        """Should include service name from mcp_server_name."""
        from search_service import format_tool_for_reranking
        
        tool = {
            "name": "huly_create_issue",
            "description": "Create Huly issue",
            "mcp_server_name": "huly"
        }
        
        result = format_tool_for_reranking(tool)
        
        assert "Service: huly" in result
    
    def test_format_with_mcp_tag(self):
        """Should extract service from mcp: tag."""
        from search_service import format_tool_for_reranking
        
        tool = {
            "name": "ghost_publish",
            "description": "Publish to Ghost",
            "tags": ["mcp:ghost", "blogging"]
        }
        
        result = format_tool_for_reranking(tool)
        
        assert "Service: ghost" in result
    
    def test_format_truncates_long_description(self):
        """Should truncate descriptions over 500 chars."""
        from search_service import format_tool_for_reranking
        
        long_desc = "A" * 600
        tool = {
            "name": "test_tool",
            "description": long_desc
        }
        
        result = format_tool_for_reranking(tool)
        
        # Should be truncated with ellipsis
        assert "..." in result
        assert len(result) < len(long_desc) + 100


class TestRerankerTest:
    """Tests for test_reranker utility function."""
    
    @pytest.mark.asyncio
    async def test_reranker_disabled(self, mock_weaviate_client):
        """Should report disabled when reranking is off."""
        import search_service
        from search_service import SearchConfig, RerankerConfig
        
        config = SearchConfig(
            reranker=RerankerConfig(enabled=False)
        )
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        
        status = await search_service.test_reranker()
        
        assert status["reranking_enabled"] is False
        assert status["test_passed"] is False
        assert "disabled" in status["error"]
    
    @pytest.mark.asyncio
    async def test_reranker_not_initialized(self, mock_weaviate_client):
        """Should report error when reranker not initialized."""
        import search_service
        from search_service import SearchConfig, RerankerConfig
        
        config = SearchConfig(
            reranker=RerankerConfig(enabled=True)
        )
        search_service.configure(
            weaviate_client=mock_weaviate_client,
            config=config
        )
        # Force reranker to None
        search_service._reranker = None
        
        status = await search_service.test_reranker()
        
        assert status["test_passed"] is False
        assert "not initialized" in status["error"]


# ============================================================================
# RerankResult Tests
# ============================================================================

class TestRerankResult:
    """Tests for RerankResult dataclass."""
    
    def test_create_rerank_result(self):
        """Should create RerankResult with all fields."""
        from search_service import RerankResult
        
        result = RerankResult(
            index=0,
            relevance_score=0.95,
            document="Test document"
        )
        
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document == "Test document"
    
    def test_create_rerank_result_minimal(self):
        """Should create RerankResult with minimal fields."""
        from search_service import RerankResult
        
        result = RerankResult(index=1, relevance_score=0.5)
        
        assert result.index == 1
        assert result.relevance_score == 0.5
        assert result.document is None
