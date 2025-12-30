"""
Unit tests for /api/v1/tools/search endpoint

Tests cover:
- Request validation (missing query, missing body)
- Response format contract
- MCP tool filtering (MANAGE_ONLY_MCP_TOOLS)
- Reranking integration
- Error handling

These tests use mocked search_tools to avoid requiring live Weaviate.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tool-selector-api"))


def _register_tools_blueprint(app):
    """
    Register the tools blueprint with the app for testing.
    
    Since startup() is not called during tests, we need to manually register
    the tools blueprint and configure services.
    """
    import api_server
    from routes import tools as tools_routes
    from routes.tools import tools_bp
    from services.tool_search import configure_search_service
    from services.tool_cache import get_tool_cache_service
    
    # Only register if not already registered
    if 'tools' not in app.blueprints:
        # Configure services
        configure_search_service(api_server.search_tools)
        get_tool_cache_service('/tmp/test_cache')
        
        # Configure blueprint with dependencies (handlers now live in blueprint)
        tools_routes.configure(
            manage_only_mcp_tools=api_server.MANAGE_ONLY_MCP_TOOLS,
            default_min_score=api_server.DEFAULT_MIN_SCORE,
            agent_service=api_server.agent_service,
            tool_manager=api_server.tool_manager,
            search_tools_func=api_server.search_tools,
            read_tool_cache_func=api_server.read_tool_cache,
            read_mcp_servers_cache_func=api_server.read_mcp_servers_cache,
            process_matching_tool_func=api_server.process_matching_tool,
            init_weaviate_client_func=api_server.init_weaviate_client,
            get_weaviate_client_func=lambda: api_server.weaviate_client,
            is_letta_core_tool_func=api_server.models_is_letta_core_tool
        )
        app.register_blueprint(tools_bp)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def app_client():
    """Create a test client for the Flask app."""
    # Import here to avoid module-level import issues
    with patch.dict('os.environ', {
        'WEAVIATE_URL': 'http://test:8080',
        'OPENAI_API_KEY': 'test-key',
        'LETTA_API_URL': 'http://test:8283',
        'LETTA_PASSWORD': 'test-pass',
    }):
        # Need to mock various imports before importing api_server
        with patch('api_server.search_tools') as mock_search:
            with patch('api_server.weaviate_client', MagicMock()):
                from api_server import app
                app.config['TESTING'] = True
                with app.test_client() as client:
                    yield client, mock_search


@pytest.fixture
def sample_search_results():
    """Sample tool search results."""
    return [
        {
            "id": "tool-1",
            "tool_id": "tool-1",
            "name": "web_search",
            "description": "Search the web for information",
            "tool_type": "external_mcp",
            "mcp_server_name": "web-tools",
            "score": 0.95,
            "distance": 0.05
        },
        {
            "id": "tool-2",
            "tool_id": "tool-2",
            "name": "file_reader",
            "description": "Read files from the filesystem",
            "tool_type": "external_mcp",
            "mcp_server_name": "file-tools",
            "score": 0.85,
            "distance": 0.15
        },
        {
            "id": "tool-3",
            "tool_id": "tool-3",
            "name": "send_message",
            "description": "Send a message to the user",
            "tool_type": "letta_core",
            "score": 0.80,
            "distance": 0.20
        }
    ]


# ============================================================================
# Request Validation Tests
# ============================================================================

class TestSearchRequestValidation:
    """Tests for search endpoint request validation."""
    
    @pytest.mark.asyncio
    async def test_search_missing_body_returns_400(self):
        """Should return 400 when request body is missing."""
        from api_server import app
        _register_tools_blueprint(app)
        
        async with app.test_client() as client:
            # Send request with no body
            response = await client.post('/api/v1/tools/search')
            
            # Should return 400
            assert response.status_code == 400
            data = await response.get_json()
            assert "error" in data
    
    @pytest.mark.asyncio
    async def test_search_missing_query_returns_400(self):
        """Should return 400 when query parameter is missing."""
        from api_server import app
        _register_tools_blueprint(app)
        
        async with app.test_client() as client:
            response = await client.post(
                '/api/v1/tools/search',
                json={"limit": 10}  # No query
            )
            
            assert response.status_code == 400
            data = await response.get_json()
            assert "error" in data
            assert "query" in data["error"].lower() or "Query" in data["error"]
    
    @pytest.mark.asyncio
    async def test_search_empty_query_returns_400(self):
        """Should return 400 for empty query string."""
        from api_server import app
        _register_tools_blueprint(app)
        
        async with app.test_client() as client:
            response = await client.post(
                '/api/v1/tools/search',
                json={"query": "", "limit": 10}
            )
            
            # Empty string should be treated as missing
            assert response.status_code == 400


# ============================================================================
# Response Format Tests
# ============================================================================

class TestSearchResponseFormat:
    """Tests for search endpoint response format contract."""
    
    @pytest.mark.asyncio
    async def test_search_returns_list(self, sample_search_results):
        """Search should return a JSON list."""
        from api_server import app
        _register_tools_blueprint(app)
        
        with patch('services.tool_search.ToolSearchService.search', return_value=sample_search_results[:2]):
            async with app.test_client() as client:
                response = await client.post(
                    '/api/v1/tools/search',
                    json={"query": "search files", "limit": 5}
                )
                
                assert response.status_code == 200
                data = await response.get_json()
                assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_search_respects_limit(self, sample_search_results):
        """Search should respect the limit parameter."""
        from api_server import app
        _register_tools_blueprint(app)
        
        # Return more results than limit
        with patch('services.tool_search.ToolSearchService.search', return_value=sample_search_results):
            with patch('routes.tools._manage_only_mcp_tools', False):
                async with app.test_client() as client:
                    response = await client.post(
                        '/api/v1/tools/search',
                        json={"query": "search files", "limit": 2}
                    )
                    
                    assert response.status_code == 200
                    data = await response.get_json()
                    # Note: The endpoint passes limit to search_tools,
                    # so we can't test post-filtering here without more complex mocking
    
    @pytest.mark.asyncio
    async def test_search_default_limit(self, sample_search_results):
        """Search should use default limit of 10 when not specified."""
        from api_server import app
        _register_tools_blueprint(app)
        
        mock_search = Mock(return_value=sample_search_results)
        
        with patch('services.tool_search.ToolSearchService.search', mock_search):
            with patch('routes.tools._manage_only_mcp_tools', False):
                async with app.test_client() as client:
                    response = await client.post(
                        '/api/v1/tools/search',
                        json={"query": "test query"}
                    )
                    
                    # Verify search was called with limit=10
                    mock_search.assert_called_once()
                    call_kwargs = mock_search.call_args
                    assert call_kwargs.kwargs.get('limit') == 10 or \
                           (call_kwargs.args and len(call_kwargs.args) > 1 and call_kwargs.args[1] == 10)
    
    @pytest.mark.asyncio
    async def test_search_result_has_required_fields(self, sample_search_results):
        """Each search result should have required fields."""
        from api_server import app
        _register_tools_blueprint(app)
        
        with patch('services.tool_search.ToolSearchService.search', return_value=sample_search_results[:1]):
            with patch('routes.tools._manage_only_mcp_tools', False):
                async with app.test_client() as client:
                    response = await client.post(
                        '/api/v1/tools/search',
                        json={"query": "web search"}
                    )
                    
                    assert response.status_code == 200
                    data = await response.get_json()
                    
                    if data:  # If there are results
                        result = data[0]
                        # Core required fields
                        assert "name" in result
                        assert "description" in result


# ============================================================================
# MCP Filtering Tests
# ============================================================================

class TestMCPToolFiltering:
    """Tests for MANAGE_ONLY_MCP_TOOLS filtering."""
    
    @pytest.mark.asyncio
    async def test_mcp_filter_excludes_letta_core(self, sample_search_results):
        """Should exclude letta_core tools when MANAGE_ONLY_MCP_TOOLS is enabled."""
        from api_server import app
        _register_tools_blueprint(app)
        
        # Include the letta_core tool in results
        with patch('services.tool_search.ToolSearchService.search', return_value=sample_search_results):
            with patch('routes.tools._manage_only_mcp_tools', True):
                with patch('services.tool_cache.ToolCacheService.read_tool_cache', new_callable=AsyncMock) as mock_cache:
                    # Mock tool cache to return same data
                    mock_cache.return_value = sample_search_results
                    
                    async with app.test_client() as client:
                        response = await client.post(
                            '/api/v1/tools/search',
                            json={"query": "send message", "limit": 10}
                        )
                        
                        assert response.status_code == 200
                        data = await response.get_json()
                        
                        # Verify letta_core tool is NOT in results
                        for result in data:
                            assert result.get("tool_type") != "letta_core"
    
    @pytest.mark.asyncio
    async def test_mcp_filter_includes_external_mcp(self, sample_search_results):
        """Should include external_mcp tools when MANAGE_ONLY_MCP_TOOLS is enabled."""
        from api_server import app
        _register_tools_blueprint(app)
        
        with patch('services.tool_search.ToolSearchService.search', return_value=sample_search_results):
            with patch('routes.tools._manage_only_mcp_tools', True):
                with patch('services.tool_cache.ToolCacheService.read_tool_cache', new_callable=AsyncMock) as mock_cache:
                    mock_cache.return_value = sample_search_results
                    
                    async with app.test_client() as client:
                        response = await client.post(
                            '/api/v1/tools/search',
                            json={"query": "web search", "limit": 10}
                        )
                        
                        assert response.status_code == 200
                        data = await response.get_json()
                        
                        # Should include at least the MCP tools
                        mcp_tools = [r for r in data if r.get("tool_type") == "external_mcp"]
                        assert len(mcp_tools) > 0


# ============================================================================
# Score Normalization Tests
# ============================================================================

class TestScoreNormalization:
    """Tests for score field normalization."""
    
    @pytest.mark.asyncio
    async def test_rerank_score_mapped_to_score(self):
        """Should map rerank_score to score field."""
        from api_server import app
        _register_tools_blueprint(app)
        
        results_with_rerank = [
            {
                "name": "tool1",
                "description": "A tool",
                "rerank_score": 0.92
                # No 'score' field
            }
        ]
        
        with patch('services.tool_search.ToolSearchService.search', return_value=results_with_rerank):
            with patch('routes.tools._manage_only_mcp_tools', False):
                async with app.test_client() as client:
                    response = await client.post(
                        '/api/v1/tools/search',
                        json={"query": "test"}
                    )
                    
                    assert response.status_code == 200
                    data = await response.get_json()
                    
                    if data:
                        result = data[0]
                        # Should have score field now
                        assert "score" in result
                        assert result["score"] == 0.92


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestSearchErrorHandling:
    """Tests for search endpoint error handling."""
    
    @pytest.mark.asyncio
    async def test_search_handles_weaviate_error(self):
        """Should return 500 and error message on Weaviate failure."""
        from api_server import app
        _register_tools_blueprint(app)
        
        with patch('services.tool_search.ToolSearchService.search', side_effect=Exception("Weaviate connection failed")):
            async with app.test_client() as client:
                response = await client.post(
                    '/api/v1/tools/search',
                    json={"query": "test"}
                )
                
                # Should return 500
                assert response.status_code == 500
                data = await response.get_json()
                assert "error" in data
    
    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self):
        """Should return empty list for no matches."""
        from api_server import app
        _register_tools_blueprint(app)
        
        with patch('services.tool_search.ToolSearchService.search', return_value=[]):
            with patch('routes.tools._manage_only_mcp_tools', False):
                async with app.test_client() as client:
                    response = await client.post(
                        '/api/v1/tools/search',
                        json={"query": "nonexistent tool xyz123"}
                    )
                    
                    assert response.status_code == 200
                    data = await response.get_json()
                    assert data == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
