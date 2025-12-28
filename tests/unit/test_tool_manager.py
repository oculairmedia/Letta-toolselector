"""
Unit tests for the tool_manager module.

Tests cover:
- attach_tool function
- detach_tool function
- process_tools batch operation
- configure function
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lettaaugment-source"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_agent_id():
    """Test agent ID."""
    return "agent-test-12345"


@pytest.fixture
def sample_tool():
    """Sample tool data."""
    return {
        "id": "tool-abc123",
        "tool_id": "tool-abc123",
        "name": "sample_tool",
        "description": "A sample tool",
        "tool_type": "external_mcp",
        "distance": 0.1
    }


@pytest.fixture
def mock_http_response_success():
    """Create a mock successful HTTP response."""
    mock_response = MagicMock()
    mock_response.status = 200
    
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_response
    async_cm.__aexit__.return_value = None
    
    return async_cm


@pytest.fixture
def mock_http_session(mock_http_response_success):
    """Create a mock HTTP session."""
    session = MagicMock()
    session.patch.return_value = mock_http_response_success
    return session


# ============================================================================
# Configuration Tests
# ============================================================================

class TestToolManagerConfiguration:
    """Tests for tool_manager.configure function."""
    
    def test_configure_sets_module_state(self, mock_http_session):
        """Should set module-level state variables."""
        import tool_manager
        
        tool_manager.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={"Authorization": "Bearer test"},
            use_letta_sdk=False,
            get_letta_sdk_client_func=None
        )
        
        assert tool_manager._http_session is mock_http_session
        assert tool_manager._letta_url == "http://test:8283"
        assert tool_manager._use_letta_sdk is False
    
    def test_configure_with_sdk(self, mock_http_session):
        """Should configure SDK mode."""
        import tool_manager
        
        mock_sdk_func = Mock()
        
        tool_manager.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=True,
            get_letta_sdk_client_func=mock_sdk_func
        )
        
        assert tool_manager._use_letta_sdk is True
        assert tool_manager._get_letta_sdk_client is mock_sdk_func


# ============================================================================
# Attach Tool Tests
# ============================================================================

class TestAttachToolDirect:
    """Tests for tool_manager.attach_tool function directly."""
    
    @pytest.mark.asyncio
    async def test_attach_tool_success(self, test_agent_id, sample_tool):
        """Should successfully attach a tool via HTTP."""
        import tool_manager
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.status = 200
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool manager
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={"Authorization": "Bearer test"},
            use_letta_sdk=False
        )
        
        result = await tool_manager.attach_tool(test_agent_id, sample_tool)
        
        assert result["success"] is True
        assert result["tool_id"] == sample_tool["id"]
        assert "match_score" in result
    
    @pytest.mark.asyncio
    async def test_attach_tool_missing_id(self, test_agent_id):
        """Should fail when tool has no ID."""
        import tool_manager
        
        tool_without_id = {"name": "no_id_tool"}
        
        tool_manager.configure(
            http_session=MagicMock(),
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.attach_tool(test_agent_id, tool_without_id)
        
        assert result["success"] is False
        assert "No tool ID" in result["error"]
    
    @pytest.mark.asyncio
    async def test_attach_tool_no_session(self, test_agent_id, sample_tool):
        """Should fail gracefully when no session configured."""
        import tool_manager
        
        tool_manager.configure(
            http_session=None,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.attach_tool(test_agent_id, sample_tool)
        
        assert result["success"] is False
        assert "HTTP session not available" in result["error"]


# ============================================================================
# Detach Tool Tests
# ============================================================================

class TestDetachToolDirect:
    """Tests for tool_manager.detach_tool function directly."""
    
    @pytest.mark.asyncio
    async def test_detach_tool_success(self, test_agent_id):
        """Should successfully detach a tool."""
        import tool_manager
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "detached"})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.detach_tool(test_agent_id, "tool-123")
        
        assert result["success"] is True
        assert result["tool_id"] == "tool-123"
    
    @pytest.mark.asyncio
    async def test_detach_tool_404_is_success(self, test_agent_id):
        """Should treat 404 as success (already detached)."""
        import tool_manager
        
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"error": "not found"})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.detach_tool(test_agent_id, "tool-123")
        
        assert result["success"] is True
        assert "warning" in result
    
    @pytest.mark.asyncio
    async def test_detach_tool_no_session(self, test_agent_id):
        """Should fail gracefully when no session."""
        import tool_manager
        
        tool_manager.configure(
            http_session=None,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.detach_tool(test_agent_id, "tool-123")
        
        assert result["success"] is False
        assert "HTTP session not available" in result["error"]


# ============================================================================
# Process Tools Tests
# ============================================================================

class TestProcessToolsDirect:
    """Tests for tool_manager.process_tools function directly."""
    
    @pytest.mark.asyncio
    async def test_process_tools_batch_operations(self, test_agent_id):
        """Should perform batch attach/detach operations."""
        import tool_manager
        
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        current_tools = [
            {"id": "tool-old-1", "name": "old_tool_1", "tool_type": "external_mcp"},
            {"id": "tool-old-2", "name": "old_tool_2", "tool_type": "external_mcp"},
        ]
        
        new_tools = [
            {"id": "tool-new-1", "name": "new_tool_1", "tool_type": "external_mcp"},
        ]
        
        result = await tool_manager.process_tools(
            test_agent_id,
            current_tools,
            new_tools,
            keep_tools=[]
        )
        
        assert "detached_tools" in result
        assert "successful_attachments" in result
        assert "failed_attachments" in result
    
    @pytest.mark.asyncio
    async def test_process_tools_respects_keep_list(self, test_agent_id):
        """Should not detach tools in keep list."""
        import tool_manager
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        current_tools = [
            {"id": "tool-keep", "name": "keep_me", "tool_type": "external_mcp"},
            {"id": "tool-remove", "name": "remove_me", "tool_type": "external_mcp"},
        ]
        
        new_tools = [
            {"id": "tool-new", "name": "new_tool", "tool_type": "external_mcp"},
        ]
        
        result = await tool_manager.process_tools(
            test_agent_id,
            current_tools,
            new_tools,
            keep_tools=["tool-keep"]
        )
        
        # tool-keep should NOT be in detached list
        assert "tool-keep" not in result["detached_tools"]
    
    @pytest.mark.asyncio
    async def test_process_tools_no_session_returns_error(self, test_agent_id):
        """Should return error structure when no session."""
        import tool_manager
        
        tool_manager.configure(
            http_session=None,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        result = await tool_manager.process_tools(
            test_agent_id,
            [{"id": "tool-1"}],
            [{"id": "tool-2"}],
            keep_tools=[]
        )
        
        assert "error" in result
        assert result["error"] == "HTTP session not available"


# ============================================================================
# Fetch Agent Tools Tests
# ============================================================================

class TestFetchAgentToolsDirect:
    """Tests for tool_manager.fetch_agent_tools function directly."""
    
    @pytest.mark.asyncio
    async def test_fetch_agent_tools_success(self, test_agent_id):
        """Should successfully fetch agent tools."""
        import tool_manager
        
        expected_tools = [
            {"id": "tool-1", "name": "tool_one", "tool_type": "external_mcp"},
            {"id": "tool-2", "name": "tool_two", "tool_type": "letta_core"},
        ]
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=expected_tools)
        mock_response.raise_for_status = MagicMock()
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.get.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={"Authorization": "Bearer test"},
            use_letta_sdk=False
        )
        
        result = await tool_manager.fetch_agent_tools(test_agent_id)
        
        assert result == expected_tools
        assert len(result) == 2
        mock_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_agent_tools_no_session(self, test_agent_id):
        """Should raise ConnectionError when no session configured."""
        import tool_manager
        
        tool_manager.configure(
            http_session=None,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        with pytest.raises(ConnectionError) as exc_info:
            await tool_manager.fetch_agent_tools(test_agent_id)
        
        assert "HTTP session not available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_agent_tools_api_error(self, test_agent_id):
        """Should propagate exceptions from API errors."""
        import tool_manager
        from aiohttp import ClientResponseError
        
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.raise_for_status = MagicMock(side_effect=Exception("Server Error"))
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.get.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=False
        )
        
        with pytest.raises(Exception) as exc_info:
            await tool_manager.fetch_agent_tools(test_agent_id)
        
        assert "Server Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_fetch_agent_tools_correct_url(self, test_agent_id):
        """Should call correct URL endpoint."""
        import tool_manager
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])
        mock_response.raise_for_status = MagicMock()
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.get.return_value = async_cm
        
        tool_manager.configure(
            http_session=mock_session,
            letta_url="http://test:8283",
            headers={"X-Custom": "header"},
            use_letta_sdk=False
        )
        
        await tool_manager.fetch_agent_tools(test_agent_id)
        
        # Verify the URL called
        call_args = mock_session.get.call_args
        expected_url = f"http://test:8283/agents/{test_agent_id}/tools"
        assert call_args[0][0] == expected_url
        assert call_args[1]["headers"]["X-Custom"] == "header"


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_min_mcp_tools_default(self):
        """Should return default MIN_MCP_TOOLS."""
        import tool_manager
        
        with patch.dict('os.environ', {}, clear=True):
            # Clear any existing value
            import os
            if 'MIN_MCP_TOOLS' in os.environ:
                del os.environ['MIN_MCP_TOOLS']
            
            result = tool_manager.get_min_mcp_tools()
            assert result == 7  # Default
    
    def test_get_min_mcp_tools_from_env(self):
        """Should read MIN_MCP_TOOLS from environment."""
        import tool_manager
        
        with patch.dict('os.environ', {'MIN_MCP_TOOLS': '5'}):
            result = tool_manager.get_min_mcp_tools()
            assert result == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
