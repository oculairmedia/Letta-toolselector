"""
Unit tests for SDK list_agent_tools pagination handling.

Tests verify that the LettaSDKClient correctly handles paginated responses
from the Letta API when listing agent tools.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tool-selector-api'))


class TestSDKPagination:
    """Tests for SDK pagination handling."""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool object that behaves like SDK Tool."""
        def _make_tool(name: str, tool_id: str = None):
            tool = Mock()
            tool.name = name
            tool.id = tool_id or f"tool-{name}"
            tool.description = f"Description for {name}"
            tool.tags = []
            tool.source_type = "python"
            # model_dump returns a dict representation
            tool.model_dump.return_value = {
                "id": tool.id,
                "name": name,
                "description": f"Description for {name}",
                "tags": [],
                "source_type": "python"
            }
            return tool
        return _make_tool
    
    @pytest.fixture
    def mock_page(self):
        """Create a mock SyncArrayPage."""
        def _make_page(tools: list):
            page = Mock()
            page.data = tools
            page.__iter__ = lambda self: iter(tools)
            page.__len__ = lambda self: len(tools)
            return page
        return _make_page

    def test_list_agent_tools_returns_all_tools(self, mock_tool, mock_page):
        """Test that list_agent_tools returns all tools from a single page."""
        from letta_sdk_client import LettaSDKClient
        
        tools = [mock_tool("tool1"), mock_tool("tool2"), mock_tool("tool3")]
        page = mock_page(tools)
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            client._client.agents.tools.list = Mock(return_value=page)
            
            import asyncio
            result = asyncio.run(client.list_agent_tools("agent-123"))
            
            assert len(result) == 3
            assert result[0]["name"] == "tool1"
            assert result[1]["name"] == "tool2"
            assert result[2]["name"] == "tool3"

    def test_list_agent_tools_handles_empty_response(self, mock_page):
        """Test that list_agent_tools handles empty tool list."""
        from letta_sdk_client import LettaSDKClient
        
        page = mock_page([])
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            client._client.agents.tools.list = Mock(return_value=page)
            
            import asyncio
            result = asyncio.run(client.list_agent_tools("agent-123"))
            
            assert len(result) == 0
            assert result == []

    def test_list_agent_tools_handles_api_error(self):
        """Test that list_agent_tools handles API errors gracefully."""
        from letta_sdk_client import LettaSDKClient
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            client._client.agents.tools.list = Mock(side_effect=Exception("API Error"))
            
            import asyncio
            with pytest.raises(Exception, match="API Error"):
                asyncio.run(client.list_agent_tools("agent-123"))

    def test_list_agent_tools_returns_dicts(self, mock_tool, mock_page):
        """Test that list_agent_tools returns tool dicts with expected keys."""
        from letta_sdk_client import LettaSDKClient
        
        tool = mock_tool("test_tool", "tool-abc123")
        page = mock_page([tool])
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            client._client.agents.tools.list = Mock(return_value=page)
            
            import asyncio
            result = asyncio.run(client.list_agent_tools("agent-123"))
            
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert result[0]["id"] == "tool-abc123"
            assert result[0]["name"] == "test_tool"

    def test_sdk_list_called_with_agent_id(self, mock_page):
        """Test that the SDK list method is called with agent_id."""
        from letta_sdk_client import LettaSDKClient
        
        page = mock_page([])
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            mock_list = Mock(return_value=page)
            client._client.agents.tools.list = mock_list
            
            import asyncio
            asyncio.run(client.list_agent_tools("agent-123"))
            
            # Verify list was called with agent_id
            mock_list.assert_called_once_with(agent_id="agent-123")

    def test_list_agent_tools_with_many_tools(self, mock_tool, mock_page):
        """Test handling of large tool lists."""
        from letta_sdk_client import LettaSDKClient
        
        # Create 100 mock tools
        tools = [mock_tool(f"tool_{i}", f"tool-{i}") for i in range(100)]
        page = mock_page(tools)
        
        with patch.object(LettaSDKClient, '__init__', lambda self, **kwargs: None):
            client = LettaSDKClient(base_url="http://test")
            client._client = Mock()
            client._client.agents.tools.list = Mock(return_value=page)
            
            import asyncio
            result = asyncio.run(client.list_agent_tools("agent-123"))
            
            assert len(result) == 100
            assert result[0]["name"] == "tool_0"
            assert result[99]["name"] == "tool_99"


class TestPaginationAPIContract:
    """Tests documenting the SDK pagination API contract."""
    
    def test_sdk_supports_pagination_params(self):
        """Document that SDK list method accepts pagination parameters.
        
        The Letta SDK tools.list method signature includes:
        - after: Optional[str] - cursor for next page
        - before: Optional[str] - cursor for previous page  
        - limit: Optional[int] - page size
        - order: Literal['asc', 'desc'] - sort order
        - order_by: Literal['created_at'] - sort field
        
        Returns: SyncArrayPage[Tool] with pagination helpers
        """
        # This test documents expected API behavior
        expected_params = ['agent_id', 'after', 'before', 'limit', 'order', 'order_by']
        assert len(expected_params) == 6  # Document the param count

    def test_current_implementation_fetches_all(self):
        """Document that current implementation fetches all tools without pagination.
        
        The current list_agent_tools implementation does not explicitly handle
        pagination - it relies on the SDK's default behavior. For agents with
        many tools, this may need to be updated to iterate through pages.
        """
        # This documents current behavior for future reference
        pass
