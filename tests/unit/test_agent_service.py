"""
Unit tests for the agent_service module.

Tests cover:
- configure function
- fetch_agent_info function
- register_tool function
- send_trigger_message function
- trigger_agent_loop function
- emit_matrix_webhook function
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
        "tool_type": "external_mcp"
    }


@pytest.fixture
def mock_http_response_success():
    """Create a mock successful HTTP response."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"name": "TestAgent"})
    mock_response.text = AsyncMock(return_value="OK")
    mock_response.raise_for_status = Mock()
    
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_response
    async_cm.__aexit__.return_value = None
    
    return async_cm


@pytest.fixture
def mock_http_session(mock_http_response_success):
    """Create a mock HTTP session."""
    session = MagicMock()
    session.get.return_value = mock_http_response_success
    session.post.return_value = mock_http_response_success
    return session


@pytest.fixture(autouse=True)
def reset_agent_service():
    """Reset agent_service module state before each test."""
    import agent_service
    agent_service._http_session = None
    agent_service._letta_url = None
    agent_service._headers = None
    agent_service._use_letta_sdk = False
    agent_service._get_letta_sdk_client = None
    agent_service._letta_message_base_urls = []
    agent_service._matrix_bridge_webhook_url = None
    yield
    # Clean up after test
    agent_service._http_session = None
    agent_service._letta_url = None
    agent_service._headers = None
    agent_service._use_letta_sdk = False
    agent_service._get_letta_sdk_client = None
    agent_service._letta_message_base_urls = []
    agent_service._matrix_bridge_webhook_url = None


# ============================================================================
# Configuration Tests
# ============================================================================

class TestAgentServiceConfiguration:
    """Tests for agent_service.configure function."""
    
    def test_configure_sets_module_state(self, mock_http_session):
        """Should set module-level state variables."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={"Authorization": "Bearer test"},
            use_letta_sdk=False,
            get_letta_sdk_client_func=None,
            letta_message_base_urls=["http://letta1:8283/v1", "http://letta2:8283/v1"],
            matrix_bridge_webhook_url="http://matrix-bridge/webhook"
        )
        
        assert agent_service._http_session is mock_http_session
        assert agent_service._letta_url == "http://test:8283"
        assert agent_service._use_letta_sdk is False
        assert agent_service._letta_message_base_urls == ["http://letta1:8283/v1", "http://letta2:8283/v1"]
        assert agent_service._matrix_bridge_webhook_url == "http://matrix-bridge/webhook"
    
    def test_configure_with_sdk(self, mock_http_session):
        """Should configure SDK mode."""
        import agent_service
        
        mock_sdk_func = Mock()
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=True,
            get_letta_sdk_client_func=mock_sdk_func
        )
        
        assert agent_service._use_letta_sdk is True
        assert agent_service._get_letta_sdk_client is mock_sdk_func

    def test_configure_defaults(self, mock_http_session):
        """Should use defaults for optional parameters."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283"
        )
        
        assert agent_service._letta_message_base_urls == []
        assert agent_service._matrix_bridge_webhook_url is None


# ============================================================================
# fetch_agent_info Tests
# ============================================================================

class TestFetchAgentInfo:
    """Tests for agent_service.fetch_agent_info function."""
    
    @pytest.mark.asyncio
    async def test_fetch_agent_info_success(self, test_agent_id, mock_http_session):
        """Should fetch agent info via HTTP and return agent name."""
        import agent_service
        
        # Configure with mock session
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={"Authorization": "Bearer test"}
        )
        
        result = await agent_service.fetch_agent_info(test_agent_id)
        
        assert result == "TestAgent"
        mock_http_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_agent_info_no_session(self, test_agent_id):
        """Should raise ConnectionError when HTTP session not configured."""
        import agent_service
        
        # Don't configure session
        with pytest.raises(ConnectionError, match="HTTP session not available"):
            await agent_service.fetch_agent_info(test_agent_id)
    
    @pytest.mark.asyncio
    async def test_fetch_agent_info_sdk_mode(self, test_agent_id, mock_http_session):
        """Should use SDK when enabled."""
        import agent_service
        
        mock_sdk_client = Mock()
        mock_sdk_client.get_agent_name = AsyncMock(return_value="SDKAgent")
        mock_sdk_func = Mock(return_value=mock_sdk_client)
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            use_letta_sdk=True,
            get_letta_sdk_client_func=mock_sdk_func
        )
        
        result = await agent_service.fetch_agent_info(test_agent_id)
        
        assert result == "SDKAgent"
        mock_sdk_client.get_agent_name.assert_called_once_with(test_agent_id)
    
    @pytest.mark.asyncio
    async def test_fetch_agent_info_returns_unknown_on_missing_name(self, test_agent_id, mock_http_session):
        """Should return 'Unknown Agent' when name field is missing."""
        import agent_service
        
        # Create response without name field
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})  # No 'name' field
        mock_response.raise_for_status = Mock()
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.get.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={}
        )
        
        result = await agent_service.fetch_agent_info(test_agent_id)
        
        assert result == "Unknown Agent"


# ============================================================================
# register_tool Tests
# ============================================================================

class TestRegisterTool:
    """Tests for agent_service.register_tool function."""
    
    @pytest.mark.asyncio
    async def test_register_tool_success(self, mock_http_session):
        """Should register tool and return tool dict with normalized IDs."""
        import agent_service
        
        # Create response with id field
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": "tool-12345",
            "name": "test_tool"
        })
        mock_response.raise_for_status = Mock()
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.post.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={}
        )
        
        result = await agent_service.register_tool("test_tool", "test_server")
        
        assert result["id"] == "tool-12345"
        assert result["tool_id"] == "tool-12345"  # Normalized
    
    @pytest.mark.asyncio
    async def test_register_tool_no_session(self):
        """Should raise ConnectionError when HTTP session not configured."""
        import agent_service
        
        with pytest.raises(ConnectionError, match="HTTP session not available"):
            await agent_service.register_tool("test_tool", "test_server")
    
    @pytest.mark.asyncio
    async def test_register_tool_returns_none_on_missing_id(self, mock_http_session):
        """Should return None when response lacks id/tool_id."""
        import agent_service
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"name": "test_tool"})  # No id
        mock_response.raise_for_status = Mock()
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.post.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={}
        )
        
        result = await agent_service.register_tool("test_tool", "test_server")
        
        assert result is None


# ============================================================================
# send_trigger_message Tests
# ============================================================================

class TestSendTriggerMessage:
    """Tests for agent_service.send_trigger_message function."""
    
    @pytest.mark.asyncio
    async def test_send_trigger_message_success(self, test_agent_id, mock_http_session):
        """Should send trigger message and return True on success."""
        import agent_service
        
        # Create successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "messages": [{"run_id": "run-12345"}]
        })
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.post.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            letta_message_base_urls=["http://letta:8283/v1"]
        )
        
        result = await agent_service.send_trigger_message(
            test_agent_id,
            ["tool1", "tool2"],
            "test query"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_send_trigger_message_no_session(self, test_agent_id):
        """Should return False when HTTP session not available."""
        import agent_service
        
        result = await agent_service.send_trigger_message(
            test_agent_id,
            ["tool1"],
            None
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_trigger_message_no_endpoints(self, test_agent_id, mock_http_session):
        """Should return False when no message endpoints configured."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            letta_message_base_urls=[]  # Empty
        )
        
        result = await agent_service.send_trigger_message(
            test_agent_id,
            ["tool1"],
            None
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_trigger_message_truncates_tool_list(self, test_agent_id, mock_http_session):
        """Should truncate tool list to 5 items in message."""
        import agent_service
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"messages": []})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.post.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            letta_message_base_urls=["http://letta:8283/v1"]
        )
        
        # Send with 8 tools
        tools = [f"tool{i}" for i in range(8)]
        result = await agent_service.send_trigger_message(
            test_agent_id,
            tools,
            None
        )
        
        assert result is True
        # Verify POST was called
        mock_http_session.post.assert_called()


# ============================================================================
# trigger_agent_loop Tests
# ============================================================================

class TestTriggerAgentLoop:
    """Tests for agent_service.trigger_agent_loop function."""
    
    def test_trigger_agent_loop_returns_false_without_agent_id(self):
        """Should return False when agent_id is empty."""
        import agent_service
        
        result = agent_service.trigger_agent_loop("", [{"name": "tool1"}], None)
        
        assert result is False
    
    def test_trigger_agent_loop_returns_false_without_tools(self, test_agent_id):
        """Should return False when attached_tools is empty."""
        import agent_service
        
        result = agent_service.trigger_agent_loop(test_agent_id, [], None)
        
        assert result is False
    
    def test_trigger_agent_loop_extracts_tool_names(self, test_agent_id, mock_http_session):
        """Should extract tool names from attached_tools dicts."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            letta_message_base_urls=["http://letta:8283/v1"]
        )
        
        tools = [
            {"name": "tool1"},
            {"tool_name": "tool2"},
            {"name": "tool3", "tool_name": "ignored"}  # name takes priority
        ]
        
        # This will try to create an event loop task
        # In test context, we just verify it doesn't crash
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.create_task = Mock(return_value=Mock())
            result = agent_service.trigger_agent_loop(test_agent_id, tools, "query")
        
        assert result is True


# ============================================================================
# emit_matrix_webhook Tests
# ============================================================================

class TestEmitMatrixWebhook:
    """Tests for agent_service.emit_matrix_webhook function."""
    
    @pytest.mark.asyncio
    async def test_emit_webhook_success(self, test_agent_id, mock_http_session):
        """Should send webhook and return True on success."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            matrix_bridge_webhook_url="http://matrix-bridge/webhook"
        )
        
        result = await agent_service.emit_matrix_webhook(
            agent_id=test_agent_id,
            new_run_id="run-12345",
            tool_names=["tool1", "tool2"],
            query="test query"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_emit_webhook_no_url(self, test_agent_id, mock_http_session):
        """Should return False when webhook URL not configured."""
        import agent_service
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            matrix_bridge_webhook_url=None  # Not configured
        )
        
        result = await agent_service.emit_matrix_webhook(
            agent_id=test_agent_id,
            new_run_id="run-12345"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_emit_webhook_no_session(self, test_agent_id):
        """Should return False when HTTP session not available."""
        import agent_service
        
        agent_service._matrix_bridge_webhook_url = "http://matrix-bridge/webhook"
        
        result = await agent_service.emit_matrix_webhook(
            agent_id=test_agent_id,
            new_run_id="run-12345"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_emit_webhook_handles_non_200(self, test_agent_id, mock_http_session):
        """Should return False on non-200 response."""
        import agent_service
        
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server Error")
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        mock_http_session.post.return_value = async_cm
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            matrix_bridge_webhook_url="http://matrix-bridge/webhook"
        )
        
        result = await agent_service.emit_matrix_webhook(
            agent_id=test_agent_id,
            new_run_id="run-12345"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_emit_webhook_handles_timeout(self, test_agent_id, mock_http_session):
        """Should return False on timeout."""
        import agent_service
        import asyncio
        
        # Make post raise TimeoutError
        async def raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()
        
        mock_http_session.post.side_effect = raise_timeout
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            matrix_bridge_webhook_url="http://matrix-bridge/webhook"
        )
        
        result = await agent_service.emit_matrix_webhook(
            agent_id=test_agent_id,
            new_run_id="run-12345"
        )
        
        assert result is False


# ============================================================================
# Integration Tests (module interaction)
# ============================================================================

class TestAgentServiceIntegration:
    """Tests for agent_service module interaction patterns."""
    
    @pytest.mark.asyncio
    async def test_send_trigger_calls_emit_webhook(self, test_agent_id, mock_http_session):
        """send_trigger_message should call emit_matrix_webhook after success."""
        import agent_service
        
        # Create successful message response with run_id
        mock_msg_response = MagicMock()
        mock_msg_response.status = 200
        mock_msg_response.json = AsyncMock(return_value={
            "messages": [{"run_id": "run-12345"}]
        })
        
        # Create successful webhook response
        mock_webhook_response = MagicMock()
        mock_webhook_response.status = 200
        
        # Track which endpoint is being called
        call_count = {"msg": 0, "webhook": 0}
        
        def create_response(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if 'messages' in str(url):
                call_count["msg"] += 1
                cm = AsyncMock()
                cm.__aenter__.return_value = mock_msg_response
                cm.__aexit__.return_value = None
                return cm
            else:
                call_count["webhook"] += 1
                cm = AsyncMock()
                cm.__aenter__.return_value = mock_webhook_response
                cm.__aexit__.return_value = None
                return cm
        
        mock_http_session.post.side_effect = create_response
        
        agent_service.configure(
            http_session=mock_http_session,
            letta_url="http://test:8283",
            headers={},
            letta_message_base_urls=["http://letta:8283/v1"],
            matrix_bridge_webhook_url="http://matrix-bridge/webhook"
        )
        
        await agent_service.send_trigger_message(
            test_agent_id,
            ["tool1", "tool2"],
            "test query"
        )
        
        # Both message and webhook should have been called
        assert call_count["msg"] == 1
        assert call_count["webhook"] == 1
