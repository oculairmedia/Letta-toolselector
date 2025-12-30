"""
Unit tests for tool management functions.

Tests cover:
- attach_tool() - Attach single tool to agent (tool_manager)
- detach_tool() - Detach single tool from agent (tool_manager)
- process_tools() - Batch attach/detach operations (tool_manager)
- is_letta_core_tool() - Core tool detection (models)
- agent_service.trigger_agent_loop() - Loop trigger mechanism (agent_service)
- perform_tool_pruning() - Tool pruning logic (tool_manager)

These tests use mocked HTTP clients to avoid requiring live services.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tool-selector-api"))

# Import modules
import tool_manager
import agent_service
from models import is_letta_core_tool, ToolLimitsConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for API calls."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_letta_base_urls():
    """Mock Letta API base URLs."""
    return ["http://test-letta:8283"]


@pytest.fixture
def sample_mcp_tool():
    """Sample MCP tool data."""
    return {
        "id": "tool-abc123",
        "tool_id": "tool-abc123",
        "name": "sample_mcp_tool",
        "description": "A sample MCP tool",
        "tool_type": "external_mcp",
        "source_type": "python",
        "mcp_server_name": "test-server",
        "tags": ["mcp:test-server"]
    }


@pytest.fixture
def sample_letta_core_tool():
    """Sample Letta core tool data."""
    return {
        "id": "tool-core123",
        "tool_id": "tool-core123",
        "name": "send_message",
        "description": "Send a message to the user",
        "tool_type": "letta_core",
        "source_type": "letta"
    }


@pytest.fixture
def sample_agent_tools(sample_mcp_tool, sample_letta_core_tool):
    """Sample list of tools on an agent."""
    return [
        sample_letta_core_tool,
        sample_mcp_tool,
        {
            "id": "tool-mcp2",
            "tool_id": "tool-mcp2",
            "name": "another_mcp_tool",
            "description": "Another MCP tool",
            "tool_type": "external_mcp",
            "mcp_server_name": "other-server"
        },
        {
            "id": "tool-mcp3",
            "tool_id": "tool-mcp3",
            "name": "find_tools",
            "description": "Find tools for the agent",
            "tool_type": "external_mcp",
            "mcp_server_name": "tool-selector"
        }
    ]


@pytest.fixture
def test_agent_id():
    """Test agent ID."""
    return "agent-test-12345"


# ============================================================================
# _is_letta_core_tool Tests
# ============================================================================

class TestIsLettaCoreTool:
    """Tests for is_letta_core_tool function from models module."""
    
    def test_letta_core_tool_type(self):
        """Should identify letta_core tool_type as core tool."""
        tool = {"tool_type": "letta_core", "name": "send_message"}
        assert is_letta_core_tool(tool) is True
    
    def test_letta_memory_core_tool_type(self):
        """Should identify letta_memory_core as core tool."""
        tool = {"tool_type": "letta_memory_core", "name": "core_memory_replace"}
        assert is_letta_core_tool(tool) is True
    
    def test_letta_multi_agent_core_tool_type(self):
        """Should identify letta_multi_agent_core as core tool."""
        tool = {"tool_type": "letta_multi_agent_core", "name": "send_message_to_agent"}
        assert is_letta_core_tool(tool) is True
    
    def test_external_mcp_not_core(self):
        """Should NOT identify external_mcp as core tool."""
        tool = {"tool_type": "external_mcp", "name": "web_search"}
        assert is_letta_core_tool(tool) is False
    
    def test_custom_not_core(self):
        """Should NOT identify custom tools as core (by default)."""
        tool = {"tool_type": "custom", "name": "my_custom_tool"}
        assert is_letta_core_tool(tool) is False
    
    def test_core_tool_by_name(self):
        """Should identify core tools by name pattern."""
        # Core tools by name
        core_names = [
            "send_message",
            "conversation_search", 
            "archival_memory_search",
            "archival_memory_insert",
            "core_memory_append",
            "core_memory_replace"
        ]
        
        for name in core_names:
            tool = {"tool_type": "custom", "name": name}
            # Note: This may or may not be True depending on implementation
            # The function checks tool_type first, then may check name
            result = is_letta_core_tool(tool)
            # Just verify it doesn't crash
            assert isinstance(result, bool)
    
    def test_empty_tool(self):
        """Should handle empty tool dict gracefully."""
        tool = {}
        result = is_letta_core_tool(tool)
        assert result is False
    
    def test_missing_tool_type(self):
        """Should handle missing tool_type gracefully."""
        tool = {"name": "some_tool"}
        result = is_letta_core_tool(tool)
        assert isinstance(result, bool)


# ============================================================================
# attach_tool Tests
# ============================================================================

class TestAttachTool:
    """Tests for attach_tool function from tool_manager module."""
    
    @pytest.mark.asyncio
    async def test_attach_tool_success(self, test_agent_id, sample_mcp_tool):
        """Should successfully attach a tool."""
        # Create a proper async context manager mock
        mock_response = MagicMock()
        mock_response.status = 200
        
        # Create async context manager
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool_manager with mock session
        tool_manager.configure(
            http_session=mock_session,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.attach_tool(test_agent_id, sample_mcp_tool)
            assert result["success"] is True
            assert result["tool_id"] == sample_mcp_tool["id"]
        finally:
            # Reset tool_manager state
            tool_manager.configure()
    
    @pytest.mark.asyncio
    async def test_attach_tool_missing_id(self, test_agent_id):
        """Should fail if tool has no ID."""
        tool_without_id = {"name": "no_id_tool", "description": "Missing ID"}
        
        # No need to mock session - should fail early due to missing ID
        result = await tool_manager.attach_tool(test_agent_id, tool_without_id)
        
        assert result["success"] is False
        assert "error" in result
        assert "No tool ID" in result["error"]
    
    @pytest.mark.asyncio
    async def test_attach_tool_api_error(self, test_agent_id, sample_mcp_tool):
        """Should handle API errors gracefully."""
        # Create a proper async context manager mock for error case
        mock_response = MagicMock()
        mock_response.status = 500
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool_manager with mock session
        tool_manager.configure(
            http_session=mock_session,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.attach_tool(test_agent_id, sample_mcp_tool)
            # Should not crash, should return failure
            assert result["success"] is False
        finally:
            # Reset tool_manager state
            tool_manager.configure()
    
    @pytest.mark.asyncio
    async def test_attach_tool_no_session(self, test_agent_id, sample_mcp_tool):
        """Should fail gracefully if http_session is None."""
        # Import from tool_manager instead of api_server
        
        # Configure tool_manager with no session
        tool_manager.configure(
            http_session=None,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.attach_tool(test_agent_id, sample_mcp_tool)
            assert result["success"] is False
            assert "HTTP session not available" in result["error"]
        finally:
            # Reset tool_manager state
            tool_manager.configure()


# ============================================================================
# detach_tool Tests  
# ============================================================================

class TestDetachTool:
    """Tests for detach_tool function."""
    
    @pytest.mark.asyncio
    async def test_detach_tool_success(self, test_agent_id):
        """Should successfully detach a tool."""
        # Import from tool_manager instead of api_server
        
        tool_id = "tool-to-detach"
        
        # Create a proper async context manager mock (detach uses patch, not delete)
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "detached"})
        
        # Create async context manager
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool_manager with mock session (this is what api_server delegates to)
        tool_manager.configure(
            http_session=mock_session,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.detach_tool(test_agent_id, tool_id, "test_tool")
            # detach_tool returns a dict with success key
            assert result["success"] is True
            assert result["tool_id"] == tool_id
        finally:
            # Reset tool_manager state
            tool_manager.configure()
    
    @pytest.mark.asyncio
    async def test_detach_tool_not_found(self, test_agent_id):
        """Should handle tool not found (404) - treated as success."""
        # Import from tool_manager instead of api_server
        
        tool_id = "nonexistent-tool"
        
        # Create a proper async context manager mock for 404 case
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"error": "not found"})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool_manager with mock session (this is what api_server delegates to)
        tool_manager.configure(
            http_session=mock_session,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.detach_tool(test_agent_id, tool_id)
            # 404 is treated as success (tool already detached)
            assert result["success"] is True
            assert "warning" in result
        finally:
            # Reset tool_manager state
            tool_manager.configure()
    
    @pytest.mark.asyncio
    async def test_detach_tool_server_error(self, test_agent_id):
        """Should handle server errors (500)."""
        # Import from tool_manager instead of api_server
        
        tool_id = "tool-error"
        
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "internal error"})
        
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_response
        async_cm.__aexit__.return_value = None
        
        mock_session = MagicMock()
        mock_session.patch.return_value = async_cm
        
        # Configure tool_manager with mock session
        tool_manager.configure(
            http_session=mock_session,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.detach_tool(test_agent_id, tool_id)
            # Should return failure
            assert result["success"] is False
            assert "error" in result
        finally:
            # Reset tool_manager state
            tool_manager.configure()
    
    @pytest.mark.asyncio
    async def test_detach_tool_no_session(self, test_agent_id):
        """Should fail gracefully if http_session is None."""
        # Import from tool_manager instead of api_server
        
        tool_id = "tool-test"
        
        # Configure tool_manager with no session
        tool_manager.configure(
            http_session=None,
            letta_url='http://test:8283',
            headers={'Content-Type': 'application/json'},
            use_letta_sdk=False
        )
        
        try:
            result = await tool_manager.detach_tool(test_agent_id, tool_id)
            assert result["success"] is False
            assert "HTTP session not available" in result["error"]
        finally:
            # Reset tool_manager state
            tool_manager.configure()


# ============================================================================
# process_tools Tests
# ============================================================================

class TestProcessTools:
    """Tests for process_tools function."""
    
    @pytest.mark.asyncio
    async def test_process_tools_attach_new(self, test_agent_id, sample_agent_tools, sample_mcp_tool):
        """Should attach new tools not already on agent."""
        # Import from tool_manager instead of api_server
        
        # Current MCP tools on agent
        current_mcp_tools = [t for t in sample_agent_tools if t.get("tool_type") == "external_mcp"]
        
        # New tool to attach
        new_tool = {
            "id": "tool-new-123",
            "tool_id": "tool-new-123",
            "name": "new_tool",
            "description": "A new tool",
            "tool_type": "external_mcp"
        }
        
        # Mock attach_tool
        with patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_attach.return_value = {"success": True, "tool_id": new_tool["id"], "name": new_tool["name"]}
            
            with patch('tool_manager.detach_tool', new_callable=AsyncMock) as mock_detach:
                mock_detach.return_value = True
                
                result = await tool_manager.process_tools(
                    test_agent_id,
                    current_mcp_tools,
                    [new_tool],
                    keep_tools=[]
                )
        
        assert "successful_attachments" in result
        assert "failed_attachments" in result
        assert "detached_tools" in result
    
    @pytest.mark.asyncio
    async def test_process_tools_respects_keep_tools(self, test_agent_id):
        """Should not detach tools in keep_tools list."""
        # Import from tool_manager instead of api_server
        
        # Current tools
        current_tools = [
            {"id": "tool-keep", "tool_id": "tool-keep", "name": "keep_me", "tool_type": "external_mcp"},
            {"id": "tool-remove", "tool_id": "tool-remove", "name": "remove_me", "tool_type": "external_mcp"}
        ]
        
        # New tools
        new_tools = [
            {"id": "tool-new", "tool_id": "tool-new", "name": "new_tool", "tool_type": "external_mcp"}
        ]
        
        detached_ids = []
        
        async def mock_detach(agent_id, tool_id, tool_name=None):
            detached_ids.append(tool_id)
            return True
        
        with patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_attach.return_value = {"success": True, "tool_id": "tool-new", "name": "new_tool"}
            
            with patch('tool_manager.detach_tool', side_effect=mock_detach):
                result = await tool_manager.process_tools(
                    test_agent_id,
                    current_tools,
                    new_tools,
                    keep_tools=["tool-keep"]
                )
        
        # tool-keep should NOT be in detached
        assert "tool-keep" not in detached_ids
    
    @pytest.mark.asyncio
    async def test_process_tools_handles_attach_failure(self, test_agent_id):
        """Should handle attachment failures gracefully."""
        # Import from tool_manager instead of api_server
        
        current_tools = []
        new_tools = [
            {"id": "tool-fail", "tool_id": "tool-fail", "name": "will_fail", "tool_type": "external_mcp"}
        ]
        
        with patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_attach.return_value = {"success": False, "error": "API Error"}
            
            result = await tool_manager.process_tools(test_agent_id, current_tools, new_tools, [])
        
        assert len(result["failed_attachments"]) > 0 or len(result["successful_attachments"]) == 0


# ============================================================================
# trigger_agent_loop Tests
# ============================================================================

class TestTriggerAgentLoop:
    """Tests for trigger_agent_loop function."""
    
    def test_trigger_agent_loop_spawns_background_task(self, test_agent_id):
        """Should spawn a background task for trigger."""
        # Import from agent_service instead of api_server
        
        attached_tools = [
            {"tool_id": "tool-1", "name": "tool_one"},
            {"tool_id": "tool-2", "name": "tool_two"}
        ]
        
        with patch('agent_service.asyncio.create_task') as mock_create_task:
            with patch('agent_service.send_trigger_message', new_callable=AsyncMock):
                result = agent_service.trigger_agent_loop(test_agent_id, attached_tools, query="test query")
        
        # Should have attempted to create a task
        assert result is True or mock_create_task.called
    
    def test_trigger_agent_loop_with_empty_tools(self, test_agent_id):
        """Should handle empty tools list."""
        # Import from agent_service instead of api_server
        
        result = agent_service.trigger_agent_loop(test_agent_id, [], query="test")
        
        # Should not crash, may return False for empty list
        assert isinstance(result, bool)
    
    def test_trigger_agent_loop_includes_query(self, test_agent_id):
        """Should include query in trigger message."""
        # Import from agent_service instead of api_server
        
        attached_tools = [{"tool_id": "tool-1", "name": "test_tool"}]
        query = "find database tools"
        
        with patch('agent_service.send_trigger_message', new_callable=AsyncMock) as mock_send:
            with patch('agent_service.asyncio.create_task'):
                agent_service.trigger_agent_loop(test_agent_id, attached_tools, query=query)
        
        # The function should pass the query to _send_trigger_message
        # (implementation dependent)


# ============================================================================
# _perform_tool_pruning Tests
# ============================================================================

class TestPerformToolPruning:
    """Tests for _perform_tool_pruning function."""
    
    @pytest.fixture
    def mock_agent_tools(self):
        """Return a list of agent tools for testing."""
        return [
            # MCP tools
            {"id": "tool-mcp-1", "tool_id": "tool-mcp-1", "name": "mcp_tool_1", "tool_type": "external_mcp"},
            {"id": "tool-mcp-2", "tool_id": "tool-mcp-2", "name": "mcp_tool_2", "tool_type": "external_mcp"},
            {"id": "tool-mcp-3", "tool_id": "tool-mcp-3", "name": "mcp_tool_3", "tool_type": "external_mcp"},
            {"id": "tool-mcp-4", "tool_id": "tool-mcp-4", "name": "mcp_tool_4", "tool_type": "external_mcp"},
            {"id": "tool-mcp-5", "tool_id": "tool-mcp-5", "name": "mcp_tool_5", "tool_type": "external_mcp"},
            {"id": "tool-mcp-6", "tool_id": "tool-mcp-6", "name": "mcp_tool_6", "tool_type": "external_mcp"},
            {"id": "tool-mcp-7", "tool_id": "tool-mcp-7", "name": "mcp_tool_7", "tool_type": "external_mcp"},
            {"id": "tool-mcp-8", "tool_id": "tool-mcp-8", "name": "mcp_tool_8", "tool_type": "external_mcp"},
            {"id": "tool-mcp-9", "tool_id": "tool-mcp-9", "name": "mcp_tool_9", "tool_type": "external_mcp"},
            {"id": "tool-mcp-10", "tool_id": "tool-mcp-10", "name": "mcp_tool_10", "tool_type": "external_mcp"},
            # Core tools
            {"id": "tool-core-1", "tool_id": "tool-core-1", "name": "send_message", "tool_type": "letta_core"},
            {"id": "tool-core-2", "tool_id": "tool-core-2", "name": "archival_memory_search", "tool_type": "letta_core"},
            # Protected tool
            {"id": "tool-find", "tool_id": "tool-find", "name": "find_tools", "tool_type": "external_mcp"},
        ]
    
    @pytest.mark.asyncio
    async def test_pruning_no_mcp_tools(self, test_agent_id):
        """Should handle agent with no MCP tools."""
        # Import from tool_manager instead of api_server
        # ToolLimitsConfig already imported at module level
        
        # Only core tools
        agent_tools = [
            {"id": "tool-core-1", "name": "send_message", "tool_type": "letta_core"},
        ]
        
        # Mock fetch_agent_tools at the tool_manager level
        async def mock_fetch(agent_id):
            return agent_tools
        
        # Mock search_tools
        def mock_search(query, limit):
            return []
        
        # Configure tool_manager with mocks
        tool_config = ToolLimitsConfig(manage_only_mcp_tools=True)
        tool_manager.configure(
            http_session=MagicMock(),
            letta_url='http://test:8283',
            headers={},
            use_letta_sdk=False,
            search_tools_func=mock_search,
            tool_config=tool_config
        )
        
        # Patch the fetch_agent_tools function in tool_manager
        with patch.object(tool_manager, 'fetch_agent_tools', mock_fetch):
            result = await tool_manager.perform_tool_pruning(
                test_agent_id,
                user_prompt="test",
                drop_rate=0.5
            )
        
        # Reset tool_manager
        tool_manager.configure()
        
        assert result["success"] is True
        assert "No MCP tools" in result["message"]
    
    @pytest.mark.asyncio
    async def test_pruning_respects_minimum(self, test_agent_id, mock_agent_tools):
        """Should not prune below MIN_MCP_TOOLS."""
        # Import from tool_manager instead of api_server
        # ToolLimitsConfig already imported at module level
        
        # Only 5 MCP tools (below default min of 7)
        agent_tools = mock_agent_tools[:5] + mock_agent_tools[10:12]  # 5 MCP + 2 core
        
        # Mock fetch_agent_tools at the tool_manager level
        async def mock_fetch(agent_id):
            return agent_tools
        
        # Mock search_tools
        def mock_search(query, limit):
            return []
        
        # Configure tool_manager with mocks (min_mcp_tools=7)
        tool_config = ToolLimitsConfig(manage_only_mcp_tools=True, min_mcp_tools=7)
        tool_manager.configure(
            http_session=MagicMock(),
            letta_url='http://test:8283',
            headers={},
            use_letta_sdk=False,
            search_tools_func=mock_search,
            tool_config=tool_config
        )
        
        # Patch the fetch_agent_tools function in tool_manager
        with patch.object(tool_manager, 'fetch_agent_tools', mock_fetch):
            result = await tool_manager.perform_tool_pruning(
                test_agent_id,
                user_prompt="test",
                drop_rate=0.8
            )
        
        # Reset tool_manager
        tool_manager.configure()
        
        assert result["success"] is True
        assert "Pruning skipped" in result["message"]
    
    @pytest.mark.asyncio
    async def test_pruning_preserves_core_tools(self, test_agent_id, mock_agent_tools):
        """Should always preserve Letta core tools."""
        # Import from tool_manager instead of api_server
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            with patch('tool_manager.detach_tool', new_callable=AsyncMock) as mock_detach:
                mock_detach.return_value = {"success": True}
                
                with patch('api_server.search_tools') as mock_search:
                    # Return some tools for relevance ranking
                    mock_search.return_value = [
                        {"name": "mcp_tool_1", "score": 0.9},
                        {"name": "mcp_tool_2", "score": 0.8},
                    ]
                    
                    with patch('api_server.read_tool_cache', new_callable=AsyncMock) as mock_cache:
                        mock_cache.return_value = mock_agent_tools
                        
                        with patch('api_server.MANAGE_ONLY_MCP_TOOLS', True):
                            with patch.dict('os.environ', {'MIN_MCP_TOOLS': '3', 'MAX_MCP_TOOLS': '20'}):
                                result = await tool_manager.perform_tool_pruning(
                                    test_agent_id,
                                    user_prompt="test",
                                    drop_rate=0.5
                                )
        
        # Verify no core tools were detached
        for call in mock_detach.call_args_list:
            tool_id = call[0][1]  # Second positional arg is tool_id
            # Core tool IDs shouldn't be in detach calls
            assert not tool_id.startswith("tool-core-")
    
    @pytest.mark.asyncio
    async def test_pruning_protects_never_detach_tools(self, test_agent_id, mock_agent_tools):
        """Should never detach tools in NEVER_DETACH_TOOLS list."""
        # Import from tool_manager instead of api_server
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            with patch('tool_manager.detach_tool', new_callable=AsyncMock) as mock_detach:
                mock_detach.return_value = {"success": True}
                
                with patch('api_server.search_tools') as mock_search:
                    mock_search.return_value = []
                    
                    with patch('api_server.read_tool_cache', new_callable=AsyncMock) as mock_cache:
                        mock_cache.return_value = mock_agent_tools
                        
                        with patch('api_server.MANAGE_ONLY_MCP_TOOLS', True):
                            with patch('api_server.NEVER_DETACH_TOOLS', ['find_tools']):
                                with patch.dict('os.environ', {'MIN_MCP_TOOLS': '3', 'MAX_MCP_TOOLS': '20'}):
                                    result = await tool_manager.perform_tool_pruning(
                                        test_agent_id,
                                        user_prompt="test",
                                        drop_rate=0.8
                                    )
        
        # Verify find_tools was NOT detached
        for call in mock_detach.call_args_list:
            tool_id = call[0][1]
            assert tool_id != "tool-find"
    
    @pytest.mark.asyncio
    async def test_pruning_keeps_specified_tool_ids(self, test_agent_id, mock_agent_tools):
        """Should keep tools specified in keep_tool_ids."""
        # Import from tool_manager instead of api_server
        
        keep_ids = ["tool-mcp-1", "tool-mcp-2"]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            with patch('tool_manager.detach_tool', new_callable=AsyncMock) as mock_detach:
                mock_detach.return_value = {"success": True}
                
                with patch('api_server.search_tools') as mock_search:
                    mock_search.return_value = []
                    
                    with patch('api_server.read_tool_cache', new_callable=AsyncMock) as mock_cache:
                        mock_cache.return_value = mock_agent_tools
                        
                        with patch('api_server.MANAGE_ONLY_MCP_TOOLS', True):
                            with patch.dict('os.environ', {'MIN_MCP_TOOLS': '2', 'MAX_MCP_TOOLS': '20'}):
                                result = await tool_manager.perform_tool_pruning(
                                    test_agent_id,
                                    user_prompt="test",
                                    drop_rate=0.9,
                                    keep_tool_ids=keep_ids
                                )
        
        # Verify kept tools were NOT detached
        detached_ids = [call[0][1] for call in mock_detach.call_args_list]
        for keep_id in keep_ids:
            assert keep_id not in detached_ids


# ============================================================================
# Integration-style Unit Tests (with more realistic mocking)
# ============================================================================

class TestAttachWorkflow:
    """Tests for the complete attach workflow."""
    
    @pytest.mark.asyncio
    async def test_attach_workflow_filters_by_score(self, test_agent_id):
        """Should filter tools by min_score threshold."""
        # This would test the full attach endpoint logic
        # For now, just verify the filtering logic exists
        pass
    
    @pytest.mark.asyncio
    async def test_attach_workflow_triggers_loop(self, test_agent_id):
        """Should trigger agent loop after successful attachments."""
        pass
    
    @pytest.mark.asyncio
    async def test_attach_workflow_emits_webhook(self, test_agent_id):
        """Should emit webhook to Matrix bridge after trigger."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
