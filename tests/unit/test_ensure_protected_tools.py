"""
Unit tests for ensure_protected_tools function.

Tests that protected tools are automatically attached to agents if missing.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tool-selector-api'))

from tool_manager import ensure_protected_tools


class TestEnsureProtectedTools:
    """Tests for ensure_protected_tools function."""
    
    @pytest.mark.asyncio
    async def test_no_protected_tools_returns_success(self):
        """When no protected tools specified, returns success immediately."""
        result = await ensure_protected_tools("agent-123", [])
        
        assert result["success"] is True
        assert result["already_attached"] == []
        assert result["newly_attached"] == []
        assert result["failed"] == []
    
    @pytest.mark.asyncio
    async def test_all_tools_already_attached(self):
        """When all protected tools already on agent, returns success."""
        mock_agent_tools = [
            {"name": "find_tools", "id": "tool-1"},
            {"name": "conversation_search", "id": "tool-2"},
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools", "conversation_search"]
            )
        
        assert result["success"] is True
        assert set(result["already_attached"]) == {"find_tools", "conversation_search"}
        assert result["newly_attached"] == []
        assert result["failed"] == []
    
    @pytest.mark.asyncio
    async def test_attaches_missing_protected_tool(self):
        """Missing protected tools are attached from available tools."""
        mock_agent_tools = [
            {"name": "other_tool", "id": "tool-1"},
        ]
        available_tools = [
            {"name": "find_tools", "id": "tool-find", "tool_type": "external_mcp"},
            {"name": "other_tool", "id": "tool-1", "tool_type": "external_mcp"},
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch, \
             patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_fetch.return_value = mock_agent_tools
            mock_attach.return_value = {"success": True, "tool_id": "tool-find", "name": "find_tools"}
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"],
                available_tools=available_tools
            )
        
        assert result["success"] is True
        assert result["newly_attached"] == ["find_tools"]
        assert result["failed"] == []
        mock_attach.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_protected_tool_not_in_available_tools(self):
        """When protected tool not found in available tools, marks as failed."""
        mock_agent_tools = []
        available_tools = [
            {"name": "some_other_tool", "id": "tool-1"},
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"],
                available_tools=available_tools
            )
        
        assert result["success"] is False
        assert result["failed"] == ["find_tools"]
        assert any("not found" in err for err in result["errors"])
    
    @pytest.mark.asyncio
    async def test_attach_failure_marks_tool_as_failed(self):
        """When attach fails, marks tool as failed with error."""
        mock_agent_tools = []
        available_tools = [
            {"name": "find_tools", "id": "tool-find"},
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch, \
             patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_fetch.return_value = mock_agent_tools
            mock_attach.return_value = {"success": False, "error": "Agent not found"}
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"],
                available_tools=available_tools
            )
        
        assert result["success"] is False
        assert result["failed"] == ["find_tools"]
        assert any("Agent not found" in err for err in result["errors"])
    
    @pytest.mark.asyncio
    async def test_partial_success_some_attached_some_failed(self):
        """Some tools attach successfully, others fail."""
        mock_agent_tools = [
            {"name": "existing_tool", "id": "tool-0"},
        ]
        available_tools = [
            {"name": "find_tools", "id": "tool-find"},
            {"name": "conversation_search", "id": "tool-search"},
            # missing_tool not in available_tools
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch, \
             patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_fetch.return_value = mock_agent_tools
            mock_attach.return_value = {"success": True, "tool_id": "tool-find", "name": "find_tools"}
            
            result = await ensure_protected_tools(
                "agent-123",
                ["existing_tool", "find_tools", "missing_tool"],
                available_tools=available_tools
            )
        
        assert result["success"] is False  # Because missing_tool failed
        assert "existing_tool" in result["already_attached"]
        assert "find_tools" in result["newly_attached"]
        assert "missing_tool" in result["failed"]
    
    @pytest.mark.asyncio
    async def test_case_insensitive_tool_matching(self):
        """Tool name matching is case-insensitive."""
        mock_agent_tools = [
            {"name": "Find_Tools", "id": "tool-1"},  # Different case
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_agent_tools
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"]  # lowercase
            )
        
        assert result["success"] is True
        assert result["already_attached"] == ["find_tools"]
    
    @pytest.mark.asyncio
    async def test_fetch_agent_tools_failure(self):
        """When fetching agent tools fails, returns error."""
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"]
            )
        
        assert result["success"] is False
        assert any("Failed to fetch agent tools" in err for err in result["errors"])
    
    @pytest.mark.asyncio
    async def test_attach_exception_handled(self):
        """Exceptions during attach are caught and reported."""
        mock_agent_tools = []
        available_tools = [
            {"name": "find_tools", "id": "tool-find"},
        ]
        
        with patch('tool_manager.fetch_agent_tools', new_callable=AsyncMock) as mock_fetch, \
             patch('tool_manager.attach_tool', new_callable=AsyncMock) as mock_attach:
            mock_fetch.return_value = mock_agent_tools
            mock_attach.side_effect = Exception("Connection timeout")
            
            result = await ensure_protected_tools(
                "agent-123",
                ["find_tools"],
                available_tools=available_tools
            )
        
        assert result["success"] is False
        assert "find_tools" in result["failed"]
        assert any("Connection timeout" in err for err in result["errors"])
