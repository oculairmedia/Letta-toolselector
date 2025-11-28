"""Tests for tools module."""

from unittest.mock import MagicMock, patch

import pytest
import responses

from letta_toolkit.config import LettaConfig
from letta_toolkit.tools import (
    BatchOperationResult,
    ToolOperationResult,
    attach_tool_to_agent,
    batch_attach_tools,
    batch_detach_tools,
    detach_tool_from_agent,
    list_agent_tools,
)


@pytest.fixture
def config():
    """Create test config."""
    return LettaConfig(
        base_url="https://letta.test.com",
        api_key="test-key",
        timeout=5,
    )


class TestListAgentTools:
    """Tests for list_agent_tools function."""

    @responses.activate
    def test_returns_all_tools(self, config):
        """Test list_agent_tools returns all tools."""
        responses.add(
            responses.GET,
            "https://letta.test.com/v1/agents/agent-123/tools",
            json=[
                {"id": "tool-1", "name": "tool_one"},
                {"id": "tool-2", "name": "tool_two"},
            ],
            status=200,
        )
        
        result = list_agent_tools("agent-123", config=config)
        
        assert len(result) == 2
        assert result[0]["name"] == "tool_one"

    @responses.activate
    def test_uses_limit_param(self, config):
        """Test list_agent_tools passes limit parameter."""
        responses.add(
            responses.GET,
            "https://letta.test.com/v1/agents/agent-123/tools",
            json=[],
            status=200,
        )
        
        list_agent_tools("agent-123", limit=100, config=config)
        
        assert "limit=100" in responses.calls[0].request.url

    def test_returns_empty_for_no_agent_id(self, config):
        """Test returns empty list when no agent_id provided."""
        result = list_agent_tools("", config=config)
        
        assert result == []


class TestAttachToolToAgent:
    """Tests for attach_tool_to_agent function."""

    @responses.activate
    def test_successful_attach(self, config):
        """Test successful tool attachment."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/attach/tool-456",
            json={"success": True},
            status=200,
        )
        
        result = attach_tool_to_agent("agent-123", "tool-456", config=config)
        
        assert result.success is True
        assert result.tool_id == "tool-456"

    @responses.activate
    def test_already_attached_treated_as_success(self, config):
        """Test 409 conflict (already attached) treated as success."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/attach/tool-456",
            json={"error": "Already attached"},
            status=409,
        )
        
        result = attach_tool_to_agent("agent-123", "tool-456", config=config)
        
        assert result.success is True

    @responses.activate
    def test_failure_returns_error(self, config):
        """Test failure returns error details."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/attach/tool-456",
            json={"error": "Server error"},
            status=500,
        )
        
        result = attach_tool_to_agent("agent-123", "tool-456", config=config)
        
        assert result.success is False
        assert result.error is not None


class TestDetachToolFromAgent:
    """Tests for detach_tool_from_agent function."""

    @responses.activate
    def test_successful_detach(self, config):
        """Test successful tool detachment."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/detach/tool-456",
            json={"success": True},
            status=200,
        )
        
        result = detach_tool_from_agent("agent-123", "tool-456", config=config)
        
        assert result.success is True

    @responses.activate
    def test_not_found_treated_as_success(self, config):
        """Test 404 (not attached) treated as success."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/detach/tool-456",
            json={"error": "Not found"},
            status=404,
        )
        
        result = detach_tool_from_agent("agent-123", "tool-456", config=config)
        
        assert result.success is True


class TestBatchOperations:
    """Tests for batch attach/detach functions."""

    @responses.activate
    def test_batch_attach_multiple(self, config):
        """Test batch attach handles multiple tools."""
        for tool_id in ["tool-1", "tool-2", "tool-3"]:
            responses.add(
                responses.PATCH,
                f"https://letta.test.com/v1/agents/agent-123/tools/attach/{tool_id}",
                json={"success": True},
                status=200,
            )
        
        result = batch_attach_tools(
            "agent-123",
            ["tool-1", "tool-2", "tool-3"],
            config=config,
        )
        
        assert len(result.successful) == 3
        assert len(result.failed) == 0
        assert result.all_success is True

    @responses.activate
    def test_batch_attach_partial_failure(self, config):
        """Test batch attach handles partial failures."""
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/attach/tool-1",
            json={"success": True},
            status=200,
        )
        responses.add(
            responses.PATCH,
            "https://letta.test.com/v1/agents/agent-123/tools/attach/tool-2",
            json={"error": "Failed"},
            status=500,
        )
        
        result = batch_attach_tools(
            "agent-123",
            ["tool-1", "tool-2"],
            config=config,
        )
        
        assert len(result.successful) == 1
        assert len(result.failed) == 1
        assert result.all_success is False


class TestToolOperationResult:
    """Tests for ToolOperationResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = ToolOperationResult(success=True, tool_id="tool-123")
        
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = ToolOperationResult(
            success=False,
            tool_id="tool-123",
            error="Connection failed"
        )
        
        assert result.success is False
        assert result.error == "Connection failed"


class TestBatchOperationResult:
    """Tests for BatchOperationResult dataclass."""

    def test_all_success_true_when_no_failures(self):
        """Test all_success is True when no failures."""
        result = BatchOperationResult(
            successful=[ToolOperationResult(success=True, tool_id="1")]
        )
        
        assert result.all_success is True

    def test_all_success_false_when_failures(self):
        """Test all_success is False when failures exist."""
        result = BatchOperationResult(
            successful=[ToolOperationResult(success=True, tool_id="1")],
            failed=[ToolOperationResult(success=False, tool_id="2")]
        )
        
        assert result.all_success is False

    def test_total_counts_all_operations(self):
        """Test total counts both successful and failed."""
        result = BatchOperationResult(
            successful=[ToolOperationResult(success=True, tool_id="1")],
            failed=[
                ToolOperationResult(success=False, tool_id="2"),
                ToolOperationResult(success=False, tool_id="3"),
            ]
        )
        
        assert result.total == 3
