"""Unit tests for pruning scheduler safety logic."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os
import sys

# Add tool-selector-api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tool-selector-api'))

from pruning_scheduler import (
    PruningScheduler,
    PruningSchedulerConfig,
    PruningResult,
)


class TestPruningSchedulerConfig:
    """Test PruningSchedulerConfig defaults and validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PruningSchedulerConfig()
        
        assert config.enabled is False  # Disabled by default for safety
        assert config.interval_hours == 24.0
        assert config.dry_run is True  # Default to dry run for safety
        assert config.min_mcp_tools == 5
        assert config.drop_rate == 0.3
        assert config.batch_size == 10
        
    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict(os.environ, {
            "PRUNING_SCHEDULER_ENABLED": "true",
            "PRUNING_SCHEDULER_DRY_RUN": "false",
            "PRUNING_SCHEDULER_MIN_MCP_TOOLS": "3",
        }):
            config = PruningSchedulerConfig.from_env()
            assert config.enabled is True
            assert config.dry_run is False
            assert config.min_mcp_tools == 3


class TestPruningSchedulerSafety:
    """Test safety mechanisms in pruning scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a scheduler with mock dependencies."""
        config = PruningSchedulerConfig(
            enabled=True,
            dry_run=True,
        )
        scheduler = PruningScheduler(config)
        return scheduler
    
    def test_is_mcp_tool_external_mcp_type(self, scheduler):
        """Test MCP tool identification via tool_type."""
        tool = {"name": "huly_create", "tool_type": "external_mcp", "tags": []}
        assert scheduler._is_mcp_tool(tool) is True
        
    def test_is_mcp_tool_with_mcp_tag(self, scheduler):
        """Test MCP tool identification via mcp: tag."""
        tool = {"name": "some_tool", "tool_type": "custom", "tags": ["mcp:huly"]}
        assert scheduler._is_mcp_tool(tool) is True
        
    def test_is_not_mcp_tool_letta_core(self, scheduler):
        """Test Letta core tools are NOT MCP."""
        tool = {"name": "send_message", "tool_type": "letta_core", "tags": []}
        assert scheduler._is_mcp_tool(tool) is False
        
    def test_is_not_mcp_tool_custom_no_tag(self, scheduler):
        """Test custom tools without mcp: tag are NOT MCP."""
        tool = {"name": "custom_func", "tool_type": "custom", "tags": ["user"]}
        assert scheduler._is_mcp_tool(tool) is False
        
    def test_is_letta_core_tool(self, scheduler):
        """Test Letta core tool identification."""
        core_types = [
            "letta_core",
            "letta_memory_core", 
            "letta_multi_agent_core",
            "letta_sleeptime_core",
        ]
        for tool_type in core_types:
            tool = {"name": "test", "tool_type": tool_type}
            assert scheduler._is_letta_core_tool(tool) is True
            
    def test_is_protected_tool_via_env(self, scheduler):
        """Test protected tools from NEVER_DETACH_TOOLS."""
        # find_tools is default protected
        assert scheduler._is_protected_tool("find_tools") is True
        
    def test_get_prunable_mcp_tools_filters_correctly(self, scheduler):
        """Test that only MCP tools are returned as prunable."""
        tools = [
            {"name": "send_message", "tool_type": "letta_core", "tags": []},
            {"name": "huly_create", "tool_type": "external_mcp", "tags": []},
            {"name": "matrix_send", "tool_type": "external_mcp", "tags": []},
            {"name": "custom_func", "tool_type": "custom", "tags": []},
            {"name": "find_tools", "tool_type": "external_mcp", "tags": []},  # Protected
        ]
        
        prunable = scheduler._get_prunable_mcp_tools(tools)
        
        prunable_names = [t["name"] for t in prunable]
        assert "send_message" not in prunable_names  # Letta core
        assert "custom_func" not in prunable_names  # Custom non-MCP
        assert "find_tools" not in prunable_names  # Protected
        assert "huly_create" in prunable_names  # MCP, not protected
        assert "matrix_send" in prunable_names  # MCP, not protected


class TestPruningSchedulerDryRun:
    """Test dry run mode."""
    
    def test_dry_run_enabled_by_default(self):
        """Verify dry run is enabled by default for safety."""
        config = PruningSchedulerConfig()
        assert config.dry_run is True
        
    def test_scheduler_respects_dry_run(self):
        """Verify scheduler tracks dry_run state."""
        config = PruningSchedulerConfig(dry_run=True)
        scheduler = PruningScheduler(config)
        assert scheduler.config.dry_run is True


class TestPruningResult:
    """Test PruningResult dataclass."""
    
    def test_result_structure(self):
        """Test result captures all necessary info."""
        result = PruningResult(
            agent_id="agent-123",
            success=True,
            dry_run=True,
            mcp_tools_before=10,
            mcp_tools_after=7,
            tools_pruned=3,
            tools_protected=["find_tools"],
            error=None,
            skipped_reason=None,
        )
        
        assert result.agent_id == "agent-123"
        assert result.mcp_tools_before - result.mcp_tools_after == 3
        assert result.tools_pruned == 3
        assert result.dry_run is True
        assert result.success is True
        assert result.error is None
        
    def test_result_with_skip(self):
        """Test result when agent is skipped."""
        result = PruningResult(
            agent_id="agent-456",
            success=True,
            dry_run=False,
            skipped_reason="Agent in skip list",
        )
        
        assert result.skipped_reason == "Agent in skip list"
        assert result.tools_pruned == 0


class TestMinMcpToolsThreshold:
    """Test minimum MCP tools threshold."""
    
    def test_threshold_prevents_over_pruning(self):
        """Test that min_mcp_tools prevents pruning below threshold."""
        config = PruningSchedulerConfig(
            min_mcp_tools=5,
            drop_rate=0.5,  # Would normally drop 50%
        )
        scheduler = PruningScheduler(config)
        
        # With 6 MCP tools and min=5, should only allow pruning 1
        # Even with 50% drop rate
        tools = [
            {"name": f"tool_{i}", "tool_type": "external_mcp", "tags": []}
            for i in range(6)
        ]
        
        prunable = scheduler._get_prunable_mcp_tools(tools)
        max_to_prune = scheduler._calculate_prune_count(len(prunable))
        
        # Should respect min threshold
        remaining_after_prune = len(prunable) - max_to_prune
        assert remaining_after_prune >= config.min_mcp_tools
