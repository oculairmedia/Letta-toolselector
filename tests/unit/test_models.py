"""
Unit tests for the models module.

Tests cover:
- Tool model creation and validation
- SearchResult scoring logic
- Letta core tool detection
- Request/response model validation
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lettaaugment-source"))


# ============================================================================
# Tool Model Tests
# ============================================================================

class TestToolModel:
    """Tests for the Tool model."""
    
    def test_tool_creation_minimal(self):
        """Should create tool with minimal required fields."""
        from models import Tool
        
        tool = Tool(name="test_tool")
        assert tool.name == "test_tool"
        assert tool.id is None
        assert tool.tool_id is None
    
    def test_tool_creation_full(self):
        """Should create tool with all fields."""
        from models import Tool
        
        tool = Tool(
            id="tool-123",
            tool_id="tool-123",
            name="web_search",
            description="Search the web",
            tool_type="external_mcp",
            mcp_server_name="web-tools",
            tags=["search", "web"]
        )
        
        assert tool.id == "tool-123"
        assert tool.name == "web_search"
        assert tool.tool_type == "external_mcp"
        assert "search" in tool.tags
    
    def test_tool_effective_id_prefers_id(self):
        """Should prefer id over tool_id."""
        from models import Tool
        
        tool = Tool(id="id-1", tool_id="tool-id-2", name="test")
        assert tool.effective_id == "id-1"
    
    def test_tool_effective_id_falls_back_to_tool_id(self):
        """Should use tool_id when id is None."""
        from models import Tool
        
        tool = Tool(tool_id="tool-id-2", name="test")
        assert tool.effective_id == "tool-id-2"
    
    def test_tool_is_letta_core_by_type(self):
        """Should detect Letta core tool by tool_type."""
        from models import Tool
        
        tool = Tool(name="send_message", tool_type="letta_core")
        assert tool.is_letta_core() is True
        
        # Memory core
        tool2 = Tool(name="core_memory_append", tool_type="letta_memory_core")
        assert tool2.is_letta_core() is True
    
    def test_tool_is_letta_core_by_name(self):
        """Should detect Letta core tool by name."""
        from models import Tool
        
        # Core tool names should be detected even with custom type
        core_names = [
            "send_message", "conversation_search", 
            "archival_memory_insert", "archival_memory_search"
        ]
        
        for name in core_names:
            tool = Tool(name=name, tool_type="custom")
            assert tool.is_letta_core() is True, f"{name} should be detected as core"
    
    def test_tool_is_not_letta_core(self):
        """Should correctly identify non-core tools."""
        from models import Tool
        
        tool = Tool(name="web_search", tool_type="external_mcp")
        assert tool.is_letta_core() is False
    
    def test_tool_allows_extra_fields(self):
        """Should allow extra fields for flexibility."""
        from models import Tool
        
        tool = Tool(
            name="test",
            custom_field="custom_value",
            another_field=123
        )
        assert tool.name == "test"
        # Extra fields should be stored
        assert tool.model_extra.get("custom_field") == "custom_value"


# ============================================================================
# SearchResult Model Tests
# ============================================================================

class TestSearchResultModel:
    """Tests for the SearchResult model."""
    
    def test_search_result_creation(self):
        """Should create search result with scoring fields."""
        from models import SearchResult
        
        result = SearchResult(
            name="web_search",
            description="Search the web",
            score=95.5,
            distance=0.045
        )
        
        assert result.name == "web_search"
        assert result.score == 95.5
        assert result.distance == 0.045
    
    def test_effective_score_prefers_rerank(self):
        """Should prefer rerank_score over score."""
        from models import SearchResult
        
        result = SearchResult(
            name="test",
            rerank_score=98.0,
            score=85.0,
            distance=0.15
        )
        
        assert result.effective_score == 98.0
    
    def test_effective_score_uses_score_when_no_rerank(self):
        """Should use score when no rerank_score."""
        from models import SearchResult
        
        result = SearchResult(
            name="test",
            score=85.0,
            distance=0.15
        )
        
        assert result.effective_score == 85.0
    
    def test_effective_score_calculates_from_distance(self):
        """Should calculate score from distance when no score fields."""
        from models import SearchResult
        
        result = SearchResult(
            name="test",
            distance=0.2  # Should give 80% score
        )
        
        assert result.effective_score == 80.0
    
    def test_effective_score_zero_when_no_data(self):
        """Should return 0 when no scoring data available."""
        from models import SearchResult
        
        result = SearchResult(name="test")
        assert result.effective_score == 0.0


# ============================================================================
# is_letta_core_tool Function Tests
# ============================================================================

class TestIsLettaCoreTool:
    """Tests for the is_letta_core_tool utility function."""
    
    def test_dict_with_letta_core_type(self):
        """Should detect core tool from dict with letta_core type."""
        from models import is_letta_core_tool
        
        tool = {"name": "something", "tool_type": "letta_core"}
        assert is_letta_core_tool(tool) is True
    
    def test_dict_with_memory_core_type(self):
        """Should detect core tool from dict with letta_memory_core type."""
        from models import is_letta_core_tool
        
        tool = {"name": "core_memory_replace", "tool_type": "letta_memory_core"}
        assert is_letta_core_tool(tool) is True
    
    def test_dict_with_core_name(self):
        """Should detect core tool from dict by name."""
        from models import is_letta_core_tool
        
        tool = {"name": "send_message", "tool_type": "custom"}
        assert is_letta_core_tool(tool) is True
    
    def test_dict_not_core(self):
        """Should return False for non-core tool dict."""
        from models import is_letta_core_tool
        
        tool = {"name": "web_search", "tool_type": "external_mcp"}
        assert is_letta_core_tool(tool) is False
    
    def test_tool_model_input(self):
        """Should work with Tool model input."""
        from models import is_letta_core_tool, Tool
        
        tool = Tool(name="archival_memory_search", tool_type="letta_core")
        assert is_letta_core_tool(tool) is True


# ============================================================================
# Request Model Validation Tests
# ============================================================================

class TestRequestModels:
    """Tests for API request model validation."""
    
    def test_search_request_valid(self):
        """Should accept valid search request."""
        from models import SearchRequest
        
        req = SearchRequest(query="find database tools", limit=5)
        assert req.query == "find database tools"
        assert req.limit == 5
    
    def test_search_request_empty_query_fails(self):
        """Should reject empty query."""
        from models import SearchRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            SearchRequest(query="")
    
    def test_search_request_limit_bounds(self):
        """Should enforce limit bounds."""
        from models import SearchRequest
        from pydantic import ValidationError
        
        # Valid bounds
        req = SearchRequest(query="test", limit=100)
        assert req.limit == 100
        
        # Too high
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=101)
        
        # Too low
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)
    
    def test_attach_tools_request_defaults(self):
        """Should have sensible defaults."""
        from models import AttachToolsRequest
        
        req = AttachToolsRequest(agent_id="agent-123", query="test")
        assert req.limit == 10
        assert req.min_score == 35.0
        assert req.auto_prune is True
        assert req.drop_rate == 0.6
    
    def test_prune_tools_request_validation(self):
        """Should validate prune request fields."""
        from models import PruneToolsRequest
        from pydantic import ValidationError
        
        # Valid request
        req = PruneToolsRequest(agent_id="agent-123", drop_rate=0.5)
        assert req.drop_rate == 0.5
        
        # Invalid drop_rate
        with pytest.raises(ValidationError):
            PruneToolsRequest(agent_id="agent-123", drop_rate=1.5)


# ============================================================================
# Configuration Model Tests
# ============================================================================

class TestConfigModels:
    """Tests for configuration models."""
    
    def test_reranker_config_defaults(self):
        """Should have sensible defaults."""
        from models import RerankerConfig
        
        config = RerankerConfig()
        assert config.enabled is True
        assert config.provider == "vllm"
        assert config.top_k == 20
    
    def test_tool_selector_config_defaults(self):
        """Should have sensible defaults."""
        from models import ToolSelectorConfig
        
        config = ToolSelectorConfig()
        assert config.max_total_tools == 30
        assert config.max_mcp_tools == 20
        assert config.min_mcp_tools == 7
        assert config.default_drop_rate == 0.6
        assert "find_tools" in config.never_detach_tools


class TestToolLimitsConfig:
    """Tests for ToolLimitsConfig dataclass."""
    
    def test_tool_limits_config_defaults(self):
        """Should have sensible defaults."""
        from models import ToolLimitsConfig
        
        config = ToolLimitsConfig()
        assert config.max_total_tools == 30
        assert config.max_mcp_tools == 20
        assert config.min_mcp_tools == 7
        assert config.manage_only_mcp_tools is False
        assert "find_tools" in config.never_detach_tools
    
    def test_tool_limits_config_from_env(self):
        """Should load from environment variables."""
        from models import ToolLimitsConfig
        
        with patch.dict('os.environ', {
            'MAX_TOTAL_TOOLS': '40',
            'MAX_MCP_TOOLS': '25',
            'MIN_MCP_TOOLS': '5',
            'MANAGE_ONLY_MCP_TOOLS': 'true',
            'NEVER_DETACH_TOOLS': 'find_tools,custom_tool'
        }):
            config = ToolLimitsConfig.from_env()
            assert config.max_total_tools == 40
            assert config.max_mcp_tools == 25
            assert config.min_mcp_tools == 5
            assert config.manage_only_mcp_tools is True
            assert 'find_tools' in config.never_detach_tools
            assert 'custom_tool' in config.never_detach_tools
    
    def test_tool_limits_config_protected_tools_alias(self):
        """Should support PROTECTED_TOOLS as alias for NEVER_DETACH_TOOLS."""
        from models import ToolLimitsConfig
        
        with patch.dict('os.environ', {
            'PROTECTED_TOOLS': 'my_protected_tool'
        }, clear=False):
            # Remove NEVER_DETACH_TOOLS if present to test fallback
            import os
            os.environ.pop('NEVER_DETACH_TOOLS', None)
            
            config = ToolLimitsConfig.from_env()
            assert 'my_protected_tool' in config.never_detach_tools
    
    def test_should_protect_tool(self):
        """Should correctly identify protected tools."""
        from models import ToolLimitsConfig
        
        config = ToolLimitsConfig(never_detach_tools=['find_tools', 'important_tool'])
        
        assert config.should_protect_tool('find_tools') is True
        assert config.should_protect_tool('FIND_TOOLS') is True  # Case insensitive
        assert config.should_protect_tool('my_find_tools_helper') is True  # Contains pattern
        assert config.should_protect_tool('important_tool') is True
        assert config.should_protect_tool('random_tool') is False


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for model conversion utilities."""
    
    def test_dict_to_tool(self):
        """Should convert dict to Tool model."""
        from models import dict_to_tool
        
        data = {
            "id": "tool-123",
            "name": "web_search",
            "tool_type": "external_mcp"
        }
        
        tool = dict_to_tool(data)
        assert tool.id == "tool-123"
        assert tool.name == "web_search"
    
    def test_dict_to_search_result(self):
        """Should convert dict to SearchResult model."""
        from models import dict_to_search_result
        
        data = {
            "name": "web_search",
            "score": 95.0,
            "distance": 0.05
        }
        
        result = dict_to_search_result(data)
        assert result.name == "web_search"
        assert result.score == 95.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
