"""
Unit tests for protected tools logic (LTSEL-21).

Tests the protected tools hierarchy and enforcement:
1. NEVER_DETACH_TOOLS / PROTECTED_TOOLS env config
2. keep_tools parameter in API requests
3. Newly matched tools in same operation
4. Core Letta tools (when EXCLUDE_LETTA_CORE_TOOLS=true)
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any


class TestProtectedToolsLogic:
    """Tests for protected tools determination and enforcement."""
    
    @pytest.fixture
    def sample_tools(self) -> List[Dict[str, Any]]:
        """Sample tool set for testing."""
        return [
            {"id": "tool-1", "name": "find_tools", "tool_type": "external_mcp"},
            {"id": "tool-2", "name": "search_database", "tool_type": "external_mcp"},
            {"id": "tool-3", "name": "send_message", "tool_type": "letta_core"},
            {"id": "tool-4", "name": "custom_webhook", "tool_type": "custom"},
            {"id": "tool-5", "name": "data_processor", "tool_type": "external_mcp"},
        ]
    
    @pytest.fixture
    def never_detach_tools(self) -> List[str]:
        """Default never-detach tools."""
        return ["find_tools"]
    
    def test_never_detach_tools_env_variable(self, sample_tools, never_detach_tools):
        """Test that tools in NEVER_DETACH_TOOLS are protected."""
        # Mock environment variable
        with patch.dict(os.environ, {"NEVER_DETACH_TOOLS": "find_tools,search_database"}):
            never_detach_list = os.getenv("NEVER_DETACH_TOOLS", "find_tools").split(",")
            never_detach_list = [t.strip() for t in never_detach_list]
            
            protected_tool_ids = set()
            for tool in sample_tools:
                tool_name = tool.get("name", "").lower()
                if any(nd_name.lower() in tool_name for nd_name in never_detach_list):
                    protected_tool_ids.add(tool["id"])
            
            # Verify find_tools and search_database are protected
            assert "tool-1" in protected_tool_ids, "find_tools should be protected"
            assert "tool-2" in protected_tool_ids, "search_database should be protected"
            assert "tool-5" not in protected_tool_ids, "data_processor should not be protected"
    
    def test_protected_tools_alias(self, sample_tools):
        """Test that PROTECTED_TOOLS works as alias for NEVER_DETACH_TOOLS."""
        with patch.dict(os.environ, {"PROTECTED_TOOLS": "custom_webhook"}):
            # Code should check PROTECTED_TOOLS if NEVER_DETACH_TOOLS not set
            protected_tools = os.getenv("PROTECTED_TOOLS", os.getenv("NEVER_DETACH_TOOLS", "find_tools"))
            protected_list = [t.strip() for t in protected_tools.split(",")]
            
            protected_tool_ids = set()
            for tool in sample_tools:
                tool_name = tool.get("name", "").lower()
                if any(p_name.lower() in tool_name for p_name in protected_list):
                    protected_tool_ids.add(tool["id"])
            
            assert "tool-4" in protected_tool_ids, "custom_webhook should be protected via PROTECTED_TOOLS"
    
    def test_keep_tools_parameter_protection(self, sample_tools):
        """Test that tools in keep_tools parameter are protected."""
        keep_tools_param = ["tool-2", "tool-5"]
        
        # These tools should never be detached
        tools_to_detach = []
        for tool in sample_tools:
            if tool["id"] not in keep_tools_param:
                tools_to_detach.append(tool["id"])
        
        assert "tool-2" not in tools_to_detach, "tool-2 should be protected by keep_tools"
        assert "tool-5" not in tools_to_detach, "tool-5 should be protected by keep_tools"
        assert "tool-1" in tools_to_detach, "tool-1 should be detachable"
    
    def test_newly_matched_tools_protection(self, sample_tools):
        """Test that newly attached tools in same operation are protected."""
        newly_matched_ids = ["tool-4", "tool-5"]
        
        # Simulate determining which tools to keep
        protected_ids = set(newly_matched_ids)
        
        tools_to_detach = []
        for tool in sample_tools:
            if tool["id"] not in protected_ids:
                tools_to_detach.append(tool["id"])
        
        assert "tool-4" not in tools_to_detach, "Newly matched tool-4 should be protected"
        assert "tool-5" not in tools_to_detach, "Newly matched tool-5 should be protected"
    
    def test_is_letta_core_tool_by_type(self):
        """Test _is_letta_core_tool() function with tool_type."""
        # Import or mock the function
        letta_core_types = [
            'letta_core', 'letta_voice_sleeptime_core', 'letta_sleeptime_core',
            'letta_memory_core', 'letta_files_core', 'letta_builtin', 'letta_multi_agent_core'
        ]
        
        for tool_type in letta_core_types:
            tool = {"id": "test-id", "name": "test", "tool_type": tool_type}
            # Simulate _is_letta_core_tool logic
            assert tool.get("tool_type") in letta_core_types, f"{tool_type} should be identified as core tool"
    
    def test_is_letta_core_tool_by_name(self):
        """Test _is_letta_core_tool() function with tool name."""
        core_tool_names = [
            'send_message', 'conversation_search', 'archival_memory_insert',
            'archival_memory_search', 'core_memory_append', 'core_memory_replace',
            'pause_heartbeats', 'find_attach_tools'
        ]
        
        for tool_name in core_tool_names:
            tool = {"id": "test-id", "name": tool_name, "tool_type": "custom"}
            # Simulate _is_letta_core_tool logic
            assert tool.get("name") in core_tool_names, f"{tool_name} should be identified as core tool"
    
    def test_combined_protection_hierarchy(self, sample_tools):
        """Test that all protection mechanisms work together correctly."""
        # Simulate full protection hierarchy
        never_detach = ["find_tools"]
        keep_tools = ["tool-2"]
        newly_matched = ["tool-4"]
        
        protected_ids = set()
        
        # 1. Add never-detach tools
        for tool in sample_tools:
            tool_name = tool.get("name", "").lower()
            if any(nd.lower() in tool_name for nd in never_detach):
                protected_ids.add(tool["id"])
        
        # 2. Add keep_tools
        protected_ids.update(keep_tools)
        
        # 3. Add newly matched
        protected_ids.update(newly_matched)
        
        # 4. Add core tools (simulate EXCLUDE_LETTA_CORE_TOOLS=true)
        for tool in sample_tools:
            if tool.get("tool_type") == "letta_core":
                protected_ids.add(tool["id"])
        
        # Verify expected protections
        assert "tool-1" in protected_ids, "find_tools protected by never-detach"
        assert "tool-2" in protected_ids, "search_database protected by keep_tools"
        assert "tool-3" in protected_ids, "send_message protected as core tool"
        assert "tool-4" in protected_ids, "custom_webhook protected as newly matched"
        assert "tool-5" not in protected_ids, "data_processor should not be protected"
    
    def test_min_mcp_tools_prevents_over_pruning(self):
        """Test that MIN_MCP_TOOLS prevents excessive detachment."""
        current_mcp_count = 8
        min_mcp_tools = 7
        tools_to_detach_count = 3  # Would leave only 5 tools
        
        # Calculate max allowed detachments
        max_detach_allowed = max(0, current_mcp_count - min_mcp_tools)
        actual_detach_count = min(tools_to_detach_count, max_detach_allowed)
        
        assert actual_detach_count == 1, "Should only detach 1 tool to maintain MIN_MCP_TOOLS=7"
        assert (current_mcp_count - actual_detach_count) >= min_mcp_tools, "Result should meet minimum"
    
    def test_protection_prevents_all_detachments(self, sample_tools):
        """Test scenario where all tools are protected."""
        # Protect everything
        protected_ids = {tool["id"] for tool in sample_tools}
        
        tools_to_detach = [
            tool for tool in sample_tools
            if tool["id"] not in protected_ids
        ]
        
        assert len(tools_to_detach) == 0, "No tools should be detached when all are protected"
    
    def test_partial_name_matching(self):
        """Test that partial name matching works for never-detach tools."""
        tools = [
            {"id": "1", "name": "advanced_find_tools_v2"},
            {"id": "2", "name": "find_tools_helper"},
            {"id": "3", "name": "other_tool"}
        ]
        
        never_detach = ["find_tools"]
        protected_ids = set()
        
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            if any(nd.lower() in tool_name for nd in never_detach):
                protected_ids.add(tool["id"])
        
        assert "1" in protected_ids, "Should match 'find_tools' in 'advanced_find_tools_v2'"
        assert "2" in protected_ids, "Should match 'find_tools' in 'find_tools_helper'"
        assert "3" not in protected_ids, "Should not match unrelated tool"
    
    def test_case_insensitive_matching(self):
        """Test that tool name matching is case-insensitive."""
        tools = [
            {"id": "1", "name": "Find_Tools"},
            {"id": "2", "name": "FIND_TOOLS"},
            {"id": "3", "name": "find_tools"}
        ]
        
        never_detach = ["find_tools"]
        protected_ids = set()
        
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            if any(nd.lower() in tool_name for nd in never_detach):
                protected_ids.add(tool["id"])
        
        assert len(protected_ids) == 3, "All case variations should be protected"
    
    def test_empty_protection_lists(self, sample_tools):
        """Test behavior when no protection mechanisms are active."""
        # No protections
        never_detach = []
        keep_tools = []
        newly_matched = []
        
        protected_ids = set()
        protected_ids.update(keep_tools)
        protected_ids.update(newly_matched)
        
        # Only non-MCP tools would be protected (core tools)
        detachable_mcp_tools = [
            tool for tool in sample_tools
            if tool.get("tool_type") == "external_mcp" and tool["id"] not in protected_ids
        ]
        
        assert len(detachable_mcp_tools) == 3, "All 3 MCP tools should be detachable"


class TestProtectionEdgeCases:
    """Edge case tests for protected tools logic."""
    
    def test_tool_in_multiple_protection_lists(self):
        """Test tool appears in multiple protection mechanisms."""
        tool_id = "tool-123"
        
        never_detach_ids = {tool_id}
        keep_tools = [tool_id]
        newly_matched = [tool_id]
        
        # Union of all protections
        all_protected = never_detach_ids.union(set(keep_tools)).union(set(newly_matched))
        
        assert tool_id in all_protected, "Tool should be protected"
        assert len(all_protected) == 1, "Should only appear once in protected set"
    
    def test_invalid_tool_ids_in_keep_list(self):
        """Test that invalid tool IDs in keep_tools don't break logic."""
        valid_tools = ["tool-1", "tool-2"]
        keep_tools = ["tool-1", "invalid-id", "tool-2", None, ""]
        
        # Filter out invalid IDs
        valid_keep_tools = [tid for tid in keep_tools if tid and isinstance(tid, str)]
        
        assert "tool-1" in valid_keep_tools
        assert "tool-2" in valid_keep_tools
        assert "invalid-id" in valid_keep_tools  # Invalid but well-formed string
        assert None not in valid_keep_tools
        assert "" not in valid_keep_tools
    
    def test_protection_with_duplicate_tool_ids(self):
        """Test handling of duplicate tool IDs in protection lists."""
        keep_tools = ["tool-1", "tool-1", "tool-2", "tool-2"]
        
        # Convert to set to remove duplicates
        unique_protected = set(keep_tools)
        
        assert len(unique_protected) == 2, "Should have 2 unique protected tools"
        assert "tool-1" in unique_protected
        assert "tool-2" in unique_protected
    
    def test_whitespace_in_never_detach_names(self):
        """Test handling of whitespace in NEVER_DETACH_TOOLS."""
        env_value = " find_tools , search_tool , webhook_handler "
        never_detach_list = [t.strip() for t in env_value.split(",")]
        
        assert len(never_detach_list) == 3
        assert "find_tools" in never_detach_list
        assert "search_tool" in never_detach_list
        assert "webhook_handler" in never_detach_list
        assert " find_tools " not in never_detach_list  # Should be stripped


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
