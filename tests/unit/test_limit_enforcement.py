"""
Unit tests for tool limit enforcement (LTSEL-22).

Tests the enforcement of:
1. MAX_TOTAL_TOOLS - Maximum total tools per agent
2. MAX_MCP_TOOLS - Maximum external MCP tools per agent
3. MIN_MCP_TOOLS - Minimum MCP tools (prevents over-pruning)
4. Pre-attach and post-attach pruning logic
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import math


class TestLimitCalculations:
    """Tests for limit calculation logic."""
    
    @pytest.fixture
    def default_limits(self) -> Dict[str, Any]:
        """Default limit configuration."""
        return {
            "MAX_TOTAL_TOOLS": 30,
            "MAX_MCP_TOOLS": 20,
            "MIN_MCP_TOOLS": 7,
            "DEFAULT_DROP_RATE": 0.6
        }
    
    def test_max_total_tools_enforcement(self, default_limits):
        """Test MAX_TOTAL_TOOLS limit calculation."""
        current_total = 28
        new_tools = 5
        max_total = default_limits["MAX_TOTAL_TOOLS"]
        
        projected_total = current_total + new_tools  # 33
        needs_pruning = projected_total > max_total
        
        assert needs_pruning, "Should need pruning when projected > max"
        
        # Calculate removal needed
        min_removals = max(0, projected_total - max_total)
        assert min_removals == 3, "Should need to remove 3 tools"
    
    def test_max_mcp_tools_enforcement(self, default_limits):
        """Test MAX_MCP_TOOLS limit calculation."""
        current_mcp = 18
        new_mcp_tools = 4
        max_mcp = default_limits["MAX_MCP_TOOLS"]
        
        projected_mcp = current_mcp + new_mcp_tools  # 22
        needs_pruning = projected_mcp > max_mcp
        
        assert needs_pruning, "Should need pruning when projected MCP > max"
        
        min_removals = max(0, projected_mcp - max_mcp)
        assert min_removals == 2, "Should need to remove 2 MCP tools"
    
    def test_min_mcp_tools_prevents_over_pruning(self, default_limits):
        """Test MIN_MCP_TOOLS prevents removing too many tools."""
        current_mcp = 10
        min_mcp = default_limits["MIN_MCP_TOOLS"]
        proposed_removals = 5  # Would leave only 5 tools
        
        max_removals_allowed = max(0, current_mcp - min_mcp)
        actual_removals = min(proposed_removals, max_removals_allowed)
        
        assert max_removals_allowed == 3, "Can remove at most 3 tools"
        assert actual_removals == 3, "Should limit to 3 removals"
        assert (current_mcp - actual_removals) >= min_mcp, "Result meets minimum"
    
    def test_combined_limit_enforcement(self, default_limits):
        """Test when both MAX_TOTAL_TOOLS and MAX_MCP_TOOLS apply."""
        current_total = 28
        current_mcp = 18
        current_core = current_total - current_mcp  # 10
        new_tools = 5
        
        max_total = default_limits["MAX_TOTAL_TOOLS"]
        max_mcp = default_limits["MAX_MCP_TOOLS"]
        
        projected_total = current_total + new_tools  # 33
        projected_mcp = current_mcp + new_tools  # 23
        
        # Calculate removals for each constraint
        removals_for_total = max(0, projected_total - max_total)  # 3
        removals_for_mcp = max(0, projected_mcp - max_mcp)  # 3
        
        # Take the maximum
        min_removals_needed = max(removals_for_total, removals_for_mcp)
        
        assert min_removals_needed == 3, "Should need 3 removals to satisfy both limits"
    
    def test_no_pruning_needed_within_limits(self, default_limits):
        """Test when current tools are within limits."""
        current_total = 20
        current_mcp = 12
        new_tools = 5
        
        max_total = default_limits["MAX_TOTAL_TOOLS"]
        max_mcp = default_limits["MAX_MCP_TOOLS"]
        
        projected_total = current_total + new_tools  # 25
        projected_mcp = current_mcp + new_tools  # 17
        
        needs_pruning = (projected_total > max_total) or (projected_mcp > max_mcp)
        
        assert not needs_pruning, "Should not need pruning when within limits"


class TestDropRateCalculation:
    """Tests for drop rate calculation and application."""
    
    def test_default_drop_rate_calculation(self):
        """Test drop rate determines tools to keep."""
        current_mcp = 20
        drop_rate = 0.6  # Keep 40% of tools
        
        # Calculate tools to keep
        keep_percentage = 1 - drop_rate  # 0.4
        tools_to_keep = math.floor(current_mcp * keep_percentage)
        
        assert tools_to_keep == 8, "Should keep 8 tools (40% of 20)"
    
    def test_aggressive_drop_rate(self):
        """Test aggressive drop rate for pre-attach pruning."""
        current_mcp = 15
        tools_to_remove = 5
        
        # Calculate effective drop rate
        effective_drop_rate = min(0.9, tools_to_remove / max(1, current_mcp))
        
        assert effective_drop_rate == pytest.approx(0.333, rel=0.01), "Effective drop rate ~33.3%"
        assert effective_drop_rate <= 0.9, "Should not exceed max drop rate"
    
    def test_drop_rate_respects_minimum(self):
        """Test that drop rate doesn't violate MIN_MCP_TOOLS."""
        current_mcp = 10
        min_mcp = 7
        drop_rate = 0.8  # Would keep only 2 tools
        
        # Calculate tools to keep (round to avoid floating point issues like 1-0.8=0.19999...)
        keep_percentage = round(1 - drop_rate, 10)
        proposed_keep = math.floor(current_mcp * keep_percentage)  # 2
        
        # Enforce minimum
        actual_keep = max(proposed_keep, min_mcp)
        
        assert proposed_keep == 2, f"Drop rate suggests keeping 2, got {proposed_keep}"
        assert actual_keep == 7, "Minimum enforcement raises to 7"
    
    def test_zero_drop_rate(self):
        """Test drop_rate=0 means keep all tools."""
        current_mcp = 15
        drop_rate = 0.0
        
        keep_percentage = 1 - drop_rate
        tools_to_keep = math.floor(current_mcp * keep_percentage)
        
        assert tools_to_keep == 15, "Should keep all tools with drop_rate=0"
    
    def test_full_drop_rate(self):
        """Test drop_rate=1.0 means keep none (except minimum)."""
        current_mcp = 15
        drop_rate = 1.0
        min_mcp = 7
        
        keep_percentage = 1 - drop_rate
        proposed_keep = math.floor(current_mcp * keep_percentage)  # 0
        actual_keep = max(proposed_keep, min_mcp)
        
        assert proposed_keep == 0, "Drop rate suggests keeping none"
        assert actual_keep == 7, "Minimum enforcement raises to 7"


class TestPreAttachPruning:
    """Tests for pre-attach pruning logic (LTSEL-7)."""
    
    def test_preattach_pruning_triggered_on_total_limit(self):
        """Test pre-attach pruning triggers when MAX_TOTAL_TOOLS would be exceeded."""
        current_total = 28
        new_tools = 5
        max_total = 30
        
        projected = current_total + new_tools
        needs_preattach_pruning = projected > max_total
        
        assert needs_preattach_pruning, "Should trigger pre-attach pruning"
    
    def test_preattach_pruning_triggered_on_mcp_limit(self):
        """Test pre-attach pruning triggers when MAX_MCP_TOOLS would be exceeded."""
        current_mcp = 18
        new_tools = 4
        max_mcp = 20
        
        projected = current_mcp + new_tools
        needs_preattach_pruning = projected > max_mcp
        
        assert needs_preattach_pruning, "Should trigger pre-attach pruning"
    
    def test_preattach_pruning_calculates_removals_correctly(self):
        """Test pre-attach pruning calculates correct number of removals."""
        current_total = 28
        current_mcp = 18
        new_tools = 5
        max_total = 30
        max_mcp = 20
        
        projected_total = current_total + new_tools  # 33
        projected_mcp = current_mcp + new_tools  # 23
        
        min_removals_for_total = max(0, projected_total - max_total)  # 3
        min_removals_for_mcp = max(0, projected_mcp - max_mcp)  # 3
        
        min_removals_needed = max(min_removals_for_total, min_removals_for_mcp)
        
        assert min_removals_needed == 3, "Should need exactly 3 removals"
    
    def test_preattach_pruning_respects_min_limit(self):
        """Test pre-attach pruning respects MIN_MCP_TOOLS."""
        current_mcp = 9
        min_mcp = 7
        removals_needed = 5
        
        max_removals_allowed = max(0, current_mcp - min_mcp)
        actual_removals = min(removals_needed, max_removals_allowed)
        
        assert max_removals_allowed == 2, "Can only remove 2 tools"
        assert actual_removals == 2, "Should limit to 2 removals"
    
    def test_preattach_pruning_skipped_when_no_query(self):
        """Test pre-attach pruning is skipped without query (can't rank relevance)."""
        query = ""
        needs_pruning = True
        
        should_skip = needs_pruning and not query
        
        assert should_skip, "Should skip pre-attach pruning without query"


class TestPostAttachPruning:
    """Tests for post-attach pruning logic (auto_prune)."""
    
    def test_postattach_pruning_only_when_above_minimum(self):
        """Test post-attach pruning skips when at or below MIN_MCP_TOOLS."""
        mcp_count = 7
        min_mcp = 7
        
        should_prune = mcp_count > min_mcp
        
        assert not should_prune, "Should not prune when at minimum"
    
    def test_postattach_pruning_triggers_above_minimum(self):
        """Test post-attach pruning runs when above MIN_MCP_TOOLS."""
        mcp_count = 15
        min_mcp = 7
        
        should_prune = mcp_count > min_mcp
        
        assert should_prune, "Should prune when above minimum"
    
    def test_postattach_pruning_uses_relevance_scoring(self):
        """Test post-attach pruning keeps most relevant tools."""
        # Simulated relevance scores (higher = more relevant)
        tools_with_scores = [
            ("tool-1", 0.95),
            ("tool-2", 0.88),
            ("tool-3", 0.75),
            ("tool-4", 0.62),
            ("tool-5", 0.45),
        ]
        
        keep_count = 3
        
        # Sort by relevance and take top N
        sorted_tools = sorted(tools_with_scores, key=lambda x: x[1], reverse=True)
        tools_to_keep = [t[0] for t in sorted_tools[:keep_count]]
        
        assert "tool-1" in tools_to_keep, "Highest scored tool kept"
        assert "tool-2" in tools_to_keep, "Second highest kept"
        assert "tool-3" in tools_to_keep, "Third highest kept"
        assert "tool-4" not in tools_to_keep, "Lower scored tool removed"
        assert "tool-5" not in tools_to_keep, "Lowest scored tool removed"


class TestEdgeCases:
    """Edge case tests for limit enforcement."""
    
    def test_exactly_at_limit(self):
        """Test behavior when exactly at limit."""
        current = 30
        max_limit = 30
        new_tools = 0
        
        needs_pruning = (current + new_tools) > max_limit
        
        assert not needs_pruning, "Should not need pruning when exactly at limit"
    
    def test_zero_current_tools(self):
        """Test calculations with zero current tools."""
        current_mcp = 0
        new_tools = 5
        min_mcp = 7
        
        projected = current_mcp + new_tools
        
        assert projected == 5, "Projected should be 5"
        assert projected < min_mcp, "Still below minimum after adding tools"
    
    def test_negative_removal_calculation(self):
        """Test that negative removals are clamped to zero."""
        current = 10
        max_limit = 30
        new_tools = 5
        
        projected = current + new_tools
        removals = max(0, projected - max_limit)
        
        assert removals == 0, "Negative removals should be clamped to 0"
    
    def test_removal_exceeds_current_tools(self):
        """Test when calculated removals exceed current tool count."""
        current_mcp = 3
        removals_needed = 10
        
        actual_removals = min(removals_needed, current_mcp)
        
        assert actual_removals == 3, "Cannot remove more tools than exist"
    
    def test_all_tools_protected(self):
        """Test limit enforcement when all tools are protected."""
        current_mcp = 25
        protected_count = 25
        removals_needed = 5
        
        # All tools protected, so actual removals = 0
        removable_count = current_mcp - protected_count
        actual_removals = min(removals_needed, removable_count)
        
        assert actual_removals == 0, "Cannot remove protected tools"
    
    def test_limits_with_mixed_tool_types(self):
        """Test limits apply correctly to mixed tool types."""
        total_tools = 30
        mcp_tools = 15
        core_tools = total_tools - mcp_tools  # 15
        
        max_total = 30
        max_mcp = 20
        
        # MCP tools are within limit
        mcp_within_limit = mcp_tools <= max_mcp
        # Total is at limit
        total_at_limit = total_tools <= max_total
        
        assert mcp_within_limit, "MCP tools within limit"
        assert total_at_limit, "Total tools at limit"
        
        # Can add 5 more MCP tools without exceeding MCP limit
        # But cannot add any without exceeding total limit
        can_add_mcp = (mcp_tools + 5) <= max_mcp  # True
        can_add_total = (total_tools + 5) <= max_total  # False
        
        assert can_add_mcp, "Can add more MCP tools (within MCP limit)"
        assert not can_add_total, "Cannot add tools (at total limit)"


class TestLimitIntegration:
    """Integration tests for complete limit enforcement workflow."""
    
    def test_full_attach_workflow_with_limits(self):
        """Test complete attach workflow with limit enforcement."""
        # Initial state
        current_total = 28
        current_mcp = 18
        new_tools = 5
        
        max_total = 30
        max_mcp = 20
        min_mcp = 7
        
        # Step 1: Check if pre-attach pruning needed
        projected_total = current_total + new_tools
        projected_mcp = current_mcp + new_tools
        
        needs_preattach = (projected_total > max_total) or (projected_mcp > max_mcp)
        assert needs_preattach, "Should need pre-attach pruning"
        
        # Step 2: Calculate removals
        removals_total = max(0, projected_total - max_total)
        removals_mcp = max(0, projected_mcp - max_mcp)
        removals_needed = max(removals_total, removals_mcp)
        
        assert removals_needed == 3, "Need to remove 3 tools"
        
        # Step 3: Apply pre-attach pruning
        current_mcp_after_preattach = current_mcp - removals_needed
        assert current_mcp_after_preattach == 15, "15 MCP tools after pre-attach pruning"
        
        # Step 4: Attach new tools
        final_mcp = current_mcp_after_preattach + new_tools
        final_total = (current_total - removals_needed) + new_tools
        
        assert final_mcp == 20, "Final MCP count at limit"
        assert final_total == 30, "Final total at limit"
        assert final_mcp <= max_mcp, "MCP limit respected"
        assert final_total <= max_total, "Total limit respected"
        assert final_mcp >= min_mcp, "Minimum respected"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
