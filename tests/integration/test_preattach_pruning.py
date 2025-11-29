"""
Integration tests for pre-attach pruning behavior (LTSEL-7).

These tests verify that the system proactively prunes tools BEFORE
attachment when adding new tools would exceed MAX_TOTAL_TOOLS or MAX_MCP_TOOLS.
"""

import pytest
import os
import asyncio
from typing import List, Dict, Any

# Import the necessary functions from api_server
# Note: You may need to adjust imports based on your project structure
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lettaaugment-source'))

from api_server import fetch_agent_tools, _is_letta_core_tool


@pytest.mark.integration
class TestPreAttachPruning:
    """Tests for pre-attach pruning enforcement (LTSEL-7)."""
    
    @pytest.fixture
    def api_client(self):
        """Fixture to get API client for testing."""
        import aiohttp
        return aiohttp.ClientSession()
    
    @pytest.fixture
    async def test_agent_id(self):
        """
        Fixture to provide a test agent ID.
        Replace with actual test agent creation logic.
        """
        # TODO: Create a test agent or use existing one
        test_agent_id = os.getenv('TEST_AGENT_ID', 'agent-test-uuid')
        return test_agent_id
    
    @pytest.fixture
    def api_base_url(self):
        """API base URL for testing."""
        return os.getenv('API_SERVER_URL', 'http://localhost:8020')
    
    async def _get_tool_counts(self, agent_id: str) -> Dict[str, int]:
        """Helper to get current tool counts for an agent."""
        tools = await fetch_agent_tools(agent_id)
        
        mcp_count = sum(1 for tool in tools 
                       if tool.get("tool_type") == "external_mcp" or 
                       (not _is_letta_core_tool(tool) and tool.get("tool_type") == "custom"))
        core_count = len(tools) - mcp_count
        
        return {
            "total": len(tools),
            "mcp": mcp_count,
            "core": core_count
        }
    
    async def _attach_tools_via_api(self, api_client, api_base_url: str, 
                                    agent_id: str, query: str, limit: int = 10,
                                    auto_prune: bool = False) -> Dict[str, Any]:
        """Helper to call the attach tools API endpoint."""
        url = f"{api_base_url}/api/v1/tools/attach"
        payload = {
            "agent_id": agent_id,
            "query": query,
            "limit": limit,
            "auto_prune": auto_prune
        }
        
        async with api_client.post(url, json=payload) as response:
            return await response.json()
    
    @pytest.mark.asyncio
    async def test_preattach_pruning_enforces_max_total_tools(self, api_client, test_agent_id, api_base_url):
        """
        Test that pre-attach pruning prevents exceeding MAX_TOTAL_TOOLS.
        
        Scenario:
        1. Get current tool count
        2. Set MAX_TOTAL_TOOLS to current count + 2 (via env or config)
        3. Try to attach 5 new tools
        4. Verify that pre-attach pruning runs
        5. Verify final count <= MAX_TOTAL_TOOLS
        """
        # Get initial state
        initial_counts = await self._get_tool_counts(test_agent_id)
        max_total = initial_counts["total"] + 2
        
        # Store original env value
        original_max = os.getenv('MAX_TOTAL_TOOLS')
        os.environ['MAX_TOTAL_TOOLS'] = str(max_total)
        
        try:
            # Try to attach tools that would exceed the limit
            result = await self._attach_tools_via_api(
                api_client, api_base_url, test_agent_id,
                query="data processing and analysis tools",
                limit=5
            )
            
            # Check final state
            final_counts = await self._get_tool_counts(test_agent_id)
            
            # Verify pre-attach pruning prevented limit violation
            assert final_counts["total"] <= max_total, \
                f"Total tools ({final_counts['total']}) exceeds MAX_TOTAL_TOOLS ({max_total})"
            
            # Verify the API call succeeded (even though pruning happened)
            assert result.get("success") == True, \
                f"API call failed: {result.get('error')}"
        
        finally:
            # Restore original env value
            if original_max:
                os.environ['MAX_TOTAL_TOOLS'] = original_max
            else:
                del os.environ['MAX_TOTAL_TOOLS']
    
    @pytest.mark.asyncio
    async def test_preattach_pruning_enforces_max_mcp_tools(self, api_client, test_agent_id, api_base_url):
        """
        Test that pre-attach pruning prevents exceeding MAX_MCP_TOOLS.
        
        Scenario:
        1. Get current MCP tool count
        2. Set MAX_MCP_TOOLS to current count + 1
        3. Try to attach 3 new MCP tools
        4. Verify pre-attach pruning runs
        5. Verify final MCP count <= MAX_MCP_TOOLS
        """
        # Get initial state
        initial_counts = await self._get_tool_counts(test_agent_id)
        max_mcp = initial_counts["mcp"] + 1
        
        # Store original env values
        original_max_mcp = os.getenv('MAX_MCP_TOOLS')
        os.environ['MAX_MCP_TOOLS'] = str(max_mcp)
        
        try:
            # Try to attach MCP tools that would exceed the limit
            result = await self._attach_tools_via_api(
                api_client, api_base_url, test_agent_id,
                query="external API integration tools",
                limit=3
            )
            
            # Check final state
            final_counts = await self._get_tool_counts(test_agent_id)
            
            # Verify pre-attach pruning prevented MCP limit violation
            assert final_counts["mcp"] <= max_mcp, \
                f"MCP tools ({final_counts['mcp']}) exceeds MAX_MCP_TOOLS ({max_mcp})"
            
            # Verify the API call succeeded
            assert result.get("success") == True, \
                f"API call failed: {result.get('error')}"
        
        finally:
            # Restore original env value
            if original_max_mcp:
                os.environ['MAX_MCP_TOOLS'] = original_max_mcp
            else:
                del os.environ['MAX_MCP_TOOLS']
    
    @pytest.mark.asyncio
    async def test_preattach_pruning_respects_min_mcp_tools(self, api_client, test_agent_id, api_base_url):
        """
        Test that pre-attach pruning never removes tools below MIN_MCP_TOOLS.
        
        Scenario:
        1. Set MAX_MCP_TOOLS to MIN_MCP_TOOLS + 2
        2. Ensure agent has MIN_MCP_TOOLS + 1 tools
        3. Try to attach 3 new tools
        4. Verify agent never goes below MIN_MCP_TOOLS during pruning
        """
        min_mcp = int(os.getenv('MIN_MCP_TOOLS', '7'))
        max_mcp = min_mcp + 2
        
        # Store original env
        original_max_mcp = os.getenv('MAX_MCP_TOOLS')
        os.environ['MAX_MCP_TOOLS'] = str(max_mcp)
        
        try:
            # Initial counts
            initial_counts = await self._get_tool_counts(test_agent_id)
            
            # Attach tools
            result = await self._attach_tools_via_api(
                api_client, api_base_url, test_agent_id,
                query="utility and helper tools",
                limit=3
            )
            
            # Check final state
            final_counts = await self._get_tool_counts(test_agent_id)
            
            # Verify MIN_MCP_TOOLS respected
            assert final_counts["mcp"] >= min_mcp, \
                f"MCP tools ({final_counts['mcp']}) fell below MIN_MCP_TOOLS ({min_mcp})"
            
            # Verify API succeeded
            assert result.get("success") == True, \
                f"API call failed: {result.get('error')}"
        
        finally:
            # Restore original env
            if original_max_mcp:
                os.environ['MAX_MCP_TOOLS'] = original_max_mcp
            else:
                del os.environ['MAX_MCP_TOOLS']
    
    @pytest.mark.asyncio
    async def test_preattach_pruning_skipped_when_no_query(self, api_client, test_agent_id, api_base_url):
        """
        Test that pre-attach pruning is skipped when no query is provided.
        
        Pre-attach pruning requires a query to determine tool relevance.
        Without a query, it should skip pruning and may fail attachment
        if limits would be exceeded.
        """
        # This test verifies the warning path when query is missing
        # In production, this should either:
        # 1. Fail the attachment request, OR
        # 2. Warn but allow attachment (current behavior)
        
        # Get initial state
        initial_counts = await self._get_tool_counts(test_agent_id)
        max_total = initial_counts["total"] + 1
        
        # Store original env
        original_max = os.getenv('MAX_TOTAL_TOOLS')
        os.environ['MAX_TOTAL_TOOLS'] = str(max_total)
        
        try:
            # Try to attach without query (should skip pre-attach pruning)
            # Note: This may fail or warn depending on implementation
            result = await self._attach_tools_via_api(
                api_client, api_base_url, test_agent_id,
                query="",  # Empty query
                limit=3
            )
            
            # Either way, verify system doesn't crash
            assert "error" in result or "success" in result, \
                "API should return structured response even without query"
        
        finally:
            # Restore original env
            if original_max:
                os.environ['MAX_TOTAL_TOOLS'] = original_max
            else:
                del os.environ['MAX_TOTAL_TOOLS']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
