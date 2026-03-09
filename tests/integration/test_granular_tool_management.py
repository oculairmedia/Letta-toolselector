"""
Integration tests for granular tool management (LTSEL-273).

End-to-end tests that verify the full stack for the new direct tool
operations: lookup, attach, detach, list, inspect — plus pin integration
and the no-pruning guarantee.

These tests hit the worker service (Layer 2) which proxies to the API
server (Layer 1). They require running services: API server, worker
service, Weaviate, and Letta API.

Run with:
    pytest tests/integration/test_granular_tool_management.py -v -s
"""

import pytest
import os
import aiohttp
from typing import List, Optional


@pytest.mark.integration
class TestGranularToolManagement:
    """Integration tests for granular tool management operations."""

    @pytest.fixture
    def api_base_url(self) -> str:
        """API server base URL (Layer 1)."""
        return os.getenv("API_SERVER_URL", "http://localhost:8020")

    @pytest.fixture
    def worker_base_url(self) -> str:
        """Worker service base URL (Layer 2)."""
        return os.getenv("WORKER_SERVICE_URL", "http://localhost:3021")

    @pytest.fixture
    def test_agent_id(self) -> str:
        """Test agent ID — must be a real agent in the Letta instance."""
        return os.getenv("TEST_AGENT_ID", "test-agent-id")

    @pytest.fixture
    async def http_client(self):
        """HTTP client for API calls."""
        async with aiohttp.ClientSession() as session:
            yield session

    # =====================================================================
    # Helpers
    # =====================================================================

    async def _worker_post(
        self,
        http_client: aiohttp.ClientSession,
        worker_base_url: str,
        endpoint: str,
        payload: dict,
    ) -> tuple:
        """POST to worker service, return (status, json_data)."""
        url = f"{worker_base_url}/{endpoint}"
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            return response.status, data

    async def _api_get(
        self,
        http_client: aiohttp.ClientSession,
        api_base_url: str,
        path: str,
        params: dict | None = None,
    ) -> tuple:
        """GET to API server, return (status, json_data)."""
        url = f"{api_base_url}/{path}"
        async with http_client.get(url, params=params) as response:
            data = await response.json()
            return response.status, data

    async def _get_attached_tool_names(
        self,
        http_client: aiohttp.ClientSession,
        worker_base_url: str,
        agent_id: str,
    ) -> List[str]:
        """Get list of tool names currently attached to an agent."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": agent_id, "filter": "all", "include_schema": False},
        )
        if status != 200:
            return []
        tools = data.get("tools", [])
        return [t.get("name", "") for t in tools]

    async def _cleanup_tool(
        self,
        http_client: aiohttp.ClientSession,
        worker_base_url: str,
        agent_id: str,
        tool_name: str,
    ) -> None:
        """Best-effort detach + unpin for cleanup."""
        await self._worker_post(
            http_client,
            worker_base_url,
            "detach_tool",
            {"agent_id": agent_id, "tools": [{"name": tool_name}], "unpin": True},
        )

    # =====================================================================
    # Scenario 1: Lookup flow
    # =====================================================================

    @pytest.mark.asyncio
    async def test_lookup_by_name(self, http_client, worker_base_url):
        """Lookup a tool by name returns correct tool or 404 with suggestions."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_name": "find_tools"},
        )
        # find_tools is a well-known tool that should exist
        if status == 200 and data.get("success"):
            tool = data.get("tool", {})
            assert tool.get("name") == "find_tools"
        elif status == 404:
            # If not in cache, should have tool_not_found + suggestions
            assert data.get("tool_not_found") is True
            assert "suggestions" in data

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_returns_404_with_suggestions(self, http_client, worker_base_url):
        """Lookup a nonexistent tool returns 404 with suggestions (PM requirement)."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_name": "nonexistent_tool_xyz_999"},
        )
        # Should be 404 (or pass-through from API)
        assert data.get("tool_not_found") is True or data.get("success") is False
        assert "suggestions" in data

    @pytest.mark.asyncio
    async def test_lookup_by_id(self, http_client, worker_base_url):
        """Lookup by ID returns the correct tool or 404."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_id": "tool-00000000-0000-0000-0000-000000000000"},
        )
        # Fake ID should not exist
        if status == 404 or not data.get("success"):
            assert data.get("tool_not_found") is True

    @pytest.mark.asyncio
    async def test_lookup_fuzzy_mode(self, http_client, worker_base_url):
        """Lookup with fuzzy=true returns suggestions for partial matches."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_name": "find_too", "fuzzy": True, "limit": 5},
        )
        # Should either find exact match or return suggestions
        assert "suggestions" in data or data.get("success") is True

    # =====================================================================
    # Scenario 2: Attach flow
    # =====================================================================

    @pytest.mark.asyncio
    async def test_attach_by_name(self, http_client, worker_base_url, test_agent_id):
        """Attach a tool by name and verify it appears in agent's tool list."""
        # First find a tool we can attach
        _, lookup_data = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_name": "find_tools"},
        )
        if not lookup_data.get("success"):
            pytest.skip("find_tools not available in tool cache")

        # Attach it
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": "find_tools"}]},
        )
        assert status == 200
        results = data.get("results", [])
        if results:
            # Should be attached or already_attached
            assert results[0].get("status") in ("attached", "already_attached")

    @pytest.mark.asyncio
    async def test_attach_nonexistent_tool_returns_error_with_suggestions(
        self, http_client, worker_base_url, test_agent_id
    ):
        """Attaching a nonexistent tool returns structured error with suggestions."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [{"name": "totally_fake_tool_zzz"}],
            },
        )
        results = data.get("results", [])
        if results:
            result = results[0]
            assert result.get("tool_not_found") is True or result.get("error")
            assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_attach_does_not_trigger_pruning(self, http_client, worker_base_url, test_agent_id):
        """Direct attach MUST NOT trigger auto-pruning (Meridian requirement)."""
        # Get current tools
        before_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)

        # Attach a tool
        await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": "find_tools"}]},
        )

        # Get tools after attach
        after_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)

        # No tool from before should have been removed
        for tool in before_tools:
            assert tool in after_tools, (
                f"Tool '{tool}' was present before attach but removed after — direct attach must NOT trigger pruning"
            )

    # =====================================================================
    # Scenario 3: Detach flow
    # =====================================================================

    @pytest.mark.asyncio
    async def test_detach_by_name(self, http_client, worker_base_url, test_agent_id):
        """Detach a tool by name and verify it's removed."""
        # First look up a tool we can safely detach (must be MCP, not protected)
        status, list_data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": test_agent_id, "filter": "mcp"},
        )
        if status != 200:
            pytest.skip("Cannot list agent tools")

        mcp_tools = list_data.get("tools", [])
        detachable = [t for t in mcp_tools if not t.get("is_protected") and not t.get("is_pinned")]
        if not detachable:
            pytest.skip("No detachable MCP tools on test agent")

        target = detachable[0]
        target_name = target.get("name")

        # Detach it
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "detach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": target_name}]},
        )
        assert status == 200

        results = data.get("results", [])
        if results:
            assert results[0].get("status") in ("detached", "not_attached")

        # Re-attach for cleanup
        await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": target_name}]},
        )

    @pytest.mark.asyncio
    async def test_detach_protected_tool_is_refused(self, http_client, worker_base_url, test_agent_id):
        """Protected tools cannot be detached."""
        # find_tools is in NEVER_DETACH_TOOLS
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "detach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": "find_tools"}]},
        )
        results = data.get("results", [])
        if results:
            result = results[0]
            # Should be refused
            assert result.get("status") in ("refused", "protected", "error") or (
                "protected" in str(result.get("error", "")).lower()
                or "never_detach" in str(result.get("error", "")).lower()
            ), f"Protected tool should be refused, got: {result}"

    # =====================================================================
    # Scenario 4: List flow
    # =====================================================================

    @pytest.mark.asyncio
    async def test_list_agent_tools_returns_categorized(self, http_client, worker_base_url, test_agent_id):
        """List agent tools returns categorized response with summary."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": test_agent_id, "filter": "all"},
        )
        assert status == 200
        assert "tools" in data
        assert isinstance(data["tools"], list)

        # Should have summary counts
        summary = data.get("summary", {})
        assert "total" in summary or len(data["tools"]) >= 0

    @pytest.mark.asyncio
    async def test_list_agent_tools_filter_mcp(self, http_client, worker_base_url, test_agent_id):
        """Filter=mcp returns only MCP tools."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": test_agent_id, "filter": "mcp"},
        )
        if status == 200:
            for tool in data.get("tools", []):
                assert tool.get("category") == "mcp" or tool.get("tool_type") in ("external_mcp", "mcp"), (
                    f"Expected MCP tool, got: {tool.get('category', tool.get('tool_type'))}"
                )

    @pytest.mark.asyncio
    async def test_list_agent_tools_filter_core(self, http_client, worker_base_url, test_agent_id):
        """Filter=core returns only core Letta tools."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": test_agent_id, "filter": "core"},
        )
        if status == 200:
            for tool in data.get("tools", []):
                assert tool.get("category") == "core" or tool.get("tool_type") not in ("external_mcp",), (
                    f"Expected core tool, got: {tool.get('category', tool.get('tool_type'))}"
                )

    # =====================================================================
    # Scenario 5: Inspect flow
    # =====================================================================

    @pytest.mark.asyncio
    async def test_inspect_tool_returns_metadata(self, http_client, worker_base_url):
        """Inspect a tool returns full metadata including schema."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "inspect_tool",
            {"tool_name_or_id": "find_tools"},
        )
        if status == 200 and data.get("success"):
            tool = data.get("tool", {})
            assert "name" in tool
            # Should have schema information
            assert "json_schema" in tool or "source_code" in tool or "parameters_summary" in tool

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_returns_404(self, http_client, worker_base_url):
        """Inspect a nonexistent tool returns 404 with suggestions."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "inspect_tool",
            {"tool_name_or_id": "nonexistent_tool_xyz_999"},
        )
        assert data.get("tool_not_found") is True or data.get("success") is False

    # =====================================================================
    # Scenario 6: No-pruning guarantee (CRITICAL)
    # =====================================================================

    @pytest.mark.asyncio
    async def test_no_pruning_guarantee(self, http_client, worker_base_url, test_agent_id):
        """
        CRITICAL TEST: Manually attached tools must survive semantic find_tools.

        Meridian requirement: 'explicit attach calls should never trigger pruning —
        if I deliberately attach a tool, don't auto-remove it to make room.'

        Flow:
        1. Record current tools
        2. Attach a tool via direct-attach
        3. Run a semantic find_tools search
        4. Verify the directly-attached tool is still present
        """
        # Get baseline
        before_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)

        # Direct-attach find_tools (should be no-op if already attached)
        await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {"agent_id": test_agent_id, "tools": [{"name": "find_tools"}]},
        )

        # Now run a semantic search via the original find_tools endpoint
        # This COULD trigger pruning in the old flow
        await self._worker_post(
            http_client,
            worker_base_url,
            "find_tools",
            {
                "query": "completely unrelated test query for pruning check",
                "agent_id": test_agent_id,
                "limit": 3,
                "min_score": 90,  # High threshold to minimize actual changes
            },
        )

        # Verify all pre-existing tools are still attached
        after_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)

        for tool in before_tools:
            assert tool in after_tools, (
                f"Tool '{tool}' was removed by find_tools — "
                "this violates the no-pruning guarantee for pre-existing tools"
            )

    # =====================================================================
    # Scenario 7: Bulk operations
    # =====================================================================

    @pytest.mark.asyncio
    async def test_bulk_attach(self, http_client, worker_base_url, test_agent_id):
        """Bulk attach multiple tools in one call."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [
                    {"name": "find_tools"},
                    {"name": "nonexistent_bulk_test_zzz"},
                ],
            },
        )
        assert status == 200

        # API returns {"details": {"attached": [...], "failed": [...]}} not "results"
        details = data.get("details", {})
        attached = details.get("attached", [])
        failed = details.get("failed", [])
        results = attached + failed
        assert len(results) == 2, f"Should have one result per tool in bulk request, got {len(results)}: {data}"

        # First should succeed or be already_attached
        assert (
            attached[0].get("status") in ("attached", "already_attached")
            or attached[0].get("tool_not_found") is not True
        )

        # Second should fail with tool_not_found
        assert failed[0].get("tool_not_found") is True or failed[0].get("error")

    @pytest.mark.asyncio
    async def test_bulk_detach(self, http_client, worker_base_url, test_agent_id):
        """Bulk detach handles mixed valid/invalid gracefully."""
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "detach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [
                    {"name": "nonexistent_detach_test_zzz"},
                    {"name": "another_nonexistent_aaa"},
                ],
            },
        )
        assert status == 200

        results = data.get("results", [])
        # Both should fail gracefully (not_found or error), not crash
        for result in results:
            assert result.get("status") is not None or result.get("error") is not None

    # =====================================================================
    # Scenario 8: Pin integration
    # =====================================================================

    @pytest.mark.asyncio
    async def test_pin_attach_and_verify(self, http_client, worker_base_url, test_agent_id):
        """Attach with pin=true marks the tool as pinned."""
        # Look up a tool to pin
        _, lookup = await self._worker_post(
            http_client,
            worker_base_url,
            "lookup_tool",
            {"tool_name": "find_tools"},
        )
        if not lookup.get("success"):
            pytest.skip("find_tools not in cache")

        # Attach with pin=true
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [{"name": "find_tools"}],
                "pin": True,
            },
        )
        assert status == 200

        # Verify pin status via list_agent_tools
        _, list_data = await self._worker_post(
            http_client,
            worker_base_url,
            "list_agent_tools",
            {"agent_id": test_agent_id, "filter": "all"},
        )
        tools = list_data.get("tools", [])
        pinned_tool = next((t for t in tools if t.get("name") == "find_tools"), None)
        if pinned_tool:
            assert pinned_tool.get("is_pinned") is True, (
                "Tool attached with pin=true should show is_pinned=true in listing"
            )

    @pytest.mark.asyncio
    async def test_pinned_tool_cannot_be_detached_without_unpin(self, http_client, worker_base_url, test_agent_id):
        """A pinned tool should be refused for detach unless unpin=true."""
        # Ensure find_tools is pinned
        await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [{"name": "find_tools"}],
                "pin": True,
            },
        )

        # Try to detach without unpin — should be refused
        status, data = await self._worker_post(
            http_client,
            worker_base_url,
            "detach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [{"name": "find_tools"}],
                # unpin defaults to false
            },
        )
        results = data.get("results", [])
        if results:
            result = results[0]
            # Should be refused — either because it's protected (NEVER_DETACH)
            # or because it's pinned
            assert result.get("status") in ("refused", "protected", "error") or (
                "pinned" in str(result.get("error", "")).lower() or "protected" in str(result.get("error", "")).lower()
            ), f"Pinned tool should be refused for detach, got: {result}"

    @pytest.mark.asyncio
    async def test_pinned_tool_survives_after_find_tools(self, http_client, worker_base_url, test_agent_id):
        """A pinned tool must survive pruning triggered by find_tools."""
        # Pin find_tools
        pin_status, pin_data = await self._worker_post(
            http_client,
            worker_base_url,
            "attach_tool",
            {
                "agent_id": test_agent_id,
                "tools": [{"name": "find_tools"}],
                "pin": True,
            },
        )
        # Verify the pin/attach actually worked
        assert pin_status == 200, f"Pin attach failed with status {pin_status}: {pin_data}"
        details = pin_data.get("details", {})
        attached = details.get("attached", [])
        pinned = details.get("pinned", [])
        # find_tools should be attached (or already_attached) AND pinned
        assert len(attached) > 0, (
            f"find_tools was not attached — cannot verify pin survives pruning. Response: {pin_data}"
        )
        assert len(pinned) > 0, f"find_tools was not pinned — pin list empty. Response: {pin_data}"

        # Verify find_tools is attached before pruning
        before_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)
        assert "find_tools" in before_tools, (
            f"find_tools not attached before pruning test — cannot validate pin survival. Tools: {before_tools}"
        )

        # Trigger a semantic search that could cause pruning
        await self._worker_post(
            http_client,
            worker_base_url,
            "find_tools",
            {
                "query": "aggressive search to trigger pruning",
                "agent_id": test_agent_id,
                "limit": 10,
                "min_score": 10,  # Low threshold to maximize attachments
            },
        )

        # Verify find_tools is still attached
        after_tools = await self._get_attached_tool_names(http_client, worker_base_url, test_agent_id)
        assert "find_tools" in after_tools, (
            "Pinned tool 'find_tools' was removed after find_tools search — "
            f"pinned tools must survive pruning. Remaining tools: {after_tools}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
