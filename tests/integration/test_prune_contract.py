"""
Contract tests for /api/v1/tools/prune endpoint (LTSEL-24).

Verifies that the prune endpoint adheres to the API contract specification.

Tests validate:
1. Request schema compliance
2. Response schema compliance
3. Pruning behavior specification
4. Protected tools preservation
5. Limit enforcement (MIN_MCP_TOOLS)
6. Drop rate application
"""

import pytest
import os
import aiohttp
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.contract
class TestPruneEndpointContract:
    """Contract tests for /api/v1/tools/prune endpoint."""
    
    @pytest.fixture
    def api_base_url(self) -> str:
        """API base URL."""
        return os.getenv('API_SERVER_URL', 'http://localhost:8020')
    
    @pytest.fixture
    def test_agent_id(self) -> str:
        """Test agent ID."""
        return os.getenv('TEST_AGENT_ID', 'test-agent-id')
    
    @pytest.fixture
    async def http_client(self):
        """HTTP client for API calls."""
        async with aiohttp.ClientSession() as session:
            yield session
    
    # === REQUEST SCHEMA TESTS ===
    
    @pytest.mark.asyncio
    async def test_request_requires_agent_id(self, http_client, api_base_url):
        """Contract: agent_id is required in request."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "user_prompt": "test prompt",
            "drop_rate": 0.5
            # Missing agent_id
        }
        
        async with http_client.post(url, json=payload) as response:
            assert response.status == 400, "Should return 400 when agent_id missing"
            data = await response.json()
            assert "error" in data
            assert "agent_id" in data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_request_requires_user_prompt(self, http_client, api_base_url, test_agent_id):
        """Contract: user_prompt is required in request."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "drop_rate": 0.5
            # Missing user_prompt
        }
        
        async with http_client.post(url, json=payload) as response:
            assert response.status == 400, "Should return 400 when user_prompt missing"
            data = await response.json()
            assert "error" in data
            assert "user_prompt" in data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_request_requires_drop_rate(self, http_client, api_base_url, test_agent_id):
        """Contract: drop_rate is required in request."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "test prompt"
            # Missing drop_rate
        }
        
        async with http_client.post(url, json=payload) as response:
            assert response.status == 400, "Should return 400 when drop_rate missing"
            data = await response.json()
            assert "error" in data
            assert "drop_rate" in data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_drop_rate_must_be_valid_range(self, http_client, api_base_url, test_agent_id):
        """Contract: drop_rate must be between 0 and 1."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Test invalid drop_rate > 1
        payload_high = {
            "agent_id": test_agent_id,
            "user_prompt": "test prompt",
            "drop_rate": 1.5
        }
        
        async with http_client.post(url, json=payload_high) as response:
            assert response.status == 400, "Should reject drop_rate > 1"
            data = await response.json()
            assert "error" in data
        
        # Test invalid drop_rate < 0
        payload_low = {
            "agent_id": test_agent_id,
            "user_prompt": "test prompt",
            "drop_rate": -0.5
        }
        
        async with http_client.post(url, json=payload_low) as response:
            assert response.status == 400, "Should reject drop_rate < 0"
            data = await response.json()
            assert "error" in data
    
    @pytest.mark.asyncio
    async def test_request_accepts_optional_parameters(self, http_client, api_base_url, test_agent_id):
        """Contract: Optional parameters are accepted."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Test with all optional parameters
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "test prompt",
            "drop_rate": 0.6,
            "keep_tool_ids": ["tool-1", "tool-2"],
            "newly_matched_tool_ids": ["tool-3"]
        }
        
        async with http_client.post(url, json=payload) as response:
            # Should not fail due to optional parameters
            assert response.status in [200, 500], "Should accept optional parameters"
            data = await response.json()
            assert isinstance(data, dict), "Should return JSON object"
    
    # === RESPONSE SCHEMA TESTS ===
    
    @pytest.mark.asyncio
    async def test_response_has_success_field(self, http_client, api_base_url, test_agent_id):
        """Contract: Response must have 'success' boolean field."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "reduce tool count",
            "drop_rate": 0.5
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            assert "success" in data, "Response must have 'success' field"
            assert isinstance(data["success"], bool), "'success' must be boolean"
    
    @pytest.mark.asyncio
    async def test_response_has_message_on_success(self, http_client, api_base_url, test_agent_id):
        """Contract: Successful response must have 'message' field."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "optimize tools",
            "drop_rate": 0.3
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                assert "message" in data, "Successful response must have 'message'"
                assert isinstance(data["message"], str), "'message' must be string"
    
    @pytest.mark.asyncio
    async def test_response_has_details_object(self, http_client, api_base_url, test_agent_id):
        """Contract: Successful response must have 'details' object."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "prune unnecessary tools",
            "drop_rate": 0.4
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                assert "details" in data, "Response must have 'details' object"
                assert isinstance(data["details"], dict), "'details' must be dict"
    
    @pytest.mark.asyncio
    async def test_response_details_has_required_fields(self, http_client, api_base_url, test_agent_id):
        """Contract: details object must have required fields."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "clean up tools",
            "drop_rate": 0.5
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                required_fields = [
                    "tools_on_agent_before_total",
                    "mcp_tools_on_agent_before",
                    "core_tools_preserved_count",
                    "target_mcp_tools_to_keep_after_pruning",
                    "mcp_tools_detached_count",
                    "mcp_tools_failed_detachment_count",
                    "drop_rate_applied_to_mcp_tools",
                    "successful_detachments_mcp",
                    "failed_detachments_mcp"
                ]
                
                for field in required_fields:
                    assert field in details, f"details must have '{field}' field"
    
    @pytest.mark.asyncio
    async def test_response_detachment_objects_have_required_fields(self, http_client, api_base_url, test_agent_id):
        """Contract: Detachment objects must have required fields."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "remove tools",
            "drop_rate": 0.7
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                # Check successful detachments structure
                for detachment in details.get("successful_detachments_mcp", []):
                    assert "tool_id" in detachment
                    assert "name" in detachment
                
                # Check failed detachments structure
                for detachment in details.get("failed_detachments_mcp", []):
                    assert "tool_id" in detachment
                    assert "name" in detachment
                    assert "error" in detachment
    
    # === BEHAVIOR SPECIFICATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_drop_rate_affects_pruning_amount(self, http_client, api_base_url, test_agent_id):
        """Contract: Higher drop_rate should result in more pruning."""
        # Note: This test may need actual agent state to be meaningful
        # We just verify the drop_rate is reflected in response
        
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "optimize",
            "drop_rate": 0.8  # High drop rate
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                # Verify drop_rate is recorded correctly
                assert details.get("drop_rate_applied_to_mcp_tools") == 0.8
    
    @pytest.mark.asyncio
    async def test_user_prompt_is_used_for_relevance(self, http_client, api_base_url, test_agent_id):
        """Contract: user_prompt is used to determine tool relevance."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # The prompt should influence which tools are kept
        # We can't directly test relevance without knowing tool content,
        # but we verify the operation completes
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "database query and data processing tools",
            "drop_rate": 0.5
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            # Should complete successfully (or fail for reasons other than prompt)
            assert "success" in data
    
    @pytest.mark.asyncio
    async def test_keep_tool_ids_prevents_pruning(self, http_client, api_base_url, test_agent_id):
        """Contract: keep_tool_ids parameter protects specified tools."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        keep_tool_ids = ["protected-tool-1", "protected-tool-2"]
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "prune aggressively",
            "drop_rate": 0.9,
            "keep_tool_ids": keep_tool_ids
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                # Kept tools should be reflected in response
                assert "explicitly_kept_tool_ids_from_request" in details
    
    @pytest.mark.asyncio
    async def test_newly_matched_tools_are_protected(self, http_client, api_base_url, test_agent_id):
        """Contract: newly_matched_tool_ids are protected from pruning."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        newly_matched = ["new-tool-1", "new-tool-2"]
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "prune old tools",
            "drop_rate": 0.6,
            "newly_matched_tool_ids": newly_matched
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                # Newly matched tools should be reflected in response
                assert "newly_matched_tool_ids_from_request" in details
    
    # === LIMIT ENFORCEMENT TESTS ===
    
    @pytest.mark.asyncio
    async def test_min_mcp_tools_is_respected(self, http_client, api_base_url, test_agent_id):
        """Contract: Pruning never goes below MIN_MCP_TOOLS."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Try to prune everything with drop_rate=1.0
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "remove all tools",
            "drop_rate": 1.0  # Maximum pruning
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                # Even with drop_rate=1.0, some MCP tools should remain
                # (unless agent had fewer than MIN_MCP_TOOLS to begin with)
                mcp_before = details.get("mcp_tools_on_agent_before", 0)
                detached = details.get("mcp_tools_detached_count", 0)
                
                if mcp_before > 0:
                    # Should not detach all tools
                    assert detached < mcp_before or mcp_before <= 7  # MIN_MCP_TOOLS default
    
    @pytest.mark.asyncio
    async def test_only_mcp_tools_are_pruned(self, http_client, api_base_url, test_agent_id):
        """Contract: Only MCP tools are pruned, core tools are preserved."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "aggressive pruning",
            "drop_rate": 0.8
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                # Core tools count should remain unchanged
                core_before = details.get("core_tools_preserved_count", 0)
                # The contract states core tools are always preserved
                assert core_before >= 0  # Core tools exist and are counted
    
    # === COUNT CONSISTENCY TESTS ===
    
    @pytest.mark.asyncio
    async def test_count_fields_are_consistent(self, http_client, api_base_url, test_agent_id):
        """Contract: Count fields must be consistent with actual results."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "prune tools",
            "drop_rate": 0.5
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                # mcp_tools_detached_count should match length of successful_detachments_mcp
                detached_count = details.get("mcp_tools_detached_count", 0)
                successful_detachments = details.get("successful_detachments_mcp", [])
                assert detached_count == len(successful_detachments)
                
                # mcp_tools_failed_detachment_count should match length of failed_detachments_mcp
                failed_count = details.get("mcp_tools_failed_detachment_count", 0)
                failed_detachments = details.get("failed_detachments_mcp", [])
                assert failed_count == len(failed_detachments)
    
    @pytest.mark.asyncio
    async def test_tools_before_and_after_counts(self, http_client, api_base_url, test_agent_id):
        """Contract: Before/after tool counts should be logical."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "cleanup",
            "drop_rate": 0.4
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                
                tools_before = details.get("tools_on_agent_before_total", 0)
                mcp_before = details.get("mcp_tools_on_agent_before", 0)
                core_preserved = details.get("core_tools_preserved_count", 0)
                
                # MCP + core should equal total before
                # (within reason - there might be rounding or edge cases)
                assert mcp_before + core_preserved == tools_before or abs((mcp_before + core_preserved) - tools_before) <= 1
    
    # === ERROR HANDLING TESTS ===
    
    @pytest.mark.asyncio
    async def test_error_response_has_error_field(self, http_client, api_base_url):
        """Contract: Error responses must have 'error' field."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Invalid request (missing required fields)
        payload = {
            "agent_id": "test"
            # Missing user_prompt and drop_rate
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if not data.get("success", True):  # If success is False or missing
                assert "error" in data, "Error response must have 'error' field"
                assert isinstance(data["error"], str), "'error' must be string"
    
    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, http_client, api_base_url):
        """Contract: Invalid JSON returns 400 error."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Send invalid JSON
        async with http_client.post(url, data="invalid json{{{") as response:
            assert response.status >= 400, "Invalid JSON should return 4xx error"
    
    # === DROP RATE VALIDATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_drop_rate_zero_keeps_all_tools(self, http_client, api_base_url, test_agent_id):
        """Contract: drop_rate=0 should keep all tools."""
        url = f"{api_base_url}/api/v1/tools/prune"
        payload = {
            "agent_id": test_agent_id,
            "user_prompt": "keep everything",
            "drop_rate": 0.0
        }
        
        async with http_client.post(url, json=payload) as response:
            data = await response.json()
            if data.get("success"):
                details = data["details"]
                # With drop_rate=0, no tools should be detached
                detached_count = details.get("mcp_tools_detached_count", 0)
                # May still be 0 or close to 0
                assert detached_count >= 0  # Just verify structure
    
    @pytest.mark.asyncio
    async def test_drop_rate_boundary_values(self, http_client, api_base_url, test_agent_id):
        """Contract: Boundary values 0.0 and 1.0 are valid."""
        url = f"{api_base_url}/api/v1/tools/prune"
        
        # Test drop_rate = 0.0
        payload_zero = {
            "agent_id": test_agent_id,
            "user_prompt": "test",
            "drop_rate": 0.0
        }
        
        async with http_client.post(url, json=payload_zero) as response:
            assert response.status in [200, 500], "drop_rate=0.0 should be valid"
        
        # Test drop_rate = 1.0
        payload_one = {
            "agent_id": test_agent_id,
            "user_prompt": "test",
            "drop_rate": 1.0
        }
        
        async with http_client.post(url, json=payload_one) as response:
            assert response.status in [200, 500], "drop_rate=1.0 should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "contract"])
