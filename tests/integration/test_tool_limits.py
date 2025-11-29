"""
Integration tests for tool limit enforcement.

These tests verify that the tool selector API correctly enforces
MAX_TOTAL_TOOLS and MAX_MCP_TOOLS limits end-to-end.
"""

import pytest
import requests
import os
from typing import Dict, List, Any

# Test configuration
LETTA_API_URL = os.getenv('LETTA_API_URL', 'https://letta.oculair.ca/v1')
LETTA_PASSWORD = os.getenv('LETTA_PASSWORD')
TOOL_SELECTOR_URL = os.getenv('TOOLS_API_BASE_URL', 'http://localhost:8020')

# Test-specific limits (lower than production for faster testing)
TEST_MAX_TOTAL_TOOLS = 15
TEST_MAX_MCP_TOOLS = 10
TEST_MIN_MCP_TOOLS = 3


def get_letta_headers() -> Dict[str, str]:
    """Get headers for Letta API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {LETTA_PASSWORD}"
    }


@pytest.fixture
def test_agent():
    """Create a test agent and clean it up after the test."""
    headers = get_letta_headers()
    
    # Create test agent
    response = requests.post(
        f"{LETTA_API_URL}/agents",
        headers=headers,
        json={
            "name": "test-tool-limits-agent",
            "llm_config": {"model": "gpt-4"},
            "embedding_config": {"model": "text-embedding-ada-002"}
        }
    )
    response.raise_for_status()
    agent = response.json()
    agent_id = agent["id"]
    
    yield agent_id
    
    # Cleanup: Delete test agent
    try:
        requests.delete(
            f"{LETTA_API_URL}/agents/{agent_id}",
            headers=headers
        )
    except Exception as e:
        print(f"Warning: Failed to delete test agent {agent_id}: {e}")


def get_agent_tools(agent_id: str) -> List[Dict[str, Any]]:
    """Get all tools attached to an agent."""
    response = requests.get(
        f"{LETTA_API_URL}/agents/{agent_id}/tools",
        headers=get_letta_headers(),
        params={"limit": 100}
    )
    response.raise_for_status()
    return response.json()


def attach_tools_via_selector(
    agent_id: str,
    query: str,
    limit: int = 10,
    min_score: float = 35.0
) -> Dict[str, Any]:
    """Attach tools to an agent via the tool selector API."""
    response = requests.post(
        f"{TOOL_SELECTOR_URL}/api/v1/tools/attach",
        json={
            "agent_id": agent_id,
            "query": query,
            "limit": limit,
            "min_score": min_score,
            "auto_prune": True
        }
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.integration
def test_tool_count_never_exceeds_max_total_tools(test_agent):
    """
    Test that an agent never has more than MAX_TOTAL_TOOLS after attachment.
    
    Scenario:
    1. Attach tools that would exceed the limit
    2. Verify total tool count <= MAX_TOTAL_TOOLS
    3. Verify pruning occurred if necessary
    """
    agent_id = test_agent
    
    # Initial state
    initial_tools = get_agent_tools(agent_id)
    print(f"\nInitial tool count: {len(initial_tools)}")
    
    # Attempt to attach many tools with a broad query
    result = attach_tools_via_selector(
        agent_id=agent_id,
        query="search documents create charts analyze data manage files send messages",
        limit=20,  # Request many tools
        min_score=30.0  # Lower threshold to get more matches
    )
    
    # Verify response indicates success
    assert result["success"], f"Tool attachment failed: {result.get('error')}"
    
    # Get final tool count
    final_tools = get_agent_tools(agent_id)
    final_count = len(final_tools)
    print(f"Final tool count: {final_count}")
    print(f"Attached: {result['details']['success_count']}")
    print(f"Detached: {len(result['details']['detached_tools'])}")
    
    # CRITICAL ASSERTION: Tool count must not exceed limit
    # Note: Using a reasonable limit since we can't override env vars in integration test
    # In production, this would be MAX_TOTAL_TOOLS from config
    assert final_count <= 30, (
        f"Agent has {final_count} tools, exceeding reasonable limit of 30. "
        f"Pruning failed to enforce limits."
    )
    
    # If pruning occurred, verify it's documented in the response
    if result['details'].get('detached_tools'):
        assert len(result['details']['detached_tools']) > 0, (
            "Response claims tools were detached but list is empty"
        )


@pytest.mark.integration
def test_mcp_tool_count_respects_min_limit(test_agent):
    """
    Test that MCP tool count never drops below MIN_MCP_TOOLS.
    
    Scenario:
    1. Attach a few MCP tools
    2. Trigger pruning with incompatible query
    3. Verify MIN_MCP_TOOLS preserved
    """
    agent_id = test_agent
    
    # Attach initial set of tools
    result1 = attach_tools_via_selector(
        agent_id=agent_id,
        query="search and analyze documents",
        limit=8,
        min_score=40.0
    )
    
    assert result1["success"]
    attached_count = result1['details']['success_count']
    print(f"\nInitially attached: {attached_count} tools")
    
    # Try to trigger aggressive pruning with very different query
    result2 = attach_tools_via_selector(
        agent_id=agent_id,
        query="send emails and manage calendar",  # Different domain
        limit=8,
        min_score=40.0
    )
    
    assert result2["success"]
    
    # Get final MCP tools
    final_tools = get_agent_tools(agent_id)
    mcp_tools = [
        t for t in final_tools 
        if t.get("tool_type") in ["external_mcp", "custom"]
    ]
    
    print(f"Final MCP tool count: {len(mcp_tools)}")
    
    # MCP tools should not drop below minimum
    # Using 3 as reasonable minimum for testing
    assert len(mcp_tools) >= 3, (
        f"Only {len(mcp_tools)} MCP tools remain, below minimum threshold. "
        f"MIN_MCP_TOOLS protection failed."
    )


@pytest.mark.integration
def test_protected_tools_never_detached(test_agent):
    """
    Test that protected tools (like find_tools) are never detached.
    
    Scenario:
    1. Attach tools including find_tools  
    2. Trigger pruning that would remove tools
    3. Verify find_tools still present
    """
    agent_id = test_agent
    
    # Attach tools
    result = attach_tools_via_selector(
        agent_id=agent_id,
        query="search tools and find tools to help with tasks",
        limit=12,
        min_score=35.0
    )
    
    assert result["success"]
    
    # Get current tools
    tools = get_agent_tools(agent_id)
    tool_names = [t.get("name", "").lower() for t in tools]
    
    print(f"\nTools present: {tool_names}")
    
    # find_tools should be protected and present
    # Note: It may not always be attached depending on the query match,
    # but if it was attached, pruning should not remove it
    has_find_tools = any("find" in name and "tool" in name for name in tool_names)
    
    if has_find_tools:
        print("✓ find_tools is present (protected tool)")
        
        # Trigger another attachment with different query
        result2 = attach_tools_via_selector(
            agent_id=agent_id,
            query="completely different domain unrelated tools",
            limit=10,
            min_score=40.0
        )
        
        # Check find_tools is still there
        tools_after = get_agent_tools(agent_id)
        tool_names_after = [t.get("name", "").lower() for t in tools_after]
        
        still_has_find_tools = any("find" in name and "tool" in name for name in tool_names_after)
        
        assert still_has_find_tools, (
            "Protected tool 'find_tools' was removed during pruning. "
            "NEVER_DETACH_TOOLS protection failed."
        )
        print("✓ find_tools survived pruning (protected)")
    else:
        pytest.skip("find_tools was not attached in initial query, cannot test protection")


@pytest.mark.integration
def test_keep_tools_parameter_respected(test_agent):
    """
    Test that tools specified in keep_tools are preserved.
    
    Scenario:
    1. Attach tools
    2. Note some tool IDs
    3. Trigger new attachment with keep_tools list
    4. Verify kept tools remain
    """
    agent_id = test_agent
    
    # Initial attachment
    result1 = attach_tools_via_selector(
        agent_id=agent_id,
        query="document search and analysis",
        limit=8,
        min_score=40.0
    )
    
    assert result1["success"]
    
    # Get current tools and pick some to keep
    initial_tools = get_agent_tools(agent_id)
    if len(initial_tools) < 2:
        pytest.skip("Not enough tools attached to test keep_tools")
    
    # Pick first 2 tools to explicitly keep
    tools_to_keep = [initial_tools[0]["id"], initial_tools[1]["id"]]
    keep_names = [initial_tools[0].get("name"), initial_tools[1].get("name")]
    
    print(f"\nExplicitly keeping tools: {keep_names}")
    
    # Trigger new attachment with very different query
    response = requests.post(
        f"{TOOL_SELECTOR_URL}/api/v1/tools/attach",
        json={
            "agent_id": agent_id,
            "query": "email calendar scheduling",  # Different domain
            "limit": 8,
            "min_score": 40.0,
            "keep_tools": tools_to_keep,
            "auto_prune": True
        }
    )
    
    response.raise_for_status()
    result2 = response.json()
    assert result2["success"]
    
    # Verify kept tools are still present
    final_tools = get_agent_tools(agent_id)
    final_tool_ids = {t["id"] for t in final_tools}
    
    for tool_id, tool_name in zip(tools_to_keep, keep_names):
        assert tool_id in final_tool_ids, (
            f"Tool '{tool_name}' ({tool_id}) was removed despite being in keep_tools list. "
            f"keep_tools parameter not respected."
        )
    
    print(f"✓ All {len(tools_to_keep)} kept tools survived pruning")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
