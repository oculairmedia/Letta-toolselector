#!/usr/bin/env python3
"""Test script to verify MIN_MCP_TOOLS enforcement in auto-detachment."""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8020"
LETTA_API_URL = os.getenv("LETTA_API_URL", "").rstrip('/')
LETTA_PASSWORD = os.getenv("LETTA_PASSWORD")
MIN_MCP_TOOLS = int(os.getenv("MIN_MCP_TOOLS", "21"))

headers = {
    "Authorization": f"Bearer {LETTA_PASSWORD}",
    "Content-Type": "application/json"
}

def get_agent_tools(agent_id):
    """Get all tools attached to an agent."""
    response = requests.get(
        f"{LETTA_API_URL}/agents/{agent_id}/tools",
        headers=headers
    )
    response.raise_for_status()
    return response.json()

def count_mcp_tools(tools):
    """Count MCP tools (external_mcp or custom non-core tools)."""
    mcp_count = 0
    for tool in tools:
        if tool.get("tool_type") == "external_mcp":
            mcp_count += 1
        elif tool.get("tool_type") == "custom":
            # Check if it's not a core tool
            core_names = ['send_message', 'conversation_search', 'archival_memory_insert', 
                         'archival_memory_search', 'core_memory_append', 'core_memory_replace', 
                         'pause_heartbeats', 'find_attach_tools']
            if tool.get("name") not in core_names:
                mcp_count += 1
    return mcp_count

def test_min_mcp_tools_enforcement():
    """Test that MIN_MCP_TOOLS is respected during auto-detachment."""
    print(f"Testing MIN_MCP_TOOLS enforcement (MIN_MCP_TOOLS={MIN_MCP_TOOLS})")
    print("=" * 60)
    
    # Test agent ID - using the one from logs
    test_agent_id = "agent-e54fc601-4773-4116-9c6c-cf45da2e269e"
    
    # Get current tools
    print(f"\nGetting current tools for agent {test_agent_id}...")
    try:
        current_tools = get_agent_tools(test_agent_id)
        mcp_tool_count = count_mcp_tools(current_tools)
        print(f"Current MCP tools: {mcp_tool_count}")
        print(f"Total tools: {len(current_tools)}")
        
        # List MCP tools
        print("\nCurrent MCP tools:")
        for tool in current_tools:
            if tool.get("tool_type") == "external_mcp" or (tool.get("tool_type") == "custom" and 
                tool.get("name") not in ['send_message', 'conversation_search', 'archival_memory_insert', 
                                       'archival_memory_search', 'core_memory_append', 'core_memory_replace', 
                                       'pause_heartbeats', 'find_attach_tools']):
                print(f"  - {tool.get('name')} (ID: {tool.get('id')})")
        
        # Test attaching a new tool with a query that would normally trigger aggressive detachment
        print(f"\nTesting tool attachment with MIN_MCP_TOOLS enforcement...")
        attach_payload = {
            "agent_id": test_agent_id,
            "query": "web search tools",  # Specific query that should trigger detachment of unrelated tools
            "limit": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/tools/attach",
            json=attach_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAttachment result:")
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            
            details = result.get('details', {})
            print(f"\nDetached tools: {len(details.get('detached_tools', []))}")
            if details.get('detached_tools'):
                print("Detached tool IDs:", details.get('detached_tools'))
            
            print(f"Attached tools: {len(details.get('successful_attachments', []))}")
            
            # Get updated tool count
            updated_tools = get_agent_tools(test_agent_id)
            updated_mcp_count = count_mcp_tools(updated_tools)
            print(f"\nUpdated MCP tool count: {updated_mcp_count}")
            print(f"Updated total tools: {len(updated_tools)}")
            
            # Verify MIN_MCP_TOOLS was respected
            if updated_mcp_count >= MIN_MCP_TOOLS:
                print(f"\n✅ SUCCESS: MIN_MCP_TOOLS ({MIN_MCP_TOOLS}) was respected! Current count: {updated_mcp_count}")
            else:
                print(f"\n❌ FAILURE: MCP tools dropped below MIN_MCP_TOOLS! Current: {updated_mcp_count}, Required: {MIN_MCP_TOOLS}")
                
        else:
            print(f"\n❌ Error attaching tools: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_min_mcp_tools_enforcement()