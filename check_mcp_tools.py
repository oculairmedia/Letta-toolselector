#!/usr/bin/env python3
"""Check MCP servers and find the tool ID mapping"""

import os
import requests
import json

# Set environment variables
os.environ['LETTA_API_URL'] = 'https://letta.oculair.ca/v1'
os.environ['LETTA_PASSWORD'] = 'lettaSecurePass123'

LETTA_URL = 'https://letta.oculair.ca/v1'
LETTA_API_KEY = 'lettaSecurePass123'

headers = {
    "Authorization": f"Bearer {LETTA_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# First, get MCP servers
mcp_url = f"{LETTA_URL}/tools/mcp/servers"
mcp_response = requests.get(mcp_url, headers=headers, timeout=10)

if mcp_response.status_code == 200:
    mcp_servers = mcp_response.json()
    print(f"Found {len(mcp_servers)} MCP servers")
    
    # Check toolfinder specifically
    if 'toolfinder' in mcp_servers:
        print("\nToolfinder MCP server details:")
        print(json.dumps(mcp_servers['toolfinder'], indent=2))
        
        # Get tools from toolfinder
        tools_url = f"{LETTA_URL}/tools/mcp/servers/toolfinder/tools"
        tools_response = requests.get(tools_url, headers=headers, timeout=10)
        
        if tools_response.status_code == 200:
            toolfinder_tools = tools_response.json()
            print(f"\nToolfinder has {len(toolfinder_tools)} tools:")
            
            for tool in toolfinder_tools:
                print(f"\nTool: {tool.get('name')}")
                print(f"Description: {tool.get('description', '')[:100]}...")
                print(f"Full tool info:")
                print(json.dumps(tool, indent=2))
                
    # Now check toolselector if it exists
    if 'toolselector' in mcp_servers:
        print("\n\nToolselector MCP server details:")
        print(json.dumps(mcp_servers['toolselector'], indent=2))
        
        # Get tools from toolselector
        tools_url = f"{LETTA_URL}/tools/mcp/servers/toolselector/tools"
        tools_response = requests.get(tools_url, headers=headers, timeout=10)
        
        if tools_response.status_code == 200:
            toolselector_tools = tools_response.json()
            print(f"\nToolselector has {len(toolselector_tools)} tools:")
            
            for tool in toolselector_tools:
                print(f"\nTool: {tool.get('name')}")
                print(f"Description: {tool.get('description', '')[:100]}...")

# Now get all regular tools and search for MCP-sourced ones
print("\n\nSearching regular tools for MCP-sourced find_tools...")
tools_url = f"{LETTA_URL}/tools"
tools_response = requests.get(tools_url, headers=headers, timeout=10)

if tools_response.status_code == 200:
    all_tools = tools_response.json()
    
    # Look for tools that might be from MCP servers
    for tool in all_tools:
        tool_source = str(tool.get('source', '')).lower()
        tool_name = tool.get('name', '').lower()
        tool_id = tool.get('id', '')
        
        # Check if this is from an MCP server and related to find_tools
        if ('mcp' in tool_source or 'toolfinder' in tool_source or 'toolselector' in tool_source):
            if 'find' in tool_name or 'tool' in tool_name:
                print(f"\nFound MCP-sourced tool:")
                print(f"  ID: {tool_id}")
                print(f"  Name: {tool.get('name')}")
                print(f"  Source: {tool.get('source')}")
                print(f"  Description: {tool.get('description', '')[:100]}...")