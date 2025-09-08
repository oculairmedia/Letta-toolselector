#!/usr/bin/env python3
"""Find tools by checking their source_type"""

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

# Get all tools
tools_url = f"{LETTA_URL}/tools"
tools_response = requests.get(tools_url, headers=headers, timeout=10)

if tools_response.status_code == 200:
    all_tools = tools_response.json()
    print(f"Found {len(all_tools)} total tools\n")
    
    # Group by source_type
    by_source_type = {}
    for tool in all_tools:
        source_type = tool.get('source_type', 'unknown')
        if source_type not in by_source_type:
            by_source_type[source_type] = []
        by_source_type[source_type].append(tool)
    
    print("Tools grouped by source_type:")
    for source_type, tools in by_source_type.items():
        print(f"\n{source_type}: {len(tools)} tools")
        if source_type == 'mcp' or 'mcp' in source_type.lower():
            # Show all MCP tools
            for tool in tools:
                print(f"  - {tool.get('name')} (ID: {tool.get('id')})")
                if 'find' in tool.get('name', '').lower() or 'tool' in tool.get('name', '').lower():
                    print(f"    Description: {tool.get('description', '')[:80]}...")
                    print(f"    Tags: {tool.get('tags', [])}")
                    print(f"    Source: {tool.get('source', 'N/A')}")
    
    # Also check for tools with specific tags or metadata
    print("\n\nLooking for tools with MCP-related metadata...")
    for tool in all_tools:
        # Check various fields that might indicate MCP origin
        tags = tool.get('tags', [])
        source = str(tool.get('source', '')).lower()
        metadata = tool.get('metadata', {})
        
        is_mcp_related = (
            'mcp' in str(tags).lower() or
            'toolfinder' in source or
            'toolselector' in source or
            (metadata and 'mcp' in str(metadata).lower())
        )
        
        if is_mcp_related and ('find' in tool.get('name', '').lower() or 'attach' in tool.get('name', '').lower()):
            print(f"\nPotential match:")
            print(f"  ID: {tool.get('id')}")
            print(f"  Name: {tool.get('name')}")
            print(f"  Source Type: {tool.get('source_type')}")
            print(f"  Source: {tool.get('source')}")
            print(f"  Tags: {tags}")
            print(f"  Description: {tool.get('description', '')[:100]}...")
else:
    print(f"Error: {tools_response.status_code}")
    print(tools_response.text)