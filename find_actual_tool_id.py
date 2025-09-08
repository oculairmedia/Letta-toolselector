#!/usr/bin/env python3
"""Find the actual tool ID for find_tools in the Letta system"""

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

print("Searching for find_tools tool ID...")
print()

# Get all tools
url = f"{LETTA_URL}/tools"
response = requests.get(url, headers=headers, timeout=10)

if response.status_code == 200:
    tools = response.json()
    print(f"Found {len(tools)} total tools")
    print()
    
    # Search for find_tools
    find_tools_candidates = []
    
    for tool in tools:
        tool_name = tool.get('name', '')
        tool_id = tool.get('id', '')
        tool_description = tool.get('description', '')
        tool_source = tool.get('source', '')
        
        # Look for find_tools by name or description
        if 'find_tools' in tool_name.lower() or 'find tools' in tool_name.lower():
            find_tools_candidates.append(tool)
            print(f"Found candidate by name:")
            print(f"  ID: {tool_id}")
            print(f"  Name: {tool_name}")
            print(f"  Description: {tool_description[:100]}...")
            print(f"  Source: {tool_source}")
            print()
        elif 'tool' in tool_name.lower() and ('find' in tool_name.lower() or 'search' in tool_name.lower() or 'attach' in tool_name.lower()):
            find_tools_candidates.append(tool)
            print(f"Found potential candidate:")
            print(f"  ID: {tool_id}")
            print(f"  Name: {tool_name}")
            print(f"  Description: {tool_description[:100]}...")
            print(f"  Source: {tool_source}")
            print()
        elif 'toolfinder' in str(tool_source).lower() or 'tool finder' in str(tool_source).lower():
            find_tools_candidates.append(tool)
            print(f"Found candidate by source:")
            print(f"  ID: {tool_id}")
            print(f"  Name: {tool_name}")
            print(f"  Description: {tool_description[:100]}...")
            print(f"  Source: {tool_source}")
            print()
    
    if not find_tools_candidates:
        print("No find_tools candidates found!")
        print("\nListing all tools with 'tool' in name:")
        for tool in tools:
            if 'tool' in tool.get('name', '').lower():
                print(f"  - {tool.get('name')} (ID: {tool.get('id')})")
else:
    print(f"Error getting tools: {response.status_code}")
    print(response.text)