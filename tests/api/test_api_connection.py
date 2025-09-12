#!/usr/bin/env python3
import os
import requests
import json

# Load environment
LETTA_URL = os.environ.get('LETTA_API_URL', 'https://letta.oculair.ca/v1').replace('http://', 'https://')
if not LETTA_URL.endswith('/v1'):
    LETTA_URL = LETTA_URL.rstrip('/') + '/v1'

LETTA_API_KEY = os.environ.get('LETTA_PASSWORD')

print(f"Using Letta URL: {LETTA_URL}")
print(f"API Key available: {bool(LETTA_API_KEY)}")

# Prepare headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

if LETTA_API_KEY:
    headers["X-BARE-PASSWORD"] = f"password {LETTA_API_KEY}"

try:
    # Test connection
    print("\nTesting connection to Letta API...")
    response = requests.get(f"{LETTA_URL}/tools", headers=headers, timeout=10)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        tools = response.json()
        print(f"Found {len(tools)} tools")
        
        # Look for find_tools
        for tool in tools:
            if 'find' in tool.get('name', '').lower():
                print(f"  - {tool.get('name')} (ID: {tool.get('id')})")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")