#!/usr/bin/env python3
"""Test direct API connection"""

import os
import requests

# Set environment variables
os.environ['LETTA_API_URL'] = 'https://letta.oculair.ca/v1'
os.environ['LETTA_PASSWORD'] = 'lettaSecurePass123'

url = "https://letta.oculair.ca/v1/tools"
headers = {
    "Authorization": "Bearer lettaSecurePass123",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

print(f"Testing direct connection to: {url}")
print(f"Headers: {headers}")

try:
    # Disable redirect following to see what's happening
    response = requests.get(url, headers=headers, timeout=10, allow_redirects=False)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code in [301, 302, 303, 307, 308]:
        print(f"Redirect to: {response.headers.get('Location')}")
    
    # Now follow redirects
    response = requests.get(url, headers=headers, timeout=10)
    print(f"\nAfter following redirects:")
    print(f"Final URL: {response.url}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        tools = response.json()
        print(f"Success! Found {len(tools)} tools")
        # List first 3 tools
        for i, tool in enumerate(tools[:3]):
            print(f"  Tool {i+1}: {tool.get('name', 'Unknown')} (ID: {tool.get('id', 'Unknown')})")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")