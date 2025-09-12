#!/usr/bin/env python3
"""Test script to verify dynamic tool ID lookup is working"""

import os
import sys
from letta_tool_utils import get_find_tools_id, get_find_tools_id_with_fallback

# Set environment variables if not already set
if not os.environ.get('LETTA_API_URL'):
    os.environ['LETTA_API_URL'] = 'https://letta.oculair.ca/v1'

if not os.environ.get('LETTA_PASSWORD'):
    print("Warning: LETTA_PASSWORD not set, API calls may fail")

print("Testing dynamic tool ID lookup...")
print(f"API URL: {os.environ.get('LETTA_API_URL', 'Not set')}")
print(f"API Key set: {'Yes' if os.environ.get('LETTA_PASSWORD') else 'No'}")
print()

# Test direct lookup
print("1. Testing direct lookup (get_find_tools_id):")
tool_id = get_find_tools_id()
if tool_id:
    print(f"   Success! Found tool ID: {tool_id}")
else:
    print("   Failed to find tool ID")

print()

# Test with fallback
print("2. Testing lookup with fallback:")
fallback_id = "tool-d4a7c168-3123-4b19-91b5-809320fdddf8"
tool_id_with_fallback = get_find_tools_id_with_fallback(fallback_id)
print(f"   Result: {tool_id_with_fallback}")
print(f"   Used fallback: {'Yes' if tool_id_with_fallback == fallback_id else 'No'}")

print()
print("3. Testing with custom fallback:")
custom_fallback = "tool-custom-fallback-id"
custom_result = get_find_tools_id_with_fallback(custom_fallback)
print(f"   Result: {custom_result}")
print(f"   Used custom fallback: {'Yes' if custom_result == custom_fallback else 'No'}")

print("\nTest complete!")