#!/usr/bin/env python3
"""Test agent-based tool ID lookup"""

import os
import sys
from letta_tool_utils import get_find_tools_id, get_find_tools_id_with_fallback

# Set environment variables
os.environ['LETTA_API_URL'] = 'https://letta.oculair.ca/v1'
os.environ['LETTA_PASSWORD'] = 'lettaSecurePass123'

# Test with the agent ID from the conversation
test_agent_id = "agent-e54fc601-4773-4116-9c6c-cf45da2e269e"

print(f"Testing tool lookup for agent: {test_agent_id}")
print()

# Test direct lookup with agent
print("1. Testing direct lookup with agent ID:")
tool_id = get_find_tools_id(agent_id=test_agent_id)
if tool_id:
    print(f"   Success! Found tool ID: {tool_id}")
else:
    print("   No tool found via agent lookup")

print()

# Test with fallback and agent
print("2. Testing lookup with fallback and agent ID:")
tool_id_with_fallback = get_find_tools_id_with_fallback(agent_id=test_agent_id)
print(f"   Result: {tool_id_with_fallback}")

print()

# Test without agent ID
print("3. Testing without agent ID (global search):")
global_tool_id = get_find_tools_id()
if global_tool_id:
    print(f"   Found globally: {global_tool_id}")
else:
    print("   No tool found globally")

print("\nTest complete!")