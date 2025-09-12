#!/usr/bin/env python3
import sys
from letta_tool_utils import get_find_tools_id, get_find_tools_id_with_fallback

def test_dynamic_lookup():
    """Test the dynamic tool ID lookup functionality."""
    print("Testing dynamic tool ID lookup...")
    
    # Test without fallback
    tool_id = get_find_tools_id()
    if tool_id:
        print(f"✅ Successfully found find_tools ID: {tool_id}")
    else:
        print("❌ Could not find find_tools ID (will use fallback)")
    
    # Test with fallback
    tool_id_with_fallback = get_find_tools_id_with_fallback()
    print(f"✅ Tool ID with fallback: {tool_id_with_fallback}")
    
    # Test with custom fallback
    custom_fallback = "tool-custom-fallback-id"
    tool_id_custom = get_find_tools_id_with_fallback(custom_fallback)
    print(f"✅ Tool ID with custom fallback: {tool_id_custom}")

if __name__ == "__main__":
    test_dynamic_lookup()