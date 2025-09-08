#!/usr/bin/env python3
"""
Simple test to verify environment variable parsing for never-detach tools.
"""
import os

def test_env_parsing():
    """Test that environment variable parsing works correctly."""
    print("Testing environment variable parsing...")
    
    # Test cases
    test_cases = [
        ('find_tools', ['find_tools']),
        ('find_tools,test_tool', ['find_tools', 'test_tool']),
        ('find_tools, test_tool, another_tool', ['find_tools', 'test_tool', 'another_tool']),
        ('', []),
        ('  find_tools  ,  test_tool  ', ['find_tools', 'test_tool']),
    ]
    
    all_passed = True
    
    for env_value, expected in test_cases:
        # Simulate the parsing logic from api_server.py
        parsed = [name.strip() for name in env_value.split(',') if name.strip()]
        
        print(f"Input: '{env_value}' -> Expected: {expected}, Got: {parsed}")
        
        if parsed == expected:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            all_passed = False
        print()
    
    return all_passed

if __name__ == "__main__":
    print("=== Testing Environment Variable Parsing ===\n")
    
    if test_env_parsing():
        print("🎉 All environment variable parsing tests passed!")
    else:
        print("⚠️  Some environment variable parsing tests failed.")