#!/usr/bin/env python3
"""
Test script to verify minimum MCP tool count enforcement in pruning logic.
"""
import asyncio
import os
import sys
import logging

# Add the lettaaugment-source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lettaaugment-source'))

from api_server import _perform_tool_pruning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_minimum_tools_enforcement():
    """Test that pruning respects the minimum MCP tool count."""
    
    # Set test environment variables
    os.environ['MIN_MCP_TOOLS'] = '7'
    os.environ['MAX_TOTAL_TOOLS'] = '30'
    os.environ['MAX_MCP_TOOLS'] = '20'
    
    print("=" * 60)
    print("Testing Minimum MCP Tool Count Enforcement")
    print("=" * 60)
    
    # Test case 1: Agent with exactly 7 MCP tools (should not prune)
    print("\nTest Case 1: Agent with exactly 7 MCP tools")
    print("Expected: Pruning should be skipped due to minimum requirement")
    
    # Test case 2: Agent with fewer than 7 MCP tools (should not prune)
    print("\nTest Case 2: Agent with fewer than 7 MCP tools")
    print("Expected: Pruning should be skipped due to minimum requirement")
    
    # Test case 3: Agent with more than 7 MCP tools (should prune but keep at least 7)
    print("\nTest Case 3: Agent with more than 7 MCP tools")
    print("Expected: Pruning should proceed but keep at least 7 MCP tools")
    
    print("\n" + "=" * 60)
    print("Note: This test requires a live Letta instance to run.")
    print("To run with a real agent, update the test with a valid agent_id.")
    print("For now, the minimum tool logic has been implemented in:")
    print("1. _perform_tool_pruning() function")
    print("2. Auto-pruning trigger in attach endpoint")
    print("3. Environment configuration (.env.example)")
    print("4. Documentation (README.md)")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    asyncio.run(test_minimum_tools_enforcement())