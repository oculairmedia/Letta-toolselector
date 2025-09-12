import pytest
#!/usr/bin/env python3
"""
Test script for the new tool pruning endpoint.
Tests the /api/v1/tools/prune endpoint with a sample query and agent ID.
"""

import asyncio
import aiohttp
import json
import os
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8020"
LETTA_BASE_URL = "http://192.168.50.90:8289"

# Test data
TEST_AGENT_ID = "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444"  # Use the agent ID from previous tests
TEST_QUERY = "I need to search for remote software engineering jobs"
DEFAULT_DROP_RATE = 0.6

@pytest.mark.asyncio
async def test_pruning_endpoint():
    """Test the tool pruning endpoint"""
    print("=" * 60)
    print("Testing Tool Pruning Endpoint")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test the pruning endpoint
            url = f"{API_BASE_URL}/api/v1/tools/prune"
            
            payload = {
                "user_prompt": TEST_QUERY,
                "agent_id": TEST_AGENT_ID,
                "drop_rate": DEFAULT_DROP_RATE
            }
            
            print(f"Making request to: {url}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            print()
            
            async with session.post(url, json=payload) as response:
                print(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Pruning successful!")
                    print(f"Response: {json.dumps(result, indent=2)}")
                    
                    # Display results summary
                    if "summary" in result:
                        summary = result["summary"]
                        print("\n" + "=" * 40)
                        print("PRUNING SUMMARY")
                        print("=" * 40)
                        print(f"Tools before pruning: {summary.get('tools_before_pruning', 'N/A')}")
                        print(f"Tools kept (relevant): {summary.get('tools_kept', 'N/A')}")
                        print(f"Tools newly matched: {summary.get('tools_newly_matched', 'N/A')}")
                        print(f"Tools pruned: {summary.get('tools_pruned', 'N/A')}")
                        print(f"Tools after pruning: {summary.get('tools_after_pruning', 'N/A')}")
                        print(f"Drop rate used: {summary.get('drop_rate_used', 'N/A')}")
                    
                    if "details" in result:
                        details = result["details"]
                        
                        if "tools_kept" in details and details["tools_kept"]:
                            print(f"\n📌 Tools Kept ({len(details['tools_kept'])}):")
                            for tool in details["tools_kept"]:
                                print(f"  - {tool['name']} (ID: {tool['id']}) - Score: {tool.get('score', 'N/A'):.3f}")
                        
                        if "tools_newly_matched" in details and details["tools_newly_matched"]:
                            print(f"\n🆕 Tools Newly Matched ({len(details['tools_newly_matched'])}):")
                            for tool in details["tools_newly_matched"]:
                                print(f"  - {tool['name']} (ID: {tool['id']}) - Score: {tool.get('score', 'N/A'):.3f}")
                        
                        if "tools_pruned" in details and details["tools_pruned"]:
                            print(f"\n🗑️  Tools Pruned ({len(details['tools_pruned'])}):")
                            for tool in details["tools_pruned"]:
                                print(f"  - {tool['name']} (ID: {tool['id']}) - Score: {tool.get('score', 'N/A'):.3f}")
                
                else:
                    error_text = await response.text()
                    print(f"❌ Request failed with status {response.status}")
                    print(f"Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Error during request: {str(e)}")

async def check_api_health():
    """Check if the API server is running"""
    print("Checking API server health...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE_URL}/api/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ API server is healthy: {health_data}")
                    return True
                else:
                    print(f"❌ API server health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Could not connect to API server: {str(e)}")
            return False

async def main():
    """Main test function"""
    print("Tool Pruning Endpoint Test")
    print("=" * 60)
    
    # Check API health first
    if not await check_api_health():
        print("\n❌ API server is not available. Please make sure it's running.")
        return
    
    print()
    
    # Test the pruning endpoint
    await test_pruning_endpoint()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())