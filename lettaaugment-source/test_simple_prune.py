#!/usr/bin/env python3
"""
Simple test for pruning endpoint with minimal drop rate to see if it works at all
"""

import asyncio
import aiohttp
import json

# Configuration
API_BASE_URL = "http://localhost:8020"

async def test_simple_prune():
    """Test with a very low drop rate to see basic functionality"""
    
    payload = {
        "user_prompt": "search for jobs",  # Simple query
        "agent_id": "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444",
        "drop_rate": 0.1  # Very low drop rate
    }
    
    print(f"Testing with payload: {json.dumps(payload, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{API_BASE_URL}/api/v1/tools/prune"
            
            async with session.post(url, json=payload) as response:
                print(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Success!")
                    print(json.dumps(result, indent=2))
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_prune())