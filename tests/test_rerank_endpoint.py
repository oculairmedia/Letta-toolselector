#!/usr/bin/env python3
"""
Test the rerank endpoint in the dashboard backend.
This tests the complete flow: Frontend → Dashboard Backend → API Server → Reranker
"""
import asyncio
import aiohttp
import json
import time

async def test_rerank_endpoint():
    """Test the dashboard backend's rerank endpoint."""

    # Test data
    test_query = "data analysis tools for financial modeling"

    request_payload = {
        "query": test_query,
        "agent_id": "agent-e54fc601-4773-4116-9c6c-cf45da2e269e",
        "limit": 5,
        "enable_reranking": True,
        "min_score": 0.1,
        "include_metadata": True
    }

    print(f"Testing rerank endpoint with query: '{test_query}'")
    print(f"Request payload: {json.dumps(request_payload, indent=2)}")

    try:
        async with aiohttp.ClientSession() as session:
            # Test the rerank endpoint
            start_time = time.time()
            async with session.post(
                "http://192.168.50.90:8030/api/v1/tools/search/rerank",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                duration = (time.time() - start_time) * 1000
                print(f"\n=== Rerank Endpoint Response ===")
                print(f"Status: {response.status}")
                print(f"Duration: {duration:.2f}ms")

                if response.status == 200:
                    data = await response.json()
                    print(f"Response type: {type(data)}")

                    if isinstance(data, list):
                        print(f"✓ Got direct array format (expected for frontend)")
                        print(f"✓ Found {len(data)} results")

                        # Show first few results
                        for i, result in enumerate(data[:3]):
                            print(f"\n--- Result {i+1} ---")
                            print(f"Name: {result.get('name', 'N/A')}")
                            print(f"Tool ID: {result.get('tool_id', 'N/A')}")
                            print(f"Score: {result.get('score', 'N/A')}")
                            print(f"Rerank Score: {result.get('rerank_score', 'N/A')}")
                            print(f"Rank: {result.get('rank', 'N/A')}")
                            print(f"Reranked: {result.get('reranked', False)}")
                            print(f"Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
                    else:
                        print(f"✗ Got structured format: {json.dumps(data, indent=2)[:500]}...")

                else:
                    text = await response.text()
                    print(f"✗ Error response: {text}")

            # Also test regular search for comparison
            print(f"\n=== Testing Regular Search for Comparison ===")
            regular_payload = request_payload.copy()
            regular_payload["enable_reranking"] = False

            async with session.post(
                "http://192.168.50.90:8030/api/v1/tools/search",
                json=regular_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Regular search status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list):
                        print(f"✓ Regular search also returns direct array with {len(data)} results")
                        if data:
                            first_result = data[0]
                            print(f"First regular result: {first_result.get('name', 'N/A')} (score: {first_result.get('score', 'N/A')})")
                    else:
                        print(f"✗ Regular search returned: {type(data)}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rerank_endpoint())