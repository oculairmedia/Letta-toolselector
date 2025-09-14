#!/usr/bin/env python3
"""
Compare baseline vs reranked search results to verify reranking is working.
"""
import asyncio
import aiohttp
import json
import time

async def test_reranking_comparison():
    """Compare baseline vs reranked search to verify different ordering."""

    queries = [
        "data analysis tools for financial modeling",
        "web scraping tools for extracting data",
        "file operations and text processing",
        "database integration and SQL queries"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Testing Query: '{query}'")
        print('='*80)

        request_payload = {
            "query": query,
            "agent_id": "agent-e54fc601-4773-4116-9c6c-cf45da2e269e",
            "limit": 8,
            "enable_reranking": False,  # Will be overridden
            "min_score": 0.1,
            "include_metadata": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Test baseline search
                baseline_payload = request_payload.copy()
                baseline_payload["enable_reranking"] = False

                start_time = time.time()
                async with session.post(
                    "http://192.168.50.90:8030/api/v1/tools/search",
                    json=baseline_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    baseline_duration = (time.time() - start_time) * 1000
                    baseline_results = []
                    if response.status == 200:
                        baseline_results = await response.json()

                # Test reranked search
                rerank_payload = request_payload.copy()
                rerank_payload["enable_reranking"] = True

                start_time = time.time()
                async with session.post(
                    "http://192.168.50.90:8030/api/v1/tools/search/rerank",
                    json=rerank_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    rerank_duration = (time.time() - start_time) * 1000
                    rerank_results = []
                    if response.status == 200:
                        rerank_results = await response.json()

                # Compare results
                print(f"\n--- Baseline Search Results ---")
                print(f"Count: {len(baseline_results)}, Duration: {baseline_duration:.2f}ms")
                for i, result in enumerate(baseline_results[:5]):
                    print(f"{i+1}. {result.get('name', 'N/A')} (score: {result.get('score', 0):.3f})")

                print(f"\n--- Reranked Search Results ---")
                print(f"Count: {len(rerank_results)}, Duration: {rerank_duration:.2f}ms")
                for i, result in enumerate(rerank_results[:5]):
                    reranked_flag = "🎯" if result.get('reranked') else ""
                    print(f"{i+1}. {result.get('name', 'N/A')} (score: {result.get('score', 0):.3f}, rerank: {result.get('rerank_score', 0):.3f}) {reranked_flag}")

                # Check for differences
                baseline_names = [r.get('name') for r in baseline_results[:5]]
                rerank_names = [r.get('name') for r in rerank_results[:5]]

                if baseline_names != rerank_names:
                    print(f"\n✅ RERANKING WORKING: Different result ordering detected!")

                    # Find position changes
                    for name in set(baseline_names + rerank_names):
                        baseline_pos = baseline_names.index(name) + 1 if name in baseline_names else None
                        rerank_pos = rerank_names.index(name) + 1 if name in rerank_names else None

                        if baseline_pos and rerank_pos and baseline_pos != rerank_pos:
                            direction = "↑" if rerank_pos < baseline_pos else "↓"
                            print(f"  • {name}: #{baseline_pos} → #{rerank_pos} {direction}")
                        elif baseline_pos and not rerank_pos:
                            print(f"  • {name}: #{baseline_pos} → removed ❌")
                        elif not baseline_pos and rerank_pos:
                            print(f"  • {name}: new → #{rerank_pos} ✨")
                else:
                    print(f"\n⚠️  Same ordering (might still have different scores)")

        except Exception as e:
            print(f"✗ Test failed for query '{query}': {e}")

if __name__ == "__main__":
    asyncio.run(test_reranking_comparison())