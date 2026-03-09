#!/usr/bin/env python3
"""
Quick test to verify reranking is working after networking fixes.
Tests both baseline search and reranked search to see if results differ.
"""
import requests
import json
import sys

def test_search_endpoints(query="data analysis tools"):
    """Test both baseline and reranked search endpoints."""

    base_url = "http://192.168.50.90:8030"  # Dashboard backend

    print(f"🔍 Testing query: '{query}'")
    print("=" * 60)

    # Test 1: Baseline search (no reranking)
    print("\n📊 BASELINE SEARCH (no reranking):")
    print("-" * 50)

    try:
        baseline_payload = {
            "query": query,
            "limit": 5,
            "enable_reranking": False
        }

        response = requests.post(f"{base_url}/api/v1/search", json=baseline_payload, timeout=30)

        if response.status_code == 200:
            baseline_data = response.json()
            if baseline_data.get("success") and baseline_data.get("data", {}).get("tools"):
                baseline_tools = baseline_data["data"]["tools"]
                for i, tool in enumerate(baseline_tools[:5], 1):
                    name = tool.get("name", "Unknown")[:40]
                    score = tool.get("score", 0)
                    print(f"{i:2}. {name:40} | Score: {score:.4f}")
            else:
                print("❌ No tools found or invalid response format")
                print(f"Response: {json.dumps(baseline_data, indent=2)[:500]}...")
        else:
            print(f"❌ Baseline search failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")

    except Exception as e:
        print(f"❌ Baseline search error: {e}")

    # Test 2: Reranked search
    print("\n🎯 RERANKED SEARCH (with reranking):")
    print("-" * 50)

    try:
        reranked_payload = {
            "query": query,
            "limit": 5,
            "enable_reranking": True
        }

        response = requests.post(f"{base_url}/api/v1/search", json=reranked_payload, timeout=30)

        if response.status_code == 200:
            reranked_data = response.json()
            if reranked_data.get("success") and reranked_data.get("data", {}).get("tools"):
                reranked_tools = reranked_data["data"]["tools"]
                reranking_applied = reranked_data["data"].get("reranking_applied", False)

                print(f"Reranking Applied: {'✅ YES' if reranking_applied else '❌ NO'}")
                print()

                for i, tool in enumerate(reranked_tools[:5], 1):
                    name = tool.get("name", "Unknown")[:40]
                    score = tool.get("score", 0)
                    rerank_score = tool.get("rerank_score")
                    if rerank_score is not None:
                        print(f"{i:2}. {name:40} | Score: {score:.4f} | Rerank: {rerank_score:.4f}")
                    else:
                        print(f"{i:2}. {name:40} | Score: {score:.4f}")

                # Compare result ordering
                print("\n📈 COMPARISON:")
                print("-" * 50)
                if 'baseline_tools' in locals():
                    baseline_names = [t.get("name", "") for t in baseline_tools[:5]]
                    reranked_names = [t.get("name", "") for t in reranked_tools[:5]]

                    if baseline_names == reranked_names:
                        print("⚠️  Results are IDENTICAL - reranking may not be working!")
                    else:
                        print("✅ Results are DIFFERENT - reranking appears to be working!")
                        changes = 0
                        for i, name in enumerate(reranked_names, 1):
                            if i <= len(baseline_names) and baseline_names[i-1] != name:
                                try:
                                    old_pos = baseline_names.index(name) + 1
                                    change = old_pos - i
                                    if change > 0:
                                        print(f"  ↑ {name[:30]:30} moved up {change} positions (#{old_pos} → #{i})")
                                    elif change < 0:
                                        print(f"  ↓ {name[:30]:30} moved down {abs(change)} positions (#{old_pos} → #{i})")
                                    changes += 1
                                except ValueError:
                                    print(f"  ★ {name[:30]:30} NEW in reranked results")
                                    changes += 1

                        print(f"\nTotal position changes: {changes}")
            else:
                print("❌ No tools found or invalid response format")
                print(f"Response: {json.dumps(reranked_data, indent=2)[:500]}...")
        else:
            print(f"❌ Reranked search failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")

    except Exception as e:
        print(f"❌ Reranked search error: {e}")

def test_reranker_health():
    """Test if the reranker adapter is accessible."""
    print("\n🩺 RERANKER HEALTH CHECK:")
    print("-" * 50)

    # Test external access to reranker
    try:
        response = requests.get("http://192.168.50.90:8091/health", timeout=10)
        if response.status_code == 200:
            print("✅ Reranker adapter accessible externally")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Reranker health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Reranker health check error: {e}")

if __name__ == "__main__":
    print("🧪 RERANKING FUNCTIONALITY TEST")
    print("Testing after networking fixes...")

    # Test reranker health first
    test_reranker_health()

    # Test different queries
    test_queries = [
        "data analysis tools",
        "blog content creation",
        "file operations",
        "web scraping"
    ]

    for query in test_queries:
        test_search_endpoints(query)
        print("\n" + "="*80 + "\n")

    print("✅ Testing complete!")