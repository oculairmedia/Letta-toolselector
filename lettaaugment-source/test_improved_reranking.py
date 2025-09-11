#!/usr/bin/env python3
import requests
import json

API_URL = "http://localhost:8020/api/v1/tools/search"

def test_search(query, use_reranking=False):
    """Test search with or without reranking."""
    payload = {
        "query": query,
        "limit": 5,
        "enable_reranking": use_reranking
    }
    
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return []

def format_results(results, title):
    """Format and display results."""
    print(f"\n{title}")
    print("-" * 50)
    for i, tool in enumerate(results[:5], 1):
        score = tool.get('rerank_score', tool.get('score', 0))
        name = tool.get('name', 'Unknown')
        print(f"{i}. {name:35} Score: {score:.4f}")

# Test queries
queries = [
    "create blog post",
    "file operations",
    "agent memory management"
]

for query in queries:
    print("\n" + "="*60)
    print(f"QUERY: '{query}'")
    print("="*60)
    
    # Regular search
    regular = test_search(query, use_reranking=False)
    format_results(regular, "📊 REGULAR SEARCH (Hybrid Vector+Keyword):")
    
    # Reranked search with improvements
    reranked = test_search(query, use_reranking=True)
    format_results(reranked, "🎯 RERANKED SEARCH (With Improvements):")
    
    # Check if results differ
    if regular and reranked:
        regular_names = [t['name'] for t in regular[:5]]
        reranked_names = [t['name'] for t in reranked[:5]]
        
        if regular_names != reranked_names:
            print("\n📈 RANKING CHANGES DETECTED!")
            for i, name in enumerate(reranked_names, 1):
                if name in regular_names:
                    old_pos = regular_names.index(name) + 1
                    if old_pos != i:
                        change = old_pos - i
                        arrow = "↑" if change > 0 else "↓"
                        print(f"  {arrow} {name}: #{old_pos} → #{i}")
                else:
                    print(f"  ★ {name}: NEW in top 5")

print("\n" + "="*60)
print("✅ Testing complete!")
