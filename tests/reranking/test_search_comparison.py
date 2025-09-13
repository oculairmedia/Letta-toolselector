import json
import requests

def search_tools(query, limit=5, enable_reranking=False):
    """Search tools with or without reranking."""
    if enable_reranking:
        # Use the dedicated rerank endpoint
        url = "http://localhost:8020/api/v1/tools/search/rerank"
        payload = {
            "query": query,
            "limit": limit,
            "reranker_config": {
                "enabled": True,
                "model": "bge-reranker-v2-m3",
                "base_url": "http://localhost:8091"
            }
        }
    else:
        # Use regular search endpoint
        url = "http://localhost:8020/api/v1/tools/search"
        payload = {
            "query": query,
            "limit": limit,
        }

    response = requests.post(url, json=payload)
    return response.json()

def compare_searches(query):
    """Compare regular and reranked search results."""
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    
    # Regular search
    regular = search_tools(query, limit=10, enable_reranking=False)
    print("\n📊 REGULAR SEARCH (Vector + Keyword Hybrid):")
    print("-" * 50)
    for i, tool in enumerate(regular[:5], 1):
        score = tool.get('score', 0)
        print(f"{i:2}. {tool['name']:35} Score: {score:.4f}")
    
    # Reranked search
    reranked = search_tools(query, limit=10, enable_reranking=True)
    print("\n🎯 RERANKED SEARCH (With Ollama Reranker):")
    print("-" * 50)
    for i, tool in enumerate(reranked[:5], 1):
        score = tool.get('rerank_score', tool.get('score', 0))
        original_score = tool.get('score', 0) if 'rerank_score' not in tool else None
        print(f"{i:2}. {tool['name']:35} Score: {score:.4f}")
    
    # Show changes
    regular_names = [t['name'] for t in regular[:5]]
    reranked_names = [t['name'] for t in reranked[:5]]
    
    print("\n📈 RANKING CHANGES:")
    print("-" * 50)
    for i, name in enumerate(reranked_names, 1):
        if name in regular_names:
            old_pos = regular_names.index(name) + 1
            change = old_pos - i
            if change > 0:
                print(f"  ↑ {name}: #{old_pos} → #{i} (up {change})")
            elif change < 0:
                print(f"  ↓ {name}: #{old_pos} → #{i} (down {abs(change)})")
            else:
                print(f"  = {name}: stayed at #{i}")
        else:
            print(f"  ★ {name}: NEW in top 5")

# Test various queries
queries = [
    "create blog post",
    "file operations",
    "memory management",
    "huly project issues",
    "agent configuration"
]

for query in queries:
    compare_searches(query)

print("\n" + "="*60)
print("✅ Search comparison complete!")
