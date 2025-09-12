#!/usr/bin/env python3
"""
Test script to verify that reranking is working correctly after MCP URL fix.
This will make parallel API calls to compare original vs reranked results.
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any, List
import time

# Test configuration
API_BASE_URL = "http://localhost:8030"  # Dashboard backend
TEST_QUERIES = [
    "data analysis tools",
    "file operations", 
    "web scraping",
    "database management"
]

async def search_tools(session: aiohttp.ClientSession, query: str, enable_reranking: bool) -> Dict[str, Any]:
    """Search for tools with or without reranking."""
    url = f"{API_BASE_URL}/api/v1/search"
    payload = {
        "query": query,
        "limit": 10,
        "enable_reranking": enable_reranking,
        "min_score": 0.0,
        "include_metadata": True
    }
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                print(f"Error {response.status}: {error_text}")
                return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

async def compare_results(query: str) -> None:
    """Compare original and reranked results for a query."""
    print(f"\n{'='*80}")
    print(f"Query: '{query}'")
    print('='*80)
    
    async with aiohttp.ClientSession() as session:
        # Make parallel requests
        original_task = search_tools(session, query, False)
        reranked_task = search_tools(session, query, True)
        
        original_result, reranked_result = await asyncio.gather(original_task, reranked_task)
        
        if not original_result or not reranked_result:
            print("❌ Failed to get results")
            return
        
        # Extract tools
        original_tools = original_result.get("results", [])
        reranked_tools = reranked_result.get("results", [])
        
        print(f"\n📊 Results Summary:")
        print(f"  Original: {len(original_tools)} tools found")
        print(f"  Reranked: {len(reranked_tools)} tools found")
        print(f"  Reranking Applied: {reranked_result.get('reranking_applied', False)}")
        
        # Compare top 5 results
        print(f"\n🔍 Top 5 Results Comparison:")
        print("-" * 80)
        
        for i in range(min(5, max(len(original_tools), len(reranked_tools)))):
            print(f"\nPosition {i+1}:")
            
            if i < len(original_tools):
                orig = original_tools[i]
                print(f"  Original: {orig['name'][:40]:40} | Score: {orig['score']:.4f}")
            else:
                print(f"  Original: (no result)")
            
            if i < len(reranked_tools):
                rerank = reranked_tools[i]
                rerank_score = rerank.get('rerank_score', 'N/A')
                if isinstance(rerank_score, float):
                    rerank_score = f"{rerank_score:.4f}"
                print(f"  Reranked: {rerank['name'][:40]:40} | Score: {rerank['score']:.4f} | Rerank: {rerank_score}")
            else:
                print(f"  Reranked: (no result)")
        
        # Check if results are identical
        if len(original_tools) == len(reranked_tools):
            identical = all(
                orig['id'] == rerank['id'] and orig['score'] == rerank['score']
                for orig, rerank in zip(original_tools, reranked_tools)
            )
            
            if identical:
                print("\n⚠️  WARNING: Results are IDENTICAL - reranking may not be working!")
            else:
                # Calculate position changes
                orig_positions = {tool['id']: i for i, tool in enumerate(original_tools)}
                position_changes = []
                
                for i, tool in enumerate(reranked_tools):
                    if tool['id'] in orig_positions:
                        change = orig_positions[tool['id']] - i
                        if change != 0:
                            position_changes.append((tool['name'], change))
                
                if position_changes:
                    print("\n✅ Results are DIFFERENT - reranking is working!")
                    print("\n📈 Position Changes (positive = moved up):")
                    for name, change in sorted(position_changes, key=lambda x: abs(x[1]), reverse=True)[:5]:
                        arrow = "↑" if change > 0 else "↓"
                        print(f"  {arrow} {name[:40]:40} | {abs(change):+2} positions")
                else:
                    print("\n⚠️  Same tools but no position changes detected")

async def test_mcp_endpoint() -> None:
    """Test the MCP endpoint directly."""
    print("\n" + "="*80)
    print("Testing MCP Endpoint Directly")
    print("="*80)
    
    mcp_url = "http://localhost:3020/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "find_tools",
            "arguments": {
                "query": "data analysis",
                "limit": 5,
                "enable_reranking": True,
                "min_score": 0.0,
                "detailed_response": True
            }
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(mcp_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        tools = result["result"].get("tools", [])
                        print(f"✅ MCP endpoint working - found {len(tools)} tools")
                        if result["result"].get("reranking_applied"):
                            print("✅ Reranking was applied")
                        else:
                            print("⚠️  Reranking was NOT applied")
                    else:
                        print(f"❌ MCP error: {result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ MCP endpoint returned status {response.status}")
        except Exception as e:
            print(f"❌ Failed to connect to MCP endpoint: {e}")

async def main():
    """Run all tests."""
    print("\n🧪 Testing Improved Reranking System")
    print("="*80)
    
    # Test MCP endpoint first
    await test_mcp_endpoint()
    
    # Test each query
    for query in TEST_QUERIES:
        await compare_results(query)
        await asyncio.sleep(0.5)  # Small delay between tests
    
    print("\n" + "="*80)
    print("✅ Testing complete!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
