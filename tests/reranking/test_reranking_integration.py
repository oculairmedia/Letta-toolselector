#!/usr/bin/env python3
"""
Integration test for Ollama reranker with Weaviate
Tests the complete two-stage retrieval pipeline
"""
import os
import sys
import json
import time
import requests
from typing import List, Dict, Any

# Add lettaaugment-source to path
sys.path.append('/opt/stacks/lettatoolsselector/lettaaugment-source')

def test_adapter_health():
    """Test if the Ollama reranker adapter is healthy"""
    adapter_url = "http://localhost:8091"
    
    try:
        response = requests.get(f"{adapter_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Adapter Health Check:")
            print(f"   Status: {health.get('status')}")
            print(f"   Model: {health.get('model')}")
            print(f"   Ollama: {health.get('ollama')}")
            return True
        else:
            print(f"❌ Adapter health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to adapter: {e}")
        print("   Make sure the adapter is running on port 8091")
        return False

def test_adapter_reranking():
    """Test the adapter's reranking functionality directly"""
    adapter_url = "http://localhost:8091"
    
    test_request = {
        "query": "tool for creating blog posts",
        "documents": [
            "Ghost CMS - A powerful blogging platform with API for creating and managing blog posts",
            "FileManager - Basic file operations tool",
            "WordPress API - Create and manage WordPress blog posts programmatically",
            "Calculator - Perform mathematical calculations"
        ]
    }
    
    try:
        print("\n🔄 Testing Direct Adapter Reranking...")
        start_time = time.time()
        
        response = requests.post(
            f"{adapter_url}/rerank",
            json=test_request,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            scores = result.get("scores", [])
            
            print(f"✅ Reranking successful in {elapsed:.2f}s")
            print("\nReranked scores:")
            for i, (doc, score) in enumerate(zip(test_request["documents"], scores)):
                print(f"   [{score:.3f}] {doc[:60]}...")
            
            # Check if blog-related tools scored higher
            if scores[0] > 0.7 and scores[2] > 0.7:  # Ghost and WordPress should score high
                print("\n✅ Relevance scoring looks correct!")
                return True
            else:
                print("\n⚠️ Unexpected relevance scores")
                return False
        else:
            print(f"❌ Reranking failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Adapter reranking error: {e}")
        return False

def test_weaviate_with_reranking():
    """Test Weaviate search with reranking enabled"""
    
    # Set environment variables for reranking
    os.environ["ENABLE_RERANKING"] = "true"
    os.environ["RERANK_INITIAL_LIMIT"] = "20"
    os.environ["RERANK_TOP_K"] = "5"
    
    try:
        from weaviate_tool_search_with_reranking import (
            search_tools_with_reranking,
            test_reranking_capability
        )
        
        print("\n🔍 Testing Weaviate Integration with Reranking...")
        
        # First test configuration
        print("\n1. Checking reranking configuration:")
        config = test_reranking_capability()
        print(json.dumps(config, indent=2))
        
        if not config.get("reranking_enabled"):
            print("❌ Reranking is not enabled")
            return False
        
        # Test actual search
        print("\n2. Performing search with reranking:")
        
        test_queries = [
            "create blog posts",
            "manage github repositories",
            "send email notifications"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            print("   " + "-" * 40)
            
            # Search with reranking
            results_with = search_tools_with_reranking(
                query=query,
                limit=3,
                use_reranking=True
            )
            
            # Search without reranking (for comparison)
            results_without = search_tools_with_reranking(
                query=query,
                limit=3,
                use_reranking=False
            )
            
            print("   With Reranking:")
            for i, tool in enumerate(results_with[:3], 1):
                score = tool.get('rerank_score', tool.get('score', 0))
                print(f"      {i}. [{score:.3f}] {tool.get('name', 'Unknown')}")
            
            print("   Without Reranking:")
            for i, tool in enumerate(results_without[:3], 1):
                score = tool.get('score', 0)
                print(f"      {i}. [{score:.3f}] {tool.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weaviate integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance characteristics of reranking"""
    
    print("\n⚡ Testing Performance Characteristics...")
    
    adapter_url = "http://localhost:8091"
    
    # Test with varying document counts
    test_sizes = [5, 10, 20]
    
    for size in test_sizes:
        test_request = {
            "query": "data analysis and visualization tool",
            "documents": [
                f"Tool{i} - A tool for various data operations and analysis tasks"
                for i in range(1, size + 1)
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{adapter_url}/rerank",
                json=test_request,
                timeout=60
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   {size} documents: {elapsed:.2f}s ({elapsed/size:.3f}s per doc)")
            else:
                print(f"   {size} documents: Failed")
                
        except Exception as e:
            print(f"   {size} documents: Error - {e}")
    
    # Get metrics
    try:
        response = requests.get(f"{adapter_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print("\n📊 Adapter Metrics:")
            print(f"   Total requests: {metrics.get('total_requests', 0)}")
            print(f"   Success rate: {metrics.get('success_rate', 0):.1%}")
            print(f"   Average latency: {metrics.get('average_latency_ms', 0):.1f}ms")
            print(f"   Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
    except:
        pass

def main():
    """Run all integration tests"""
    
    print("=" * 60)
    print("Ollama Reranker Integration Test Suite")
    print("=" * 60)
    
    # Track test results
    tests_passed = []
    tests_failed = []
    
    # Test 1: Adapter Health
    print("\n[Test 1: Adapter Health Check]")
    if test_adapter_health():
        tests_passed.append("Adapter Health")
    else:
        tests_failed.append("Adapter Health")
        print("\n⚠️ Stopping tests - adapter not available")
        print("Please ensure:")
        print("1. The reranker adapter is built and running")
        print("2. It's accessible on port 8083")
        print("\nTo start the adapter:")
        print("   docker-compose -f compose-with-reranker.yaml up -d reranker-ollama-adapter")
        return
    
    # Test 2: Direct Adapter Reranking
    print("\n[Test 2: Direct Adapter Reranking]")
    if test_adapter_reranking():
        tests_passed.append("Direct Reranking")
    else:
        tests_failed.append("Direct Reranking")
    
    # Test 3: Weaviate Integration
    print("\n[Test 3: Weaviate Integration]")
    if test_weaviate_with_reranking():
        tests_passed.append("Weaviate Integration")
    else:
        tests_failed.append("Weaviate Integration")
    
    # Test 4: Performance
    print("\n[Test 4: Performance Testing]")
    test_performance()
    tests_passed.append("Performance")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {len(tests_passed)} tests")
    if tests_passed:
        for test in tests_passed:
            print(f"   - {test}")
    
    if tests_failed:
        print(f"\n❌ Failed: {len(tests_failed)} tests")
        for test in tests_failed:
            print(f"   - {test}")
    
    print("\n" + "=" * 60)
    
    if not tests_failed:
        print("🎉 All tests passed! Reranking integration is working.")
        print("\nNext steps:")
        print("1. Deploy with: docker-compose -f compose-with-reranker.yaml up -d")
        print("2. Monitor metrics at: http://localhost:8083/metrics")
        print("3. Set ENABLE_RERANKING=true in production")
    else:
        print("⚠️ Some tests failed. Review the output above for details.")

if __name__ == "__main__":
    main()