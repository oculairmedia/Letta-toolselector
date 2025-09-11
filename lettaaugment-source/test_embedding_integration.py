#!/usr/bin/env python3
"""
Test script to verify embedding provider integration with existing functions
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fallback_embedding import get_embedding_for_text_direct
from weaviate_tool_search import _get_embedding_direct_provider
import asyncio
from embedding_providers import EmbeddingProviderFactory, EmbeddingProviderContext

def test_fallback_embedding():
    """Test the updated fallback embedding function"""
    print("🧪 Testing fallback embedding function...")
    
    test_text = "I need to search for remote software engineering jobs"
    embedding = get_embedding_for_text_direct(test_text)
    
    if embedding and len(embedding) > 0:
        print(f"✅ Fallback embedding successful! Length: {len(embedding)}")
        return True
    else:
        print("❌ Fallback embedding failed")
        return False

def test_weaviate_search_embedding():
    """Test the updated weaviate search embedding function"""
    print("🧪 Testing weaviate search embedding function...")
    
    test_text = "Create a new blog post with Ghost CMS"
    embedding = _get_embedding_direct_provider(test_text)
    
    if embedding and len(embedding) > 0:
        print(f"✅ Weaviate search embedding successful! Length: {len(embedding)}")
        return True
    else:
        print("❌ Weaviate search embedding failed")
        return False

@pytest.mark.asyncio
async def test_provider_factory():
    """Test the embedding provider factory directly"""
    print("🧪 Testing embedding provider factory...")
    
    try:
        async with EmbeddingProviderContext() as provider:
            test_text = "API endpoint management and configuration"
            embedding = await provider.get_single_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                print(f"✅ Provider factory successful! Provider: {provider.provider_name}, Length: {len(embedding)}")
                return True
            else:
                print("❌ Provider factory returned empty embedding")
                return False
    except Exception as e:
        print(f"❌ Provider factory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Embedding Provider Integration\n")
    
    tests = [
        ("Fallback Embedding", test_fallback_embedding),
        ("Weaviate Search Embedding", test_weaviate_search_embedding),
        ("Provider Factory", lambda: asyncio.run(test_provider_factory()))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
            print()
    
    print("📊 Test Results Summary:")
    print("=" * 40)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All embedding integration tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - check the logs above")
        return 1

if __name__ == "__main__":
    exit(main())