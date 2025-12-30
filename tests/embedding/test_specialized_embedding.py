#!/usr/bin/env python3
"""
Test the specialized embedding integration with Weaviate tool search.
"""

import sys
import os

# Add the tool-selector-api directory to the path
sys.path.append('/opt/stacks/lettatoolsselector/tool-selector-api')

os.environ.setdefault('USE_QWEN3_FORMAT', 'true')

from specialized_embedding import (
    SpecializedEmbeddingPrompter,
    enhance_query_for_embedding,
    enhance_tool_for_embedding,
    format_query_for_qwen3,
)
from weaviate_tool_search import search_tools, get_embedding_for_text


def test_prompt_enhancement():
    """Test the prompt enhancement functionality."""
    print("🧪 Testing Specialized Embedding Prompt Enhancement")
    print("=" * 60)
    
    # Test tool description enhancement
    tool_desc = "Creates and manages GitHub issues, pull requests, and repositories"
    enhanced_tool = enhance_tool_for_embedding(
        tool_description=tool_desc,
        tool_name="GitHub MCP",
        tool_type="mcp"
    )
    
    print("📝 Tool Description Enhancement:")
    print(f"Original: {tool_desc}")
    print(f"Enhanced: {enhanced_tool}")
    print()

    assert enhanced_tool == tool_desc.strip(), "Tool description should remain unchanged for Qwen3 format"

    # Test query enhancement
    query = "find tools to create blog posts"
    enhanced_query = enhance_query_for_embedding(query)

    print("🔍 Query Enhancement:")
    print(f"Original: {query}")
    print(f"Enhanced: {enhanced_query}")
    print()

    expected_query_line = f"Query: {format_query_for_qwen3(query)}"
    assert enhanced_query.startswith("Instruct:"), "Enhanced query should start with instruction prefix"
    assert expected_query_line in enhanced_query.splitlines(), "Enhanced query should include cleaned query line"

    return True


def test_search_integration():
    """Test the integration with Weaviate search (if available)."""
    print("🔍 Testing Search Integration")
    print("=" * 60)
    
    try:
        # Test a simple search query
        test_queries = [
            "create blog post",
            "manage GitHub repositories",
            "send email notifications",
            "process data files"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            
            # This will use the enhanced query processing
            results = search_tools(query, limit=3)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, tool in enumerate(results, 1):
                    name = tool.get('name', 'Unknown')
                    distance = tool.get('distance', 'N/A')
                    print(f"  {i}. {name} (distance: {distance})")
            else:
                print("❌ No results found")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Search test failed (expected if Weaviate not available): {e}")
        return False


def test_embedding_generation():
    """Test direct embedding generation with enhancement."""
    print("🎯 Testing Embedding Generation")
    print("=" * 60)
    
    try:
        test_text = "create a blog post about AI"
        
        # Test without enhancement
        print("Testing without enhancement...")
        embedding_plain = get_embedding_for_text(test_text, enhance_prompt=False)
        
        # Test with enhancement
        print("Testing with enhancement...")
        embedding_enhanced = get_embedding_for_text(test_text, enhance_prompt=True)
        
        if embedding_plain and embedding_enhanced:
            print(f"✅ Plain embedding length: {len(embedding_plain)}")
            print(f"✅ Enhanced embedding length: {len(embedding_enhanced)}")
            
            # Check if embeddings are different (they should be)
            if embedding_plain != embedding_enhanced:
                print("✅ Embeddings are different (enhancement working)")
            else:
                print("⚠️ Embeddings are identical (enhancement may not be working)")
            
            return True
        else:
            print("❌ Failed to generate embeddings")
            return False
            
    except Exception as e:
        print(f"⚠️ Embedding test failed (expected if Weaviate not available): {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Specialized Embedding Integration Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Prompt enhancement
    print("TEST 1: Prompt Enhancement")
    results.append(test_prompt_enhancement())
    print()
    
    # Test 2: Search integration
    print("TEST 2: Search Integration")
    results.append(test_search_integration())
    print()
    
    # Test 3: Embedding generation
    print("TEST 3: Embedding Generation")
    results.append(test_embedding_generation())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("🏁 Test Results Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Specialized embedding integration is working.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)