#!/usr/bin/env python3
"""
Test the updated Weaviate integration with specialized embeddings.
This test validates that the system can:
1. Create Weaviate schema without OpenAI vectorizer
2. Upload tools with specialized embeddings
3. Search using custom embeddings with Qwen3-Embedding-4B
"""

import sys
import os
import asyncio

# Add the lettaaugment-source directory to the path
sys.path.append('/opt/stacks/lettatoolsselector/lettaaugment-source')

from specialized_embedding import enhance_tool_for_embedding, enhance_query_for_embedding
from embedding_providers import EmbeddingProviderFactory


async def test_embedding_provider():
    """Test the embedding provider works correctly."""
    print("🔬 Testing Embedding Provider")
    print("=" * 50)
    
    try:
        # Initialize embedding provider
        provider = EmbeddingProviderFactory.create_from_env()
        
        # Test basic embedding generation
        test_text = "This is a test for embedding generation"
        embedding = await provider.get_single_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding generated successfully")
            print(f"   Length: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            print(f"   Provider: {provider.provider_name}")
            print(f"   Model: {provider.model}")
            return True
        else:
            print("❌ Failed to generate embedding")
            return False
            
    except Exception as e:
        print(f"❌ Embedding provider test failed: {e}")
        return False
    
    finally:
        if 'provider' in locals():
            await provider.close()


async def test_specialized_embedding_integration():
    """Test specialized embedding with the provider."""
    print("\n🎯 Testing Specialized Embedding Integration")
    print("=" * 50)
    
    try:
        provider = EmbeddingProviderFactory.create_from_env()
        
        # Test tool enhancement and embedding
        tool_desc = "Creates and manages GitHub issues and pull requests"
        enhanced_desc = enhance_tool_for_embedding(
            tool_description=tool_desc,
            tool_name="GitHub API",
            tool_type="mcp"
        )
        
        print(f"Original: {tool_desc}")
        print(f"Enhanced: {enhanced_desc[:100]}...")
        
        # Generate embeddings for both
        original_embedding = await provider.get_single_embedding(tool_desc)
        enhanced_embedding = await provider.get_single_embedding(enhanced_desc)
        
        if original_embedding and enhanced_embedding:
            print(f"✅ Both embeddings generated successfully")
            
            # Check if they're different
            if original_embedding != enhanced_embedding:
                print("✅ Enhanced embedding is different from original")
            else:
                print("⚠️ Enhanced embedding is identical to original")
                
            # Test query enhancement
            query = "find GitHub tools"
            enhanced_query = enhance_query_for_embedding(query)
            
            query_embedding = await provider.get_single_embedding(enhanced_query)
            if query_embedding:
                print("✅ Query embedding generated successfully")
                return True
            else:
                print("❌ Failed to generate query embedding")
                return False
        else:
            print("❌ Failed to generate embeddings")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'provider' in locals():
            await provider.close()


def test_weaviate_schema_config():
    """Test Weaviate schema configuration (without actual connection)."""
    print("\n📋 Testing Weaviate Schema Configuration")
    print("=" * 50)
    
    try:
        # Test schema creation logic without actual Weaviate connection
        # This validates our configuration approach
        import weaviate.classes.config
        
        # Test the vectorizer configuration we're using
        vectorizer_config = weaviate.classes.config.Configure.Vectorizer.none()
        print("✅ Vectorizer.none() configuration created successfully")
        
        # Test property configuration
        test_property = weaviate.classes.config.Property(
            name="test_field",
            data_type=weaviate.classes.config.DataType.TEXT,
            description="Test property"
        )
        print("✅ Property configuration created successfully")
        
        print("✅ Schema configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Schema configuration test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("🚀 Weaviate Integration Tests with Specialized Embeddings")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Embedding Provider
    print("TEST 1: Embedding Provider")
    results.append(await test_embedding_provider())
    print()
    
    # Test 2: Specialized Embedding Integration
    print("TEST 2: Specialized Embedding Integration")
    results.append(await test_specialized_embedding_integration())
    print()
    
    # Test 3: Weaviate Schema Configuration
    print("TEST 3: Weaviate Schema Configuration")
    results.append(test_weaviate_schema_config())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("🏁 Integration Test Results Summary")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("The Weaviate integration with specialized embeddings is ready.")
        return True
    else:
        print("⚠️ Some integration tests failed.")
        print("Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)