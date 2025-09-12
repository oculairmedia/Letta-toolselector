#!/usr/bin/env python3
"""
Test script to verify Weaviate embedding functionality
"""

import os
from weaviate_tool_search import init_client, get_embedding_for_text

def test_weaviate_connection():
    """Test basic Weaviate connection"""
    print("Testing Weaviate connection...")
    
    client = None
    try:
        client = init_client()
        
        # Check if Tool collection exists
        collections = client.collections.list_all()
        tool_collection_exists = any(collection.name == "Tool" for collection in collections)
        print(f"Tool collection exists: {tool_collection_exists}")
        
        if tool_collection_exists:
            # Get Tool collection and check count
            tool_collection = client.collections.get("Tool")
            count_result = tool_collection.aggregate.over_all(total_count=True)
            tool_count = count_result.total_count if count_result else 0
            print(f"Number of tools in database: {tool_count}")
            
            if tool_count > 0:
                # Get a sample tool
                sample_tools = tool_collection.query.fetch_objects(limit=3)
                if sample_tools.objects:
                    print(f"Sample tools:")
                    for tool in sample_tools.objects:
                        print(f"  - {tool.properties.get('name', 'Unknown')} (ID: {tool.uuid})")
            else:
                print("No tools found in database - this might be why embedding fails")
        
        return True
        
    except Exception as e:
        print(f"Error testing Weaviate connection: {e}")
        return False
        
    finally:
        if client:
            try:
                client.close()
            except:
                pass

def test_embedding_generation():
    """Test embedding generation"""
    print("\nTesting embedding generation...")
    
    test_text = "I need to search for remote software engineering jobs"
    print(f"Test text: '{test_text}'")
    
    try:
        embedding = get_embedding_for_text(test_text)
        if embedding:
            print(f"✅ Embedding generated successfully!")
            print(f"Embedding length: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ Embedding generation failed - empty result")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_openai_key():
    """Test if OpenAI API key is available"""
    print("\nTesting OpenAI API key...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key found: {openai_key[:20]}...")
        return True
    else:
        print("❌ OpenAI API key not found in environment")
        return False

def main():
    """Main test function"""
    print("Weaviate Embedding Test")
    print("=" * 50)
    
    # Test OpenAI key
    openai_ok = test_openai_key()
    
    # Test Weaviate connection
    weaviate_ok = test_weaviate_connection()
    
    # Test embedding generation
    embedding_ok = test_embedding_generation()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"OpenAI API Key: {'✅' if openai_ok else '❌'}")
    print(f"Weaviate Connection: {'✅' if weaviate_ok else '❌'}")
    print(f"Embedding Generation: {'✅' if embedding_ok else '❌'}")
    
    if all([openai_ok, weaviate_ok, embedding_ok]):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed - check the issues above")

if __name__ == "__main__":
    main()