#!/usr/bin/env python3
"""
Test embedding generation locally (outside Docker) to debug the issue
"""

import weaviate
import os
from dotenv import load_dotenv
import requests
import json

def init_client_local():
    """Initialize Weaviate client for local testing (connects to localhost)"""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Use localhost instead of "weaviate" for local testing
    client = weaviate.connect_to_local(
        host="localhost",  # Changed from "weaviate" to "localhost"
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-Api-Key": openai_api_key
        },
        skip_init_checks=True
    )
    
    return client

def test_weaviate_graphql():
    """Test the GraphQL query that's failing in Docker"""
    client = None
    try:
        print("Connecting to Weaviate locally...")
        client = init_client_local()
        
        text = "search for jobs"
        print(f"Testing GraphQL embedding query for: '{text}'")
        
        # This is the exact query that's failing in Docker
        query = """
        {
          Get {
            Tool(
              limit: 1
              nearText: {
                concepts: [""" + f'"{text}"' + """]
              }
            ) {
              _additional {
                vector
              }
            }
          }
        }
        """
        
        print("Sending GraphQL query...")
        result = client.graphql_raw_query(query)
        
        print("GraphQL Result:")
        print(json.dumps(result, indent=2))
        
        # Check if we got a vector
        if (result and
            'data' in result and
            'Get' in result['data'] and
            'Tool' in result['data']['Get'] and
            len(result['data']['Get']['Tool']) > 0 and
            '_additional' in result['data']['Get']['Tool'][0] and
            'vector' in result['data']['Get']['Tool'][0]['_additional']):
            
            vector = result['data']['Get']['Tool'][0]['_additional']['vector']
            print(f"✅ Successfully got embedding! Length: {len(vector)}")
            return vector
        else:
            print("❌ No vector in GraphQL response")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

def test_direct_openai():
    """Test direct OpenAI API call"""
    try:
        print("\nTesting direct OpenAI API call...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ OpenAI API key not found")
            return []
            
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "text-embedding-3-small",
            "input": "search for jobs"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                print(f"✅ Direct OpenAI successful! Length: {len(embedding)}")
                return embedding
        else:
            print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Direct OpenAI error: {e}")
        
    return []

def test_weaviate_collection_info():
    """Check what's in the Weaviate Tool collection"""
    client = None
    try:
        print("\nChecking Weaviate Tool collection...")
        client = init_client_local()
        
        # Get collection info
        collection = client.collections.get("Tool")
        
        # Get total count
        count_result = collection.aggregate.over_all(total_count=True)
        tool_count = count_result.total_count if count_result else 0
        print(f"Tools in database: {tool_count}")
        
        if tool_count > 0:
            # Get a sample tool
            sample_tools = collection.query.fetch_objects(limit=3, include_vector=True)
            if sample_tools.objects:
                print(f"Sample tools:")
                for tool in sample_tools.objects:
                    name = tool.properties.get('name', 'Unknown')
                    has_vector = hasattr(tool, 'vector') and tool.vector is not None
                    vector_length = len(tool.vector) if has_vector else 0
                    print(f"  - {name} (Vector: {has_vector}, Length: {vector_length})")
        
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        
    finally:
        if client:
            try:
                client.close()
            except:
                pass

def main():
    """Run all tests"""
    load_dotenv()
    
    print("Testing Embedding Generation Outside Docker")
    print("=" * 50)
    
    # Test 1: Check what's in Weaviate
    test_weaviate_collection_info()
    
    # Test 2: Try the failing GraphQL query
    test_weaviate_graphql()
    
    # Test 3: Try direct OpenAI
    test_direct_openai()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()