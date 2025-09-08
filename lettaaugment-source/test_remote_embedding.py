#!/usr/bin/env python3
"""
Test embedding generation specifically on the remote server
"""

import requests
import json
import weaviate
import os
from dotenv import load_dotenv

def test_embedding_via_remote_weaviate():
    """Test embedding generation using remote Weaviate directly"""
    print("Testing Embedding Generation via Remote Weaviate")
    print("=" * 60)
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
        
    try:
        # Connect to remote Weaviate using v4 client
        client = weaviate.connect_to_custom(
            http_host="192.168.50.90",
            http_port=8080,
            http_secure=False,
            grpc_host="192.168.50.90", 
            grpc_port=50051,
            grpc_secure=False,
            headers={
                "X-OpenAI-Api-Key": openai_api_key
            },
            skip_init_checks=True
        )
        
        print("✅ Connected to remote Weaviate")
        
        # Test GraphQL query for embedding generation
        text = "search for jobs"
        query = """
        {
          Get {
            Tool(
              limit: 1
              nearText: {
                concepts: [""" + f'"{text}"' + """]
              }
            ) {
              name
              description
              _additional {
                vector
                distance
              }
            }
          }
        }
        """
        
        print(f"Testing nearText query with: '{text}'")
        result = client.graphql_raw_query(query)
        
        print(f"GraphQL result type: {type(result)}")
        print(f"Has 'get' attribute: {hasattr(result, 'get')}")
        
        if hasattr(result, 'get') and result.get:
            tools = result.get.get('Tool', [])
            print(f"Found {len(tools)} tools in nearText result")
            
            if tools and '_additional' in tools[0]:
                additional = tools[0]['_additional']
                if 'vector' in additional:
                    vector = additional['vector']
                    print(f"✅ SUCCESS! Got vector with length: {len(vector)}")
                    print(f"First tool: {tools[0].get('name', 'Unknown')}")
                    print(f"Distance: {additional.get('distance', 'N/A')}")
                    return True
                else:
                    print(f"❌ No vector in _additional. Keys: {list(additional.keys())}")
            else:
                print("❌ No tools found or no _additional data")
        else:
            print("❌ No result.get or empty result")
            
        # Check for errors
        if hasattr(result, 'errors') and result.errors:
            print(f"GraphQL errors: {result.errors}")
            
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            client.close()
        except:
            pass

def test_tool_count_and_embeddings():
    """Check how many tools exist and if they have embeddings"""
    print("\nChecking Tool Count and Embeddings")
    print("=" * 60)
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        # Connect to remote Weaviate
        client = weaviate.connect_to_custom(
            http_host="192.168.50.90",
            http_port=8080,
            http_secure=False,
            grpc_host="192.168.50.90", 
            grpc_port=50051,
            grpc_secure=False,
            headers={
                "X-OpenAI-Api-Key": openai_api_key
            },
            skip_init_checks=True
        )
        
        # Get Tool collection
        collection = client.collections.get("Tool")
        
        # Get a few tools to check
        response = collection.query.fetch_objects(limit=5)
        
        print(f"Found {len(response.objects)} tools in collection")
        
        for i, obj in enumerate(response.objects):
            name = obj.properties.get('name', 'Unknown')
            has_vector = hasattr(obj, 'vector') and obj.vector is not None
            print(f"  {i+1}. {name} - Vector: {'✅' if has_vector else '❌'}")
            
            if has_vector:
                print(f"     Vector length: {len(obj.vector)}")
        
        return len(response.objects) > 0
        
    except Exception as e:
        print(f"❌ Error checking tools: {e}")
        return False
    finally:
        try:
            client.close()
        except:
            pass

def test_direct_embedding_request():
    """Test making a direct embedding request to the remote API"""
    print("\nTesting Direct Embedding via API")
    print("=" * 60)
    
    # Test if the remote server has an embedding endpoint
    api_base_url = "http://192.168.50.90:8020"
    
    # Check what endpoints are available
    try:
        # Try to get available endpoints (common patterns)
        test_endpoints = [
            "/api/v1/embedding",
            "/api/v1/tools/embedding", 
            "/embedding",
            "/health",
            "/",
        ]
        
        for endpoint in test_endpoints:
            url = f"{api_base_url}{endpoint}"
            try:
                response = requests.get(url, timeout=5)
                print(f"{endpoint}: {response.status_code}")
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"  Response: {json.dumps(data, indent=2)[:100]}...")
                    except:
                        print(f"  Response: {response.text[:100]}...")
            except:
                print(f"{endpoint}: Connection failed")
                
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")

def main():
    """Run embedding-specific tests on remote server"""
    print("Remote Server Embedding Testing")
    print("=" * 70)
    print("Remote Weaviate: http://192.168.50.90:8080")
    print("Remote API: http://192.168.50.90:8020")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Embedding generation via Weaviate
    results['embedding_generation'] = test_embedding_via_remote_weaviate()
    
    # Test 2: Check existing tools and their embeddings
    results['tool_inspection'] = test_tool_count_and_embeddings()
    
    # Test 3: Check API endpoints
    test_direct_embedding_request()
    
    print("\n" + "=" * 70)
    print("EMBEDDING TEST SUMMARY")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    if results.get('embedding_generation', False):
        print("\n🎉 Embedding generation is working on remote server!")
    else:
        print("\n⚠️ Embedding generation needs investigation")

if __name__ == "__main__":
    main()