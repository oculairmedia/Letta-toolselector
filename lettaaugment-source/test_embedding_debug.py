#!/usr/bin/env python3
"""
Debug the GraphQL embedding issue with better error handling
"""

import weaviate
import os
from dotenv import load_dotenv
import requests

def init_client_local():
    """Initialize Weaviate client for local testing"""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-Api-Key": openai_api_key
        },
        skip_init_checks=True
    )
    
    return client

def test_graphql_with_proper_handling():
    """Test GraphQL with proper result handling"""
    client = None
    try:
        print("Testing GraphQL embedding generation...")
        client = init_client_local()
        
        text = "search for jobs"
        
        # This is the exact query from the failing code
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
        
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        
        # Try to access the result properly
        if hasattr(result, 'data'):
            print("✅ Result has data attribute")
            data = result.data
            print(f"Data type: {type(data)}")
            
            if data and 'Get' in data and 'Tool' in data['Get']:
                tools = data['Get']['Tool']
                print(f"Found {len(tools)} tools in response")
                
                if len(tools) > 0 and '_additional' in tools[0]:
                    if 'vector' in tools[0]['_additional']:
                        vector = tools[0]['_additional']['vector']
                        print(f"✅ Got vector! Length: {len(vector)}")
                        print(f"First 5 values: {vector[:5]}")
                        return vector
                    else:
                        print("❌ No vector in _additional")
                else:
                    print("❌ No _additional in tool or no tools")
            else:
                print("❌ No Get/Tool in data")
        else:
            print("❌ Result has no data attribute")
        
        if hasattr(result, 'errors'):
            print(f"Errors: {result.errors}")
            
        return []
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

def test_alternative_embedding_approach():
    """Try a different approach using Weaviate's modules API"""
    client = None
    try:
        print("\nTesting alternative module approach...")
        client = init_client_local()
        
        # Try to use the vectorizer directly
        text = "search for jobs"
        
        # Method 1: Try using the collection's vectorizer
        collection = client.collections.get("Tool")
        print("Got Tool collection")
        
        # Method 2: Check if we can get the vectorizer module
        try:
            modules = client.modules
            print(f"Available modules: {dir(modules)}")
            
            openai_module = modules.get("text2vec-openai")
            print(f"OpenAI module: {openai_module}")
            
            if openai_module:
                vectorize_result = openai_module.vectorize_query(text)
                print(f"Vectorize result: {vectorize_result}")
                
                if vectorize_result and 'vector' in vectorize_result:
                    vector = vectorize_result['vector']
                    print(f"✅ Module vectorization successful! Length: {len(vector)}")
                    return vector
                    
        except Exception as module_error:
            print(f"Module approach failed: {module_error}")
        
        return []
        
    except Exception as e:
        print(f"❌ Alternative approach error: {e}")
        return []
        
    finally:
        if client:
            try:
                client.close()
            except:
                pass

def main():
    """Run debug tests"""
    load_dotenv()
    
    print("Debugging Weaviate Embedding Generation")
    print("=" * 50)
    
    # Test the GraphQL approach with proper handling
    result1 = test_graphql_with_proper_handling()
    
    # Test alternative module approach
    result2 = test_alternative_embedding_approach()
    
    print("\n" + "=" * 50)
    print("Debug Results:")
    print(f"GraphQL approach: {'✅ Success' if result1 else '❌ Failed'}")
    print(f"Module approach: {'✅ Success' if result2 else '❌ Failed'}")

if __name__ == "__main__":
    main()