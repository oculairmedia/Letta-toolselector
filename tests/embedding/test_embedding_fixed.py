#!/usr/bin/env python3
"""
Test embedding generation with corrected GraphQL result handling
"""

import weaviate
import os
from dotenv import load_dotenv

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

def test_corrected_graphql():
    """Test GraphQL with corrected result access"""
    client = None
    try:
        print("Testing corrected GraphQL embedding generation...")
        client = init_client_local()
        
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
        print(f"Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
        
        # Try accessing via .get attribute instead of .data
        if hasattr(result, 'get'):
            print("✅ Result has 'get' attribute")
            get_data = result.get
            print(f"Get data type: {type(get_data)}")
            print(f"Get data: {get_data}")
            
            if get_data and 'Tool' in get_data:
                tools = get_data['Tool']
                print(f"Found {len(tools)} tools")
                
                if len(tools) > 0 and '_additional' in tools[0]:
                    if 'vector' in tools[0]['_additional']:
                        vector = tools[0]['_additional']['vector']
                        print(f"✅ Got vector! Length: {len(vector)}")
                        print(f"First 5 values: {vector[:5]}")
                        return vector
                    else:
                        print("❌ No vector in _additional")
                        print(f"_additional contents: {tools[0]['_additional']}")
                else:
                    print("❌ No _additional in tool or no tools found")
                    if len(tools) > 0:
                        print(f"Tool contents: {tools[0]}")
            else:
                print("❌ No Tool data in get result")
                print(f"Get result keys: {list(get_data.keys()) if get_data else 'None'}")
        
        # Check errors
        if hasattr(result, 'errors') and result.errors:
            print(f"❌ GraphQL errors: {result.errors}")
            
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

def test_collection_query():
    """Test using the collection query API instead of raw GraphQL"""
    client = None
    try:
        print("\nTesting collection query API...")
        client = init_client_local()
        
        text = "search for jobs"
        
        # Get Tool collection
        collection = client.collections.get("Tool")
        
        # Try a nearText query using the collection API
        response = collection.query.near_text(
            query=text,
            limit=1,
            return_metadata=['vector']
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        if hasattr(response, 'objects') and response.objects:
            print(f"Found {len(response.objects)} objects")
            obj = response.objects[0]
            
            print(f"Object type: {type(obj)}")
            print(f"Object attributes: {[attr for attr in dir(obj) if not attr.startswith('_')]}")
            
            # Check if object has vector in metadata
            if hasattr(obj, 'metadata') and obj.metadata:
                print(f"Metadata: {obj.metadata}")
                if hasattr(obj.metadata, 'vector') and obj.metadata.vector:
                    vector = obj.metadata.vector
                    print(f"✅ Got vector from metadata! Length: {len(vector)}")
                    return vector
            
            # Check if object has vector directly
            if hasattr(obj, 'vector') and obj.vector:
                vector = obj.vector
                print(f"✅ Got vector directly! Length: {len(vector)}")
                return vector
                
        print("❌ No vector found in collection query")
        return []
        
    except Exception as e:
        print(f"❌ Collection query error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        if client:
            try:
                client.close()
            except:
                pass

def main():
    """Run corrected tests"""
    load_dotenv()
    
    print("Testing Corrected Embedding Generation")
    print("=" * 50)
    
    # Test corrected GraphQL approach
    result1 = test_corrected_graphql()
    
    # Test collection API approach
    result2 = test_collection_query()
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"Corrected GraphQL: {'✅ Success' if result1 else '❌ Failed'}")
    print(f"Collection API: {'✅ Success' if result2 else '❌ Failed'}")
    
    if result1 or result2:
        print("🎉 Found a working approach for embedding generation!")
    else:
        print("⚠️ Need to use direct OpenAI fallback")

if __name__ == "__main__":
    main()