#!/usr/bin/env python3
"""
Debug existing embeddings in Weaviate to understand the structure
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

def test_existing_tool_embeddings():
    """Get embeddings for existing tools to understand structure"""
    client = None
    try:
        print("Connecting to Weaviate...")
        client = init_client_local()
        
        # Get some tools with their vectors
        collection = client.collections.get("Tool")
        
        # Try to fetch tools with vectors using the proper API
        from weaviate.classes.query import MetadataQuery
        
        response = collection.query.fetch_objects(
            limit=3,
            return_metadata=MetadataQuery(distance=True, certainty=True)
        )
        
        print(f"Found {len(response.objects)} tools")
        
        for i, obj in enumerate(response.objects):
            print(f"\nTool {i+1}:")
            print(f"  Name: {obj.properties.get('name', 'Unknown')}")
            print(f"  Description: {obj.properties.get('description', 'No description')[:100]}...")
            print(f"  UUID: {obj.uuid}")
            
            # Check if vector is available
            if hasattr(obj, 'vector') and obj.vector:
                print(f"  Vector length: {len(obj.vector)}")
                print(f"  First 5 vector values: {obj.vector[:5]}")
            else:
                print("  No vector found")
                
        # Now try a GraphQL query to get the vector for the first tool
        if response.objects:
            first_tool_uuid = response.objects[0].uuid
            print(f"\nTesting GraphQL vector retrieval for tool: {first_tool_uuid}")
            
            query = f"""
            {{
              Get {{
                Tool(
                  where: {{
                    path: ["tool_id"]
                    operator: Equal
                    valueText: "{response.objects[0].properties.get('tool_id', '')}"
                  }}
                  limit: 1
                ) {{
                  name
                  description
                  _additional {{
                    id
                    vector
                  }}
                }}
              }}
            }}
            """
            
            result = client.graphql_raw_query(query)
            print(f"GraphQL result type: {type(result)}")
            print(f"GraphQL result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            if hasattr(result, 'get') and result.get:
                tools_result = result.get.get('Tool', [])
                print(f"Found {len(tools_result)} tools in GraphQL result")
                
                if tools_result and '_additional' in tools_result[0]:
                    additional = tools_result[0]['_additional']
                    if 'vector' in additional:
                        vector = additional['vector']
                        print(f"✅ GraphQL vector found! Length: {len(vector)}")
                        print(f"First 5 values: {vector[:5]}")
                    else:
                        print("❌ No vector in _additional")
                        print(f"_additional keys: {list(additional.keys())}")
                else:
                    print("❌ No _additional in result")
                    
            else:
                print("❌ No 'get' attribute in GraphQL result")
        
        return response.objects
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

def test_neartext_query():
    """Test a nearText query to see how it works"""
    client = None
    try:
        print("\n" + "="*50)
        print("Testing nearText query with job-related search...")
        client = init_client_local()
        
        # Try nearText query for job-related tools
        query = """
        {
          Get {
            Tool(
              limit: 2
              nearText: {
                concepts: ["job", "search", "work", "employment"]
              }
            ) {
              name
              description
              _additional {
                distance
                vector
              }
            }
          }
        }
        """
        
        result = client.graphql_raw_query(query)
        
        if hasattr(result, 'get') and result.get:
            tools = result.get.get('Tool', [])
            print(f"Found {len(tools)} job-related tools")
            
            for i, tool in enumerate(tools):
                print(f"\nJob-related tool {i+1}:")
                print(f"  Name: {tool.get('name', 'Unknown')}")
                print(f"  Description: {tool.get('description', 'No description')[:100]}...")
                
                if '_additional' in tool:
                    additional = tool['_additional']
                    if 'distance' in additional:
                        print(f"  Distance: {additional['distance']}")
                    if 'vector' in additional:
                        vector = additional['vector']
                        print(f"  Vector length: {len(vector)}")
                        return vector  # Return this vector as our embedding!
                        
        return []
        
    except Exception as e:
        print(f"❌ NearText error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

def main():
    """Run debug tests"""
    load_dotenv()
    
    print("Debugging Existing Embeddings in Weaviate")
    print("=" * 50)
    
    # Test 1: Check existing tools and their embeddings
    # tools = test_existing_tool_embeddings()  # Result currently unused
    
    # Test 2: Try nearText query to see if we can get vectors that way
    vector = test_neartext_query()
    
    if vector:
        print("\n🎉 SUCCESS! Found a way to get embeddings via nearText query!")
        print(f"Vector length: {len(vector)}")
    else:
        print("\n⚠️ No vector found via nearText query")

if __name__ == "__main__":
    main()