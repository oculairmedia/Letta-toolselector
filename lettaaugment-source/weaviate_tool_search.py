import weaviate
from weaviate.classes.query import MetadataQuery, HybridFusion
import os
from dotenv import load_dotenv
from typing import List
import requests
import json

def init_client():
    """Initialize Weaviate client using v4 API."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        # Fallback or raise error if critical
        print("Warning: OPENAI_API_KEY environment variable not set. Vectorizer might fail.")
        # raise ValueError("OPENAI_API_KEY environment variable not set.") # Uncomment if key is strictly required

    # Determine Weaviate host and port
    # Default to Docker service name 'weaviate' if specific env vars are not set
    # These can be overridden by .env file if needed for other environments
    weaviate_http_host = os.getenv("WEAVIATE_HTTP_HOST", "weaviate")
    weaviate_http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    weaviate_grpc_host = os.getenv("WEAVIATE_GRPC_HOST", "weaviate") # Often same as HTTP host for gRPC
    weaviate_grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    # For local development outside Docker, you might set WEAVIATE_HTTP_HOST=localhost in your .env

    print(f"Attempting to connect to Weaviate -> HTTP: {weaviate_http_host}:{weaviate_http_port}, GRPC: {weaviate_grpc_host}:{weaviate_grpc_port}")
    
    client_headers = {}
    if openai_api_key:
        client_headers["X-OpenAI-Api-Key"] = openai_api_key
    else:
        # If no OpenAI key, Weaviate might still work if vectorizer is not OpenAI or if pre-vectorized
        print("No OpenAI API key provided to Weaviate client; vectorization may rely on Weaviate's config or fail if OpenAI is default.")

    try:
        client = weaviate.connect_to_custom(
            http_host=weaviate_http_host,
            http_port=weaviate_http_port,
            http_secure=False,
            grpc_host=weaviate_grpc_host,
            grpc_port=weaviate_grpc_port,
            grpc_secure=False,
            headers=client_headers
            # skip_init_checks=True # It's often better to let it check
            # timeout_config not supported in this version of weaviate-client
        )
        
        # Check connection
        if not client.is_ready():
            print("❌ Weaviate client failed to connect or is not ready.")
            # Potentially raise an error here or return None, depending on desired handling
            # For now, let's allow it to return the client and let caller handle
        else:
            print("✅ Successfully connected to Weaviate and client is ready.")
            
    except Exception as e:
        print(f"❌ Failed to initialize Weaviate client: {e}")
        client = None # Ensure client is None if connection fails

    return client

def preprocess_query(query: str) -> str:
    """
    Expand query with common synonyms and related terms for more robust matching.
    Add or modify expansions as needed to capture domain-specific language.
    """
    expansions = {
        "create": ["add", "new", "publish", "post", "initiate", "build"],
        "post": ["publish", "entry", "article"],
        "list": ["get", "fetch", "show", "display", "view", "enumerate"],
        "delete": ["remove", "destroy", "clear", "erase", "purge"],
        "update": ["edit", "modify", "change", "revise", "upgrade"],
        "search": ["find", "query", "lookup", "locate", "explore"],
        "manage": ["organize", "handle", "control", "track", "administer"],
        "api": ["integration", "service", "endpoint", "connection"],
        "content": ["post", "article", "page", "data", "material", "resource"],
        "tool": ["utility", "function", "capability", "feature"],
        "blog": ["article", "posts", "ghost", "cms", "write-up"],
        "integration": ["api", "service", "connector", "plugin"],
        "configure": ["setup", "initialize", "customize"],
        "ghost": ["blogging", "headless", "cms"],
        "web": ["online", "internet", "site", "webpage"],
    }
    
    words = query.lower().split()
    expanded = set(words)
    
    for word in words:
        if word in expansions:
            expanded.update(expansions[word])
    
    return " ".join(expanded)

def search_tools(query: str, limit: int = 10) -> list:
    """
    Search tools by semantic similarity using hybrid search with query expansion.
    The result is a list of tools containing key metadata and score information.
    Initializes and closes its own Weaviate client.
    """
    client = None
    try:
        client = init_client()
        
        try:
            # Get the Tool collection
            collection = client.collections.get("Tool")
            
            # Expand query with related terms
            expanded_query = preprocess_query(query)
            
            # Perform hybrid search using v4 API
            result = collection.query.hybrid(
                query=expanded_query,
                alpha=0.75,  # 75% vector search, 25% keyword search
                limit=limit,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                query_properties=["name^2", "description^1.5", "tags"],
                return_metadata=MetadataQuery(score=True)
            )

            # Process results
            tools = []
            if result and hasattr(result, 'objects'):
                for obj in result.objects:
                    tool_data = obj.properties
                    if hasattr(obj, 'metadata') and obj.metadata is not None:
                        score = getattr(obj.metadata, 'score', 0.5)
                        tool_data["distance"] = 1 - (score if score is not None else 0.5)
                    else:
                        tool_data["distance"] = 0.5
                    tools.append(tool_data)
            return tools
            
        except Exception as e:
            print(f"Error in collection query: {e}")
            return []
            
    except Exception as e:
        print(f"Error in search_tools: {e}")
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

def _get_embedding_direct_openai(text: str) -> List[float]:
    """
    Fallback function to get embeddings directly from OpenAI API
    when Weaviate's vectorizer fails.
    """
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("OpenAI API key not found for direct embedding")
            return []
            
        # Make direct request to OpenAI API
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "text-embedding-3-small",
            "input": text
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
                print(f"✅ Direct OpenAI embedding successful, length: {len(embedding)}")
                return embedding
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error getting direct OpenAI embedding: {e}")
        
    return []

def get_embedding_for_text(text: str) -> list:
    """
    Get vector embedding for a given text string using the Weaviate client.
    Uses the text2vec-openai vectorizer with text-embedding-3-small model
    as configured for the Tool collection.
    
    Args:
        text (str): The text string to get embedding for
        
    Returns:
        list: Vector embedding as a list of floats, or empty list if error occurs
    """
    client = None
    try:
        client = init_client()
        
        # Use GraphQL to get vector representation of the text
        # This leverages the text2vec-openai vectorizer configured for the Tool collection
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
        
        result = client.graphql_raw_query(query)
        
        # Debug the result structure
        print(f"GraphQL result type: {type(result)}")
        print(f"Has 'get' attribute: {hasattr(result, 'get')}")
        
        if hasattr(result, 'get') and result.get:
            print(f"result.get keys: {list(result.get.keys()) if result.get else 'None'}")
            
            if 'Tool' in result.get:
                tools = result.get['Tool']
                print(f"Found {len(tools)} tools in result")
                
                if len(tools) > 0 and '_additional' in tools[0] and 'vector' in tools[0]['_additional']:
                    vector = tools[0]['_additional']['vector']
                    print(f"✅ Got embedding via GraphQL! Length: {len(vector)}")
                    return vector if isinstance(vector, list) else []
                else:
                    print(f"❌ No suitable tools found or no vector in result")
                    if len(tools) > 0:
                        print(f"First tool keys: {list(tools[0].keys())}")
                        if '_additional' in tools[0]:
                            print(f"_additional keys: {list(tools[0]['_additional'].keys())}")
            else:
                print("❌ No 'Tool' key in result.get")
        else:
            print("❌ No result.get or result.get is empty")
            
        # Check for GraphQL errors
        if hasattr(result, 'errors') and result.errors:
            print(f"GraphQL errors: {result.errors}")
        
        # If GraphQL approach fails, use direct OpenAI fallback
        print("GraphQL approach failed, using direct OpenAI fallback...")
        return _get_embedding_direct_openai(text)
        
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        # Final fallback: Use OpenAI directly
        print("Exception occurred, using direct OpenAI fallback...")
        return _get_embedding_direct_openai(text)
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

def get_tool_embedding_by_id(tool_id: str) -> List[float]:
    """
    Retrieve the stored vector embedding for a tool given its ID.
    
    Args:
        tool_id (str): The unique identifier for the tool (Weaviate UUID)
        
    Returns:
        List[float]: Vector embedding as a list of floats, or empty list if not found or error occurs
    """
    client = None
    try:
        client = init_client()
        
        # Get the Tool collection
        collection = client.collections.get("Tool")
        
        # Query for the specific tool by ID, including vector data
        response = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_id().equal(tool_id),
            limit=1,
            include_vector=True
        )
        
        # Check if we found the tool and it has a vector
        if response.objects and len(response.objects) > 0:
            tool_obj = response.objects[0]
            # Also check if the vector is not empty and has a reasonable length (e.g. > 1)
            if hasattr(tool_obj, 'vector') and tool_obj.vector is not None and len(tool_obj.vector) > 1:
                vector = tool_obj.vector
                if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
                    print(f"✅ Retrieved vector for tool {tool_id} via fetch_objects. Length: {len(vector)}")
                    return vector
                else:
                    print(f"⚠️ Vector for tool {tool_id} (from fetch_objects) is not a list of numbers: {type(vector)}")
            else:
                print(f"⚠️ No valid vector attribute for tool {tool_id} via fetch_objects (vector: {tool_obj.vector if hasattr(tool_obj, 'vector') else 'N/A'}).")
        else:
            print(f"⚠️ Tool {tool_id} not found via fetch_objects.")
        
        # Fallback to GraphQL if fetch_objects didn't yield a valid vector
        print(f"Trying GraphQL fallback for tool {tool_id} as fetch_objects did not return a valid vector.")
        
        # The tool_id passed to this function is the Weaviate UUID.
        # The GraphQL query needs to filter by the internal 'id' (which is the UUID)
        graphql_query = f"""
        {{
          Get {{
            Tool(where: {{operator: Equal, path: ["id"], valueString: "{tool_id}"}}, limit: 1) {{
              _additional {{
                vector
              }}
            }}
          }}
        }}
        """
        print(f"Executing GraphQL query for tool UUID: {tool_id}")
        gql_response = client.graphql_raw_query(graphql_query)

        if hasattr(gql_response, 'get') and gql_response.get and \
           'Tool' in gql_response.get and len(gql_response.get['Tool']) > 0 and \
           '_additional' in gql_response.get['Tool'][0] and \
           'vector' in gql_response.get['Tool'][0]['_additional']:
            vector = gql_response.get['Tool'][0]['_additional']['vector']
            if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector) and len(vector) > 1:
                print(f"✅ Retrieved vector for tool {tool_id} via GraphQL fallback. Length: {len(vector)}")
                return vector
            else:
                print(f"⚠️ GraphQL fallback for tool {tool_id} returned invalid vector (type: {type(vector)}, len: {len(vector) if isinstance(vector, list) else 'N/A'}).")
        else:
            print(f"⚠️ GraphQL fallback for tool {tool_id} failed to retrieve vector. Response: {gql_response}")
            
        return []
        
    except Exception as e:
        print(f"Error retrieving tool embedding by ID '{tool_id}': {e}")
        return []
        
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Error closing client: {e}")
