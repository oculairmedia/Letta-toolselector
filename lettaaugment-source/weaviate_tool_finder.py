import os
import json
import weaviate
from typing import List, Dict, Any
from dotenv import load_dotenv

class WeaviateConnection:
    def __init__(self):
        load_dotenv()
        # Use connect_to_local with v4 API
        # Keep grpc_port as an integer
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,  # Must be an integer
            skip_init_checks=True  # Skip initialization checks
        )
        
        # Add OpenAI API key for vectorization
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.client.set_additional_headers({
                "X-OpenAI-Api-Key": openai_api_key
            })

    def close(self):
        if self.client:
            self.client.close()


def create_tool_uploader():
    """Create a tool definition for uploading tools to Weaviate Cloud"""
    return {
        "id": "upload_tools_to_weaviate",
        "name": "Upload Tools to Weaviate Cloud",
        "description": "Uploads tool definitions to Weaviate Cloud for semantic search. Requires a JSON file containing tool definitions.",
        "source_type": "python",
        "tags": ["tools", "weaviate", "upload"],
        "json_schema": {
            "type": "object",
            "properties": {
                "tools_file": {
                    "type": "string",
                    "description": "Path to the JSON file containing tool definitions"
                }
            },
            "required": ["tools_file"]
        }
    }


def create_tool_finder():
    """Create a tool definition for finding tools using Weaviate Cloud"""
    return {
        "id": "find_tools_weaviate",
        "name": "Find Tools Using Weaviate",
        "description": "Performs a semantic search using Weaviate Cloud to find relevant tools based on a query. Returns the most relevant tools matching the query.",
        "source_type": "python",
        "tags": ["tools", "weaviate", "search", "semantic"],
        "json_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the desired tool functionality"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tools to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }


def upload_tools_to_weaviate(tools_file: str) -> Dict[str, Any]:
    """Implementation of the upload_tools_to_weaviate tool"""
    try:
        conn = WeaviateConnection()
        
        # Create schema for tools using v4 API
        try:
            # First try to delete the collection if it exists
            try:
                conn.client.collections.delete("Tool")
                print("Deleted existing Tool collection")
            except Exception as e:
                print(f"Note: Could not delete existing Tool collection: {e}")
            
            # Create the collection with properties
            conn.client.collections.create_from_dict({
                "class": "Tool",
                "description": "A Letta tool with its metadata and description",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada-002",
                        "modelVersion": "002",
                        "type": "text"
                    }
                },
                "properties": [
                    {
                        "name": "tool_id",
                        "dataType": ["string"],
                        "description": "The unique identifier of the tool",
                    },
                    {
                        "name": "name",
                        "dataType": ["string"],
                        "description": "The name of the tool",
                    },
                    {
                        "name": "description",
                        "dataType": ["text"],
                        "description": "The description of what the tool does",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "source_type",
                        "dataType": ["string"],
                        "description": "The type of tool (python, mcp, etc)",
                    },
                    {
                        "name": "tags",
                        "dataType": ["string[]"],
                        "description": "Tags associated with the tool",
                    },
                    {
                        "name": "json_schema",
                        "dataType": ["text"],
                        "description": "The JSON schema defining the tool's interface",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    }
                ]
            })
            print("Created Tool collection")
            
        except Exception as e:
            print(f"Warning: Error creating schema: {e}")
            print("Attempting to proceed with existing schema...")

        # Load and upload tools
        with open(tools_file, 'r') as f:
            tools = json.load(f)

        # Batch import tools using v4 API
        with conn.client.batch.fixed_size(batch_size=100) as batch:
            for tool in tools:
                properties = {
                    "tool_id": tool["id"],
                    "name": tool["name"],
                    "description": tool["description"],
                    "source_type": tool["source_type"],
                    "tags": tool.get("tags", []),
                    "json_schema": json.dumps(tool["json_schema"]) if tool.get("json_schema") else ""
                }
                
                batch.add_object(
                    properties=properties,
                    collection="Tool"
                )

        conn.close()
        return {"status": "success", "message": f"Successfully uploaded {len(tools)} tools"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def find_tools_weaviate(query: str, limit: int = 5) -> Dict[str, Any]:
    """Implementation of the find_tools tool using Weaviate"""
    try:
        conn = WeaviateConnection()
        
        # Get the Tool collection
        collection = conn.client.collections.get("Tool")
        
        # Perform semantic search using v4 API
        result = collection.query.near_text(
            query=query,
            limit=limit
        )

        # Extract tools from response
        tools = []
        if result and hasattr(result, 'objects'):
            for obj in result.objects:
                tool = obj.properties
                # Convert JSON schema string back to object
                if "json_schema" in tool and tool["json_schema"]:
                    try:
                        tool["json_schema"] = json.loads(tool["json_schema"])
                    except:
                        pass
                tools.append(tool)

        conn.close()
        return {
            "status": "success",
            "tools": tools
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Export tool definitions and implementations
tool_definitions = {
    "upload_tools_to_weaviate": create_tool_uploader(),
    "find_tools_weaviate": create_tool_finder()
}

tool_implementations = {
    "upload_tools_to_weaviate": upload_tools_to_weaviate,
    "find_tools_weaviate": find_tools_weaviate
}                       "dataType": ["text"],
                        "description": "The JSON schema defining the tool's interface",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    }
                ]
            })
            print("Created Tool collection")
            
        except Exception as e:
            print(f"Warning: Error creating schema: {e}")
            print("Attempting to proceed with existing schema...")

        # Load and upload tools
        with open(tools_file, 'r') as f:
            tools = json.load(f)

        # Batch import tools using v4 API
        with conn.client.batch.fixed_size(batch_size=100) as batch:
            for tool in tools:
                properties = {
                    "tool_id": tool["id"],
                    "name": tool["name"],
                    "description": tool["description"],
                    "source_type": tool["source_type"],
                    "tags": tool.get("tags", []),
                    "json_schema": json.dumps(tool["json_schema"]) if tool.get("json_schema") else ""
                }
                
                batch.add_object(
                    properties=properties,
                    collection="Tool"
                )

        conn.close()
        return {"status": "success", "message": f"Successfully uploaded {len(tools)} tools"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def find_tools_weaviate(query: str, limit: int = 5) -> Dict[str, Any]:
    """Implementation of the find_tools tool using Weaviate"""
    try:
        conn = WeaviateConnection()
        
        # Get the Tool collection
        collection = conn.client.collections.get("Tool")
        
        # Perform semantic search using v4 API
        result = collection.query.near_text(
            query=query,
            limit=limit
        )

        # Extract tools from response
        tools = []
        if result and hasattr(result, 'objects'):
            for obj in result.objects:
                tool = obj.properties
                # Convert JSON schema string back to object
                if "json_schema" in tool and tool["json_schema"]:
                    try:
                        tool["json_schema"] = json.loads(tool["json_schema"])
                    except:
                        pass
                tools.append(tool)

        conn.close()
        return {
            "status": "success",
            "tools": tools
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Export tool definitions and implementations
tool_definitions = {
    "upload_tools_to_weaviate": create_tool_uploader(),
    "find_tools_weaviate": create_tool_finder()
}

tool_implementations = {
    "upload_tools_to_weaviate": upload_tools_to_weaviate,
    "find_tools_weaviate": find_tools_weaviate
}