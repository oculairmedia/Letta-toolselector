import weaviate
import os
from dotenv import load_dotenv
from embedding_config import OPENAI_EMBEDDING_MODEL, WEAVIATE_VECTORIZER

def init_weaviate_schema():
    """Initialize Weaviate schema with the Tool collection."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    print("Connecting to Weaviate at 192.168.50.90:8080...")
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

    try:
        # Check if the Tool collection already exists
        try:
            # More robust check for existing collection
            if client.collections.exists("Tool"):
                print("Tool collection already exists. Deleting it...")
                client.collections.delete("Tool")
                print("Tool collection deleted.")
            else:
                print("Tool collection does not exist. Proceeding to create.")
        except Exception as e:
            print(f"Error during pre-creation check/delete of Tool collection: {e}")
            # Decide if this is fatal or if we can try to create anyway
            # For now, let's try to proceed with creation

        # Create the Tool collection
        print("Creating Tool collection...")
        client.collections.create_from_dict({
            "class": "Tool",
            "description": "A Letta tool with its metadata and description",
            "vectorizer": WEAVIATE_VECTORIZER,
            "moduleConfig": {
                "text2vec-openai": {
                    "model": OPENAI_EMBEDDING_MODEL,
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
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
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
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
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
        print("Tool collection created successfully.")

        # Add a sample tool for testing
        print("Adding a sample tool...")
        client.collections.get("Tool").data.insert({
            "tool_id": "sample_tool",
            "name": "Sample Tool",
            "description": "This is a sample tool for testing the Weaviate connection.",
            "source_type": "python",
            "tags": ["sample", "test"],
            "json_schema": "{\"type\": \"object\", \"properties\": {\"input\": {\"type\": \"string\"}}}"
        })
        print("Sample tool added successfully.")

        print("Weaviate schema initialized successfully.")
    finally:
        client.close()

if __name__ == "__main__":
    init_weaviate_schema()