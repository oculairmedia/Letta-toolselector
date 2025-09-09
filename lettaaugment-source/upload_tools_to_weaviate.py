import weaviate
import json
import os
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import asyncio # Added asyncio
from weaviate.collections import Collection
import weaviate.classes.query as wq
from dotenv import load_dotenv
# Import the new async function
from fetch_all_tools import fetch_all_tools_async
# Import embedding configuration constants
from embedding_config import OPENAI_EMBEDDING_MODEL, WEAVIATE_VECTORIZER
# Import specialized embedding functionality
from specialized_embedding import enhance_tool_for_embedding
from embedding_providers import EmbeddingProviderFactory

def get_or_create_tool_schema(client) -> Collection:
    """Get existing schema or create new one if it doesn't exist."""
    
    collection_name = "Tool"

    try:
        # Check if collection exists and delete it to ensure correct schema
        if client.collections.exists(collection_name):
            print(f"Deleting existing '{collection_name}' collection to ensure correct schema...")
            client.collections.delete(collection_name)
            print(f"Collection '{collection_name}' deleted.")
        # If deletion was successful or collection didn't exist, proceed to create
        # The original code would return here if collection existed, now we fall through to create
    except Exception as e:
        # Handle potential errors during check/delete, e.g., permissions
        print(f"Note: Could not check/delete existing collection '{collection_name}': {e}. Proceeding to create.")

    # Always attempt to create after checking/deleting
    try:
        print(f"Attempting to create new '{collection_name}' schema...")
        # Create the schema with Ollama vectorizer for automatic embedding generation
        collection = client.collections.create(
            name="Tool",
            description="A Letta tool with its metadata and description",
            # Use Ollama vectorizer with Qwen3-Embedding-4B model from environment
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_ollama(
                api_endpoint=f"http://{os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}:11434",
                model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                vectorize_collection_name=False  # Don't include collection name in embedding
            ),
            properties=[
                weaviate.classes.config.Property(
                    name="tool_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The unique identifier of the tool",
                ),
                weaviate.classes.config.Property(
                    name="name",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The name of the tool",
                ),
                weaviate.classes.config.Property(
                    name="description",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The description of what the tool does",
                    vectorize_property_name=False  # Don't vectorize the property name
                    # The property value will be vectorized as part of the object
                ),
                weaviate.classes.config.Property(
                    name="enhanced_description", 
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Enhanced tool description with specialized prompting for better embeddings",
                    vectorize_property_name=False
                ),
                weaviate.classes.config.Property(
                    name="source_type",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The type of tool (python, mcp, etc)",
                ),
                weaviate.classes.config.Property(
                    name="tool_type",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The specific tool type (custom, external_mcp, etc)",
                ),
                weaviate.classes.config.Property(
                    name="tags",
                    data_type=weaviate.classes.config.DataType.TEXT_ARRAY,
                    description="Tags associated with the tool",
                ),
                weaviate.classes.config.Property(
                    name="json_schema",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="The JSON schema defining the tool's interface",
                    vectorize_property_name=False
                )
            ]
        )
        print("Schema created successfully")
        return collection
    except Exception as e:
        # If creation fails, it's a fatal error for this script's purpose
        print(f"FATAL: Failed to create schema '{collection_name}': {e}")
        raise  # Re-raise the exception to halt the script

async def upload_tools(): # Make the function async
    """Upload tools to Weaviate."""
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        return
    
    try:
        # Initialize Weaviate client
        print("\nConnecting to Weaviate at 192.168.50.90:8080...")
        client = weaviate.connect_to_custom(
            http_host="192.168.50.90",
            http_port=8080,
            http_secure=False,
            grpc_host="192.168.50.90",
            grpc_port=50051,
            grpc_secure=False,
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            },
            skip_init_checks=True
        )

        # Get or create schema
        print("Getting/Creating schema...")
        collection = get_or_create_tool_schema(client)
        client.connect()  # Ensure we're connected after schema creation

        # Fetch all tools using the async function
        print("\nFetching all tools (async)...")
        # Note: This runs the full fetch/register/save process from fetch_all_tools_async
        # We might want to refactor fetch_all_tools_async later to *only* return data
        tools = await fetch_all_tools_async()
        if not tools:
             print("Error: Failed to fetch tools. Aborting upload.")
             if 'client' in locals() and client.is_connected(): client.close()
             return

        print(f"Found {len(tools)} tools fetched/registered to process for Weaviate upload")

        # Prepare batch import with Weaviate's automatic vectorization
        print("\nUploading tools with Ollama-powered specialized embeddings...")
        successful_uploads = 0
        skipped_uploads = 0
        failed_uploads = 0

        with collection.batch.dynamic() as batch:
            for i, tool in enumerate(tools, 1):
                try:
                    # Check if tool already exists by querying for name
                    name_filter = wq.Filter.by_property("name").equal(tool["name"])
                    query = collection.query.fetch_objects(
                        limit=1,
                        filters=name_filter
                    )

                    # If tool exists, skip it
                    if query.objects:
                        skipped_uploads += 1
                        if i % 25 == 0:
                            print(f"Progress: {i}/{len(tools)} tools processed...")
                        continue

                    # Enhanced tool description with specialized prompting
                    raw_description = tool.get("description", "")
                    enhanced_description = enhance_tool_for_embedding(
                        tool_description=raw_description,
                        tool_name=tool["name"],
                        tool_type=tool.get("tool_type", "general"),
                        tool_source=tool.get("source_type", "python")
                    )

                    # Prepare tool properties - Weaviate will automatically vectorize enhanced_description
                    properties = {
                        "tool_id": tool.get("id", ""),
                        "name": tool["name"],
                        "description": raw_description,  # Original description for display
                        "enhanced_description": enhanced_description,  # This will be automatically vectorized by Weaviate
                        "source_type": tool.get("source_type", "python"),
                        "tool_type": tool.get("tool_type", "external_mcp"),
                        "tags": tool.get("tags", []),
                        "json_schema": json.dumps(tool.get("json_schema", {})) if tool.get("json_schema") else ""
                    }
                    
                    # Add object to batch - Weaviate will handle embedding generation
                    batch.add_object(properties=properties)
                    successful_uploads += 1

                    if i % 25 == 0:
                        print(f"Progress: {i}/{len(tools)} tools processed...")
                        
                except Exception as e:
                    failed_uploads += 1
                    print(f"Error uploading tool {tool.get('name', 'Unknown')}: {str(e)}")

        print(f"\nUpload complete:")
        print(f"- Successfully uploaded: {successful_uploads} tools")
        print(f"- Skipped (already exist): {skipped_uploads} tools")
        if failed_uploads > 0:
            print(f"- Failed uploads: {failed_uploads} tools")
        
        client.close()

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    # Run the async upload function
    asyncio.run(upload_tools())