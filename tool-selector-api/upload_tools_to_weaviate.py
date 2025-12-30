import weaviate
import json
import os
import sys
import time
from pathlib import Path
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

# Import LLM enhancement framework (experimental integration)
try:
    sys.path.append(str(Path(__file__).parent.parent / "experiments" / "tool_description_enhancement"))
    from enhancement_prompts import EnhancementPrompts, ToolContext
    from ollama_client import OllamaClient
    from enhancement_cache import get_cache, cache_tool_enhancement, get_cached_tool_enhancement
    LLM_ENHANCEMENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LLM Enhancement framework not available: {e}")
    print("   Falling back to basic enhancement only.")
    LLM_ENHANCEMENT_AVAILABLE = False

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


class EnhancedToolUploader:
    """Tool uploader with optional LLM enhancement integration"""

    def __init__(self):
        # Check environment variables for LLM enhancement settings
        self.enable_llm_enhancement = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_LLM_BASE_URL", "http://100.81.139.20:11434/v1")
        self.ollama_model = os.getenv("OLLAMA_LLM_MODEL", "gemma3:12b")
        self.batch_size = int(os.getenv("LLM_BATCH_SIZE", "3"))

        # Initialize LLM client if enhancement is enabled and available
        self.llm_client = None
        self.cache = None

        if self.enable_llm_enhancement and LLM_ENHANCEMENT_AVAILABLE:
            try:
                self.llm_client = OllamaClient(
                    base_url=self.ollama_base_url,
                    model=self.ollama_model,
                    timeout=300,  # 5 minute timeout for complex descriptions
                    max_retries=3
                )
                self.cache = get_cache()
                print(f"🤖 LLM Enhancement ENABLED using {self.ollama_model}")
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM client: {e}")
                print("   Falling back to basic enhancement only.")
                self.enable_llm_enhancement = False
        else:
            if self.enable_llm_enhancement:
                print("⚠️  LLM Enhancement requested but framework not available")
            print("📝 Using basic specialized enhancement only")

        # Statistics tracking
        self.stats = {
            "tools_processed": 0,
            "llm_enhancements_successful": 0,
            "llm_enhancements_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "basic_enhancements": 0,
            "uploads_successful": 0,
            "uploads_skipped": 0,
            "uploads_failed": 0,
            "total_enhancement_time": 0.0
        }

    async def enhance_tool_description(self, tool: dict) -> str:
        """
        Enhance tool description using LLM (if available) or basic enhancement.

        Args:
            tool: Tool dictionary with metadata

        Returns:
            Enhanced description string
        """
        start_time = time.time()

        raw_description = tool.get("description", "")
        tool_name = tool.get("name", "")

        # Try LLM enhancement first if enabled
        if self.enable_llm_enhancement and self.llm_client and self.cache:
            try:
                # Check cache first
                cache_key = f"{tool_name}:{hash(raw_description)}"
                cached_result = get_cached_tool_enhancement(self.cache, cache_key)

                if cached_result:
                    self.stats["cache_hits"] += 1
                    enhancement_time = time.time() - start_time
                    self.stats["total_enhancement_time"] += enhancement_time
                    return cached_result

                self.stats["cache_misses"] += 1

                # Create tool context for LLM enhancement
                tool_context = ToolContext(
                    name=tool_name,
                    description=raw_description,
                    tool_type=tool.get("tool_type", "general"),
                    source_type=tool.get("source_type", "python"),
                    tags=tool.get("tags", []),
                    json_schema=tool.get("json_schema", {})
                )

                # Generate LLM enhancement
                enhanced_description = await self.llm_client.enhance_tool_description(tool_context)

                if enhanced_description and enhanced_description.strip():
                    # Cache the successful enhancement
                    cache_tool_enhancement(self.cache, cache_key, enhanced_description)
                    self.stats["llm_enhancements_successful"] += 1

                    enhancement_time = time.time() - start_time
                    self.stats["total_enhancement_time"] += enhancement_time

                    return enhanced_description
                else:
                    raise ValueError("Empty enhancement returned")

            except Exception as e:
                print(f"⚠️  LLM enhancement failed for '{tool_name}': {e}")
                self.stats["llm_enhancements_failed"] += 1

        # Fallback to basic specialized enhancement
        self.stats["basic_enhancements"] += 1
        enhanced_description = enhance_tool_for_embedding(
            tool_description=raw_description,
            tool_name=tool_name,
            tool_type=tool.get("tool_type", "general"),
            tool_source=tool.get("source_type", "python")
        )

        enhancement_time = time.time() - start_time
        self.stats["total_enhancement_time"] += enhancement_time

        return enhanced_description

    def print_stats(self):
        """Print enhancement and upload statistics"""
        print(f"\n{'='*60}")
        print("📊 UPLOAD STATISTICS")
        print(f"{'='*60}")
        print(f"Tools processed: {self.stats['tools_processed']}")
        print(f"Uploads successful: {self.stats['uploads_successful']}")
        print(f"Uploads skipped: {self.stats['uploads_skipped']}")
        print(f"Uploads failed: {self.stats['uploads_failed']}")
        print(f"\n🤖 ENHANCEMENT STATISTICS")
        print(f"LLM enhancements successful: {self.stats['llm_enhancements_successful']}")
        print(f"LLM enhancements failed: {self.stats['llm_enhancements_failed']}")
        print(f"Basic enhancements: {self.stats['basic_enhancements']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Cache misses: {self.stats['cache_misses']}")
        print(f"Total enhancement time: {self.stats['total_enhancement_time']:.2f}s")
        if self.stats['tools_processed'] > 0:
            avg_time = self.stats['total_enhancement_time'] / self.stats['tools_processed']
            print(f"Average enhancement time: {avg_time:.2f}s per tool")


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

        # Initialize enhanced uploader with LLM integration
        uploader = EnhancedToolUploader()

        # Prepare batch import with enhanced descriptions
        print(f"\nUploading tools with enhanced descriptions...")
        print(f"   LLM Enhancement: {'ENABLED' if uploader.enable_llm_enhancement else 'DISABLED'}")
        print(f"   Batch size: {uploader.batch_size}")

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
                        uploader.stats["uploads_skipped"] += 1
                        if i % 25 == 0:
                            print(f"Progress: {i}/{len(tools)} tools processed...")
                        continue

                    # Enhanced tool description using new integrated system
                    enhanced_description = await uploader.enhance_tool_description(tool)
                    uploader.stats["tools_processed"] += 1

                    # Prepare tool properties - Weaviate will automatically vectorize enhanced_description
                    properties = {
                        "tool_id": tool.get("id", ""),
                        "name": tool["name"],
                        "description": tool.get("description", ""),  # Original description for display
                        "enhanced_description": enhanced_description,  # This will be automatically vectorized by Weaviate
                        "source_type": tool.get("source_type", "python"),
                        "tool_type": tool.get("tool_type", "external_mcp"),
                        "tags": tool.get("tags", []),
                        "json_schema": json.dumps(tool.get("json_schema", {})) if tool.get("json_schema") else ""
                    }

                    # Add object to batch - Weaviate will handle embedding generation
                    batch.add_object(properties=properties)
                    uploader.stats["uploads_successful"] += 1

                    if i % 25 == 0:
                        print(f"Progress: {i}/{len(tools)} tools processed...")
                        
                except Exception as e:
                    uploader.stats["uploads_failed"] += 1
                    print(f"Error uploading tool {tool.get('name', 'Unknown')}: {str(e)}")

        # Print comprehensive statistics
        uploader.print_stats()
        
        client.close()

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    # Run the async upload function
    asyncio.run(upload_tools())