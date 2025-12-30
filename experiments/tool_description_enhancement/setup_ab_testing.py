#!/usr/bin/env python3
"""
Setup Script for A/B Testing Framework

This script helps set up the A/B testing environment by:
1. Creating a copy of the current Tool collection as baseline
2. Running enhancement on tools to create enhanced collection
3. Verifying both collections are ready for testing
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add the parent directories to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "tool-selector-api"))

from upload_tools_to_weaviate import EnhancedToolUploader, get_or_create_tool_schema
import weaviate
from dotenv import load_dotenv
from fetch_all_tools import fetch_all_tools_async

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ABTestingSetup:
    """Setup utility for A/B testing framework"""

    def __init__(self):
        load_dotenv()
        self.weaviate_client = None
        self.uploader = EnhancedToolUploader()

    def connect_weaviate(self):
        """Connect to Weaviate"""
        logger.info("Connecting to Weaviate...")
        self.weaviate_client = weaviate.connect_to_custom(
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

    def create_baseline_collection(self):
        """Create baseline collection (copy of existing Tool collection)"""
        logger.info("Creating baseline Tool collection...")

        # The baseline collection is the existing "Tool" collection
        # We don't need to do anything special - it already exists
        collection_name = "Tool"

        try:
            if self.weaviate_client.collections.exists(collection_name):
                logger.info(f"✅ Baseline collection '{collection_name}' already exists")
                return True
            else:
                logger.error(f"❌ Baseline collection '{collection_name}' does not exist")
                logger.error("Please run upload_tools_to_weaviate.py first to create the baseline collection")
                return False
        except Exception as e:
            logger.error(f"Error checking baseline collection: {e}")
            return False

    def create_enhanced_collection(self):
        """Create enhanced collection with LLM-enhanced descriptions"""
        logger.info("Creating enhanced Tool collection with LLM-enhanced descriptions...")

        collection_name = "ToolEnhanced"

        try:
            # Check if enhanced collection exists and delete it to recreate
            if self.weaviate_client.collections.exists(collection_name):
                logger.info(f"Deleting existing '{collection_name}' collection...")
                self.weaviate_client.collections.delete(collection_name)
                logger.info(f"Collection '{collection_name}' deleted.")

            # Create the enhanced schema (same as regular Tool schema but different name)
            logger.info(f"Creating new '{collection_name}' schema...")
            collection = self.weaviate_client.collections.create(
                name=collection_name,
                description="A Letta tool with LLM-enhanced metadata and description",
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_ollama(
                    api_endpoint=f"http://{os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')}:11434",
                    model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                    vectorize_collection_name=False
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
                        vectorize_property_name=False
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
            logger.info("✅ Enhanced collection schema created successfully")
            return collection

        except Exception as e:
            logger.error(f"❌ Failed to create enhanced collection: {e}")
            raise

    async def populate_enhanced_collection(self):
        """Populate enhanced collection with LLM-enhanced tool descriptions"""
        logger.info("Populating enhanced collection with LLM-enhanced descriptions...")

        # Enable LLM enhancement
        self.uploader.enable_llm_enhancement = True
        os.environ['ENABLE_LLM_ENHANCEMENT'] = 'true'

        # Get enhanced collection
        collection = self.weaviate_client.collections.get("ToolEnhanced")

        # Fetch all tools
        tools = await fetch_all_tools_async()
        if not tools:
            logger.error("❌ Failed to fetch tools")
            return False

        logger.info(f"Processing {len(tools)} tools with LLM enhancement...")

        # Process tools with enhanced descriptions
        with collection.batch.dynamic() as batch:
            for i, tool in enumerate(tools, 1):
                try:
                    # Enhanced tool description using LLM
                    enhanced_description = await self.uploader.enhance_tool_description(tool)

                    # Prepare tool properties
                    properties = {
                        "tool_id": tool.get("id", ""),
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "enhanced_description": enhanced_description,
                        "source_type": tool.get("source_type", "python"),
                        "tool_type": tool.get("tool_type", "external_mcp"),
                        "tags": tool.get("tags", []),
                        "json_schema": tool.get("json_schema", "") if isinstance(tool.get("json_schema"), str) else str(tool.get("json_schema", {}))
                    }

                    # Add to batch
                    batch.add_object(properties=properties)

                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(tools)} tools processed...")

                except Exception as e:
                    logger.error(f"Error processing tool {tool.get('name', 'Unknown')}: {str(e)}")

        logger.info("✅ Enhanced collection populated successfully")
        return True

    def verify_collections(self):
        """Verify both collections are ready for testing"""
        logger.info("Verifying collections for A/B testing...")

        collections_to_check = ["Tool", "ToolEnhanced"]
        results = {}

        for collection_name in collections_to_check:
            try:
                if self.weaviate_client.collections.exists(collection_name):
                    collection = self.weaviate_client.collections.get(collection_name)

                    # Get sample of objects to count
                    result = collection.query.fetch_objects(limit=1)

                    # Estimate total count (Weaviate doesn't have direct count)
                    # We'll use aggregate to get more accurate count
                    try:
                        aggregate_result = collection.aggregate.over_all(total_count=True)
                        total_count = aggregate_result.total_count
                    except:
                        # Fallback: estimate from sample
                        total_count = "Unknown"

                    results[collection_name] = {
                        "exists": True,
                        "count": total_count,
                        "sample_tool": result.objects[0].properties if result.objects else None
                    }
                    logger.info(f"✅ Collection '{collection_name}': {total_count} tools")
                else:
                    results[collection_name] = {"exists": False, "count": 0, "sample_tool": None}
                    logger.error(f"❌ Collection '{collection_name}' does not exist")

            except Exception as e:
                logger.error(f"Error checking collection '{collection_name}': {e}")
                results[collection_name] = {"exists": False, "error": str(e)}

        return results

    async def run_setup(self):
        """Run complete A/B testing setup"""
        logger.info("🚀 Starting A/B Testing Framework Setup")

        try:
            # Connect to Weaviate
            self.connect_weaviate()

            # Check baseline collection
            if not self.create_baseline_collection():
                return False

            # Create enhanced collection
            enhanced_collection = self.create_enhanced_collection()
            if not enhanced_collection:
                return False

            # Populate enhanced collection
            if not await self.populate_enhanced_collection():
                return False

            # Verify both collections
            verification_results = self.verify_collections()

            # Print results
            logger.info("\n" + "="*60)
            logger.info("SETUP COMPLETE")
            logger.info("="*60)

            for collection_name, result in verification_results.items():
                if result.get("exists"):
                    logger.info(f"✅ {collection_name}: {result.get('count', 'Unknown')} tools ready")
                else:
                    logger.error(f"❌ {collection_name}: Not ready")

            all_ready = all(r.get("exists", False) for r in verification_results.values())

            if all_ready:
                logger.info("\n🎉 A/B Testing Framework is ready!")
                logger.info("   You can now run: python ab_testing_framework.py")

                # Print usage instructions
                logger.info("\n📋 Usage Instructions:")
                logger.info("   1. Modify test queries in ab_testing_framework.py if needed")
                logger.info("   2. Run: python ab_testing_framework.py")
                logger.info("   3. Check results in ab_test_results/ directory")

                return True
            else:
                logger.error("\n❌ Setup incomplete. Check errors above.")
                return False

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
        finally:
            if self.weaviate_client:
                self.weaviate_client.close()


async def main():
    """Main function"""
    setup = ABTestingSetup()
    await setup.run_setup()


if __name__ == "__main__":
    asyncio.run(main())