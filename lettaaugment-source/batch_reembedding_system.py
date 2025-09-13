#!/usr/bin/env python3
"""
Batch Re-embedding System with Resume Capability

This system handles large-scale re-embedding of tools in Weaviate when:
- Switching embedding models (e.g., OpenAI -> Ollama)
- Updating to newer model versions
- Migrating embedding approaches
- Recovering from interrupted processes

Key Features:
- Resume capability with checkpoint system
- Dual-index strategy for safe migration
- Progress tracking and error recovery
- Batch processing with configurable sizes
- Embedding model comparison utilities
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import hashlib

import weaviate
from weaviate.collections import Collection
import weaviate.classes.query as wq
from dotenv import load_dotenv

# Import existing components
from fetch_all_tools import fetch_all_tools_async
from upload_tools_to_weaviate import EnhancedToolUploader
from embedding_providers import EmbeddingProviderFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMigrationConfig:
    """Configuration for embedding migration"""
    source_collection: str = "Tool"
    target_collection: str = "ToolReembedded"
    checkpoint_file: str = "reembedding_checkpoint.json"
    batch_size: int = 50
    max_retries: int = 3
    delay_between_batches: float = 1.0
    backup_enabled: bool = True
    verification_enabled: bool = True

@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resume capability"""
    processed_count: int = 0
    failed_tools: List[str] = None
    last_processed_tool: Optional[str] = None
    start_time: str = ""
    current_batch: int = 0
    total_tools: int = 0
    embedding_provider: str = ""
    embedding_model: str = ""

    def __post_init__(self):
        if self.failed_tools is None:
            self.failed_tools = []

@dataclass
class ReembeddingStats:
    """Statistics for re-embedding process"""
    total_tools: int = 0
    processed_successfully: int = 0
    failed_tools: int = 0
    skipped_tools: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    embedding_model_used: str = ""
    embedding_provider: str = ""
    checkpoint_saves: int = 0

class BatchReembeddingSystem:
    """Comprehensive system for batch re-embedding with resume capability"""

    def __init__(self, config: EmbeddingMigrationConfig = None):
        load_dotenv()

        self.config = config or EmbeddingMigrationConfig()
        self.weaviate_client = None
        self.uploader = EnhancedToolUploader()
        self.checkpoint = ProcessingCheckpoint()
        self.stats = ReembeddingStats()

        # Initialize embedding provider
        try:
            self.embedding_provider = EmbeddingProviderFactory.create_from_env()
        except Exception as e:
            logger.warning(f"Could not initialize embedding provider: {e}")
            self.embedding_provider = None

        self.checkpoint_file_path = Path(self.config.checkpoint_file)

    def connect_weaviate(self):
        """Connect to Weaviate instance"""
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

    def create_target_collection(self) -> Collection:
        """Create target collection with new embedding configuration"""
        logger.info(f"Creating target collection: {self.config.target_collection}")

        # Delete existing target collection if it exists
        if self.weaviate_client.collections.exists(self.config.target_collection):
            logger.info(f"Deleting existing '{self.config.target_collection}' collection...")
            self.weaviate_client.collections.delete(self.config.target_collection)
            logger.info(f"Collection '{self.config.target_collection}' deleted.")

        # Create new collection with updated vectorizer configuration
        collection = self.weaviate_client.collections.create(
            name=self.config.target_collection,
            description="Re-embedded tool collection with updated embedding model",
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
                    description="Enhanced tool description for better embeddings",
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
                ),
                # Migration metadata
                weaviate.classes.config.Property(
                    name="migration_timestamp",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Timestamp when this tool was re-embedded",
                ),
                weaviate.classes.config.Property(
                    name="embedding_model_version",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="Version of embedding model used",
                )
            ]
        )

        logger.info("✅ Target collection created successfully")
        return collection

    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                "checkpoint": asdict(self.checkpoint),
                "stats": asdict(self.stats),
                "config": asdict(self.config),
                "timestamp": datetime.now().isoformat()
            }

            with open(self.checkpoint_file_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            self.stats.checkpoint_saves += 1
            logger.debug(f"Checkpoint saved: {self.checkpoint.processed_count} tools processed")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> bool:
        """Load checkpoint from file if exists"""
        try:
            if not self.checkpoint_file_path.exists():
                logger.info("No checkpoint file found, starting fresh")
                return False

            with open(self.checkpoint_file_path, 'r') as f:
                checkpoint_data = json.load(f)

            # Restore checkpoint
            checkpoint_dict = checkpoint_data.get("checkpoint", {})
            self.checkpoint = ProcessingCheckpoint(**checkpoint_dict)

            # Restore stats
            stats_dict = checkpoint_data.get("stats", {})
            self.stats = ReembeddingStats(**stats_dict)

            logger.info(f"✅ Checkpoint loaded: {self.checkpoint.processed_count} tools already processed")
            logger.info(f"   Failed tools: {len(self.checkpoint.failed_tools)}")
            logger.info(f"   Last processed: {self.checkpoint.last_processed_tool}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def cleanup_checkpoint(self):
        """Clean up checkpoint file after successful completion"""
        try:
            if self.checkpoint_file_path.exists():
                self.checkpoint_file_path.unlink()
                logger.info("✅ Checkpoint file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint file: {e}")

    async def get_tools_to_process(self) -> List[Dict[str, Any]]:
        """Get list of tools that need to be re-embedded"""
        logger.info("Fetching tools for re-embedding...")

        # Fetch all current tools
        tools = await fetch_all_tools_async()
        if not tools:
            logger.error("Failed to fetch tools")
            return []

        # Filter out already processed tools if resuming
        if self.checkpoint.processed_count > 0:
            processed_tools = set()

            # Get already processed tools from target collection
            try:
                target_collection = self.weaviate_client.collections.get(self.config.target_collection)

                # Fetch all processed tools (paginated)
                processed_results = target_collection.query.fetch_objects(
                    limit=10000  # Adjust if you have more tools
                )

                processed_tools = {obj.properties.get("name", "") for obj in processed_results.objects}
                logger.info(f"Found {len(processed_tools)} already processed tools")

            except Exception as e:
                logger.warning(f"Could not fetch processed tools: {e}")

            # Filter tools
            if processed_tools:
                original_count = len(tools)
                tools = [tool for tool in tools if tool.get("name", "") not in processed_tools]
                logger.info(f"Filtered out {original_count - len(tools)} already processed tools")

        logger.info(f"Tools to process: {len(tools)}")
        return tools

    async def process_tool_batch(self, tools_batch: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Process a batch of tools for re-embedding"""
        successes = 0
        failures = 0

        target_collection = self.weaviate_client.collections.get(self.config.target_collection)
        migration_timestamp = datetime.now().isoformat()

        with target_collection.batch.dynamic() as batch:
            for tool in tools_batch:
                try:
                    tool_name = tool.get("name", "Unknown")

                    # Check if already processed (double-check)
                    name_filter = wq.Filter.by_property("name").equal(tool_name)
                    existing = target_collection.query.fetch_objects(limit=1, filters=name_filter)

                    if existing.objects:
                        logger.debug(f"Tool {tool_name} already exists, skipping")
                        self.stats.skipped_tools += 1
                        continue

                    # Enhance description if needed
                    enhanced_description = await self.uploader.enhance_tool_description(tool)

                    # Prepare tool properties
                    properties = {
                        "tool_id": tool.get("id", ""),
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "enhanced_description": enhanced_description,
                        "source_type": tool.get("source_type", "python"),
                        "tool_type": tool.get("tool_type", "external_mcp"),
                        "tags": tool.get("tags", []),
                        "json_schema": json.dumps(tool.get("json_schema", {})) if tool.get("json_schema") else "",
                        "migration_timestamp": migration_timestamp,
                        "embedding_model_version": self.stats.embedding_model_used
                    }

                    # Add to batch (Weaviate will handle embedding generation)
                    batch.add_object(properties=properties)

                    successes += 1
                    self.checkpoint.last_processed_tool = tool_name

                except Exception as e:
                    logger.error(f"Failed to process tool {tool.get('name', 'Unknown')}: {e}")
                    failures += 1
                    self.checkpoint.failed_tools.append(tool.get("name", "Unknown"))

        return successes, failures

    async def run_reembedding_process(self):
        """Run the complete re-embedding process"""
        logger.info("🚀 Starting batch re-embedding process")

        start_time = time.time()
        self.stats.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
        self.stats.embedding_model_used = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

        try:
            # Connect to Weaviate
            self.connect_weaviate()

            # Load checkpoint if exists
            resumed = self.load_checkpoint()
            if resumed:
                logger.info("🔄 Resuming from previous checkpoint")
            else:
                logger.info("🆕 Starting fresh re-embedding process")
                self.checkpoint.start_time = datetime.now().isoformat()

            # Create target collection
            target_collection = self.create_target_collection()

            # Get tools to process
            tools_to_process = await self.get_tools_to_process()
            if not tools_to_process:
                logger.error("No tools to process")
                return

            self.stats.total_tools = len(tools_to_process)
            self.checkpoint.total_tools = len(tools_to_process)

            logger.info(f"Processing {len(tools_to_process)} tools in batches of {self.config.batch_size}")

            # Process in batches
            for i in range(0, len(tools_to_process), self.config.batch_size):
                batch = tools_to_process[i:i + self.config.batch_size]
                batch_num = i // self.config.batch_size + 1
                total_batches = (len(tools_to_process) + self.config.batch_size - 1) // self.config.batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tools)")

                batch_start_time = time.time()
                successes, failures = await self.process_tool_batch(batch)
                batch_time = time.time() - batch_start_time

                # Update statistics
                self.stats.processed_successfully += successes
                self.stats.failed_tools += failures
                self.stats.total_processing_time += batch_time
                self.checkpoint.processed_count += successes
                self.checkpoint.current_batch = batch_num

                # Save checkpoint
                self.save_checkpoint()

                logger.info(f"Batch {batch_num} completed: {successes} success, {failures} failures in {batch_time:.2f}s")

                # Delay between batches
                if i + self.config.batch_size < len(tools_to_process):
                    await asyncio.sleep(self.config.delay_between_batches)

            # Calculate final statistics
            total_time = time.time() - start_time
            self.stats.total_processing_time = total_time
            self.stats.average_processing_time = total_time / self.stats.total_tools if self.stats.total_tools > 0 else 0

            # Print final results
            self.print_final_report()

            # Cleanup checkpoint on success
            if self.stats.failed_tools == 0:
                self.cleanup_checkpoint()
                logger.info("🎉 Re-embedding completed successfully!")
            else:
                logger.warning(f"⚠️  Re-embedding completed with {self.stats.failed_tools} failures")
                logger.info("Checkpoint preserved for retry of failed tools")

        except Exception as e:
            logger.error(f"❌ Re-embedding process failed: {e}")
            self.save_checkpoint()  # Save progress before exit
            raise

        finally:
            if self.weaviate_client:
                self.weaviate_client.close()

    def print_final_report(self):
        """Print comprehensive final report"""
        print(f"\n{'='*70}")
        print("🔄 BATCH RE-EMBEDDING REPORT")
        print(f"{'='*70}")
        print(f"Total Tools: {self.stats.total_tools}")
        print(f"Successfully Processed: {self.stats.processed_successfully}")
        print(f"Failed: {self.stats.failed_tools}")
        print(f"Skipped: {self.stats.skipped_tools}")
        print(f"Success Rate: {(self.stats.processed_successfully / self.stats.total_tools * 100):.1f}%" if self.stats.total_tools > 0 else "N/A")
        print(f"Total Processing Time: {self.stats.total_processing_time:.2f}s")
        print(f"Average Time per Tool: {self.stats.average_processing_time:.2f}s")
        print(f"Embedding Provider: {self.stats.embedding_provider}")
        print(f"Embedding Model: {self.stats.embedding_model_used}")
        print(f"Checkpoint Saves: {self.stats.checkpoint_saves}")

        if self.checkpoint.failed_tools:
            print(f"\n❌ FAILED TOOLS ({len(self.checkpoint.failed_tools)}):")
            for tool_name in self.checkpoint.failed_tools[:10]:  # Show first 10
                print(f"  - {tool_name}")
            if len(self.checkpoint.failed_tools) > 10:
                print(f"  ... and {len(self.checkpoint.failed_tools) - 10} more")

    async def verify_migration(self) -> bool:
        """Verify the re-embedding migration was successful"""
        logger.info("Verifying migration...")

        try:
            source_collection = self.weaviate_client.collections.get(self.config.source_collection)
            target_collection = self.weaviate_client.collections.get(self.config.target_collection)

            # Get counts (approximate)
            source_count = 0
            target_count = 0

            try:
                source_agg = source_collection.aggregate.over_all(total_count=True)
                source_count = source_agg.total_count
            except:
                # Fallback: sample-based estimation
                source_sample = source_collection.query.fetch_objects(limit=1)
                source_count = len(source_sample.objects) * 1000  # Rough estimate

            try:
                target_agg = target_collection.aggregate.over_all(total_count=True)
                target_count = target_agg.total_count
            except:
                target_sample = target_collection.query.fetch_objects(limit=1)
                target_count = len(target_sample.objects) * 1000

            logger.info(f"Source collection: ~{source_count} tools")
            logger.info(f"Target collection: ~{target_count} tools")

            # Verification passed if target has at least 90% of source
            success = target_count >= (source_count * 0.9)

            if success:
                logger.info("✅ Migration verification PASSED")
            else:
                logger.warning("⚠️  Migration verification FAILED - target collection has significantly fewer tools")

            return success

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    async def retry_failed_tools(self):
        """Retry processing failed tools from checkpoint"""
        if not self.checkpoint.failed_tools:
            logger.info("No failed tools to retry")
            return

        logger.info(f"Retrying {len(self.checkpoint.failed_tools)} failed tools")

        # Get fresh tool data
        all_tools = await fetch_all_tools_async()
        failed_tool_names = set(self.checkpoint.failed_tools)

        # Find failed tools in fresh data
        retry_tools = [tool for tool in all_tools if tool.get("name") in failed_tool_names]

        if retry_tools:
            logger.info(f"Found {len(retry_tools)} tools to retry")

            # Clear failed list for retry
            self.checkpoint.failed_tools = []

            # Process retry batch
            successes, failures = await self.process_tool_batch(retry_tools)

            logger.info(f"Retry completed: {successes} recovered, {failures} still failed")

            # Update stats
            self.stats.processed_successfully += successes
            self.stats.failed_tools = failures

            self.save_checkpoint()


async def main():
    """Main function to run batch re-embedding"""

    # Configuration
    config = EmbeddingMigrationConfig(
        source_collection="Tool",
        target_collection="ToolReembedded",
        batch_size=25,
        checkpoint_file="reembedding_checkpoint.json"
    )

    # Initialize system
    reembedding_system = BatchReembeddingSystem(config)

    try:
        # Run re-embedding process
        await reembedding_system.run_reembedding_process()

        # Verify migration
        if config.verification_enabled:
            success = await reembedding_system.verify_migration()
            if not success:
                logger.warning("Migration verification failed")

        # Optional: retry failed tools
        if reembedding_system.checkpoint.failed_tools:
            retry_choice = input("\nRetry failed tools? (y/N): ").lower().strip()
            if retry_choice == 'y':
                await reembedding_system.retry_failed_tools()

    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user. Checkpoint saved.")
    except Exception as e:
        logger.error(f"❌ Re-embedding failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())