#!/usr/bin/env python3
"""
Re-embedding Manager

Command-line utility for managing batch re-embedding operations with resume capability.
Provides easy access to start, resume, verify, and manage embedding migrations.

Usage:
    python reembedding_manager.py start [--batch-size 25] [--config config.json]
    python reembedding_manager.py resume
    python reembedding_manager.py status
    python reembedding_manager.py verify
    python reembedding_manager.py cleanup
    python reembedding_manager.py retry-failed
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

from batch_reembedding_system import BatchReembeddingSystem, EmbeddingMigrationConfig, ProcessingCheckpoint

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

def create_config_from_args(args) -> EmbeddingMigrationConfig:
    """Create configuration from command line arguments"""
    config_data = {}

    # Load from config file if provided
    if hasattr(args, 'config') and args.config:
        config_data = load_config_file(args.config)

    # Override with command line arguments
    if hasattr(args, 'batch_size') and args.batch_size:
        config_data['batch_size'] = args.batch_size

    if hasattr(args, 'source_collection') and args.source_collection:
        config_data['source_collection'] = args.source_collection

    if hasattr(args, 'target_collection') and args.target_collection:
        config_data['target_collection'] = args.target_collection

    return EmbeddingMigrationConfig(**config_data)

def show_checkpoint_status(checkpoint_file: str = "reembedding_checkpoint.json"):
    """Display current checkpoint status"""
    checkpoint_path = Path(checkpoint_file)

    if not checkpoint_path.exists():
        print("📝 No active re-embedding process found")
        print("   Use 'python reembedding_manager.py start' to begin")
        return

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        checkpoint = data.get('checkpoint', {})
        stats = data.get('stats', {})

        print("🔄 ACTIVE RE-EMBEDDING STATUS")
        print("=" * 50)
        print(f"Started: {checkpoint.get('start_time', 'Unknown')}")
        print(f"Progress: {checkpoint.get('processed_count', 0)}/{checkpoint.get('total_tools', '?')} tools")

        if checkpoint.get('total_tools', 0) > 0:
            progress_pct = (checkpoint.get('processed_count', 0) / checkpoint.get('total_tools', 1)) * 100
            print(f"Completion: {progress_pct:.1f}%")

        print(f"Current Batch: {checkpoint.get('current_batch', 0)}")
        print(f"Last Processed: {checkpoint.get('last_processed_tool', 'None')}")

        failed_count = len(checkpoint.get('failed_tools', []))
        if failed_count > 0:
            print(f"⚠️  Failed Tools: {failed_count}")

        print(f"Successful: {stats.get('processed_successfully', 0)}")
        print(f"Failed: {stats.get('failed_tools', 0)}")
        print(f"Skipped: {stats.get('skipped_tools', 0)}")

        if stats.get('total_processing_time', 0) > 0:
            print(f"Processing Time: {stats.get('total_processing_time', 0):.2f}s")
            print(f"Avg Time/Tool: {stats.get('average_processing_time', 0):.3f}s")

        print(f"Embedding Model: {stats.get('embedding_model_used', 'Unknown')}")
        print(f"Provider: {stats.get('embedding_provider', 'Unknown')}")

    except Exception as e:
        print(f"Error reading checkpoint: {e}")

async def start_reembedding(args):
    """Start new re-embedding process"""
    config = create_config_from_args(args)
    system = BatchReembeddingSystem(config)

    print(f"🚀 Starting batch re-embedding process")
    print(f"   Source: {config.source_collection}")
    print(f"   Target: {config.target_collection}")
    print(f"   Batch size: {config.batch_size}")

    try:
        await system.run_reembedding_process()
        print("✅ Re-embedding completed successfully!")
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted. Progress saved to checkpoint.")
    except Exception as e:
        print(f"❌ Re-embedding failed: {e}")
        return 1

    return 0

async def resume_reembedding(args):
    """Resume interrupted re-embedding process"""
    config = create_config_from_args(args)
    system = BatchReembeddingSystem(config)

    if not Path(config.checkpoint_file).exists():
        print("❌ No checkpoint found. Cannot resume.")
        print("   Use 'start' command to begin a new process.")
        return 1

    print("🔄 Resuming re-embedding process...")

    try:
        await system.run_reembedding_process()
        print("✅ Re-embedding resumed and completed!")
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted again. Progress saved.")
    except Exception as e:
        print(f"❌ Resume failed: {e}")
        return 1

    return 0

async def verify_migration(args):
    """Verify migration was successful"""
    config = create_config_from_args(args)
    system = BatchReembeddingSystem(config)

    print("🔍 Verifying migration...")

    try:
        system.connect_weaviate()
        success = await system.verify_migration()

        if success:
            print("✅ Migration verification PASSED")
            return 0
        else:
            print("❌ Migration verification FAILED")
            return 1

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1

async def retry_failed_tools(args):
    """Retry failed tools from checkpoint"""
    config = create_config_from_args(args)
    system = BatchReembeddingSystem(config)

    if not system.load_checkpoint():
        print("❌ No checkpoint found with failed tools.")
        return 1

    if not system.checkpoint.failed_tools:
        print("✅ No failed tools to retry.")
        return 0

    print(f"🔄 Retrying {len(system.checkpoint.failed_tools)} failed tools...")

    try:
        system.connect_weaviate()
        await system.retry_failed_tools()
        print("✅ Failed tools retry completed!")
    except Exception as e:
        print(f"❌ Retry failed: {e}")
        return 1

    return 0

def cleanup_checkpoint(args):
    """Clean up checkpoint file"""
    config = create_config_from_args(args)
    checkpoint_path = Path(config.checkpoint_file)

    if not checkpoint_path.exists():
        print("📝 No checkpoint file to clean up.")
        return 0

    try:
        # Show current status first
        print("Current checkpoint status:")
        show_checkpoint_status(config.checkpoint_file)

        # Confirm cleanup
        confirm = input(f"\nDelete checkpoint file '{config.checkpoint_file}'? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Cleanup cancelled.")
            return 0

        checkpoint_path.unlink()
        print("✅ Checkpoint file deleted.")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1

    return 0

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "source_collection": "Tool",
        "target_collection": "ToolReembedded",
        "checkpoint_file": "reembedding_checkpoint.json",
        "batch_size": 25,
        "max_retries": 3,
        "delay_between_batches": 1.0,
        "backup_enabled": True,
        "verification_enabled": True
    }

    config_file = "reembedding_config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"✅ Sample configuration created: {config_file}")
    print("   Edit this file and use --config flag to customize settings.")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="Re-embedding Manager - Manage batch tool re-embedding operations"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start new re-embedding process')
    start_parser.add_argument('--batch-size', type=int, default=25, help='Batch size for processing')
    start_parser.add_argument('--source-collection', help='Source Weaviate collection name')
    start_parser.add_argument('--target-collection', help='Target Weaviate collection name')
    start_parser.add_argument('--config', help='Configuration file path')

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume interrupted process')
    resume_parser.add_argument('--config', help='Configuration file path')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show current process status')
    status_parser.add_argument('--checkpoint-file', default='reembedding_checkpoint.json',
                              help='Checkpoint file to check')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify migration success')
    verify_parser.add_argument('--config', help='Configuration file path')

    # Retry command
    retry_parser = subparsers.add_parser('retry-failed', help='Retry failed tools')
    retry_parser.add_argument('--config', help='Configuration file path')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up checkpoint file')
    cleanup_parser.add_argument('--config', help='Configuration file path')

    # Config command
    config_parser = subparsers.add_parser('create-config', help='Create sample configuration file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Handle commands
    if args.command == 'status':
        show_checkpoint_status(args.checkpoint_file)
        return 0
    elif args.command == 'create-config':
        create_sample_config()
        return 0
    elif args.command == 'cleanup':
        return cleanup_checkpoint(args)
    elif args.command == 'start':
        return asyncio.run(start_reembedding(args))
    elif args.command == 'resume':
        return asyncio.run(resume_reembedding(args))
    elif args.command == 'verify':
        return asyncio.run(verify_migration(args))
    elif args.command == 'retry-failed':
        return asyncio.run(retry_failed_tools(args))
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())