#!/usr/bin/env python3
"""
Integrated Migration System

Combines the batch re-embedding system with embedding version management
to provide a complete, tracked migration workflow with versioning.

This system:
- Automatically tracks embedding versions during migrations
- Updates migration status in version history
- Provides rollback capabilities with version tracking
- Integrates with existing checkpoint/resume functionality
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

from batch_reembedding_system import BatchReembeddingSystem, EmbeddingMigrationConfig
from embedding_version_manager import EmbeddingVersionManager, MigrationStatus

logger = logging.getLogger(__name__)

class IntegratedMigrationSystem:
    """Complete migration system with version tracking"""

    def __init__(self,
                 migration_config: EmbeddingMigrationConfig = None,
                 version_manager: EmbeddingVersionManager = None):

        self.migration_config = migration_config or EmbeddingMigrationConfig()
        self.version_manager = version_manager or EmbeddingVersionManager()
        self.reembedding_system = BatchReembeddingSystem(self.migration_config)

        self.current_migration_id: Optional[str] = None

    def detect_current_and_target_versions(self) -> Tuple[Optional[str], Optional[str]]:
        """Detect current and target embedding versions from environment"""

        # Get current version from environment
        current_version_info = self.version_manager.get_current_version_from_env()
        current_version_id = current_version_info[0] if current_version_info else None

        # For target version, we need to determine the new configuration
        # This would typically be set via environment variables for the new model
        target_provider = os.getenv("TARGET_EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "ollama"))

        if target_provider == "openai":
            target_model = os.getenv("TARGET_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            target_dimensions = int(os.getenv("TARGET_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        elif target_provider == "ollama":
            target_model = os.getenv("TARGET_OLLAMA_EMBEDDING_MODEL", "qwen3-embedding-4b")
            target_dimensions = int(os.getenv("TARGET_OLLAMA_EMBEDDING_DIMENSIONS", "768"))
        else:
            logger.warning(f"Unknown target provider: {target_provider}")
            return current_version_id, None

        # Try to find existing target version or register it
        target_version_id = None
        for version_id, version in self.version_manager.versions.items():
            if (version.provider == target_provider and
                version.model == target_model and
                version.dimensions == target_dimensions):
                target_version_id = version_id
                break

        if not target_version_id:
            # Auto-register target version
            target_version_id = self.version_manager.register_version(
                provider=target_provider,
                model=target_model,
                version="migration-target",
                dimensions=target_dimensions,
                description=f"Target version for migration from {current_version_id}"
            )

        return current_version_id, target_version_id

    async def plan_migration(self,
                           from_version_id: str = None,
                           to_version_id: str = None,
                           estimated_tools: int = None) -> dict:
        """Generate comprehensive migration plan with version tracking"""

        # Auto-detect versions if not provided
        if not from_version_id or not to_version_id:
            detected_from, detected_to = self.detect_current_and_target_versions()
            from_version_id = from_version_id or detected_from
            to_version_id = to_version_id or detected_to

        if not from_version_id or not to_version_id:
            raise ValueError("Could not determine migration versions")

        # Generate migration plan using version manager
        plan = self.version_manager.generate_migration_plan(
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            estimated_tools=estimated_tools
        )

        # Add integration-specific information
        plan["integration_info"] = {
            "source_collection": self.migration_config.source_collection,
            "target_collection": self.migration_config.target_collection,
            "checkpoint_file": self.migration_config.checkpoint_file,
            "batch_size": self.migration_config.batch_size,
            "resume_capable": True,
            "version_tracked": True
        }

        return plan

    async def start_tracked_migration(self,
                                    from_version_id: str = None,
                                    to_version_id: str = None,
                                    migration_notes: str = "") -> str:
        """Start migration with full version tracking"""

        # Auto-detect versions if needed
        if not from_version_id or not to_version_id:
            from_version_id, to_version_id = self.detect_current_and_target_versions()

        if not from_version_id or not to_version_id:
            raise ValueError("Could not determine migration versions")

        # Start migration tracking in version manager
        migration_id = self.version_manager.start_migration(
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            notes=migration_notes
        )

        self.current_migration_id = migration_id

        # Update status to in_progress
        self.version_manager.update_migration_status(
            migration_id=migration_id,
            status=MigrationStatus.IN_PROGRESS
        )

        logger.info(f"Started tracked migration {migration_id}: {from_version_id} -> {to_version_id}")

        try:
            # Run the actual re-embedding process
            await self.reembedding_system.run_reembedding_process()

            # Update migration with success statistics
            stats = self.reembedding_system.stats
            self.version_manager.update_migration_status(
                migration_id=migration_id,
                status=MigrationStatus.COMPLETED,
                tools_migrated=stats.processed_successfully,
                tools_failed=stats.failed_tools
            )

            logger.info(f"✅ Migration {migration_id} completed successfully")
            return migration_id

        except Exception as e:
            # Update migration with failure status
            stats = self.reembedding_system.stats
            self.version_manager.update_migration_status(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                tools_migrated=stats.processed_successfully,
                tools_failed=stats.failed_tools
            )

            logger.error(f"❌ Migration {migration_id} failed: {e}")
            raise

    async def resume_tracked_migration(self) -> Optional[str]:
        """Resume migration with version tracking"""

        # Check if there's an in-progress migration
        in_progress_migrations = self.version_manager.list_migrations(
            status=MigrationStatus.IN_PROGRESS
        )

        if not in_progress_migrations:
            logger.info("No in-progress migrations found")
            return None

        # Get the most recent in-progress migration
        migration_id, migration_record = in_progress_migrations[0]
        self.current_migration_id = migration_id

        logger.info(f"Resuming tracked migration {migration_id}")

        try:
            # Resume the re-embedding process
            await self.reembedding_system.run_reembedding_process()

            # Update migration with final statistics
            stats = self.reembedding_system.stats
            self.version_manager.update_migration_status(
                migration_id=migration_id,
                status=MigrationStatus.COMPLETED,
                tools_migrated=stats.processed_successfully,
                tools_failed=stats.failed_tools
            )

            logger.info(f"✅ Resumed migration {migration_id} completed")
            return migration_id

        except Exception as e:
            # Update with failure status
            stats = self.reembedding_system.stats
            self.version_manager.update_migration_status(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                tools_migrated=stats.processed_successfully,
                tools_failed=stats.failed_tools
            )

            logger.error(f"❌ Resumed migration {migration_id} failed: {e}")
            raise

    async def verify_migration_with_tracking(self, migration_id: str = None) -> bool:
        """Verify migration and update tracking"""

        migration_id = migration_id or self.current_migration_id
        if not migration_id:
            logger.warning("No migration ID for verification tracking")

        try:
            # Run verification
            success = await self.reembedding_system.verify_migration()

            if migration_id:
                # Update migration record with verification result
                migration = self.version_manager.get_migration(migration_id)
                if migration:
                    if success:
                        # Keep as completed
                        logger.info(f"✅ Migration {migration_id} verification passed")
                    else:
                        # Mark as failed
                        self.version_manager.update_migration_status(
                            migration_id=migration_id,
                            status=MigrationStatus.FAILED
                        )
                        logger.warning(f"⚠️  Migration {migration_id} verification failed")

            return success

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            if migration_id:
                self.version_manager.update_migration_status(
                    migration_id=migration_id,
                    status=MigrationStatus.FAILED
                )
            return False

    def get_migration_status(self, migration_id: str = None) -> dict:
        """Get comprehensive migration status including version info"""

        migration_id = migration_id or self.current_migration_id
        if not migration_id:
            return {"error": "No migration ID available"}

        migration = self.version_manager.get_migration(migration_id)
        if not migration:
            return {"error": f"Migration {migration_id} not found"}

        # Get re-embedding system stats if available
        reembedding_stats = {}
        if hasattr(self.reembedding_system, 'stats'):
            stats = self.reembedding_system.stats
            reembedding_stats = {
                "processed_successfully": stats.processed_successfully,
                "failed_tools": stats.failed_tools,
                "total_processing_time": stats.total_processing_time,
                "embedding_provider": stats.embedding_provider,
                "embedding_model_used": stats.embedding_model_used
            }

        # Get checkpoint info if available
        checkpoint_info = {}
        if hasattr(self.reembedding_system, 'checkpoint'):
            checkpoint = self.reembedding_system.checkpoint
            checkpoint_info = {
                "processed_count": checkpoint.processed_count,
                "current_batch": checkpoint.current_batch,
                "total_tools": checkpoint.total_tools,
                "last_processed_tool": checkpoint.last_processed_tool,
                "failed_tools_count": len(checkpoint.failed_tools)
            }

        return {
            "migration_id": migration_id,
            "status": migration.status.value,
            "from_version": {
                "provider": migration.from_version.provider,
                "model": migration.from_version.model,
                "version": migration.from_version.version,
                "dimensions": migration.from_version.dimensions
            },
            "to_version": {
                "provider": migration.to_version.provider,
                "model": migration.to_version.model,
                "version": migration.to_version.version,
                "dimensions": migration.to_version.dimensions
            },
            "started_at": migration.started_at,
            "completed_at": migration.completed_at,
            "tools_migrated": migration.tools_migrated,
            "tools_failed": migration.tools_failed,
            "reembedding_stats": reembedding_stats,
            "checkpoint_info": checkpoint_info,
            "notes": migration.notes
        }

    def print_integration_status(self):
        """Print comprehensive status of the integrated system"""
        print(f"\n{'='*70}")
        print("INTEGRATED MIGRATION SYSTEM STATUS")
        print(f"{'='*70}")

        # Current migration
        if self.current_migration_id:
            status = self.get_migration_status()
            print(f"Current Migration: {self.current_migration_id}")
            print(f"Status: {status['status']}")
            print(f"From: {status['from_version']['provider']}/{status['from_version']['model']}")
            print(f"To: {status['to_version']['provider']}/{status['to_version']['model']}")

            if status.get('checkpoint_info'):
                checkpoint = status['checkpoint_info']
                if checkpoint['total_tools'] > 0:
                    progress = (checkpoint['processed_count'] / checkpoint['total_tools']) * 100
                    print(f"Progress: {checkpoint['processed_count']}/{checkpoint['total_tools']} ({progress:.1f}%)")
        else:
            print("No active migration")

        # Recent migrations
        recent_migrations = self.version_manager.list_migrations()[:5]
        if recent_migrations:
            print(f"\nRecent Migrations:")
            for migration_id, migration in recent_migrations:
                print(f"  {migration_id}: {migration.status.value}")

        # Available versions
        versions = self.version_manager.list_versions()
        print(f"\nAvailable Versions: {len(versions)}")

        # Current environment
        current = self.version_manager.get_current_version_from_env()
        if current:
            print(f"Environment Version: {current[0]}")


async def main():
    """Main function for testing integrated system"""
    system = IntegratedMigrationSystem()

    # Show status
    system.print_integration_status()

    # Generate migration plan
    try:
        plan = await system.plan_migration(estimated_tools=100)
        print(f"\n📋 Migration Plan Available:")
        print(f"   From: {plan['from_version']['provider']}/{plan['from_version']['model']}")
        print(f"   To: {plan['to_version']['provider']}/{plan['to_version']['model']}")
        print(f"   Compatible: {'✅' if plan['compatibility']['compatible'] else '❌'}")
        print(f"   Estimated Time: {plan['estimated_time_range']}")
    except Exception as e:
        print(f"Migration plan failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())