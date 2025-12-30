#!/usr/bin/env python3
"""
Embedding Versioning System with Migration Tracking

This system provides comprehensive version management for embedding models and tracks
migration history across different embedding providers and model versions.

Key Features:
- Embedding model version tracking
- Migration history and audit trails
- Performance comparison across versions
- Rollback capabilities with version management
- Automated compatibility checking
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import weaviate
from weaviate.collections import Collection
import weaviate.classes.query as wq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MigrationStatus(Enum):
    """Migration status tracking"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

@dataclass
class EmbeddingVersion:
    """Represents an embedding model version"""
    provider: str
    model: str
    version: str
    dimensions: int
    created_at: str
    description: str = ""
    performance_metrics: Dict[str, float] = None
    compatibility_hash: str = ""

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if not self.compatibility_hash:
            self.compatibility_hash = self._generate_compatibility_hash()

    def _generate_compatibility_hash(self) -> str:
        """Generate compatibility hash for version comparison"""
        content = f"{self.provider}:{self.model}:{self.dimensions}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class MigrationRecord:
    """Represents a migration operation"""
    migration_id: str
    from_version: EmbeddingVersion
    to_version: EmbeddingVersion
    status: MigrationStatus
    started_at: str
    completed_at: Optional[str] = None
    tools_migrated: int = 0
    tools_failed: int = 0
    performance_improvement: Dict[str, float] = None
    rollback_info: Dict[str, Any] = None
    notes: str = ""

    def __post_init__(self):
        if self.performance_improvement is None:
            self.performance_improvement = {}
        if self.rollback_info is None:
            self.rollback_info = {}

class EmbeddingVersionManager:
    """Manages embedding versions and migration tracking"""

    def __init__(self, data_dir: str = "embedding_versions"):
        load_dotenv()

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.versions_file = self.data_dir / "embedding_versions.json"
        self.migrations_file = self.data_dir / "migration_history.json"

        self.weaviate_client = None
        self.versions: Dict[str, EmbeddingVersion] = {}
        self.migrations: Dict[str, MigrationRecord] = {}

        self._load_versions()
        self._load_migrations()

    def connect_weaviate(self):
        """Connect to Weaviate instance"""
        logger.info("Connecting to Weaviate for version management...")
        self.weaviate_client = weaviate.connect_to_custom(
            http_host="192.168.50.90",
            http_port=8080,
            http_secure=False,
            grpc_host="192.168.50.90",
            grpc_port=50051,
            grpc_secure=False,
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
            },
            skip_init_checks=True
        )

    def _load_versions(self):
        """Load embedding versions from file"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)

                self.versions = {
                    version_id: EmbeddingVersion(**version_data)
                    for version_id, version_data in data.items()
                }
                logger.info(f"Loaded {len(self.versions)} embedding versions")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")

    def _load_migrations(self):
        """Load migration history from file"""
        if self.migrations_file.exists():
            try:
                with open(self.migrations_file, 'r') as f:
                    data = json.load(f)

                self.migrations = {}
                for migration_id, migration_data in data.items():
                    # Convert nested dictionaries back to objects
                    migration_data['from_version'] = EmbeddingVersion(**migration_data['from_version'])
                    migration_data['to_version'] = EmbeddingVersion(**migration_data['to_version'])
                    migration_data['status'] = MigrationStatus(migration_data['status'])

                    self.migrations[migration_id] = MigrationRecord(**migration_data)

                logger.info(f"Loaded {len(self.migrations)} migration records")
            except Exception as e:
                logger.error(f"Failed to load migrations: {e}")

    def _save_versions(self):
        """Save embedding versions to file"""
        try:
            data = {
                version_id: asdict(version)
                for version_id, version in self.versions.items()
            }

            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save versions: {e}")

    def _save_migrations(self):
        """Save migration history to file"""
        try:
            data = {}
            for migration_id, migration in self.migrations.items():
                migration_dict = asdict(migration)
                migration_dict['status'] = migration.status.value
                data[migration_id] = migration_dict

            with open(self.migrations_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save migrations: {e}")

    def register_version(self,
                        provider: str,
                        model: str,
                        version: str,
                        dimensions: int,
                        description: str = "",
                        performance_metrics: Dict[str, float] = None) -> str:
        """Register a new embedding version"""

        version_id = f"{provider}_{model}_{version}".replace(":", "_").replace("/", "_")

        embedding_version = EmbeddingVersion(
            provider=provider,
            model=model,
            version=version,
            dimensions=dimensions,
            created_at=datetime.now().isoformat(),
            description=description,
            performance_metrics=performance_metrics or {}
        )

        self.versions[version_id] = embedding_version
        self._save_versions()

        logger.info(f"Registered embedding version: {version_id}")
        return version_id

    def get_version(self, version_id: str) -> Optional[EmbeddingVersion]:
        """Get specific embedding version"""
        return self.versions.get(version_id)

    def list_versions(self, provider: str = None) -> List[Tuple[str, EmbeddingVersion]]:
        """List all versions, optionally filtered by provider"""
        versions = []
        for version_id, version in self.versions.items():
            if provider is None or version.provider == provider:
                versions.append((version_id, version))

        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x[1].created_at, reverse=True)
        return versions

    def get_current_version_from_env(self) -> Optional[Tuple[str, EmbeddingVersion]]:
        """Detect current embedding version from environment"""
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

        if provider == "openai":
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            dimensions = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        elif provider == "ollama":
            model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding-4b")
            dimensions = int(os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "768"))
        else:
            logger.warning(f"Unknown embedding provider: {provider}")
            return None

        # Try to find exact match
        for version_id, version in self.versions.items():
            if (version.provider == provider and
                version.model == model and
                version.dimensions == dimensions):
                return version_id, version

        # Auto-register if not found
        version_id = self.register_version(
            provider=provider,
            model=model,
            version="auto-detected",
            dimensions=dimensions,
            description=f"Auto-detected from environment"
        )

        return version_id, self.versions[version_id]

    def start_migration(self,
                       from_version_id: str,
                       to_version_id: str,
                       notes: str = "") -> str:
        """Start a new migration between versions"""

        from_version = self.get_version(from_version_id)
        to_version = self.get_version(to_version_id)

        if not from_version or not to_version:
            raise ValueError("Invalid version IDs provided")

        migration_id = f"migration_{int(time.time())}"

        migration = MigrationRecord(
            migration_id=migration_id,
            from_version=from_version,
            to_version=to_version,
            status=MigrationStatus.PLANNED,
            started_at=datetime.now().isoformat(),
            notes=notes
        )

        self.migrations[migration_id] = migration
        self._save_migrations()

        logger.info(f"Started migration {migration_id}: {from_version_id} -> {to_version_id}")
        return migration_id

    def update_migration_status(self,
                               migration_id: str,
                               status: MigrationStatus,
                               tools_migrated: int = 0,
                               tools_failed: int = 0,
                               performance_improvement: Dict[str, float] = None):
        """Update migration progress and status"""

        if migration_id not in self.migrations:
            raise ValueError(f"Migration {migration_id} not found")

        migration = self.migrations[migration_id]
        migration.status = status
        migration.tools_migrated = tools_migrated
        migration.tools_failed = tools_failed

        if performance_improvement:
            migration.performance_improvement.update(performance_improvement)

        if status in [MigrationStatus.COMPLETED, MigrationStatus.FAILED, MigrationStatus.ROLLED_BACK]:
            migration.completed_at = datetime.now().isoformat()

        self._save_migrations()
        logger.info(f"Updated migration {migration_id}: status={status.value}")

    def get_migration(self, migration_id: str) -> Optional[MigrationRecord]:
        """Get specific migration record"""
        return self.migrations.get(migration_id)

    def list_migrations(self, status: MigrationStatus = None) -> List[Tuple[str, MigrationRecord]]:
        """List migrations, optionally filtered by status"""
        migrations = []
        for migration_id, migration in self.migrations.items():
            if status is None or migration.status == status:
                migrations.append((migration_id, migration))

        # Sort by start time (newest first)
        migrations.sort(key=lambda x: x[1].started_at, reverse=True)
        return migrations

    async def analyze_collection_version(self, collection_name: str) -> Dict[str, Any]:
        """Analyze the embedding version info in a collection"""
        if not self.weaviate_client:
            self.connect_weaviate()

        try:
            collection = self.weaviate_client.collections.get(collection_name)

            # Sample tools to analyze versions
            results = collection.query.fetch_objects(
                limit=100,
                return_properties=["migration_timestamp", "embedding_model_version", "name"]
            )

            version_analysis = {
                "total_tools": len(results.objects),
                "version_distribution": {},
                "migration_timestamps": [],
                "collection_name": collection_name
            }

            for obj in results.objects:
                props = obj.properties
                version = props.get("embedding_model_version", "unknown")
                timestamp = props.get("migration_timestamp", "unknown")

                # Track version distribution
                if version not in version_analysis["version_distribution"]:
                    version_analysis["version_distribution"][version] = 0
                version_analysis["version_distribution"][version] += 1

                # Track migration timestamps
                if timestamp != "unknown":
                    version_analysis["migration_timestamps"].append(timestamp)

            # Calculate statistics
            if version_analysis["migration_timestamps"]:
                timestamps = sorted(version_analysis["migration_timestamps"])
                version_analysis["oldest_migration"] = timestamps[0]
                version_analysis["newest_migration"] = timestamps[-1]

            return version_analysis

        except Exception as e:
            logger.error(f"Failed to analyze collection {collection_name}: {e}")
            return {"error": str(e), "collection_name": collection_name}

    def check_version_compatibility(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Check compatibility between two embedding versions"""
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)

        if not version1 or not version2:
            return {"compatible": False, "reason": "Invalid version IDs"}

        compatibility = {
            "compatible": False,
            "dimension_match": version1.dimensions == version2.dimensions,
            "provider_match": version1.provider == version2.provider,
            "compatibility_hash_match": version1.compatibility_hash == version2.compatibility_hash,
            "migration_recommended": False,
            "warnings": []
        }

        # Check dimension compatibility
        if not compatibility["dimension_match"]:
            compatibility["warnings"].append(
                f"Dimension mismatch: {version1.dimensions} vs {version2.dimensions}"
            )

        # Check provider compatibility
        if not compatibility["provider_match"]:
            compatibility["warnings"].append(
                f"Provider change: {version1.provider} -> {version2.provider}"
            )

        # Overall compatibility
        compatibility["compatible"] = (
            compatibility["dimension_match"] and
            len(compatibility["warnings"]) == 0
        )

        # Migration recommendation
        if compatibility["compatible"]:
            # Check performance metrics if available
            v1_perf = version1.performance_metrics.get("search_accuracy", 0)
            v2_perf = version2.performance_metrics.get("search_accuracy", 0)

            if v2_perf > v1_perf:
                compatibility["migration_recommended"] = True

        return compatibility

    def generate_migration_plan(self,
                              from_version_id: str,
                              to_version_id: str,
                              estimated_tools: int = None) -> Dict[str, Any]:
        """Generate a comprehensive migration plan"""

        from_version = self.get_version(from_version_id)
        to_version = self.get_version(to_version_id)

        if not from_version or not to_version:
            raise ValueError("Invalid version IDs")

        compatibility = self.check_version_compatibility(from_version_id, to_version_id)

        # Estimate migration time (assume 2-5 seconds per tool)
        if estimated_tools:
            min_time = estimated_tools * 2 / 60  # minutes
            max_time = estimated_tools * 5 / 60  # minutes
        else:
            min_time = max_time = "Unknown"

        plan = {
            "migration_id": f"planned_{int(time.time())}",
            "from_version": asdict(from_version),
            "to_version": asdict(to_version),
            "compatibility": compatibility,
            "estimated_tools": estimated_tools,
            "estimated_time_range": f"{min_time}-{max_time} minutes" if estimated_tools else "Unknown",
            "prerequisites": [],
            "risks": [],
            "rollback_strategy": {},
            "validation_steps": []
        }

        # Add prerequisites
        plan["prerequisites"] = [
            "Backup current collection",
            "Verify new embedding model accessibility",
            "Test with small batch first",
            "Monitor system resources"
        ]

        # Add risks based on compatibility
        if not compatibility["compatible"]:
            plan["risks"].extend([
                "Dimension mismatch may cause query failures",
                "Provider change requires configuration updates",
                "Search quality may change significantly"
            ])

        if compatibility["provider_match"]:
            plan["risks"].append("Minor search quality variations expected")
        else:
            plan["risks"].append("Major search behavior changes expected")

        # Rollback strategy
        plan["rollback_strategy"] = {
            "method": "dual_index",
            "steps": [
                "Keep original collection as backup",
                "Test new collection thoroughly",
                "Atomic switch when verified",
                "Immediate rollback capability maintained"
            ]
        }

        # Validation steps
        plan["validation_steps"] = [
            "Verify tool count matches",
            "Test sample queries for relevance",
            "Compare search performance metrics",
            "Validate embedding dimensions",
            "Check migration timestamps"
        ]

        return plan

    def print_version_summary(self):
        """Print summary of all registered versions"""
        print(f"\n{'='*70}")
        print("EMBEDDING VERSIONS SUMMARY")
        print(f"{'='*70}")

        if not self.versions:
            print("No embedding versions registered.")
            return

        for version_id, version in self.versions.items():
            print(f"\n📦 {version_id}")
            print(f"   Provider: {version.provider}")
            print(f"   Model: {version.model}")
            print(f"   Version: {version.version}")
            print(f"   Dimensions: {version.dimensions}")
            print(f"   Created: {version.created_at}")
            if version.description:
                print(f"   Description: {version.description}")
            if version.performance_metrics:
                print(f"   Performance: {version.performance_metrics}")

    def print_migration_summary(self):
        """Print summary of migration history"""
        print(f"\n{'='*70}")
        print("MIGRATION HISTORY")
        print(f"{'='*70}")

        if not self.migrations:
            print("No migrations recorded.")
            return

        for migration_id, migration in self.migrations.items():
            print(f"\n🔄 {migration_id}")
            print(f"   Status: {migration.status.value}")
            print(f"   From: {migration.from_version.provider}/{migration.from_version.model}")
            print(f"   To: {migration.to_version.provider}/{migration.to_version.model}")
            print(f"   Started: {migration.started_at}")
            if migration.completed_at:
                print(f"   Completed: {migration.completed_at}")
            if migration.tools_migrated:
                print(f"   Tools Migrated: {migration.tools_migrated}")
            if migration.tools_failed:
                print(f"   Tools Failed: {migration.tools_failed}")
            if migration.performance_improvement:
                print(f"   Performance Improvement: {migration.performance_improvement}")


async def main():
    """Main function for testing version management"""
    manager = EmbeddingVersionManager()

    # Register some example versions
    openai_v1 = manager.register_version(
        provider="openai",
        model="text-embedding-3-small",
        version="1.0",
        dimensions=1536,
        description="OpenAI text-embedding-3-small model",
        performance_metrics={"search_accuracy": 0.85, "speed": 0.2}
    )

    ollama_v1 = manager.register_version(
        provider="ollama",
        model="qwen3-embedding-4b",
        version="Q4_K_M",
        dimensions=768,
        description="Ollama Qwen3 Embedding model, Q4_K_M quantization",
        performance_metrics={"search_accuracy": 0.82, "speed": 0.8}
    )

    # Show version summary
    manager.print_version_summary()

    # Check current environment version
    current = manager.get_current_version_from_env()
    if current:
        print(f"\n🔍 Current Environment Version: {current[0]}")

    # Generate migration plan
    try:
        plan = manager.generate_migration_plan(openai_v1, ollama_v1, estimated_tools=1500)
        print(f"\n📋 Migration Plan:")
        print(f"   Compatibility: {'✅' if plan['compatibility']['compatible'] else '❌'}")
        print(f"   Estimated Time: {plan['estimated_time_range']}")
        print(f"   Risks: {len(plan['risks'])} identified")
    except Exception as e:
        print(f"Migration plan generation failed: {e}")

    manager.print_migration_summary()


if __name__ == "__main__":
    asyncio.run(main())