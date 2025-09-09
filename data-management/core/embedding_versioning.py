"""
Embedding Versioning System for Letta Tool Selector
Manages embedding versions, tracks changes, and ensures consistency across updates.
"""

import json
import hashlib
import asyncio
import aiofiles
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import aiohttp
import numpy as np
from openai import AsyncOpenAI


@dataclass
class EmbeddingVersion:
    """Represents a version of embeddings for a tool."""
    version_id: str
    tool_id: str
    tool_name: str
    description_hash: str
    model_name: str
    embedding_dim: int
    created_at: str
    metadata: Dict[str, Any]


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding quality assessment."""
    version_id: str
    similarity_score: float
    coherence_score: float
    diversity_score: float
    outlier_detection: bool
    quality_grade: str  # A, B, C, D, F
    computed_at: str


class EmbeddingVersionManager:
    """Manages embedding versions with automatic tracking and quality assessment."""
    
    def __init__(self, storage_path: str = "/opt/stacks/lettatoolsselector/data-management/core/versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.storage_path / "embedding_versions.json"
        self.metrics_file = self.storage_path / "embedding_metrics.json"
        self.embeddings_dir = self.storage_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.openai_client = None
        
    async def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if not self.openai_client:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        return self.openai_client
    
    def _generate_version_id(self, tool_id: str, description: str, model_name: str) -> str:
        """Generate unique version ID based on content and model."""
        content = f"{tool_id}:{description}:{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_description(self, description: str) -> str:
        """Generate hash of tool description."""
        return hashlib.md5(description.encode()).hexdigest()
    
    async def _load_versions(self) -> Dict[str, EmbeddingVersion]:
        """Load embedding versions from storage."""
        if not self.versions_file.exists():
            return {}
        
        try:
            async with aiofiles.open(self.versions_file, 'r') as f:
                data = json.loads(await f.read())
                return {
                    v_id: EmbeddingVersion(**v_data) 
                    for v_id, v_data in data.items()
                }
        except Exception as e:
            self.logger.error(f"Error loading versions: {e}")
            return {}
    
    async def _save_versions(self, versions: Dict[str, EmbeddingVersion]) -> None:
        """Save embedding versions to storage."""
        try:
            data = {v_id: asdict(version) for v_id, version in versions.items()}
            async with aiofiles.open(self.versions_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving versions: {e}")
            raise
    
    async def _load_metrics(self) -> Dict[str, EmbeddingMetrics]:
        """Load embedding metrics from storage."""
        if not self.metrics_file.exists():
            return {}
        
        try:
            async with aiofiles.open(self.metrics_file, 'r') as f:
                data = json.loads(await f.read())
                return {
                    v_id: EmbeddingMetrics(**m_data)
                    for v_id, m_data in data.items()
                }
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
            return {}
    
    async def _save_metrics(self, metrics: Dict[str, EmbeddingMetrics]) -> None:
        """Save embedding metrics to storage."""
        try:
            data = {v_id: asdict(metric) for v_id, metric in metrics.items()}
            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            raise
    
    async def _save_embedding(self, version_id: str, embedding: List[float]) -> None:
        """Save embedding vector to disk."""
        embedding_file = self.embeddings_dir / f"{version_id}.npy"
        try:
            np.save(embedding_file, np.array(embedding))
        except Exception as e:
            self.logger.error(f"Error saving embedding {version_id}: {e}")
            raise
    
    async def _load_embedding(self, version_id: str) -> Optional[np.ndarray]:
        """Load embedding vector from disk."""
        embedding_file = self.embeddings_dir / f"{version_id}.npy"
        if not embedding_file.exists():
            return None
        
        try:
            return np.load(embedding_file)
        except Exception as e:
            self.logger.error(f"Error loading embedding {version_id}: {e}")
            return None
    
    async def create_embedding_version(
        self, 
        tool_id: str, 
        tool_name: str, 
        description: str,
        model_name: str = "text-embedding-3-small",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingVersion:
        """Create a new embedding version for a tool."""
        
        # Generate embedding
        client = await self._get_openai_client()
        try:
            response = await client.embeddings.create(
                input=description,
                model=model_name
            )
            embedding = response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error creating embedding for tool {tool_id}: {e}")
            raise
        
        # Create version
        version_id = self._generate_version_id(tool_id, description, model_name)
        description_hash = self._hash_description(description)
        
        version = EmbeddingVersion(
            version_id=version_id,
            tool_id=tool_id,
            tool_name=tool_name,
            description_hash=description_hash,
            model_name=model_name,
            embedding_dim=len(embedding),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        
        # Save embedding and version
        await self._save_embedding(version_id, embedding)
        
        versions = await self._load_versions()
        versions[version_id] = version
        await self._save_versions(versions)
        
        self.logger.info(f"Created embedding version {version_id} for tool {tool_name}")
        return version
    
    async def get_version(self, version_id: str) -> Optional[EmbeddingVersion]:
        """Get a specific embedding version."""
        versions = await self._load_versions()
        return versions.get(version_id)
    
    async def get_tool_versions(self, tool_id: str) -> List[EmbeddingVersion]:
        """Get all versions for a specific tool."""
        versions = await self._load_versions()
        return [v for v in versions.values() if v.tool_id == tool_id]
    
    async def get_latest_version(self, tool_id: str) -> Optional[EmbeddingVersion]:
        """Get the latest version for a tool."""
        tool_versions = await self.get_tool_versions(tool_id)
        if not tool_versions:
            return None
        return max(tool_versions, key=lambda v: v.created_at)
    
    async def has_description_changed(self, tool_id: str, description: str) -> bool:
        """Check if tool description has changed since last version."""
        latest_version = await self.get_latest_version(tool_id)
        if not latest_version:
            return True
        
        current_hash = self._hash_description(description)
        return current_hash != latest_version.description_hash
    
    async def compute_embedding_metrics(self, version_id: str) -> Optional[EmbeddingMetrics]:
        """Compute quality metrics for an embedding version."""
        version = await self.get_version(version_id)
        if not version:
            return None
        
        embedding = await self._load_embedding(version_id)
        if embedding is None:
            return None
        
        # Load other embeddings for comparison
        versions = await self._load_versions()
        comparison_embeddings = []
        
        for v_id, v in versions.items():
            if v_id != version_id and v.tool_id != version.tool_id:
                other_emb = await self._load_embedding(v_id)
                if other_emb is not None:
                    comparison_embeddings.append(other_emb)
        
        # Compute metrics
        similarity_scores = []
        if comparison_embeddings:
            for other_emb in comparison_embeddings[:50]:  # Sample for efficiency
                similarity = np.dot(embedding, other_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_emb)
                )
                similarity_scores.append(similarity)
        
        # Quality assessment
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        coherence_score = float(np.linalg.norm(embedding))  # Vector magnitude
        diversity_score = 1.0 - avg_similarity  # How different from others
        outlier_detection = coherence_score < 0.1 or coherence_score > 2.0
        
        # Grade assignment
        if outlier_detection:
            quality_grade = "F"
        elif diversity_score > 0.8 and coherence_score > 0.5:
            quality_grade = "A"
        elif diversity_score > 0.6 and coherence_score > 0.3:
            quality_grade = "B"
        elif diversity_score > 0.4 and coherence_score > 0.2:
            quality_grade = "C"
        else:
            quality_grade = "D"
        
        metrics = EmbeddingMetrics(
            version_id=version_id,
            similarity_score=avg_similarity,
            coherence_score=coherence_score,
            diversity_score=diversity_score,
            outlier_detection=outlier_detection,
            quality_grade=quality_grade,
            computed_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Save metrics
        all_metrics = await self._load_metrics()
        all_metrics[version_id] = metrics
        await self._save_metrics(all_metrics)
        
        return metrics
    
    async def get_version_metrics(self, version_id: str) -> Optional[EmbeddingMetrics]:
        """Get metrics for a specific version."""
        all_metrics = await self._load_metrics()
        return all_metrics.get(version_id)
    
    async def cleanup_old_versions(self, keep_versions: int = 5) -> None:
        """Clean up old versions, keeping only the most recent ones per tool."""
        versions = await self._load_versions()
        metrics = await self._load_metrics()
        
        # Group by tool_id
        tools_versions = {}
        for version in versions.values():
            if version.tool_id not in tools_versions:
                tools_versions[version.tool_id] = []
            tools_versions[version.tool_id].append(version)
        
        # Keep only recent versions
        versions_to_delete = []
        for tool_versions in tools_versions.values():
            if len(tool_versions) > keep_versions:
                sorted_versions = sorted(tool_versions, key=lambda v: v.created_at, reverse=True)
                versions_to_delete.extend(sorted_versions[keep_versions:])
        
        # Delete old versions
        for version in versions_to_delete:
            version_id = version.version_id
            
            # Remove from dictionaries
            if version_id in versions:
                del versions[version_id]
            if version_id in metrics:
                del metrics[version_id]
            
            # Remove embedding file
            embedding_file = self.embeddings_dir / f"{version_id}.npy"
            if embedding_file.exists():
                embedding_file.unlink()
        
        # Save updated data
        await self._save_versions(versions)
        await self._save_metrics(metrics)
        
        self.logger.info(f"Cleaned up {len(versions_to_delete)} old versions")
    
    async def get_version_summary(self) -> Dict[str, Any]:
        """Get summary statistics about embedding versions."""
        versions = await self._load_versions()
        metrics = await self._load_metrics()
        
        # Group by tool
        tools_count = len(set(v.tool_id for v in versions.values()))
        total_versions = len(versions)
        
        # Model distribution
        model_counts = {}
        for version in versions.values():
            model = version.model_name
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Quality distribution
        quality_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for metric in metrics.values():
            quality_counts[metric.quality_grade] += 1
        
        return {
            "total_tools": tools_count,
            "total_versions": total_versions,
            "avg_versions_per_tool": total_versions / max(tools_count, 1),
            "model_distribution": model_counts,
            "quality_distribution": quality_counts,
            "storage_size_mb": sum(
                f.stat().st_size for f in self.embeddings_dir.glob("*.npy")
            ) / (1024 * 1024),
            "last_updated": max(
                (v.created_at for v in versions.values()), 
                default="Never"
            )
        }


async def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)
    
    manager = EmbeddingVersionManager()
    
    # Example: Create embedding version
    version = await manager.create_embedding_version(
        tool_id="test-tool-1",
        tool_name="Test Tool",
        description="A test tool for demonstration purposes",
        metadata={"source": "test", "category": "demo"}
    )
    
    print(f"Created version: {version.version_id}")
    
    # Compute metrics
    metrics = await manager.compute_embedding_metrics(version.version_id)
    if metrics:
        print(f"Quality grade: {metrics.quality_grade}")
        print(f"Diversity score: {metrics.diversity_score:.3f}")
    
    # Get summary
    summary = await manager.get_version_summary()
    print("Version Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())