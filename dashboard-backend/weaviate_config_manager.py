"""
LDTS-35: Weaviate hyperparameter control system

Provides comprehensive control over Weaviate configuration, hyperparameters,
and search settings for the LDTS Reranker Testing Dashboard.
"""

import weaviate
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorIndexType(Enum):
    """Weaviate vector index types"""
    HNSW = "hnsw"
    FLAT = "flat"
    DYNAMIC = "dynamic"

class DistanceMetric(Enum):
    """Distance metrics for vector similarity"""
    COSINE = "cosine"
    DOT = "dot"
    L2_SQUARED = "l2-squared"
    HAMMING = "hamming"
    MANHATTAN = "manhattan"

@dataclass
class HNSWConfig:
    """HNSW index configuration"""
    ef: int = 64                    # Size of the dynamic candidate list
    ef_construction: int = 128      # Size of the dynamic candidate list during construction
    max_connections: int = 64       # Maximum connections per element
    m: int = 16                    # Number of bi-directional links for new elements
    ml: float = 1/2.302585093       # Level generation factor
    skip: bool = False             # Skip HNSW index creation
    cleanup_interval_seconds: int = 300  # Background cleanup interval
    pq_enabled: bool = False       # Product quantization
    pq_segments: int = 0          # PQ segments
    pq_centroids: int = 256       # PQ centroids
    pq_encoder_type: str = "kmeans" # PQ encoder type
    pq_encoder_distribution: str = "log-normal"  # PQ encoder distribution

@dataclass
class InvertedIndexConfig:
    """Inverted index configuration for BM25/keyword search"""
    bm25_b: float = 0.75          # BM25 b parameter
    bm25_k1: float = 1.2          # BM25 k1 parameter
    cleanup_interval_seconds: int = 60  # Cleanup interval
    stopwords_preset: str = "en"   # Stopwords preset
    stopwords_additions: List[str] = field(default_factory=list)  # Additional stopwords
    stopwords_removals: List[str] = field(default_factory=list)   # Stopwords to remove

@dataclass
class ShardConfig:
    """Sharding configuration"""
    virtual_per_physical: int = 128  # Virtual shards per physical shard
    desired_count: int = 1          # Desired number of physical shards
    actual_count: int = 1           # Actual number of physical shards
    desired_virtual_count: int = 128  # Desired virtual shards
    actual_virtual_count: int = 128   # Actual virtual shards
    key: str = "_id"               # Sharding key
    strategy: str = "hash"         # Sharding strategy
    function: str = "murmur3"      # Hash function

@dataclass
class ReplicationConfig:
    """Replication configuration"""
    factor: int = 1               # Replication factor
    async_enabled: bool = False   # Async replication

@dataclass
class WeaviateClassConfig:
    """Complete Weaviate class configuration"""
    class_name: str = "Tool"
    description: str = "LDTS Tool objects for semantic search"
    vectorizer: str = "none"      # Using external embeddings
    vector_index_type: VectorIndexType = VectorIndexType.HNSW
    vector_index_config: HNSWConfig = field(default_factory=HNSWConfig)
    inverted_index_config: InvertedIndexConfig = field(default_factory=InvertedIndexConfig)
    shard_config: ShardConfig = field(default_factory=ShardConfig)
    replication_config: ReplicationConfig = field(default_factory=ReplicationConfig)
    module_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchConfig:
    """Search configuration and hyperparameters"""
    # Hybrid search parameters
    hybrid_alpha: float = 0.75           # 75% vector, 25% keyword
    hybrid_fusion_type: str = "rankedFusion"  # Fusion algorithm
    
    # Vector search parameters
    vector_certainty: float = 0.6        # Minimum certainty
    vector_distance: float = 0.4         # Maximum distance
    
    # Additional search parameters
    autocut: int = 1                     # Autocut parameter
    group_type: str = "merge"            # Grouping type
    group_force: float = 1.0             # Group force
    
    # BM25 parameters (overrides class config if specified)
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None
    
    # Result limits
    limit: int = 50                      # Default result limit
    offset: int = 0                      # Default offset
    
    # Query expansion
    query_expansion_enabled: bool = True  # Enable query expansion
    max_expanded_terms: int = 5          # Maximum expanded terms

@dataclass 
class PerformanceMetrics:
    """Weaviate performance metrics"""
    query_count: int = 0
    total_query_time: float = 0.0
    average_query_time: float = 0.0
    fastest_query_time: float = float('inf')
    slowest_query_time: float = 0.0
    cache_hit_rate: float = 0.0
    index_size_bytes: int = 0
    memory_usage_bytes: int = 0
    disk_usage_bytes: int = 0
    last_updated: Optional[float] = None

class WeaviateConfigManager:
    """Comprehensive Weaviate configuration and hyperparameter manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.class_config = WeaviateClassConfig()
        self.search_config = SearchConfig()
        self.performance_metrics = PerformanceMetrics()
        self._connected = False
        self._schema_exists = False
        
        # Load configuration
        self._load_config_from_dict(config)
    
    def _load_config_from_dict(self, config: Dict[str, Any]):
        """Load configuration from dictionary"""
        # Basic connection config
        self.url = config.get('url', 'http://localhost:8080')
        self.timeout_config = config.get('timeout_config', {})
        
        # Schema configuration
        schema_config = config.get('schema', {})
        if schema_config:
            self.class_config.class_name = schema_config.get('class_name', 'Tool')
            self.class_config.vectorizer = schema_config.get('vectorizer', 'none')
            self.class_config.vector_index_type = VectorIndexType(
                schema_config.get('vector_index_type', 'hnsw')
            )
            
            # HNSW configuration
            hnsw_config = schema_config.get('hnsw', {})
            self.class_config.vector_index_config = HNSWConfig(
                ef=hnsw_config.get('ef', 64),
                ef_construction=hnsw_config.get('ef_construction', 128),
                max_connections=hnsw_config.get('max_connections', 64),
                m=hnsw_config.get('m', 16),
                cleanup_interval_seconds=hnsw_config.get('cleanup_interval_seconds', 300)
            )
        
        # Search configuration
        search_config = config.get('search', {})
        if search_config:
            hybrid_config = search_config.get('hybrid', {})
            self.search_config = SearchConfig(
                hybrid_alpha=hybrid_config.get('alpha', 0.75),
                query_expansion_enabled=hybrid_config.get('query_expansion', True),
                max_expanded_terms=hybrid_config.get('max_expanded_terms', 5),
                vector_certainty=search_config.get('vector', {}).get('certainty', 0.6),
                vector_distance=search_config.get('vector', {}).get('distance', 0.4)
            )
        
        # Limits configuration
        limits_config = config.get('limits', {})
        if limits_config:
            self.search_config.limit = limits_config.get('default_limit', 50)
    
    async def initialize(self) -> bool:
        """Initialize Weaviate client and connection"""
        try:
            logger.info(f"Initializing Weaviate client for {self.url}")
            
            # Create client with timeout configuration
            timeout_config = weaviate.Config(
                timeout_config={
                    'query': self.timeout_config.get('query', 60),
                    'insert': self.timeout_config.get('insert', 120),
                    'init': self.timeout_config.get('init', 20)
                }
            )
            
            self.client = weaviate.Client(
                url=self.url,
                additional_config=timeout_config
            )
            
            # Test connection
            if await self.health_check():
                self._connected = True
                logger.info("Weaviate client initialized successfully")
                
                # Check if schema exists
                await self._check_schema_exists()
                
                return True
            else:
                logger.error("Weaviate health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Weaviate cluster health"""
        try:
            if not self.client:
                return False
            
            # Use run_in_executor for blocking Weaviate calls
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.client.cluster.get_nodes_status)
            
            # Check if at least one node is healthy
            for node in result:
                if node.get('status') == 'HEALTHY':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Weaviate health check failed: {e}")
            return False
    
    async def _check_schema_exists(self):
        """Check if the required schema exists"""
        try:
            loop = asyncio.get_event_loop()
            schema = await loop.run_in_executor(None, self.client.schema.get)
            
            # Check if our class exists
            for class_obj in schema.get('classes', []):
                if class_obj['class'] == self.class_config.class_name:
                    self._schema_exists = True
                    logger.info(f"Schema for class '{self.class_config.class_name}' found")
                    return
            
            logger.warning(f"Schema for class '{self.class_config.class_name}' not found")
            self._schema_exists = False
            
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            self._schema_exists = False
    
    async def create_schema(self, force_recreate: bool = False) -> bool:
        """Create or update Weaviate schema"""
        try:
            if not self._connected:
                raise RuntimeError("Not connected to Weaviate")
            
            loop = asyncio.get_event_loop()
            
            # Check if class already exists
            if self._schema_exists and not force_recreate:
                logger.info(f"Schema for '{self.class_config.class_name}' already exists")
                return True
            
            # Delete existing class if force recreate
            if force_recreate and self._schema_exists:
                logger.info(f"Force recreating schema for '{self.class_config.class_name}'")
                await loop.run_in_executor(
                    None, 
                    self.client.schema.delete_class, 
                    self.class_config.class_name
                )
            
            # Build class definition
            class_definition = {
                "class": self.class_config.class_name,
                "description": self.class_config.description,
                "vectorizer": self.class_config.vectorizer,
                "vectorIndexType": self.class_config.vector_index_type.value,
                "vectorIndexConfig": self._build_vector_index_config(),
                "invertedIndexConfig": self._build_inverted_index_config(),
                "shardingConfig": self._build_shard_config(),
                "replicationConfig": self._build_replication_config(),
                "properties": self._build_properties_schema()
            }
            
            # Add module config if specified
            if self.class_config.module_config:
                class_definition["moduleConfig"] = self.class_config.module_config
            
            # Create class
            await loop.run_in_executor(
                None,
                self.client.schema.create_class,
                class_definition
            )
            
            self._schema_exists = True
            logger.info(f"Successfully created schema for '{self.class_config.class_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def _build_vector_index_config(self) -> Dict[str, Any]:
        """Build vector index configuration"""
        if self.class_config.vector_index_type == VectorIndexType.HNSW:
            config = self.class_config.vector_index_config
            return {
                "ef": config.ef,
                "efConstruction": config.ef_construction,
                "maxConnections": config.max_connections,
                "m": config.m,
                "ml": config.ml,
                "skip": config.skip,
                "cleanupIntervalSeconds": config.cleanup_interval_seconds,
                "pq": {
                    "enabled": config.pq_enabled,
                    "segments": config.pq_segments,
                    "centroids": config.pq_centroids,
                    "encoder": {
                        "type": config.pq_encoder_type,
                        "distribution": config.pq_encoder_distribution
                    }
                }
            }
        else:
            return {}
    
    def _build_inverted_index_config(self) -> Dict[str, Any]:
        """Build inverted index configuration"""
        config = self.class_config.inverted_index_config
        return {
            "bm25": {
                "b": config.bm25_b,
                "k1": config.bm25_k1
            },
            "cleanupIntervalSeconds": config.cleanup_interval_seconds,
            "stopwords": {
                "preset": config.stopwords_preset,
                "additions": config.stopwords_additions,
                "removals": config.stopwords_removals
            }
        }
    
    def _build_shard_config(self) -> Dict[str, Any]:
        """Build shard configuration"""
        config = self.class_config.shard_config
        return {
            "virtualPerPhysical": config.virtual_per_physical,
            "desiredCount": config.desired_count,
            "actualCount": config.actual_count,
            "desiredVirtualCount": config.desired_virtual_count,
            "actualVirtualCount": config.actual_virtual_count,
            "key": config.key,
            "strategy": config.strategy,
            "function": config.function
        }
    
    def _build_replication_config(self) -> Dict[str, Any]:
        """Build replication configuration"""
        config = self.class_config.replication_config
        return {
            "factor": config.factor,
            "asyncEnabled": config.async_enabled
        }
    
    def _build_properties_schema(self) -> List[Dict[str, Any]]:
        """Build properties schema for the Tool class"""
        return [
            {
                "name": "name",
                "dataType": ["text"],
                "description": "Tool name",
                "indexFilterable": True,
                "indexSearchable": True
            },
            {
                "name": "description", 
                "dataType": ["text"],
                "description": "Tool description",
                "indexFilterable": False,
                "indexSearchable": True
            },
            {
                "name": "category",
                "dataType": ["text"],
                "description": "Tool category",
                "indexFilterable": True,
                "indexSearchable": True
            },
            {
                "name": "source",
                "dataType": ["text"],
                "description": "Tool source",
                "indexFilterable": True,
                "indexSearchable": False
            },
            {
                "name": "tags",
                "dataType": ["text[]"],
                "description": "Tool tags",
                "indexFilterable": True,
                "indexSearchable": True
            },
            {
                "name": "metadata",
                "dataType": ["object"],
                "description": "Additional tool metadata",
                "indexFilterable": False,
                "indexSearchable": False
            },
            {
                "name": "created_at",
                "dataType": ["date"],
                "description": "Creation timestamp",
                "indexFilterable": True,
                "indexSearchable": False
            },
            {
                "name": "updated_at",
                "dataType": ["date"],
                "description": "Last update timestamp",
                "indexFilterable": True,
                "indexSearchable": False
            }
        ]
    
    async def update_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """Update search hyperparameters"""
        try:
            # Update search config
            if 'hybrid_alpha' in hyperparameters:
                self.search_config.hybrid_alpha = float(hyperparameters['hybrid_alpha'])
            
            if 'vector_certainty' in hyperparameters:
                self.search_config.vector_certainty = float(hyperparameters['vector_certainty'])
            
            if 'vector_distance' in hyperparameters:
                self.search_config.vector_distance = float(hyperparameters['vector_distance'])
            
            if 'bm25_k1' in hyperparameters:
                self.search_config.bm25_k1 = float(hyperparameters['bm25_k1'])
            
            if 'bm25_b' in hyperparameters:
                self.search_config.bm25_b = float(hyperparameters['bm25_b'])
            
            if 'limit' in hyperparameters:
                self.search_config.limit = int(hyperparameters['limit'])
            
            logger.info(f"Updated hyperparameters: {hyperparameters}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update hyperparameters: {e}")
            return False
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get Weaviate cluster statistics"""
        try:
            if not self._connected:
                return {"error": "Not connected to Weaviate"}
            
            loop = asyncio.get_event_loop()
            
            # Get nodes status
            nodes_status = await loop.run_in_executor(None, self.client.cluster.get_nodes_status)
            
            # Get class info
            class_info = {}
            if self._schema_exists:
                try:
                    meta = await loop.run_in_executor(
                        None,
                        lambda: self.client.query.aggregate(self.class_config.class_name).with_meta_count().do()
                    )
                    class_info = {
                        "object_count": meta.get("data", {}).get("Aggregate", {}).get(self.class_config.class_name, [{}])[0].get("meta", {}).get("count", 0)
                    }
                except Exception as e:
                    logger.warning(f"Could not get class info: {e}")
                    class_info = {"error": str(e)}
            
            return {
                "connected": self._connected,
                "url": self.url,
                "nodes": nodes_status,
                "schema_exists": self._schema_exists,
                "class_name": self.class_config.class_name,
                "class_info": class_info,
                "performance_metrics": {
                    "query_count": self.performance_metrics.query_count,
                    "average_query_time_ms": self.performance_metrics.average_query_time * 1000,
                    "fastest_query_time_ms": self.performance_metrics.fastest_query_time * 1000 if self.performance_metrics.fastest_query_time != float('inf') else None,
                    "slowest_query_time_ms": self.performance_metrics.slowest_query_time * 1000,
                    "last_updated": self.performance_metrics.last_updated
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster stats: {e}")
            return {"error": str(e)}
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return {
            "connection": {
                "url": self.url,
                "connected": self._connected,
                "timeout_config": self.timeout_config
            },
            "class_config": {
                "class_name": self.class_config.class_name,
                "vectorizer": self.class_config.vectorizer,
                "vector_index_type": self.class_config.vector_index_type.value,
                "vector_index_config": {
                    "ef": self.class_config.vector_index_config.ef,
                    "ef_construction": self.class_config.vector_index_config.ef_construction,
                    "max_connections": self.class_config.vector_index_config.max_connections,
                    "m": self.class_config.vector_index_config.m,
                    "cleanup_interval_seconds": self.class_config.vector_index_config.cleanup_interval_seconds
                },
                "inverted_index_config": {
                    "bm25_b": self.class_config.inverted_index_config.bm25_b,
                    "bm25_k1": self.class_config.inverted_index_config.bm25_k1,
                    "cleanup_interval_seconds": self.class_config.inverted_index_config.cleanup_interval_seconds
                }
            },
            "search_config": {
                "hybrid_alpha": self.search_config.hybrid_alpha,
                "vector_certainty": self.search_config.vector_certainty,
                "vector_distance": self.search_config.vector_distance,
                "limit": self.search_config.limit,
                "query_expansion_enabled": self.search_config.query_expansion_enabled,
                "max_expanded_terms": self.search_config.max_expanded_terms
            }
        }
    
    def _update_performance_metrics(self, query_time: float):
        """Update performance metrics after a query"""
        self.performance_metrics.query_count += 1
        self.performance_metrics.total_query_time += query_time
        self.performance_metrics.average_query_time = (
            self.performance_metrics.total_query_time / self.performance_metrics.query_count
        )
        
        if query_time < self.performance_metrics.fastest_query_time:
            self.performance_metrics.fastest_query_time = query_time
        
        if query_time > self.performance_metrics.slowest_query_time:
            self.performance_metrics.slowest_query_time = query_time
        
        self.performance_metrics.last_updated = time.time()
    
    async def close(self):
        """Close Weaviate client connection"""
        if self.client:
            # Weaviate Python client doesn't have async close method
            self.client = None
            self._connected = False
            logger.info("Weaviate client connection closed")

# Global Weaviate manager instance
weaviate_manager: Optional[WeaviateConfigManager] = None

async def initialize_weaviate_manager(config: Dict[str, Any]) -> bool:
    """Initialize global Weaviate manager"""
    global weaviate_manager
    
    weaviate_manager = WeaviateConfigManager(config)
    return await weaviate_manager.initialize()

def get_weaviate_manager() -> WeaviateConfigManager:
    """Get global Weaviate manager instance"""
    if weaviate_manager is None:
        raise RuntimeError("Weaviate manager not initialized")
    return weaviate_manager