"""
LDTS-43: Benchmark Query Management System
Manages standard benchmark datasets and query collections for search evaluation
"""

import asyncio
import json
import uuid
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import hashlib
import zipfile
import io
from urllib.parse import urlparse

from automated_batch_evaluation import EvaluationQuery, EvaluationDataset, DatasetType

class BenchmarkSource(Enum):
    MSMARCO = "msmarco"
    TREC = "trec"
    BEIR = "beir"
    CUSTOM = "custom"
    IMPORTED = "imported"
    GENERATED = "generated"

class QueryDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class QueryType(Enum):
    FACTUAL = "factual"
    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    MULTI_HOP = "multi_hop"
    COMPOSITIONAL = "compositional"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"

@dataclass
class BenchmarkQuery:
    """Enhanced query with benchmark-specific metadata"""
    id: str
    query: str
    expected_relevant_docs: List[str]
    relevance_scores: Dict[str, float]
    
    # Benchmark metadata
    source: BenchmarkSource
    category: Optional[str] = None
    difficulty: QueryDifficulty = QueryDifficulty.MEDIUM
    query_type: QueryType = QueryType.INFORMATIONAL
    
    # Analysis metadata
    query_length: int = 0
    num_relevant_docs: int = 0
    avg_relevance_score: float = 0.0
    
    # Quality metrics
    clarity_score: Optional[float] = None  # 0-1 scale
    ambiguity_score: Optional[float] = None  # 0-1 scale (lower is better)
    
    # Temporal info
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkCollection:
    """Collection of benchmark queries organized by theme or source"""
    id: str
    name: str
    description: str
    source: BenchmarkSource
    version: str
    
    queries: List[BenchmarkQuery]
    
    # Collection statistics
    total_queries: int = 0
    avg_query_length: float = 0.0
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    overall_quality_score: Optional[float] = None
    consistency_score: Optional[float] = None
    
    # Provenance
    source_url: Optional[str] = None
    license_info: Optional[str] = None
    citation: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkTemplate:
    """Template for generating synthetic benchmark queries"""
    id: str
    name: str
    description: str
    
    # Generation parameters
    query_patterns: List[str]  # Template strings with placeholders
    entity_types: List[str]  # Types of entities to fill placeholders
    difficulty_weights: Dict[QueryDifficulty, float]  # Probability distribution
    
    # Expected answer types
    answer_types: List[str]
    relevance_criteria: List[str]
    
    # Quality constraints
    min_query_length: int = 3
    max_query_length: int = 50
    min_relevant_docs: int = 1
    max_relevant_docs: int = 20
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BenchmarkQueryManager:
    """Main manager for benchmark query collections"""
    
    def __init__(self, storage_path: str = "benchmark_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.collections_path = self.storage_path / "collections"
        self.templates_path = self.storage_path / "templates"
        self.downloads_path = self.storage_path / "downloads"
        self.cache_path = self.storage_path / "cache"
        
        for path in [self.collections_path, self.templates_path, self.downloads_path, self.cache_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Built-in benchmark sources
        self.benchmark_sources = {
            BenchmarkSource.MSMARCO: {
                'name': 'MS MARCO',
                'description': 'Microsoft Machine Reading Comprehension Dataset',
                'base_url': 'https://msmarco.blob.core.windows.net/msmarcoranking',
                'files': {
                    'queries_train': 'queries.train.tsv',
                    'queries_dev': 'queries.dev.small.tsv',
                    'qrels_train': 'qrels.train.tsv',
                    'qrels_dev': 'qrels.dev.small.tsv'
                }
            },
            BenchmarkSource.TREC: {
                'name': 'TREC',
                'description': 'Text REtrieval Conference datasets',
                'base_url': 'https://trec.nist.gov/data',
                'files': {}  # Multiple TREC tracks
            },
            BenchmarkSource.BEIR: {
                'name': 'BEIR',
                'description': 'Benchmarking Information Retrieval',
                'base_url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets',
                'files': {}  # Multiple BEIR datasets
            }
        }
    
    # Collection Management
    async def create_collection(self, collection: BenchmarkCollection) -> bool:
        """Create a new benchmark collection"""
        try:
            # Calculate collection statistics
            collection = self._calculate_collection_stats(collection)
            
            # Save collection
            collection_path = self.collections_path / f"{collection.id}.json"
            with open(collection_path, 'w') as f:
                json.dump(asdict(collection), f, indent=2, default=str)
            
            self.logger.info(f"Created benchmark collection {collection.id} with {len(collection.queries)} queries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection.id}: {e}")
            return False
    
    async def load_collection(self, collection_id: str) -> Optional[BenchmarkCollection]:
        """Load a benchmark collection"""
        try:
            collection_path = self.collections_path / f"{collection_id}.json"
            
            if not collection_path.exists():
                return None
            
            with open(collection_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            # Convert enums
            data['source'] = BenchmarkSource(data['source'])
            
            # Reconstruct queries
            queries = []
            for query_data in data['queries']:
                query_data['created_at'] = datetime.fromisoformat(query_data['created_at'])
                query_data['updated_at'] = datetime.fromisoformat(query_data['updated_at'])
                query_data['source'] = BenchmarkSource(query_data['source'])
                query_data['difficulty'] = QueryDifficulty(query_data['difficulty'])
                query_data['query_type'] = QueryType(query_data['query_type'])
                queries.append(BenchmarkQuery(**query_data))
            
            data['queries'] = queries
            
            return BenchmarkCollection(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load collection {collection_id}: {e}")
            return None
    
    async def list_collections(self, source_filter: Optional[BenchmarkSource] = None) -> List[Dict[str, Any]]:
        """List all benchmark collections"""
        collections = []
        
        for collection_file in self.collections_path.glob("*.json"):
            try:
                with open(collection_file, 'r') as f:
                    data = json.load(f)
                
                if source_filter and data['source'] != source_filter.value:
                    continue
                
                collections.append({
                    'id': data['id'],
                    'name': data['name'],
                    'description': data['description'],
                    'source': data['source'],
                    'version': data['version'],
                    'total_queries': data['total_queries'],
                    'avg_query_length': data.get('avg_query_length', 0),
                    'created_at': data['created_at']
                })
            except Exception as e:
                self.logger.error(f"Failed to load collection info from {collection_file}: {e}")
                continue
        
        return sorted(collections, key=lambda x: x['created_at'], reverse=True)
    
    def _calculate_collection_stats(self, collection: BenchmarkCollection) -> BenchmarkCollection:
        """Calculate statistics for a collection"""
        
        if not collection.queries:
            return collection
        
        collection.total_queries = len(collection.queries)
        
        # Calculate average query length
        query_lengths = []
        difficulty_counts = {}
        type_counts = {}
        quality_scores = []
        
        for query in collection.queries:
            # Update query-level stats
            query.query_length = len(query.query.split())
            query.num_relevant_docs = len(query.expected_relevant_docs)
            query.avg_relevance_score = np.mean(list(query.relevance_scores.values())) if query.relevance_scores else 0.0
            
            # Collect for collection-level stats
            query_lengths.append(query.query_length)
            
            difficulty_counts[query.difficulty.value] = difficulty_counts.get(query.difficulty.value, 0) + 1
            type_counts[query.query_type.value] = type_counts.get(query.query_type.value, 0) + 1
            
            if query.clarity_score is not None:
                quality_scores.append(query.clarity_score)
        
        collection.avg_query_length = np.mean(query_lengths) if query_lengths else 0.0
        collection.difficulty_distribution = difficulty_counts
        collection.type_distribution = type_counts
        
        if quality_scores:
            collection.overall_quality_score = np.mean(quality_scores)
            collection.consistency_score = 1.0 - np.std(quality_scores)  # Higher consistency = lower std dev
        
        collection.updated_at = datetime.utcnow()
        
        return collection
    
    # Query Management
    async def add_queries_to_collection(
        self, 
        collection_id: str, 
        queries: List[BenchmarkQuery]
    ) -> bool:
        """Add queries to an existing collection"""
        try:
            collection = await self.load_collection(collection_id)
            if not collection:
                return False
            
            # Add queries
            collection.queries.extend(queries)
            
            # Recalculate stats
            collection = self._calculate_collection_stats(collection)
            
            # Save updated collection
            return await self.create_collection(collection)
            
        except Exception as e:
            self.logger.error(f"Failed to add queries to collection {collection_id}: {e}")
            return False
    
    async def remove_queries_from_collection(
        self,
        collection_id: str,
        query_ids: List[str]
    ) -> bool:
        """Remove queries from a collection"""
        try:
            collection = await self.load_collection(collection_id)
            if not collection:
                return False
            
            # Remove queries
            query_ids_set = set(query_ids)
            collection.queries = [q for q in collection.queries if q.id not in query_ids_set]
            
            # Recalculate stats
            collection = self._calculate_collection_stats(collection)
            
            # Save updated collection
            return await self.create_collection(collection)
            
        except Exception as e:
            self.logger.error(f"Failed to remove queries from collection {collection_id}: {e}")
            return False
    
    async def search_queries(
        self,
        collection_id: Optional[str] = None,
        query_text: Optional[str] = None,
        difficulty: Optional[QueryDifficulty] = None,
        query_type: Optional[QueryType] = None,
        min_relevance_docs: Optional[int] = None,
        max_relevance_docs: Optional[int] = None,
        limit: int = 100
    ) -> List[BenchmarkQuery]:
        """Search for queries matching criteria"""
        
        matching_queries = []
        
        # Get collections to search
        if collection_id:
            collections = [await self.load_collection(collection_id)]
            collections = [c for c in collections if c is not None]
        else:
            collection_infos = await self.list_collections()
            collections = []
            for info in collection_infos:
                collection = await self.load_collection(info['id'])
                if collection:
                    collections.append(collection)
        
        # Search through collections
        for collection in collections:
            for query in collection.queries:
                # Apply filters
                if query_text and query_text.lower() not in query.query.lower():
                    continue
                
                if difficulty and query.difficulty != difficulty:
                    continue
                
                if query_type and query.query_type != query_type:
                    continue
                
                if min_relevance_docs and query.num_relevant_docs < min_relevance_docs:
                    continue
                
                if max_relevance_docs and query.num_relevant_docs > max_relevance_docs:
                    continue
                
                matching_queries.append(query)
                
                if len(matching_queries) >= limit:
                    return matching_queries
        
        return matching_queries
    
    # Benchmark Data Import
    async def import_msmarco_dataset(
        self,
        collection_name: str,
        subset: str = "dev",  # "train" or "dev" 
        limit: Optional[int] = 1000
    ) -> Optional[str]:
        """Import MS MARCO dataset"""
        
        try:
            source_info = self.benchmark_sources[BenchmarkSource.MSMARCO]
            
            # Download required files
            queries_file = f"queries.{subset}.small.tsv" if subset == "dev" else "queries.train.tsv"
            qrels_file = f"qrels.{subset}.small.tsv" if subset == "dev" else "qrels.train.tsv"
            
            queries_url = f"{source_info['base_url']}/{queries_file}"
            qrels_url = f"{source_info['base_url']}/{qrels_file}"
            
            # Download and parse files
            queries_data = await self._download_and_parse_tsv(queries_url)
            qrels_data = await self._download_and_parse_tsv(qrels_url)
            
            # Process data
            benchmark_queries = self._process_msmarco_data(queries_data, qrels_data, limit)
            
            # Create collection
            collection_id = str(uuid.uuid4())
            collection = BenchmarkCollection(
                id=collection_id,
                name=collection_name,
                description=f"MS MARCO {subset} dataset",
                source=BenchmarkSource.MSMARCO,
                version="1.0",
                queries=benchmark_queries,
                source_url=queries_url,
                license_info="Microsoft Research License",
                citation="Microsoft MARCO: A Dataset for Reading Comprehension"
            )
            
            success = await self.create_collection(collection)
            if success:
                self.logger.info(f"Imported MS MARCO {subset} dataset: {len(benchmark_queries)} queries")
                return collection_id
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to import MS MARCO dataset: {e}")
            return None
    
    async def _download_and_parse_tsv(self, url: str) -> List[List[str]]:
        """Download and parse TSV file"""
        
        try:
            # Check cache first
            cache_file = self.cache_path / self._get_url_hash(url)
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return [line.strip().split('\t') for line in f]
            
            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Cache the content
                        with open(cache_file, 'w') as f:
                            f.write(content)
                        
                        # Parse TSV
                        return [line.strip().split('\t') for line in content.strip().split('\n')]
                    else:
                        raise Exception(f"Failed to download {url}: HTTP {response.status}")
        
        except Exception as e:
            self.logger.error(f"Failed to download/parse {url}: {e}")
            return []
    
    def _get_url_hash(self, url: str) -> str:
        """Get hash for URL to use as cache filename"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _process_msmarco_data(
        self,
        queries_data: List[List[str]],
        qrels_data: List[List[str]],
        limit: Optional[int]
    ) -> List[BenchmarkQuery]:
        """Process MS MARCO TSV data into BenchmarkQuery objects"""
        
        # Build qrels lookup: query_id -> {doc_id: relevance_score}
        qrels_lookup = {}
        for row in qrels_data:
            if len(row) >= 4:
                query_id, _, doc_id, relevance = row[0], row[1], row[2], int(row[3])
                if query_id not in qrels_lookup:
                    qrels_lookup[query_id] = {}
                qrels_lookup[query_id][doc_id] = float(relevance)
        
        # Process queries
        benchmark_queries = []
        processed_count = 0
        
        for row in queries_data:
            if len(row) >= 2:
                query_id, query_text = row[0], row[1]
                
                # Get relevance judgments
                relevance_scores = qrels_lookup.get(query_id, {})
                expected_relevant = [doc_id for doc_id, score in relevance_scores.items() if score > 0]
                
                if expected_relevant:  # Only include queries with relevant documents
                    # Determine query characteristics
                    query_length = len(query_text.split())
                    difficulty = self._classify_query_difficulty(query_text, len(expected_relevant))
                    query_type = self._classify_query_type(query_text)
                    
                    benchmark_query = BenchmarkQuery(
                        id=query_id,
                        query=query_text,
                        expected_relevant_docs=expected_relevant,
                        relevance_scores=relevance_scores,
                        source=BenchmarkSource.MSMARCO,
                        category="passage_retrieval",
                        difficulty=difficulty,
                        query_type=query_type,
                        query_length=query_length,
                        num_relevant_docs=len(expected_relevant),
                        avg_relevance_score=np.mean(list(relevance_scores.values())) if relevance_scores else 0.0
                    )
                    
                    benchmark_queries.append(benchmark_query)
                    processed_count += 1
                    
                    if limit and processed_count >= limit:
                        break
        
        return benchmark_queries
    
    def _classify_query_difficulty(self, query_text: str, num_relevant: int) -> QueryDifficulty:
        """Classify query difficulty based on characteristics"""
        
        query_length = len(query_text.split())
        
        # Simple heuristic classification
        if query_length <= 3 and num_relevant >= 5:
            return QueryDifficulty.EASY
        elif query_length <= 6 and num_relevant >= 2:
            return QueryDifficulty.MEDIUM
        elif query_length <= 10:
            return QueryDifficulty.HARD
        else:
            return QueryDifficulty.EXPERT
    
    def _classify_query_type(self, query_text: str) -> QueryType:
        """Classify query type based on linguistic patterns"""
        
        query_lower = query_text.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'why', 'how']):
            return QueryType.FACTUAL
        elif any(word in query_lower for word in ['find', 'search', 'locate', 'site', 'website']):
            return QueryType.NAVIGATIONAL
        elif any(word in query_lower for word in ['buy', 'purchase', 'price', 'cost', 'shop']):
            return QueryType.TRANSACTIONAL
        elif any(word in query_lower for word in ['and', 'or', 'but', 'with', 'after', 'before']):
            return QueryType.COMPOSITIONAL
        elif any(word in query_lower for word in ['number', 'count', 'many', 'much', 'age', 'year']):
            return QueryType.NUMERICAL
        else:
            return QueryType.INFORMATIONAL
    
    # Template Management
    async def create_template(self, template: BenchmarkTemplate) -> bool:
        """Create a query generation template"""
        try:
            template_path = self.templates_path / f"{template.id}.json"
            with open(template_path, 'w') as f:
                json.dump(asdict(template), f, indent=2, default=str)
            
            self.logger.info(f"Created benchmark template {template.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create template {template.id}: {e}")
            return False
    
    async def generate_from_template(
        self,
        template_id: str,
        num_queries: int,
        collection_name: str
    ) -> Optional[str]:
        """Generate synthetic queries from template"""
        # This would be implemented with actual generation logic
        # For now, return a placeholder
        
        self.logger.info(f"Template-based generation not yet implemented")
        return None
    
    # Integration with Evaluation System
    async def export_to_evaluation_dataset(
        self,
        collection_id: str,
        dataset_name: Optional[str] = None,
        query_limit: Optional[int] = None
    ) -> Optional[str]:
        """Export benchmark collection to evaluation dataset format"""
        
        try:
            collection = await self.load_collection(collection_id)
            if not collection:
                return None
            
            # Convert benchmark queries to evaluation queries
            evaluation_queries = []
            queries_to_process = collection.queries[:query_limit] if query_limit else collection.queries
            
            for benchmark_query in queries_to_process:
                eval_query = EvaluationQuery(
                    id=benchmark_query.id,
                    query=benchmark_query.query,
                    expected_relevant_docs=benchmark_query.expected_relevant_docs,
                    relevance_scores=benchmark_query.relevance_scores,
                    category=benchmark_query.category,
                    difficulty=benchmark_query.difficulty.value,
                    metadata={
                        'source': benchmark_query.source.value,
                        'query_type': benchmark_query.query_type.value,
                        'query_length': benchmark_query.query_length,
                        'clarity_score': benchmark_query.clarity_score,
                        'ambiguity_score': benchmark_query.ambiguity_score
                    }
                )
                evaluation_queries.append(eval_query)
            
            # Create evaluation dataset
            from automated_batch_evaluation import automated_batch_evaluator
            
            dataset = EvaluationDataset(
                id=str(uuid.uuid4()),
                name=dataset_name or f"Benchmark: {collection.name}",
                description=f"Evaluation dataset exported from benchmark collection {collection.name}",
                type=DatasetType.BENCHMARK,
                queries=evaluation_queries,
                created_at=datetime.utcnow(),
                metadata={
                    'source_collection_id': collection_id,
                    'source_collection_name': collection.name,
                    'benchmark_source': collection.source.value,
                    'export_date': datetime.utcnow().isoformat()
                }
            )
            
            success = await automated_batch_evaluator.create_dataset(dataset)
            if success:
                self.logger.info(f"Exported benchmark collection {collection_id} to evaluation dataset {dataset.id}")
                return dataset.id
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to export collection {collection_id}: {e}")
            return None

# Global instance
benchmark_manager = BenchmarkQueryManager()

# Example usage
async def create_sample_collection():
    """Create a sample benchmark collection"""
    
    # Create sample queries
    sample_queries = [
        BenchmarkQuery(
            id="sample_1",
            query="What is machine learning?",
            expected_relevant_docs=["doc_ml_1", "doc_ml_2", "doc_ai_1"],
            relevance_scores={"doc_ml_1": 4.0, "doc_ml_2": 3.0, "doc_ai_1": 2.0},
            source=BenchmarkSource.CUSTOM,
            category="artificial_intelligence",
            difficulty=QueryDifficulty.EASY,
            query_type=QueryType.FACTUAL,
            clarity_score=0.9
        ),
        BenchmarkQuery(
            id="sample_2", 
            query="How do neural networks learn from data?",
            expected_relevant_docs=["doc_nn_1", "doc_ml_3", "doc_dl_1"],
            relevance_scores={"doc_nn_1": 4.0, "doc_ml_3": 3.0, "doc_dl_1": 4.0},
            source=BenchmarkSource.CUSTOM,
            category="deep_learning", 
            difficulty=QueryDifficulty.MEDIUM,
            query_type=QueryType.INFORMATIONAL,
            clarity_score=0.8
        )
    ]
    
    # Create collection
    collection = BenchmarkCollection(
        id=str(uuid.uuid4()),
        name="Sample AI/ML Collection",
        description="Sample benchmark queries for AI/ML topics",
        source=BenchmarkSource.CUSTOM,
        version="1.0",
        queries=sample_queries
    )
    
    success = await benchmark_manager.create_collection(collection)
    if success:
        print(f"Created sample collection: {collection.id}")
        return collection.id
    else:
        print("Failed to create sample collection")
        return None

if __name__ == "__main__":
    import numpy as np
    import asyncio
    asyncio.run(create_sample_collection())