"""
LDTS-40: Automated Batch Evaluation System
Implements automated evaluation of search and reranking systems at scale
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, AsyncGenerator
import logging

from manual_evaluation_interface import (
    ManualEvaluationInterface, EvaluationCriteria, DocumentEvaluation,
    QueryEvaluation, EvaluationSession
)

class BatchEvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EvaluationMode(Enum):
    COMPARATIVE = "comparative"  # Compare multiple configurations
    SINGLE = "single"           # Evaluate single configuration
    REGRESSION = "regression"   # Check for performance regression
    A_B_TEST = "a_b_test"      # A/B testing mode

class DatasetType(Enum):
    CUSTOM = "custom"
    BENCHMARK = "benchmark"
    SYNTHETIC = "synthetic"
    PRODUCTION = "production"

@dataclass
class EvaluationQuery:
    """Single query for batch evaluation"""
    id: str
    query: str
    expected_relevant_docs: List[str]  # Document IDs that should be relevant
    relevance_scores: Dict[str, float]  # doc_id -> relevance score (0-4)
    category: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationDataset:
    """Dataset for batch evaluation"""
    id: str
    name: str
    description: str
    type: DatasetType
    queries: List[EvaluationQuery]
    created_at: datetime
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemConfiguration:
    """Configuration of system being evaluated"""
    id: str
    name: str
    description: str
    embedding_provider: str
    embedding_model: str
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    search_params: Dict[str, Any] = field(default_factory=dict)
    rerank_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchEvaluationResult:
    """Results from evaluating a single query against a configuration"""
    query_id: str
    configuration_id: str
    retrieved_docs: List[str]  # Retrieved document IDs in order
    relevance_scores: Dict[str, float]  # Predicted relevance scores
    execution_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchEvaluationJob:
    """Batch evaluation job configuration and status"""
    id: str
    name: str
    description: str
    mode: EvaluationMode
    status: BatchEvaluationStatus
    dataset_id: str
    configurations: List[str]  # Configuration IDs
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    results: List[BatchEvaluationResult] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AutomatedBatchEvaluator:
    """Main class for automated batch evaluation"""
    
    def __init__(self, storage_path: str = "evaluation_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.datasets_path = self.storage_path / "datasets"
        self.configs_path = self.storage_path / "configurations"
        self.jobs_path = self.storage_path / "jobs"
        self.results_path = self.storage_path / "results"
        
        for path in [self.datasets_path, self.configs_path, self.jobs_path, self.results_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Integration with manual evaluation interface
        self.manual_evaluator = ManualEvaluationInterface()
    
    # Dataset Management
    async def create_dataset(self, dataset: EvaluationDataset) -> bool:
        """Create a new evaluation dataset"""
        try:
            dataset_path = self.datasets_path / f"{dataset.id}.json"
            
            with open(dataset_path, 'w') as f:
                json.dump(asdict(dataset), f, indent=2, default=str)
            
            self.logger.info(f"Created dataset {dataset.id} with {len(dataset.queries)} queries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset {dataset.id}: {e}")
            return False
    
    async def load_dataset(self, dataset_id: str) -> Optional[EvaluationDataset]:
        """Load an evaluation dataset"""
        try:
            dataset_path = self.datasets_path / f"{dataset_id}.json"
            
            if not dataset_path.exists():
                return None
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            
            # Reconstruct queries
            queries = []
            for query_data in data['queries']:
                queries.append(EvaluationQuery(**query_data))
            
            data['queries'] = queries
            data['type'] = DatasetType(data['type'])
            
            return EvaluationDataset(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return None
    
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        datasets = []
        
        for dataset_file in self.datasets_path.glob("*.json"):
            try:
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                
                datasets.append({
                    'id': data['id'],
                    'name': data['name'],
                    'description': data['description'],
                    'type': data['type'],
                    'query_count': len(data['queries']),
                    'created_at': data['created_at'],
                    'version': data['version']
                })
            except Exception as e:
                self.logger.error(f"Failed to load dataset info from {dataset_file}: {e}")
                continue
        
        return sorted(datasets, key=lambda x: x['created_at'], reverse=True)
    
    # Configuration Management
    async def create_configuration(self, config: SystemConfiguration) -> bool:
        """Create a new system configuration"""
        try:
            config_path = self.configs_path / f"{config.id}.json"
            
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            self.logger.info(f"Created configuration {config.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration {config.id}: {e}")
            return False
    
    async def load_configuration(self, config_id: str) -> Optional[SystemConfiguration]:
        """Load a system configuration"""
        try:
            config_path = self.configs_path / f"{config_id}.json"
            
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return SystemConfiguration(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_id}: {e}")
            return None
    
    async def list_configurations(self) -> List[Dict[str, Any]]:
        """List all available configurations"""
        configs = []
        
        for config_file in self.configs_path.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                configs.append({
                    'id': data['id'],
                    'name': data['name'],
                    'description': data['description'],
                    'embedding_provider': data['embedding_provider'],
                    'embedding_model': data['embedding_model'],
                    'reranker_provider': data.get('reranker_provider'),
                    'reranker_model': data.get('reranker_model')
                })
            except Exception as e:
                self.logger.error(f"Failed to load config info from {config_file}: {e}")
                continue
        
        return configs
    
    # Job Management
    async def create_evaluation_job(
        self,
        name: str,
        description: str,
        dataset_id: str,
        configurations: List[str],
        mode: EvaluationMode = EvaluationMode.SINGLE,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new batch evaluation job"""
        
        job_id = str(uuid.uuid4())
        
        job = BatchEvaluationJob(
            id=job_id,
            name=name,
            description=description,
            mode=mode,
            status=BatchEvaluationStatus.PENDING,
            dataset_id=dataset_id,
            configurations=configurations,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Save job
        job_path = self.jobs_path / f"{job_id}.json"
        with open(job_path, 'w') as f:
            json.dump(asdict(job), f, indent=2, default=str)
        
        self.logger.info(f"Created evaluation job {job_id}")
        return job_id
    
    async def start_evaluation_job(self, job_id: str) -> bool:
        """Start an evaluation job"""
        try:
            job = await self.load_job(job_id)
            if not job:
                self.logger.error(f"Job {job_id} not found")
                return False
            
            if job.status != BatchEvaluationStatus.PENDING:
                self.logger.error(f"Job {job_id} is not pending (status: {job.status})")
                return False
            
            # Start the job as an async task
            task = asyncio.create_task(self._execute_evaluation_job(job))
            self.active_jobs[job_id] = task
            
            self.logger.info(f"Started evaluation job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {e}")
            return False
    
    async def cancel_evaluation_job(self, job_id: str) -> bool:
        """Cancel a running evaluation job"""
        try:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]
            
            # Update job status
            job = await self.load_job(job_id)
            if job:
                job.status = BatchEvaluationStatus.CANCELLED
                await self._save_job(job)
            
            self.logger.info(f"Cancelled evaluation job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def load_job(self, job_id: str) -> Optional[BatchEvaluationJob]:
        """Load an evaluation job"""
        try:
            job_path = self.jobs_path / f"{job_id}.json"
            
            if not job_path.exists():
                return None
            
            with open(job_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('started_at'):
                data['started_at'] = datetime.fromisoformat(data['started_at'])
            if data.get('completed_at'):
                data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            
            # Convert enums
            data['mode'] = EvaluationMode(data['mode'])
            data['status'] = BatchEvaluationStatus(data['status'])
            
            # Reconstruct results
            results = []
            for result_data in data.get('results', []):
                results.append(BatchEvaluationResult(**result_data))
            data['results'] = results
            
            return BatchEvaluationJob(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load job {job_id}: {e}")
            return None
    
    async def _save_job(self, job: BatchEvaluationJob):
        """Save job state to disk"""
        job_path = self.jobs_path / f"{job.id}.json"
        with open(job_path, 'w') as f:
            json.dump(asdict(job), f, indent=2, default=str)
    
    async def list_jobs(self, status_filter: Optional[BatchEvaluationStatus] = None) -> List[Dict[str, Any]]:
        """List evaluation jobs"""
        jobs = []
        
        for job_file in self.jobs_path.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    data = json.load(f)
                
                if status_filter and data['status'] != status_filter.value:
                    continue
                
                jobs.append({
                    'id': data['id'],
                    'name': data['name'],
                    'description': data['description'],
                    'status': data['status'],
                    'mode': data['mode'],
                    'dataset_id': data['dataset_id'],
                    'configuration_count': len(data['configurations']),
                    'progress': data['progress'],
                    'created_at': data['created_at'],
                    'started_at': data.get('started_at'),
                    'completed_at': data.get('completed_at')
                })
            except Exception as e:
                self.logger.error(f"Failed to load job info from {job_file}: {e}")
                continue
        
        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)
    
    # Core Evaluation Logic
    async def _execute_evaluation_job(self, job: BatchEvaluationJob):
        """Execute an evaluation job"""
        try:
            # Update job status
            job.status = BatchEvaluationStatus.RUNNING
            job.started_at = datetime.utcnow()
            await self._save_job(job)
            
            # Load dataset
            dataset = await self.load_dataset(job.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {job.dataset_id} not found")
            
            # Load configurations
            configurations = []
            for config_id in job.configurations:
                config = await self.load_configuration(config_id)
                if not config:
                    raise ValueError(f"Configuration {config_id} not found")
                configurations.append(config)
            
            total_evaluations = len(dataset.queries) * len(configurations)
            completed_evaluations = 0
            
            self.logger.info(f"Starting evaluation job {job.id}: {total_evaluations} total evaluations")
            
            # Execute evaluations
            for config in configurations:
                for query in dataset.queries:
                    try:
                        result = await self._evaluate_single_query(query, config)
                        job.results.append(result)
                        
                        completed_evaluations += 1
                        job.progress = completed_evaluations / total_evaluations
                        
                        # Save progress periodically
                        if completed_evaluations % 10 == 0:
                            await self._save_job(job)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate query {query.id} with config {config.id}: {e}")
                        
                        # Record failed result
                        failed_result = BatchEvaluationResult(
                            query_id=query.id,
                            configuration_id=config.id,
                            retrieved_docs=[],
                            relevance_scores={},
                            execution_time_ms=0,
                            error_message=str(e)
                        )
                        job.results.append(failed_result)
                        
                        completed_evaluations += 1
                        job.progress = completed_evaluations / total_evaluations
            
            # Complete job
            job.status = BatchEvaluationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            
            self.logger.info(f"Completed evaluation job {job.id}: {len(job.results)} results")
            
        except Exception as e:
            self.logger.error(f"Evaluation job {job.id} failed: {e}")
            job.status = BatchEvaluationStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
        
        finally:
            # Save final job state
            await self._save_job(job)
            
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
    
    async def _evaluate_single_query(
        self, 
        query: EvaluationQuery, 
        config: SystemConfiguration
    ) -> BatchEvaluationResult:
        """Evaluate a single query against a configuration"""
        
        start_time = datetime.utcnow()
        
        try:
            # This is a mock implementation - in real usage, this would:
            # 1. Execute the search using the specified configuration
            # 2. Apply reranking if configured
            # 3. Return the results with relevance scores
            
            # Mock search execution
            await asyncio.sleep(0.1)  # Simulate search latency
            
            # Mock results - in reality, these would come from actual search
            retrieved_docs = [f"doc_{i}" for i in range(10)]
            relevance_scores = {doc_id: 0.8 - (i * 0.1) for i, doc_id in enumerate(retrieved_docs)}
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return BatchEvaluationResult(
                query_id=query.id,
                configuration_id=config.id,
                retrieved_docs=retrieved_docs,
                relevance_scores=relevance_scores,
                execution_time_ms=execution_time,
                metadata={
                    'search_params_used': config.search_params,
                    'rerank_params_used': config.rerank_params
                }
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return BatchEvaluationResult(
                query_id=query.id,
                configuration_id=config.id,
                retrieved_docs=[],
                relevance_scores={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    # Dataset Creation Utilities
    async def create_synthetic_dataset(
        self,
        name: str,
        description: str,
        num_queries: int = 100,
        categories: List[str] = None
    ) -> str:
        """Create a synthetic evaluation dataset"""
        
        dataset_id = str(uuid.uuid4())
        categories = categories or ["technical", "general", "specific", "complex"]
        
        queries = []
        for i in range(num_queries):
            query_id = f"synthetic_{i}"
            category = categories[i % len(categories)]
            
            # Generate synthetic query and expected results
            query = EvaluationQuery(
                id=query_id,
                query=f"Synthetic query {i} in {category} category",
                expected_relevant_docs=[f"doc_{j}" for j in range(3, 8)],
                relevance_scores={f"doc_{j}": 4.0 - (j * 0.5) for j in range(3, 8)},
                category=category,
                difficulty="medium" if i % 3 == 0 else "easy",
                metadata={"synthetic": True, "query_index": i}
            )
            queries.append(query)
        
        dataset = EvaluationDataset(
            id=dataset_id,
            name=name,
            description=description,
            type=DatasetType.SYNTHETIC,
            queries=queries,
            created_at=datetime.utcnow(),
            metadata={"num_queries": num_queries, "categories": categories}
        )
        
        await self.create_dataset(dataset)
        self.logger.info(f"Created synthetic dataset {dataset_id} with {num_queries} queries")
        
        return dataset_id
    
    async def import_from_manual_evaluation(self, session_id: str) -> Optional[str]:
        """Import dataset from manual evaluation session"""
        try:
            # Load manual evaluation session
            session = await self.manual_evaluator.load_session(session_id)
            if not session:
                return None
            
            # Convert manual evaluation to batch dataset
            queries = []
            for i, query_eval in enumerate(session.queries):
                # Extract relevant documents from manual evaluation
                expected_relevant = []
                relevance_scores = {}
                
                for doc_eval in query_eval.document_evaluations:
                    if doc_eval.overall_relevance >= 2:  # Consider relevant if score >= 2
                        expected_relevant.append(doc_eval.document_id)
                        relevance_scores[doc_eval.document_id] = float(doc_eval.overall_relevance)
                
                query = EvaluationQuery(
                    id=f"manual_{i}",
                    query=query_eval.query,
                    expected_relevant_docs=expected_relevant,
                    relevance_scores=relevance_scores,
                    metadata={
                        "imported_from_session": session_id,
                        "manual_evaluator": query_eval.evaluator_id,
                        "evaluation_date": query_eval.evaluated_at.isoformat() if query_eval.evaluated_at else None
                    }
                )
                queries.append(query)
            
            # Create dataset
            dataset_id = str(uuid.uuid4())
            dataset = EvaluationDataset(
                id=dataset_id,
                name=f"Manual Evaluation Import: {session.name}",
                description=f"Dataset imported from manual evaluation session {session_id}",
                type=DatasetType.CUSTOM,
                queries=queries,
                created_at=datetime.utcnow(),
                metadata={
                    "imported_from_session": session_id,
                    "original_session_name": session.name,
                    "import_date": datetime.utcnow().isoformat()
                }
            )
            
            await self.create_dataset(dataset)
            self.logger.info(f"Imported dataset {dataset_id} from manual evaluation session {session_id}")
            
            return dataset_id
            
        except Exception as e:
            self.logger.error(f"Failed to import from manual evaluation session {session_id}: {e}")
            return None

# Global instance
automated_batch_evaluator = AutomatedBatchEvaluator()

# Example usage and testing functions
async def create_example_configuration():
    """Create example system configuration"""
    config = SystemConfiguration(
        id="openai_basic",
        name="OpenAI Basic Configuration",
        description="Basic configuration with OpenAI embeddings",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        search_params={
            "limit": 10,
            "alpha": 0.75
        },
        metadata={"created_for": "testing"}
    )
    
    await automated_batch_evaluator.create_configuration(config)
    return config.id

async def create_example_dataset():
    """Create example evaluation dataset"""
    return await automated_batch_evaluator.create_synthetic_dataset(
        name="Test Dataset",
        description="Example dataset for testing batch evaluation",
        num_queries=20,
        categories=["search", "retrieval", "ranking"]
    )

async def run_example_evaluation():
    """Run an example batch evaluation"""
    # Create configuration and dataset
    config_id = await create_example_configuration()
    dataset_id = await create_example_dataset()
    
    # Create and start evaluation job
    job_id = await automated_batch_evaluator.create_evaluation_job(
        name="Example Evaluation",
        description="Testing batch evaluation system",
        dataset_id=dataset_id,
        configurations=[config_id],
        mode=EvaluationMode.SINGLE
    )
    
    print(f"Created evaluation job: {job_id}")
    
    # Start the job
    success = await automated_batch_evaluator.start_evaluation_job(job_id)
    if success:
        print(f"Started evaluation job: {job_id}")
        
        # Monitor progress
        while True:
            job = await automated_batch_evaluator.load_job(job_id)
            if not job:
                break
                
            print(f"Progress: {job.progress:.1%}, Status: {job.status.value}")
            
            if job.status in [BatchEvaluationStatus.COMPLETED, BatchEvaluationStatus.FAILED, BatchEvaluationStatus.CANCELLED]:
                break
                
            await asyncio.sleep(1)
        
        print(f"Evaluation completed. Results: {len(job.results)}")
    else:
        print("Failed to start evaluation job")

if __name__ == "__main__":
    asyncio.run(run_example_evaluation())