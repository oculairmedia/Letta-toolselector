"""
LDTS-81: Persist evaluation results and expose retrieval API

Comprehensive evaluation results persistence and retrieval system for storing
and analyzing evaluation metrics over time with advanced querying capabilities.
"""

import logging
import json
import sqlite3
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import time
import aiosqlite
from pathlib import Path
import hashlib
import uuid
from contextlib import asynccontextmanager

from evaluation_metrics import EvaluationSummary, MetricResult, MetricType

logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    """Status of evaluation runs"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class EvaluationRun:
    """Represents a single evaluation run"""
    run_id: str
    run_name: str
    description: str
    status: EvaluationStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_queries: int
    k_values: List[int]
    metrics_computed: List[str]
    configuration: Dict[str, Any]
    processing_time: Optional[float]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]

@dataclass
class StoredEvaluationResult:
    """Stored evaluation result with persistence metadata"""
    result_id: str
    run_id: str
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]
    ranked_doc_ids: List[str]
    metrics: Dict[str, Any]  # Serialized metric results
    scores: Optional[List[float]]
    processing_time: Optional[float]
    created_at: datetime
    metadata: Optional[Dict[str, Any]]

@dataclass
class EvaluationComparison:
    """Comparison between two evaluation runs"""
    comparison_id: str
    run_a_id: str
    run_b_id: str
    run_a_name: str
    run_b_name: str
    metric_improvements: Dict[str, float]
    statistical_significance: Dict[str, bool]
    created_at: datetime
    summary: Dict[str, Any]

class EvaluationPersistenceService:
    """
    Service for persisting and retrieving evaluation results with comprehensive
    storage, querying, and analysis capabilities.
    """
    
    def __init__(self, db_path: str = "evaluation_results.db"):
        """
        Initialize evaluation persistence service
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.statistics = {
            "total_runs": 0,
            "total_results": 0,
            "total_comparisons": 0,
            "db_size_bytes": 0,
            "last_cleanup": None
        }
        
        logger.info(f"Initialized EvaluationPersistenceService with database: {self.db_path}")
    
    async def initialize_database(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create evaluation runs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_queries INTEGER,
                    k_values TEXT,  -- JSON array
                    metrics_computed TEXT,  -- JSON array
                    configuration TEXT,  -- JSON object
                    processing_time REAL,
                    error_message TEXT,
                    metadata TEXT  -- JSON object
                )
            """)
            
            # Create evaluation results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    relevant_doc_ids TEXT,  -- JSON array
                    ranked_doc_ids TEXT,  -- JSON array
                    metrics TEXT NOT NULL,  -- JSON object
                    scores TEXT,  -- JSON array
                    processing_time REAL,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT,  -- JSON object
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
                )
            """)
            
            # Create evaluation comparisons table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    run_a_id TEXT NOT NULL,
                    run_b_id TEXT NOT NULL,
                    run_a_name TEXT NOT NULL,
                    run_b_name TEXT NOT NULL,
                    metric_improvements TEXT,  -- JSON object
                    statistical_significance TEXT,  -- JSON object
                    created_at TIMESTAMP NOT NULL,
                    summary TEXT,  -- JSON object
                    FOREIGN KEY (run_a_id) REFERENCES evaluation_runs (run_id),
                    FOREIGN KEY (run_b_id) REFERENCES evaluation_runs (run_id)
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON evaluation_runs (created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs (status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_results_run_id ON evaluation_results (run_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_results_query_id ON evaluation_results (query_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_created_at ON evaluation_comparisons (created_at)")
            
            await db.commit()
        
        logger.info("Database schema initialized successfully")
        await self._update_statistics()
    
    async def create_evaluation_run(
        self,
        run_name: str,
        description: str = "",
        configuration: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new evaluation run
        
        Args:
            run_name: Human-readable name for the run
            description: Description of the evaluation
            configuration: Configuration used for evaluation
            metadata: Additional metadata
            
        Returns:
            Created run ID
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO evaluation_runs (
                    run_id, run_name, description, status, created_at,
                    configuration, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, run_name, description, EvaluationStatus.PENDING.value,
                created_at.isoformat(),
                json.dumps(configuration or {}),
                json.dumps(metadata or {})
            ))
            await db.commit()
        
        self.statistics["total_runs"] += 1
        logger.info(f"Created evaluation run: {run_id} ({run_name})")
        
        return run_id
    
    async def update_run_status(
        self,
        run_id: str,
        status: EvaluationStatus,
        error_message: Optional[str] = None
    ):
        """Update evaluation run status"""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            
            if status == EvaluationStatus.RUNNING:
                await db.execute("""
                    UPDATE evaluation_runs 
                    SET status = ?, started_at = ?
                    WHERE run_id = ?
                """, (status.value, now, run_id))
            elif status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]:
                await db.execute("""
                    UPDATE evaluation_runs 
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE run_id = ?
                """, (status.value, now, error_message, run_id))
            else:
                await db.execute("""
                    UPDATE evaluation_runs 
                    SET status = ?
                    WHERE run_id = ?
                """, (status.value, run_id))
            
            await db.commit()
        
        logger.info(f"Updated run {run_id} status to {status.value}")
    
    async def store_evaluation_summary(
        self,
        run_id: str,
        summary: EvaluationSummary,
        queries: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store complete evaluation summary with all results
        
        Args:
            run_id: Evaluation run ID
            summary: Evaluation summary from metrics service
            queries: Original queries with ground truth
            results: Original ranking results
            
        Returns:
            List of stored result IDs
        """
        # Create query and result lookups
        query_lookup = {q["query_id"]: q for q in queries}
        result_lookup = {r["query_id"]: r for r in results}
        
        result_ids = []
        created_at = datetime.now(timezone.utc)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Store individual results
            for individual_result in summary.individual_results:
                query_id = individual_result["query_id"]
                query_data = query_lookup.get(query_id, {})
                result_data = result_lookup.get(query_id, {})
                
                result_id = str(uuid.uuid4())
                result_ids.append(result_id)
                
                await db.execute("""
                    INSERT INTO evaluation_results (
                        result_id, run_id, query_id, query_text,
                        relevant_doc_ids, ranked_doc_ids, metrics,
                        scores, processing_time, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id, run_id, query_id,
                    query_data.get("query_text", ""),
                    json.dumps(query_data.get("relevant_doc_ids", [])),
                    json.dumps(result_data.get("ranked_doc_ids", [])),
                    json.dumps(individual_result["metrics"]),
                    json.dumps(result_data.get("scores")),
                    result_data.get("processing_time"),
                    created_at.isoformat(),
                    json.dumps({
                        "query_metadata": query_data.get("metadata"),
                        "result_metadata": result_data.get("metadata")
                    })
                ))
            
            # Update run with summary information
            await db.execute("""
                UPDATE evaluation_runs 
                SET total_queries = ?, k_values = ?, metrics_computed = ?, 
                    processing_time = ?, status = ?
                WHERE run_id = ?
            """, (
                summary.total_queries,
                json.dumps(summary.k_values),
                json.dumps([m for m in summary.aggregate_metrics.keys()]),
                summary.processing_time,
                EvaluationStatus.COMPLETED.value,
                run_id
            ))
            
            await db.commit()
        
        self.statistics["total_results"] += len(result_ids)
        logger.info(f"Stored {len(result_ids)} evaluation results for run {run_id}")
        
        return result_ids
    
    async def get_evaluation_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get evaluation run by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM evaluation_runs WHERE run_id = ?
            """, (run_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return EvaluationRun(
                    run_id=row["run_id"],
                    run_name=row["run_name"],
                    description=row["description"] or "",
                    status=EvaluationStatus(row["status"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    total_queries=row["total_queries"] or 0,
                    k_values=json.loads(row["k_values"]) if row["k_values"] else [],
                    metrics_computed=json.loads(row["metrics_computed"]) if row["metrics_computed"] else [],
                    configuration=json.loads(row["configuration"]) if row["configuration"] else {},
                    processing_time=row["processing_time"],
                    error_message=row["error_message"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
    
    async def list_evaluation_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[EvaluationStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[EvaluationRun], int]:
        """
        List evaluation runs with filtering and pagination
        
        Returns:
            Tuple of (runs, total_count)
        """
        where_clauses = []
        params = []
        
        if status_filter:
            where_clauses.append("status = ?")
            params.append(status_filter.value)
        
        if start_date:
            where_clauses.append("created_at >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            where_clauses.append("created_at <= ?")
            params.append(end_date.isoformat())
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get total count
            async with db.execute(f"""
                SELECT COUNT(*) as count FROM evaluation_runs {where_clause}
            """, params) as cursor:
                total_count = (await cursor.fetchone())["count"]
            
            # Get paginated results
            async with db.execute(f"""
                SELECT * FROM evaluation_runs {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset]) as cursor:
                rows = await cursor.fetchall()
                
                runs = []
                for row in rows:
                    runs.append(EvaluationRun(
                        run_id=row["run_id"],
                        run_name=row["run_name"],
                        description=row["description"] or "",
                        status=EvaluationStatus(row["status"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                        total_queries=row["total_queries"] or 0,
                        k_values=json.loads(row["k_values"]) if row["k_values"] else [],
                        metrics_computed=json.loads(row["metrics_computed"]) if row["metrics_computed"] else [],
                        configuration=json.loads(row["configuration"]) if row["configuration"] else {},
                        processing_time=row["processing_time"],
                        error_message=row["error_message"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                    ))
                
                return runs, total_count
    
    async def get_run_results(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
        query_filter: Optional[str] = None
    ) -> Tuple[List[StoredEvaluationResult], int]:
        """Get results for a specific evaluation run"""
        where_clauses = ["run_id = ?"]
        params = [run_id]
        
        if query_filter:
            where_clauses.append("(query_id LIKE ? OR query_text LIKE ?)")
            params.extend([f"%{query_filter}%", f"%{query_filter}%"])
        
        where_clause = "WHERE " + " AND ".join(where_clauses)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get total count
            async with db.execute(f"""
                SELECT COUNT(*) as count FROM evaluation_results {where_clause}
            """, params) as cursor:
                total_count = (await cursor.fetchone())["count"]
            
            # Get paginated results
            async with db.execute(f"""
                SELECT * FROM evaluation_results {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset]) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append(StoredEvaluationResult(
                        result_id=row["result_id"],
                        run_id=row["run_id"],
                        query_id=row["query_id"],
                        query_text=row["query_text"],
                        relevant_doc_ids=json.loads(row["relevant_doc_ids"]) if row["relevant_doc_ids"] else [],
                        ranked_doc_ids=json.loads(row["ranked_doc_ids"]) if row["ranked_doc_ids"] else [],
                        metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                        scores=json.loads(row["scores"]) if row["scores"] else None,
                        processing_time=row["processing_time"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                    ))
                
                return results, total_count
    
    async def get_run_aggregate_metrics(self, run_id: str) -> Dict[str, float]:
        """Get aggregate metrics for a run by computing from stored results"""
        results, _ = await self.get_run_results(run_id, limit=10000)  # Get all results
        
        if not results:
            return {}
        
        # Aggregate metrics across all results
        metric_sums = {}
        metric_counts = {}
        
        for result in results:
            for metric_name, metric_data in result.metrics.items():
                if isinstance(metric_data, dict):
                    # Handle @K metrics
                    for k, metric_result in metric_data.items():
                        metric_key = f"{metric_name}@{k}"
                        value = metric_result.get("value", 0) if isinstance(metric_result, dict) else metric_result
                        
                        if metric_key not in metric_sums:
                            metric_sums[metric_key] = 0
                            metric_counts[metric_key] = 0
                        
                        metric_sums[metric_key] += value
                        metric_counts[metric_key] += 1
                else:
                    # Handle non-@K metrics
                    value = metric_data.get("value", 0) if isinstance(metric_data, dict) else metric_data
                    
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0
                        metric_counts[metric_name] = 0
                    
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1
        
        # Compute averages
        aggregate_metrics = {}
        for metric_name in metric_sums:
            if metric_counts[metric_name] > 0:
                aggregate_metrics[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]
        
        return aggregate_metrics
    
    async def compare_runs(
        self,
        run_a_id: str,
        run_b_id: str,
        store_comparison: bool = True
    ) -> EvaluationComparison:
        """Compare two evaluation runs"""
        # Get run information
        run_a = await self.get_evaluation_run(run_a_id)
        run_b = await self.get_evaluation_run(run_b_id)
        
        if not run_a or not run_b:
            raise ValueError("One or both runs not found")
        
        # Get aggregate metrics
        metrics_a = await self.get_run_aggregate_metrics(run_a_id)
        metrics_b = await self.get_run_aggregate_metrics(run_b_id)
        
        # Compute improvements
        metric_improvements = {}
        statistical_significance = {}
        
        for metric_name in set(metrics_a.keys()) | set(metrics_b.keys()):
            if metric_name in metrics_a and metric_name in metrics_b:
                a_value = metrics_a[metric_name]
                b_value = metrics_b[metric_name]
                
                if a_value != 0:
                    improvement = (b_value - a_value) / a_value * 100
                else:
                    improvement = 0.0
                
                metric_improvements[metric_name] = improvement
                
                # Simplified significance test (would need actual statistical testing)
                statistical_significance[metric_name] = abs(improvement) > 5.0  # 5% threshold
        
        comparison = EvaluationComparison(
            comparison_id=str(uuid.uuid4()),
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            run_a_name=run_a.run_name,
            run_b_name=run_b.run_name,
            metric_improvements=metric_improvements,
            statistical_significance=statistical_significance,
            created_at=datetime.now(timezone.utc),
            summary={
                "run_a_metrics": metrics_a,
                "run_b_metrics": metrics_b,
                "total_queries_a": run_a.total_queries,
                "total_queries_b": run_b.total_queries
            }
        )
        
        if store_comparison:
            await self._store_comparison(comparison)
        
        return comparison
    
    async def _store_comparison(self, comparison: EvaluationComparison):
        """Store comparison in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO evaluation_comparisons (
                    comparison_id, run_a_id, run_b_id, run_a_name, run_b_name,
                    metric_improvements, statistical_significance, created_at, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.comparison_id,
                comparison.run_a_id,
                comparison.run_b_id,
                comparison.run_a_name,
                comparison.run_b_name,
                json.dumps(comparison.metric_improvements),
                json.dumps(comparison.statistical_significance),
                comparison.created_at.isoformat(),
                json.dumps(comparison.summary)
            ))
            await db.commit()
        
        self.statistics["total_comparisons"] += 1
        logger.info(f"Stored comparison: {comparison.comparison_id}")
    
    async def delete_run(self, run_id: str) -> bool:
        """Delete evaluation run and all associated results"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete results first (foreign key constraint)
            await db.execute("DELETE FROM evaluation_results WHERE run_id = ?", (run_id,))
            
            # Delete comparisons involving this run
            await db.execute("""
                DELETE FROM evaluation_comparisons 
                WHERE run_a_id = ? OR run_b_id = ?
            """, (run_id, run_id))
            
            # Delete the run
            cursor = await db.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
            deleted = cursor.rowcount > 0
            
            await db.commit()
        
        if deleted:
            logger.info(f"Deleted evaluation run: {run_id}")
            await self._update_statistics()
        
        return deleted
    
    async def cleanup_old_runs(self, keep_days: int = 30) -> int:
        """Clean up evaluation runs older than specified days"""
        cutoff_date = datetime.now(timezone.utc).replace(days=-keep_days)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get runs to delete
            async with db.execute("""
                SELECT run_id FROM evaluation_runs 
                WHERE created_at < ? AND status IN (?, ?)
            """, (cutoff_date.isoformat(), EvaluationStatus.COMPLETED.value, EvaluationStatus.FAILED.value)) as cursor:
                run_ids = [row[0] for row in await cursor.fetchall()]
            
            if not run_ids:
                return 0
            
            # Delete associated data
            placeholders = ",".join("?" * len(run_ids))
            
            await db.execute(f"DELETE FROM evaluation_results WHERE run_id IN ({placeholders})", run_ids)
            await db.execute(f"DELETE FROM evaluation_comparisons WHERE run_a_id IN ({placeholders}) OR run_b_id IN ({placeholders})", run_ids + run_ids)
            await db.execute(f"DELETE FROM evaluation_runs WHERE run_id IN ({placeholders})", run_ids)
            
            await db.commit()
        
        self.statistics["last_cleanup"] = datetime.now(timezone.utc).isoformat()
        await self._update_statistics()
        
        logger.info(f"Cleaned up {len(run_ids)} old evaluation runs")
        return len(run_ids)
    
    async def _update_statistics(self):
        """Update service statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Count runs
            async with db.execute("SELECT COUNT(*) as count FROM evaluation_runs") as cursor:
                self.statistics["total_runs"] = (await cursor.fetchone())[0]
            
            # Count results
            async with db.execute("SELECT COUNT(*) as count FROM evaluation_results") as cursor:
                self.statistics["total_results"] = (await cursor.fetchone())[0]
            
            # Count comparisons
            async with db.execute("SELECT COUNT(*) as count FROM evaluation_comparisons") as cursor:
                self.statistics["total_comparisons"] = (await cursor.fetchone())[0]
        
        # Get database size
        if self.db_path.exists():
            self.statistics["db_size_bytes"] = self.db_path.stat().st_size
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        await self._update_statistics()
        return {
            **self.statistics,
            "db_path": str(self.db_path),
            "db_size_mb": round(self.statistics["db_size_bytes"] / 1024 / 1024, 2)
        }

# Global evaluation persistence service instance
evaluation_persistence_service = EvaluationPersistenceService()