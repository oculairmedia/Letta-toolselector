"""
LDTS-44: Evaluation History Tracking and Reporting
Comprehensive tracking of evaluation runs, results, and performance trends over time
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import numpy as np
from collections import defaultdict

from automated_batch_evaluation import (
    BatchEvaluationJob, BatchEvaluationResult, AutomatedBatchEvaluator
)
from ab_testing_framework import ExperimentExecution, ExperimentResult, ABTestingFramework
from metrics_calculation_engine import MetricResult, AggregateMetricResult, MetricType
from benchmark_query_management import BenchmarkCollection, BenchmarkQueryManager

class EvaluationRunType(Enum):
    SINGLE_CONFIG = "single_config"
    COMPARATIVE = "comparative"
    AB_TEST = "ab_test"
    BENCHMARK = "benchmark"
    REGRESSION = "regression"

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"

@dataclass
class EvaluationRun:
    """Single evaluation run record"""
    id: str
    name: str
    description: str
    
    # Run classification
    run_type: EvaluationRunType
    status: RunStatus
    
    # References to underlying jobs/experiments
    batch_job_ids: List[str] = field(default_factory=list)
    ab_test_execution_id: Optional[str] = None
    benchmark_collection_id: Optional[str] = None
    
    # Configurations tested
    configuration_ids: List[str] = field(default_factory=list)
    primary_configuration_id: Optional[str] = None  # Main config for single runs
    
    # Dataset information
    dataset_id: str = ""
    dataset_name: str = ""
    query_count: int = 0
    
    # Execution details
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Results summary
    primary_metric_type: Optional[MetricType] = None
    primary_metric_k: Optional[int] = None
    primary_metric_value: Optional[float] = None
    
    # Performance metrics
    total_queries_processed: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_query_time_ms: Optional[float] = None
    
    # Metadata and tags
    tags: List[str] = field(default_factory=list)
    environment: str = "development"  # development, staging, production
    version: str = "1.0"
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricTrend:
    """Trend analysis for a specific metric"""
    metric_type: MetricType
    k: Optional[int]
    configuration_id: str
    
    # Trend data
    values: List[float]
    timestamps: List[datetime]
    run_ids: List[str]
    
    # Trend analysis
    trend_direction: TrendDirection
    slope: float  # Linear regression slope
    r_squared: float  # Goodness of fit
    
    # Statistics
    current_value: float
    previous_value: Optional[float] = None
    percent_change: Optional[float] = None
    best_value: float = 0.0
    worst_value: float = 0.0
    avg_value: float = 0.0
    
    # Significance
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceRegression:
    """Detected performance regression"""
    id: str
    
    # Regression details
    metric_type: MetricType
    k: Optional[int]
    configuration_id: str
    
    # Values
    baseline_value: float
    current_value: float
    regression_amount: float  # Absolute difference
    regression_percent: float  # Percentage change
    
    # Detection details
    baseline_run_id: str
    current_run_id: str
    detected_at: datetime
    
    # Severity
    severity: str = "medium"  # low, medium, high, critical
    threshold_exceeded: float = 0.05  # Threshold that was exceeded
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    false_positive: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    id: str
    name: str
    description: str
    
    # Report scope
    run_ids: List[str]
    configuration_ids: List[str]
    date_range_start: datetime
    date_range_end: datetime
    
    # Summary statistics
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Performance summary
    best_configuration: Optional[str] = None
    best_metric_value: Optional[float] = None
    performance_trends: List[MetricTrend] = field(default_factory=list)
    
    # Regressions
    detected_regressions: List[PerformanceRegression] = field(default_factory=list)
    
    # Insights
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EvaluationHistoryTracker:
    """Main class for tracking evaluation history and generating reports"""
    
    def __init__(self, storage_path: str = "evaluation_history"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.runs_path = self.storage_path / "runs"
        self.trends_path = self.storage_path / "trends"
        self.regressions_path = self.storage_path / "regressions"
        self.reports_path = self.storage_path / "reports"
        
        for path in [self.runs_path, self.trends_path, self.regressions_path, self.reports_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Dependencies
        self.batch_evaluator = AutomatedBatchEvaluator()
        self.ab_testing_framework = ABTestingFramework()
        self.benchmark_manager = BenchmarkQueryManager()
        
        # Caching
        self._runs_cache = {}
        self._cache_expiry = timedelta(minutes=10)
        self._last_cache_update = datetime.min
    
    # Run Tracking
    async def record_batch_evaluation_run(
        self,
        batch_job_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: List[str] = None,
        environment: str = "development"
    ) -> str:
        """Record a batch evaluation run"""
        
        try:
            # Load batch job details
            batch_job = await self.batch_evaluator.load_job(batch_job_id)
            if not batch_job:
                raise ValueError(f"Batch job {batch_job_id} not found")
            
            # Load dataset info
            dataset = await self.batch_evaluator.load_dataset(batch_job.dataset_id)
            dataset_name = dataset.name if dataset else "Unknown Dataset"
            query_count = len(dataset.queries) if dataset else 0
            
            # Calculate metrics
            duration = None
            if batch_job.started_at and batch_job.completed_at:
                duration = (batch_job.completed_at - batch_job.started_at).total_seconds()
            
            successful_queries = len([r for r in batch_job.results if not r.error_message])
            failed_queries = len([r for r in batch_job.results if r.error_message])
            
            avg_query_time = None
            if batch_job.results:
                valid_times = [r.execution_time_ms for r in batch_job.results if r.execution_time_ms > 0]
                if valid_times:
                    avg_query_time = np.mean(valid_times)
            
            # Create run record
            run_id = str(uuid.uuid4())
            run = EvaluationRun(
                id=run_id,
                name=name or f"Batch Evaluation: {batch_job.name}",
                description=description or batch_job.description,
                run_type=EvaluationRunType.SINGLE_CONFIG if len(batch_job.configurations) == 1 
                         else EvaluationRunType.COMPARATIVE,
                status=RunStatus.COMPLETED if batch_job.status.value == "completed" else RunStatus.FAILED,
                batch_job_ids=[batch_job_id],
                configuration_ids=batch_job.configurations,
                primary_configuration_id=batch_job.configurations[0] if batch_job.configurations else None,
                dataset_id=batch_job.dataset_id,
                dataset_name=dataset_name,
                query_count=query_count,
                started_at=batch_job.started_at,
                completed_at=batch_job.completed_at,
                duration_seconds=duration,
                total_queries_processed=len(batch_job.results),
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                avg_query_time_ms=avg_query_time,
                tags=tags or [],
                environment=environment
            )
            
            # Save run
            await self._save_run(run)
            
            # Update trends and check for regressions
            await self._update_trends_for_run(run)
            await self._check_for_regressions(run)
            
            self.logger.info(f"Recorded batch evaluation run {run_id}")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to record batch evaluation run {batch_job_id}: {e}")
            raise
    
    async def record_ab_test_run(
        self,
        execution_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: List[str] = None,
        environment: str = "development"
    ) -> str:
        """Record an A/B test run"""
        
        try:
            # Load A/B test execution details
            execution = await self.ab_testing_framework.load_execution(execution_id)
            if not execution:
                raise ValueError(f"A/B test execution {execution_id} not found")
            
            # Calculate metrics from final result
            primary_metric_value = None
            if execution.final_result:
                primary_metric_value = execution.final_result.treatment_mean
            
            duration = None
            if execution.started_at and execution.completed_at:
                duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Create run record
            run_id = str(uuid.uuid4())
            run = EvaluationRun(
                id=run_id,
                name=name or f"A/B Test: {execution.configuration.name}",
                description=description or execution.configuration.description,
                run_type=EvaluationRunType.AB_TEST,
                status=RunStatus.COMPLETED if execution.status.value == "completed" else RunStatus.FAILED,
                ab_test_execution_id=execution_id,
                configuration_ids=[execution.configuration.control_config_id, execution.configuration.treatment_config_id],
                primary_configuration_id=execution.configuration.treatment_config_id,
                dataset_id=execution.dataset_id,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_seconds=duration,
                primary_metric_type=execution.configuration.primary_metric,
                primary_metric_k=execution.configuration.primary_metric_k,
                primary_metric_value=primary_metric_value,
                tags=tags or [],
                environment=environment
            )
            
            # Save run
            await self._save_run(run)
            
            self.logger.info(f"Recorded A/B test run {run_id}")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to record A/B test run {execution_id}: {e}")
            raise
    
    async def _save_run(self, run: EvaluationRun):
        """Save evaluation run to storage"""
        run_path = self.runs_path / f"{run.id}.json"
        with open(run_path, 'w') as f:
            json.dump(asdict(run), f, indent=2, default=str)
    
    async def load_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Load evaluation run from storage"""
        try:
            run_path = self.runs_path / f"{run_id}.json"
            if not run_path.exists():
                return None
            
            with open(run_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings and enums
            if data.get('started_at'):
                data['started_at'] = datetime.fromisoformat(data['started_at'])
            if data.get('completed_at'):
                data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            
            data['run_type'] = EvaluationRunType(data['run_type'])
            data['status'] = RunStatus(data['status'])
            
            if data.get('primary_metric_type'):
                data['primary_metric_type'] = MetricType(data['primary_metric_type'])
            
            return EvaluationRun(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load run {run_id}: {e}")
            return None
    
    # Trend Analysis
    async def _update_trends_for_run(self, run: EvaluationRun):
        """Update metric trends based on new run"""
        
        if not run.primary_configuration_id or not run.primary_metric_type:
            return
        
        # Load existing trend or create new one
        trend_id = f"{run.primary_configuration_id}_{run.primary_metric_type.value}_{run.primary_metric_k or 'none'}"
        trend = await self._load_trend(trend_id)
        
        if not trend:
            trend = MetricTrend(
                metric_type=run.primary_metric_type,
                k=run.primary_metric_k,
                configuration_id=run.primary_configuration_id,
                values=[],
                timestamps=[],
                run_ids=[],
                trend_direction=TrendDirection.UNKNOWN,
                slope=0.0,
                r_squared=0.0,
                current_value=run.primary_metric_value or 0.0
            )
        
        # Add new data point
        if run.primary_metric_value is not None and run.completed_at:
            trend.values.append(run.primary_metric_value)
            trend.timestamps.append(run.completed_at)
            trend.run_ids.append(run.id)
            
            # Update current/previous values
            trend.previous_value = trend.current_value
            trend.current_value = run.primary_metric_value
            
            if trend.previous_value is not None:
                trend.percent_change = ((trend.current_value - trend.previous_value) / trend.previous_value) * 100
        
        # Calculate trend statistics
        if len(trend.values) >= 2:
            trend = self._calculate_trend_statistics(trend)
        
        # Save updated trend
        await self._save_trend(trend_id, trend)
    
    def _calculate_trend_statistics(self, trend: MetricTrend) -> MetricTrend:
        """Calculate trend statistics and direction"""
        
        if len(trend.values) < 2:
            return trend
        
        values = np.array(trend.values)
        timestamps_numeric = np.array([(t - trend.timestamps[0]).total_seconds() 
                                      for t in trend.timestamps])
        
        # Linear regression
        if len(timestamps_numeric) > 1:
            slope, intercept = np.polyfit(timestamps_numeric, values, 1)
            predicted = slope * timestamps_numeric + intercept
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            trend.slope = slope
            trend.r_squared = r_squared
        
        # Basic statistics
        trend.best_value = np.max(values)
        trend.worst_value = np.min(values)
        trend.avg_value = np.mean(values)
        
        # Determine trend direction
        if len(values) >= 3:
            recent_values = values[-3:]  # Look at last 3 values
            if all(recent_values[i] >= recent_values[i-1] for i in range(1, len(recent_values))):
                trend.trend_direction = TrendDirection.IMPROVING
            elif all(recent_values[i] <= recent_values[i-1] for i in range(1, len(recent_values))):
                trend.trend_direction = TrendDirection.DEGRADING
            elif abs(trend.slope) < 0.001:  # Very small slope
                trend.trend_direction = TrendDirection.STABLE
            else:
                trend.trend_direction = TrendDirection.IMPROVING if trend.slope > 0 else TrendDirection.DEGRADING
        
        # Statistical significance (simple t-test on slope)
        if len(values) >= 5:
            try:
                from scipy import stats
                t_stat = trend.slope / (np.std(values) / np.sqrt(len(values)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(values) - 2))
                trend.is_statistically_significant = p_value < 0.05
                trend.p_value = p_value
            except Exception:
                trend.is_statistically_significant = False
        
        trend.calculated_at = datetime.utcnow()
        return trend
    
    async def _load_trend(self, trend_id: str) -> Optional[MetricTrend]:
        """Load metric trend from storage"""
        try:
            trend_path = self.trends_path / f"{trend_id}.json"
            if not trend_path.exists():
                return None
            
            with open(trend_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings and enums
            data['timestamps'] = [datetime.fromisoformat(ts) for ts in data['timestamps']]
            data['calculated_at'] = datetime.fromisoformat(data['calculated_at'])
            data['metric_type'] = MetricType(data['metric_type'])
            data['trend_direction'] = TrendDirection(data['trend_direction'])
            
            return MetricTrend(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load trend {trend_id}: {e}")
            return None
    
    async def _save_trend(self, trend_id: str, trend: MetricTrend):
        """Save metric trend to storage"""
        trend_path = self.trends_path / f"{trend_id}.json"
        with open(trend_path, 'w') as f:
            json.dump(asdict(trend), f, indent=2, default=str)
    
    # Regression Detection
    async def _check_for_regressions(self, run: EvaluationRun):
        """Check for performance regressions in the new run"""
        
        if not run.primary_metric_value or not run.primary_configuration_id:
            return
        
        # Find baseline runs for comparison
        baseline_runs = await self._find_baseline_runs(run)
        
        for baseline_run in baseline_runs:
            if baseline_run.primary_metric_value is None:
                continue
            
            # Calculate regression
            regression_amount = run.primary_metric_value - baseline_run.primary_metric_value
            regression_percent = (regression_amount / baseline_run.primary_metric_value) * 100
            
            # Check if this constitutes a regression (assuming higher is better)
            threshold = 0.05  # 5% regression threshold
            if regression_percent < -threshold * 100:  # Negative change = regression
                
                # Determine severity
                severity = "low"
                if abs(regression_percent) > 20:
                    severity = "critical"
                elif abs(regression_percent) > 10:
                    severity = "high"
                elif abs(regression_percent) > 5:
                    severity = "medium"
                
                # Create regression record
                regression = PerformanceRegression(
                    id=str(uuid.uuid4()),
                    metric_type=run.primary_metric_type,
                    k=run.primary_metric_k,
                    configuration_id=run.primary_configuration_id,
                    baseline_value=baseline_run.primary_metric_value,
                    current_value=run.primary_metric_value,
                    regression_amount=abs(regression_amount),
                    regression_percent=abs(regression_percent),
                    baseline_run_id=baseline_run.id,
                    current_run_id=run.id,
                    detected_at=datetime.utcnow(),
                    severity=severity,
                    threshold_exceeded=abs(regression_percent) / 100
                )
                
                await self._save_regression(regression)
                self.logger.warning(f"Performance regression detected: {regression.id}")
                break  # Only record one regression per run
    
    async def _find_baseline_runs(self, current_run: EvaluationRun, lookback_days: int = 7) -> List[EvaluationRun]:
        """Find suitable baseline runs for regression comparison"""
        
        cutoff_date = current_run.completed_at - timedelta(days=lookback_days) if current_run.completed_at else datetime.utcnow() - timedelta(days=lookback_days)
        
        baseline_runs = []
        
        # Load recent runs with same configuration and metric
        for run_file in self.runs_path.glob("*.json"):
            try:
                run = await self.load_run(run_file.stem)
                if (run and 
                    run.id != current_run.id and
                    run.primary_configuration_id == current_run.primary_configuration_id and
                    run.primary_metric_type == current_run.primary_metric_type and
                    run.primary_metric_k == current_run.primary_metric_k and
                    run.completed_at and run.completed_at >= cutoff_date and
                    run.status == RunStatus.COMPLETED):
                    baseline_runs.append(run)
            except Exception:
                continue
        
        # Sort by completion time (most recent first)
        baseline_runs.sort(key=lambda r: r.completed_at or datetime.min, reverse=True)
        
        # Return up to 3 most recent baseline runs
        return baseline_runs[:3]
    
    async def _save_regression(self, regression: PerformanceRegression):
        """Save performance regression to storage"""
        regression_path = self.regressions_path / f"{regression.id}.json"
        with open(regression_path, 'w') as f:
            json.dump(asdict(regression), f, indent=2, default=str)
    
    # Querying and Reporting
    async def list_runs(
        self,
        configuration_id: Optional[str] = None,
        run_type: Optional[EvaluationRunType] = None,
        environment: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[EvaluationRun]:
        """List evaluation runs with filtering"""
        
        runs = []
        
        for run_file in self.runs_path.glob("*.json"):
            try:
                run = await self.load_run(run_file.stem)
                if not run:
                    continue
                
                # Apply filters
                if configuration_id and configuration_id not in run.configuration_ids:
                    continue
                
                if run_type and run.run_type != run_type:
                    continue
                
                if environment and run.environment != environment:
                    continue
                
                if date_from and run.completed_at and run.completed_at < date_from:
                    continue
                
                if date_to and run.completed_at and run.completed_at > date_to:
                    continue
                
                runs.append(run)
                
                if len(runs) >= limit:
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to load run from {run_file}: {e}")
                continue
        
        return sorted(runs, key=lambda r: r.completed_at or datetime.min, reverse=True)
    
    async def get_trends(self, configuration_id: str, metric_type: MetricType, k: Optional[int] = None) -> Optional[MetricTrend]:
        """Get trend analysis for specific metric"""
        trend_id = f"{configuration_id}_{metric_type.value}_{k or 'none'}"
        return await self._load_trend(trend_id)
    
    async def list_regressions(
        self,
        configuration_id: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged_only: bool = False,
        resolved_only: bool = False
    ) -> List[PerformanceRegression]:
        """List performance regressions with filtering"""
        
        regressions = []
        
        for regression_file in self.regressions_path.glob("*.json"):
            try:
                with open(regression_file, 'r') as f:
                    data = json.load(f)
                
                # Convert datetime and enum
                data['detected_at'] = datetime.fromisoformat(data['detected_at'])
                data['metric_type'] = MetricType(data['metric_type'])
                
                regression = PerformanceRegression(**data)
                
                # Apply filters
                if configuration_id and regression.configuration_id != configuration_id:
                    continue
                
                if severity and regression.severity != severity:
                    continue
                
                if acknowledged_only and not regression.acknowledged:
                    continue
                
                if resolved_only and not regression.resolved:
                    continue
                
                regressions.append(regression)
                
            except Exception as e:
                self.logger.error(f"Failed to load regression from {regression_file}: {e}")
                continue
        
        return sorted(regressions, key=lambda r: r.detected_at, reverse=True)
    
    async def generate_evaluation_report(
        self,
        name: str,
        configuration_ids: List[str],
        date_from: datetime,
        date_to: datetime,
        description: str = ""
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        try:
            # Load runs in date range
            runs = await self.list_runs(
                date_from=date_from,
                date_to=date_to,
                limit=1000
            )
            
            # Filter to specified configurations
            relevant_runs = [r for r in runs if any(config_id in r.configuration_ids for config_id in configuration_ids)]
            
            # Calculate summary statistics
            total_runs = len(relevant_runs)
            successful_runs = len([r for r in relevant_runs if r.status == RunStatus.COMPLETED])
            failed_runs = total_runs - successful_runs
            
            # Find best configuration
            best_config = None
            best_metric_value = None
            
            config_performance = defaultdict(list)
            for run in relevant_runs:
                if run.primary_metric_value is not None and run.primary_configuration_id:
                    config_performance[run.primary_configuration_id].append(run.primary_metric_value)
            
            for config_id, values in config_performance.items():
                avg_value = np.mean(values)
                if best_metric_value is None or avg_value > best_metric_value:
                    best_config = config_id
                    best_metric_value = avg_value
            
            # Load trends for configurations
            performance_trends = []
            for config_id in configuration_ids:
                for metric_type in [MetricType.NDCG_AT_K, MetricType.PRECISION_AT_K, MetricType.MRR]:
                    trend = await self.get_trends(config_id, metric_type, k=5)
                    if trend:
                        performance_trends.append(trend)
            
            # Load regressions in date range
            all_regressions = await self.list_regressions()
            detected_regressions = [r for r in all_regressions 
                                  if date_from <= r.detected_at <= date_to
                                  and r.configuration_id in configuration_ids]
            
            # Generate insights and recommendations
            insights = []
            recommendations = []
            
            if best_config:
                insights.append(f"Best performing configuration: {best_config} with average score of {best_metric_value:.4f}")
            
            if detected_regressions:
                critical_regressions = [r for r in detected_regressions if r.severity == "critical"]
                if critical_regressions:
                    insights.append(f"Found {len(critical_regressions)} critical performance regressions")
                    recommendations.append("Investigate critical regressions immediately")
            
            improving_trends = [t for t in performance_trends if t.trend_direction == TrendDirection.IMPROVING]
            degrading_trends = [t for t in performance_trends if t.trend_direction == TrendDirection.DEGRADING]
            
            if improving_trends:
                insights.append(f"{len(improving_trends)} metrics show improving trends")
            
            if degrading_trends:
                insights.append(f"{len(degrading_trends)} metrics show degrading trends")
                recommendations.append("Review configurations with degrading performance trends")
            
            # Create report
            report_id = str(uuid.uuid4())
            report = EvaluationReport(
                id=report_id,
                name=name,
                description=description,
                run_ids=[r.id for r in relevant_runs],
                configuration_ids=configuration_ids,
                date_range_start=date_from,
                date_range_end=date_to,
                total_runs=total_runs,
                successful_runs=successful_runs,
                failed_runs=failed_runs,
                best_configuration=best_config,
                best_metric_value=best_metric_value,
                performance_trends=performance_trends,
                detected_regressions=detected_regressions,
                key_insights=insights,
                recommendations=recommendations
            )
            
            # Save report
            report_path = self.reports_path / f"{report_id}.json"
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            self.logger.info(f"Generated evaluation report {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation report: {e}")
            raise

# Global instance
evaluation_history = EvaluationHistoryTracker()

# Example usage
async def test_history_tracking():
    """Test the evaluation history tracking system"""
    
    # This would normally be called after actual evaluation runs
    print("Evaluation history tracking system initialized")
    print(f"Storage path: {evaluation_history.storage_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_history_tracking())