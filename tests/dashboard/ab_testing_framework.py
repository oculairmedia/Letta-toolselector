"""
LDTS-42: A/B Testing Framework with Statistical Significance
Implements comprehensive A/B testing for search and reranking configurations
"""

import asyncio
import json
import uuid
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import numpy as np
from scipy import stats

from automated_batch_evaluation import (
    AutomatedBatchEvaluator, BatchEvaluationJob, SystemConfiguration,
    EvaluationDataset, BatchEvaluationResult
)
from metrics_calculation_engine import (
    MetricsCalculationEngine, MetricResult, AggregateMetricResult,
    MetricConfiguration, MetricType
)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running" 
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"

class HypothesisType(Enum):
    TWO_SIDED = "two_sided"          # A ≠ B
    ONE_SIDED_GREATER = "greater"    # A > B
    ONE_SIDED_LESS = "less"         # A < B

class StatisticalTest(Enum):
    T_TEST = "t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WELCH_T_TEST = "welch_t_test"
    BOOTSTRAP = "bootstrap"

@dataclass
class ExperimentConfiguration:
    """Configuration for A/B test experiment"""
    id: str
    name: str
    description: str
    
    # Configurations being tested
    control_config_id: str  # A (baseline)
    treatment_config_id: str  # B (variant)
    
    # Primary metric for statistical testing
    primary_metric: MetricType
    primary_metric_k: Optional[int] = None
    
    # Secondary metrics to track
    secondary_metrics: List[MetricType] = field(default_factory=list)
    secondary_metric_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level
    beta: float = 0.20   # Power (1 - beta = 0.80)
    minimum_detectable_effect: float = 0.05  # 5% relative improvement
    hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    
    # Experiment design
    traffic_split: float = 0.5  # 50/50 split
    sample_size_per_group: Optional[int] = None  # Auto-calculate if None
    max_duration_days: int = 30
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_checks: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # At what completion %
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ExperimentResult:
    """Results from A/B test experiment"""
    experiment_id: str
    
    # Sample sizes and completion
    control_sample_size: int
    treatment_sample_size: int
    completion_percentage: float
    
    # Primary metric results
    control_mean: float
    treatment_mean: float
    relative_improvement: float  # (treatment - control) / control
    absolute_improvement: float  # treatment - control
    
    # Statistical significance
    p_value: float
    is_statistically_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
    effect_size: float  # Cohen's d
    statistical_power: float
    
    # Test details
    test_statistic: float
    degrees_of_freedom: Optional[float] = None
    test_method: str = ""
    
    # Decision recommendation
    recommendation: str = ""  # launch, no_launch, continue, stop
    confidence_level: float = 0.95
    
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentExecution:
    """Running A/B test experiment"""
    id: str
    configuration: ExperimentConfiguration
    status: ExperimentStatus
    
    # Execution details
    dataset_id: str
    control_batch_job_id: Optional[str] = None
    treatment_batch_job_id: Optional[str] = None
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results tracking
    intermediate_results: List[ExperimentResult] = field(default_factory=list)
    final_result: Optional[ExperimentResult] = None
    
    # Progress tracking
    progress: float = 0.0
    current_sample_sizes: Dict[str, int] = field(default_factory=dict)  # control/treatment counts
    
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ABTestingFramework:
    """Main A/B testing framework"""
    
    def __init__(self, storage_path: str = "ab_testing_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.experiments_path = self.storage_path / "experiments"
        self.executions_path = self.storage_path / "executions"
        self.results_path = self.storage_path / "results"
        
        for path in [self.experiments_path, self.executions_path, self.results_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.active_experiments: Dict[str, asyncio.Task] = {}
        
        # Dependencies
        self.batch_evaluator = AutomatedBatchEvaluator()
        self.metrics_engine = MetricsCalculationEngine()
    
    # Experiment Configuration Management
    async def create_experiment(self, config: ExperimentConfiguration) -> bool:
        """Create a new A/B test experiment configuration"""
        try:
            # Validate configuration
            validation_result = await self._validate_experiment_config(config)
            if not validation_result['valid']:
                self.logger.error(f"Invalid experiment config: {validation_result['errors']}")
                return False
            
            # Calculate sample size if not provided
            if config.sample_size_per_group is None:
                config.sample_size_per_group = self._calculate_sample_size(
                    config.minimum_detectable_effect,
                    config.alpha,
                    config.beta
                )
            
            # Save configuration
            config_path = self.experiments_path / f"{config.id}.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
            
            self.logger.info(f"Created A/B test experiment {config.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment {config.id}: {e}")
            return False
    
    async def _validate_experiment_config(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """Validate experiment configuration"""
        errors = []
        
        # Check if configurations exist
        control_config = await self.batch_evaluator.load_configuration(config.control_config_id)
        if not control_config:
            errors.append(f"Control configuration {config.control_config_id} not found")
        
        treatment_config = await self.batch_evaluator.load_configuration(config.treatment_config_id)
        if not treatment_config:
            errors.append(f"Treatment configuration {config.treatment_config_id} not found")
        
        # Validate statistical parameters
        if not 0 < config.alpha < 1:
            errors.append("Alpha must be between 0 and 1")
        
        if not 0 < config.beta < 1:
            errors.append("Beta must be between 0 and 1")
        
        if not 0 < config.traffic_split < 1:
            errors.append("Traffic split must be between 0 and 1")
        
        if config.minimum_detectable_effect <= 0:
            errors.append("Minimum detectable effect must be positive")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _calculate_sample_size(
        self,
        minimum_detectable_effect: float,
        alpha: float,
        beta: float,
        baseline_rate: float = 0.5  # Assumed baseline metric value
    ) -> int:
        """Calculate required sample size for statistical power"""
        
        # Convert to effect size (Cohen's d)
        # For proportions: d = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        # For means: d = (μ1 - μ2) / σ
        
        # Simplified calculation assuming equal variances
        effect_size = minimum_detectable_effect / baseline_rate
        
        # Use power analysis formula
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(1 - beta)
        
        # Sample size per group
        n = (2 * (z_alpha + z_beta)**2) / (effect_size**2)
        
        # Add 20% buffer for dropouts
        n = int(n * 1.2)
        
        return max(n, 30)  # Minimum 30 samples per group
    
    # Experiment Execution
    async def start_experiment(
        self,
        experiment_id: str,
        dataset_id: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Start an A/B test experiment"""
        
        try:
            # Load experiment configuration
            config = await self.load_experiment_configuration(experiment_id)
            if not config:
                raise ValueError(f"Experiment configuration {experiment_id} not found")
            
            # Verify dataset exists
            dataset = await self.batch_evaluator.load_dataset(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Create execution
            execution_id = str(uuid.uuid4())
            execution = ExperimentExecution(
                id=execution_id,
                configuration=config,
                status=ExperimentStatus.RUNNING,
                dataset_id=dataset_id,
                started_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Start experiment task
            task = asyncio.create_task(self._execute_experiment(execution))
            self.active_experiments[execution_id] = task
            
            # Save execution
            await self._save_execution(execution)
            
            self.logger.info(f"Started A/B test experiment {execution_id} for config {experiment_id}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
            raise
    
    async def _execute_experiment(self, execution: ExperimentExecution):
        """Execute A/B test experiment"""
        
        try:
            config = execution.configuration
            
            self.logger.info(f"Executing A/B test {execution.id}: {config.control_config_id} vs {config.treatment_config_id}")
            
            # Create batch evaluation jobs for both configurations
            control_job_id = await self.batch_evaluator.create_evaluation_job(
                name=f"A/B Test Control: {config.name}",
                description=f"Control group for A/B test {execution.id}",
                dataset_id=execution.dataset_id,
                configurations=[config.control_config_id],
                metadata={'ab_test_execution_id': execution.id, 'group': 'control'}
            )
            
            treatment_job_id = await self.batch_evaluator.create_evaluation_job(
                name=f"A/B Test Treatment: {config.name}",
                description=f"Treatment group for A/B test {execution.id}",
                dataset_id=execution.dataset_id,
                configurations=[config.treatment_config_id],
                metadata={'ab_test_execution_id': execution.id, 'group': 'treatment'}
            )
            
            execution.control_batch_job_id = control_job_id
            execution.treatment_batch_job_id = treatment_job_id
            await self._save_execution(execution)
            
            # Start both batch jobs
            control_success = await self.batch_evaluator.start_evaluation_job(control_job_id)
            treatment_success = await self.batch_evaluator.start_evaluation_job(treatment_job_id)
            
            if not (control_success and treatment_success):
                raise Exception("Failed to start batch evaluation jobs")
            
            # Monitor progress and perform interim analyses
            await self._monitor_experiment_progress(execution)
            
            # Perform final analysis
            final_result = await self._perform_final_analysis(execution)
            execution.final_result = final_result
            execution.status = ExperimentStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.progress = 1.0
            
            self.logger.info(f"Completed A/B test {execution.id}: {final_result.recommendation}")
            
        except Exception as e:
            self.logger.error(f"A/B test execution {execution.id} failed: {e}")
            execution.status = ExperimentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
        
        finally:
            # Save final execution state
            await self._save_execution(execution)
            
            # Remove from active experiments
            if execution.id in self.active_experiments:
                del self.active_experiments[execution.id]
    
    async def _monitor_experiment_progress(self, execution: ExperimentExecution):
        """Monitor experiment progress and perform interim analyses"""
        
        config = execution.configuration
        
        while execution.status == ExperimentStatus.RUNNING:
            # Check job completion status
            control_job = await self.batch_evaluator.load_job(execution.control_batch_job_id)
            treatment_job = await self.batch_evaluator.load_job(execution.treatment_batch_job_id)
            
            if not control_job or not treatment_job:
                break
            
            # Calculate overall progress
            overall_progress = (control_job.progress + treatment_job.progress) / 2
            execution.progress = overall_progress
            
            # Update sample sizes
            execution.current_sample_sizes = {
                'control': len(control_job.results),
                'treatment': len(treatment_job.results)
            }
            
            # Check if both jobs are completed
            if (control_job.status.value in ['completed', 'failed'] and 
                treatment_job.status.value in ['completed', 'failed']):
                break
            
            # Perform interim analysis if enabled and at checkpoint
            if config.enable_early_stopping:
                for checkpoint in config.early_stopping_checks:
                    if overall_progress >= checkpoint and not any(
                        r.completion_percentage >= checkpoint for r in execution.intermediate_results
                    ):
                        interim_result = await self._perform_interim_analysis(
                            execution, control_job.results, treatment_job.results, checkpoint
                        )
                        
                        if interim_result:
                            execution.intermediate_results.append(interim_result)
                            
                            # Check for early stopping
                            if self._should_stop_early(interim_result, config):
                                self.logger.info(f"Early stopping triggered for experiment {execution.id}")
                                await self.stop_experiment(execution.id)
                                return
            
            # Save progress
            await self._save_execution(execution)
            
            # Wait before next check
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _perform_interim_analysis(
        self,
        execution: ExperimentExecution,
        control_results: List[BatchEvaluationResult],
        treatment_results: List[BatchEvaluationResult],
        completion_percentage: float
    ) -> Optional[ExperimentResult]:
        """Perform interim statistical analysis"""
        
        try:
            return await self._calculate_experiment_result(
                execution, control_results, treatment_results, completion_percentage
            )
        except Exception as e:
            self.logger.error(f"Interim analysis failed for experiment {execution.id}: {e}")
            return None
    
    async def _perform_final_analysis(self, execution: ExperimentExecution) -> ExperimentResult:
        """Perform final statistical analysis"""
        
        # Load final results
        control_job = await self.batch_evaluator.load_job(execution.control_batch_job_id)
        treatment_job = await self.batch_evaluator.load_job(execution.treatment_batch_job_id)
        
        return await self._calculate_experiment_result(
            execution, control_job.results, treatment_job.results, 1.0
        )
    
    async def _calculate_experiment_result(
        self,
        execution: ExperimentExecution,
        control_results: List[BatchEvaluationResult],
        treatment_results: List[BatchEvaluationResult],
        completion_percentage: float
    ) -> ExperimentResult:
        """Calculate statistical results for the experiment"""
        
        config = execution.configuration
        
        # Load dataset and calculate metrics
        dataset = await self.batch_evaluator.load_dataset(execution.dataset_id)
        queries = dataset.queries
        
        # Configure metrics calculation
        metric_config = MetricConfiguration(
            metrics_to_calculate=[config.primary_metric] + config.secondary_metrics,
            k_values=[config.primary_metric_k] if config.primary_metric_k else [1, 3, 5, 10],
            use_graded_relevance=True,
            relevance_threshold=2.0
        )
        
        # Calculate metrics for both groups
        control_metrics, _ = self.metrics_engine.calculate_batch_metrics(
            queries, control_results, metric_config
        )
        treatment_metrics, _ = self.metrics_engine.calculate_batch_metrics(
            queries, treatment_results, metric_config
        )
        
        # Extract primary metric values
        primary_control = [
            m.value for m in control_metrics 
            if m.metric_type == config.primary_metric and m.k == config.primary_metric_k
        ]
        primary_treatment = [
            m.value for m in treatment_metrics
            if m.metric_type == config.primary_metric and m.k == config.primary_metric_k
        ]
        
        if not primary_control or not primary_treatment:
            raise ValueError("No primary metric values found")
        
        # Calculate basic statistics
        control_mean = np.mean(primary_control)
        treatment_mean = np.mean(primary_treatment)
        
        relative_improvement = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0
        absolute_improvement = treatment_mean - control_mean
        
        # Perform statistical test
        stat_result = self._perform_statistical_test(
            primary_control, primary_treatment, config.statistical_test, config.hypothesis_type
        )
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(
            primary_control, primary_treatment, config.alpha
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((len(primary_control) - 1) * np.var(primary_control, ddof=1) + 
             (len(primary_treatment) - 1) * np.var(primary_treatment, ddof=1)) / 
            (len(primary_control) + len(primary_treatment) - 2)
        )
        effect_size = absolute_improvement / pooled_std if pooled_std > 0 else 0
        
        # Calculate statistical power
        statistical_power = self._calculate_statistical_power(
            effect_size, len(primary_control), len(primary_treatment), config.alpha
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            stat_result['p_value'], config.alpha, relative_improvement, 
            config.minimum_detectable_effect, statistical_power
        )
        
        return ExperimentResult(
            experiment_id=execution.id,
            control_sample_size=len(primary_control),
            treatment_sample_size=len(primary_treatment),
            completion_percentage=completion_percentage,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            relative_improvement=relative_improvement,
            absolute_improvement=absolute_improvement,
            p_value=stat_result['p_value'],
            is_statistically_significant=stat_result['p_value'] < config.alpha,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            effect_size=effect_size,
            statistical_power=statistical_power,
            test_statistic=stat_result['statistic'],
            degrees_of_freedom=stat_result.get('degrees_of_freedom'),
            test_method=stat_result['method'],
            recommendation=recommendation,
            confidence_level=1 - config.alpha
        )
    
    def _perform_statistical_test(
        self,
        control_values: List[float],
        treatment_values: List[float],
        test_type: StatisticalTest,
        hypothesis_type: HypothesisType
    ) -> Dict[str, Any]:
        """Perform statistical significance test"""
        
        try:
            if test_type == StatisticalTest.T_TEST:
                statistic, p_value = stats.ttest_ind(treatment_values, control_values)
                method = "Independent t-test"
                df = len(control_values) + len(treatment_values) - 2
                
            elif test_type == StatisticalTest.WELCH_T_TEST:
                statistic, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
                method = "Welch's t-test"
                df = None  # Welch's test has complex df calculation
                
            elif test_type == StatisticalTest.MANN_WHITNEY_U:
                statistic, p_value = stats.mannwhitneyu(treatment_values, control_values, alternative='two-sided')
                method = "Mann-Whitney U test"
                df = None
                
            else:
                raise ValueError(f"Unsupported statistical test: {test_type}")
            
            # Adjust p-value for one-sided tests
            if hypothesis_type == HypothesisType.ONE_SIDED_GREATER:
                if statistic > 0:
                    p_value = p_value / 2
                else:
                    p_value = 1 - p_value / 2
            elif hypothesis_type == HypothesisType.ONE_SIDED_LESS:
                if statistic < 0:
                    p_value = p_value / 2
                else:
                    p_value = 1 - p_value / 2
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'method': method,
                'degrees_of_freedom': df
            }
            
        except Exception as e:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'method': f"Failed: {str(e)}",
                'degrees_of_freedom': None
            }
    
    def _calculate_confidence_interval(
        self,
        control_values: List[float],
        treatment_values: List[float],
        alpha: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means"""
        
        try:
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
            
            control_var = np.var(control_values, ddof=1)
            treatment_var = np.var(treatment_values, ddof=1)
            
            n_control = len(control_values)
            n_treatment = len(treatment_values)
            
            # Standard error of the difference
            se_diff = math.sqrt(control_var/n_control + treatment_var/n_treatment)
            
            # Degrees of freedom for Welch's t-test
            df = (control_var/n_control + treatment_var/n_treatment)**2 / (
                (control_var/n_control)**2/(n_control-1) + 
                (treatment_var/n_treatment)**2/(n_treatment-1)
            )
            
            # Critical value
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Confidence interval for the difference (treatment - control)
            diff = treatment_mean - control_mean
            margin_of_error = t_critical * se_diff
            
            return (diff - margin_of_error, diff + margin_of_error)
            
        except Exception:
            return (-float('inf'), float('inf'))
    
    def _calculate_statistical_power(
        self,
        effect_size: float,
        n_control: int,
        n_treatment: int,
        alpha: float
    ) -> float:
        """Calculate statistical power of the test"""
        
        try:
            # Harmonic mean of sample sizes
            n_harmonic = 2 / (1/n_control + 1/n_treatment)
            
            # Non-centrality parameter
            ncp = effect_size * math.sqrt(n_harmonic / 2)
            
            # Critical value
            t_critical = stats.t.ppf(1 - alpha/2, n_control + n_treatment - 2)
            
            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, n_control + n_treatment - 2, ncp) + \
                    stats.nct.cdf(-t_critical, n_control + n_treatment - 2, ncp)
            
            return min(max(power, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Default moderate power
    
    def _generate_recommendation(
        self,
        p_value: float,
        alpha: float,
        relative_improvement: float,
        minimum_detectable_effect: float,
        statistical_power: float
    ) -> str:
        """Generate launch recommendation based on results"""
        
        is_significant = p_value < alpha
        meets_mde = abs(relative_improvement) >= minimum_detectable_effect
        has_adequate_power = statistical_power >= 0.8
        
        if is_significant and relative_improvement > 0 and meets_mde:
            return "launch"  # Significant positive result above MDE
        elif is_significant and relative_improvement < -minimum_detectable_effect:
            return "no_launch"  # Significant negative result
        elif not is_significant and has_adequate_power:
            return "no_launch"  # Non-significant with adequate power
        elif not has_adequate_power:
            return "continue"  # Need more data
        else:
            return "no_launch"  # Default conservative decision
    
    def _should_stop_early(self, result: ExperimentResult, config: ExperimentConfiguration) -> bool:
        """Determine if experiment should stop early"""
        
        # Stop if highly significant positive result above MDE
        if (result.p_value < 0.001 and 
            result.relative_improvement > config.minimum_detectable_effect and
            result.statistical_power > 0.9):
            return True
        
        # Stop if highly significant negative result
        if (result.p_value < 0.001 and 
            result.relative_improvement < -config.minimum_detectable_effect and
            result.statistical_power > 0.9):
            return True
        
        return False
    
    # Management Operations
    async def stop_experiment(self, execution_id: str) -> bool:
        """Stop a running experiment"""
        try:
            execution = await self.load_execution(execution_id)
            if not execution:
                return False
            
            # Cancel batch jobs
            if execution.control_batch_job_id:
                await self.batch_evaluator.cancel_evaluation_job(execution.control_batch_job_id)
            if execution.treatment_batch_job_id:
                await self.batch_evaluator.cancel_evaluation_job(execution.treatment_batch_job_id)
            
            # Cancel async task
            if execution_id in self.active_experiments:
                self.active_experiments[execution_id].cancel()
                del self.active_experiments[execution_id]
            
            # Update status
            execution.status = ExperimentStatus.STOPPED
            execution.completed_at = datetime.utcnow()
            await self._save_execution(execution)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop experiment {execution_id}: {e}")
            return False
    
    async def load_experiment_configuration(self, experiment_id: str) -> Optional[ExperimentConfiguration]:
        """Load experiment configuration"""
        try:
            config_path = self.experiments_path / f"{experiment_id}.json"
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime and enums
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['hypothesis_type'] = HypothesisType(data['hypothesis_type'])
            data['statistical_test'] = StatisticalTest(data['statistical_test'])
            data['primary_metric'] = MetricType(data['primary_metric'])
            data['secondary_metrics'] = [MetricType(m) for m in data['secondary_metrics']]
            
            return ExperimentConfiguration(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment configuration {experiment_id}: {e}")
            return None
    
    async def load_execution(self, execution_id: str) -> Optional[ExperimentExecution]:
        """Load experiment execution"""
        try:
            execution_path = self.executions_path / f"{execution_id}.json"
            if not execution_path.exists():
                return None
            
            with open(execution_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime and enums
            if data.get('started_at'):
                data['started_at'] = datetime.fromisoformat(data['started_at'])
            if data.get('completed_at'):
                data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            
            data['status'] = ExperimentStatus(data['status'])
            
            # Reconstruct configuration
            config_data = data['configuration']
            config_data['created_at'] = datetime.fromisoformat(config_data['created_at'])
            config_data['hypothesis_type'] = HypothesisType(config_data['hypothesis_type'])
            config_data['statistical_test'] = StatisticalTest(config_data['statistical_test'])
            config_data['primary_metric'] = MetricType(config_data['primary_metric'])
            config_data['secondary_metrics'] = [MetricType(m) for m in config_data['secondary_metrics']]
            data['configuration'] = ExperimentConfiguration(**config_data)
            
            # Reconstruct results
            if data.get('final_result'):
                result_data = data['final_result']
                result_data['calculated_at'] = datetime.fromisoformat(result_data['calculated_at'])
                data['final_result'] = ExperimentResult(**result_data)
            
            intermediate_results = []
            for result_data in data.get('intermediate_results', []):
                result_data['calculated_at'] = datetime.fromisoformat(result_data['calculated_at'])
                intermediate_results.append(ExperimentResult(**result_data))
            data['intermediate_results'] = intermediate_results
            
            return ExperimentExecution(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load execution {execution_id}: {e}")
            return None
    
    async def _save_execution(self, execution: ExperimentExecution):
        """Save execution state"""
        execution_path = self.executions_path / f"{execution.id}.json"
        with open(execution_path, 'w') as f:
            json.dump(asdict(execution), f, indent=2, default=str)
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiment configurations"""
        experiments = []
        
        for exp_file in self.experiments_path.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                
                experiments.append({
                    'id': data['id'],
                    'name': data['name'],
                    'description': data['description'],
                    'control_config_id': data['control_config_id'],
                    'treatment_config_id': data['treatment_config_id'],
                    'primary_metric': data['primary_metric'],
                    'alpha': data['alpha'],
                    'minimum_detectable_effect': data['minimum_detectable_effect'],
                    'created_at': data['created_at']
                })
            except Exception as e:
                self.logger.error(f"Failed to load experiment info from {exp_file}: {e}")
                continue
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
    
    async def list_executions(self, status_filter: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List experiment executions"""
        executions = []
        
        for exec_file in self.executions_path.glob("*.json"):
            try:
                with open(exec_file, 'r') as f:
                    data = json.load(f)
                
                if status_filter and data['status'] != status_filter.value:
                    continue
                
                executions.append({
                    'id': data['id'],
                    'experiment_name': data['configuration']['name'],
                    'status': data['status'],
                    'progress': data['progress'],
                    'dataset_id': data['dataset_id'],
                    'started_at': data.get('started_at'),
                    'completed_at': data.get('completed_at'),
                    'has_final_result': data.get('final_result') is not None
                })
            except Exception as e:
                self.logger.error(f"Failed to load execution info from {exec_file}: {e}")
                continue
        
        return sorted(executions, key=lambda x: x.get('started_at', ''), reverse=True)

# Global instance
ab_testing_framework = ABTestingFramework()

# Example usage
async def create_example_ab_test():
    """Create example A/B test configuration"""
    
    config = ExperimentConfiguration(
        id=str(uuid.uuid4()),
        name="OpenAI vs Ollama Embedding Test",
        description="Compare OpenAI embeddings against Ollama embeddings for search quality",
        control_config_id="openai_basic",  # Assumes this exists
        treatment_config_id="ollama_basic",  # Assumes this exists  
        primary_metric=MetricType.NDCG_AT_K,
        primary_metric_k=5,
        secondary_metrics=[MetricType.PRECISION_AT_K, MetricType.MAP],
        alpha=0.05,
        beta=0.20,
        minimum_detectable_effect=0.03,  # 3% improvement
        statistical_test=StatisticalTest.T_TEST,
        enable_early_stopping=True
    )
    
    success = await ab_testing_framework.create_experiment(config)
    if success:
        print(f"Created A/B test experiment: {config.id}")
        print(f"Required sample size per group: {config.sample_size_per_group}")
        return config.id
    else:
        print("Failed to create A/B test experiment")
        return None

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_example_ab_test())