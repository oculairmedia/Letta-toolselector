"""
Core experiment framework for Letta Tool Selector optimization.

This module provides the base classes and interfaces for running experiments,
including parameter definitions, experiment execution, and result management.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ParameterType(Enum):
    """Types of parameters that can be optimized."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    """Definition of an experiment parameter."""
    name: str
    type: ParameterType
    bounds: Optional[tuple] = None  # (min, max) for continuous/discrete
    choices: Optional[List[Any]] = None  # For categorical/discrete
    default: Optional[Any] = None
    description: Optional[str] = None

    def validate_value(self, value: Any) -> bool:
        """Validate if a value is valid for this parameter."""
        if self.type == ParameterType.CONTINUOUS:
            return isinstance(value, (int, float)) and \
                   (not self.bounds or self.bounds[0] <= value <= self.bounds[1])
        elif self.type == ParameterType.DISCRETE:
            return isinstance(value, int) and \
                   (not self.bounds or self.bounds[0] <= value <= self.bounds[1])
        elif self.type == ParameterType.CATEGORICAL:
            return value in (self.choices or [])
        elif self.type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        return False


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    parameters: List[Parameter]
    objective: str  # Name of the metric to optimize
    maximize: bool = True  # Whether to maximize or minimize the objective
    max_evaluations: int = 100
    max_duration: Optional[int] = None  # Maximum duration in seconds
    parallel_jobs: int = 1
    budget_limit: Optional[float] = None  # Cost budget limit
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExperimentResult:
    """Result of a single experiment evaluation."""
    experiment_id: str
    evaluation_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    cost: Optional[float] = None
    duration: Optional[float] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def objective_value(self) -> Optional[float]:
        """Get the objective value from metrics."""
        # This will be set by the experiment based on config
        return self.metrics.get('_objective')


@dataclass
class ExperimentSummary:
    """Summary of an experiment run."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    config: ExperimentConfig
    results: List[ExperimentResult]
    best_result: Optional[ExperimentResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_cost: float = 0.0
    total_evaluations: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Total experiment duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'status': self.status.value,
            'config': asdict(self.config),
            'best_result': asdict(self.best_result) if self.best_result else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_cost': self.total_cost,
            'total_evaluations': self.total_evaluations,
            'duration': self.duration
        }


class ExperimentBackend(ABC):
    """Abstract base class for experiment backends."""

    @abstractmethod
    async def evaluate(self, parameters: Dict[str, Any]) -> ExperimentResult:
        """Evaluate a set of parameters and return results."""
        pass

    @abstractmethod
    async def setup(self) -> None:
        """Setup the backend before running experiments."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup after experiments."""
        pass


class ExperimentOptimizer(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    def suggest_parameters(
        self,
        config: ExperimentConfig,
        previous_results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Suggest next set of parameters to evaluate."""
        pass

    @abstractmethod
    def is_finished(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult]
    ) -> bool:
        """Check if optimization should stop."""
        pass


class ExperimentManager:
    """Main experiment management class."""

    def __init__(self, results_dir: str = "experiments-engine/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.running_experiments: Dict[str, ExperimentSummary] = {}

    async def run_experiment(
        self,
        config: ExperimentConfig,
        backend: ExperimentBackend,
        optimizer: ExperimentOptimizer,
        progress_callback: Optional[Callable[[ExperimentSummary], None]] = None
    ) -> ExperimentSummary:
        """Run a complete experiment."""
        experiment_id = str(uuid.uuid4())
        
        summary = ExperimentSummary(
            experiment_id=experiment_id,
            name=config.name,
            status=ExperimentStatus.CREATED,
            config=config,
            results=[],
            start_time=datetime.now()
        )
        
        self.running_experiments[experiment_id] = summary
        
        try:
            logger.info(f"Starting experiment: {config.name} ({experiment_id})")
            summary.status = ExperimentStatus.RUNNING
            
            await backend.setup()
            
            while not optimizer.is_finished(config, summary.results):
                # Check budget constraint
                if config.budget_limit and summary.total_cost >= config.budget_limit:
                    logger.info(f"Budget limit reached: {summary.total_cost}")
                    break
                
                # Check time constraint
                if config.max_duration and summary.duration and summary.duration >= config.max_duration:
                    logger.info(f"Time limit reached: {summary.duration}s")
                    break
                
                # Get next parameters to evaluate
                parameters = optimizer.suggest_parameters(config, summary.results)
                
                # Validate parameters
                if not self._validate_parameters(config, parameters):
                    logger.error(f"Invalid parameters: {parameters}")
                    continue
                
                # Evaluate parameters
                try:
                    result = await backend.evaluate(parameters)
                    result.experiment_id = experiment_id
                    
                    # Set objective value
                    if config.objective in result.metrics:
                        result.metrics['_objective'] = result.metrics[config.objective]
                    
                    summary.results.append(result)
                    summary.total_evaluations += 1
                    
                    if result.cost:
                        summary.total_cost += result.cost
                    
                    # Update best result
                    if self._is_better_result(result, summary.best_result, config.maximize):
                        summary.best_result = result
                    
                    logger.info(f"Evaluation {summary.total_evaluations}: {result.metrics}")
                    
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    error_result = ExperimentResult(
                        experiment_id=experiment_id,
                        evaluation_id=str(uuid.uuid4()),
                        parameters=parameters,
                        metrics={},
                        error=str(e)
                    )
                    summary.results.append(error_result)
                
                # Call progress callback
                if progress_callback:
                    progress_callback(summary)
            
            summary.status = ExperimentStatus.COMPLETED
            summary.end_time = datetime.now()
            
            logger.info(f"Experiment completed: {summary.total_evaluations} evaluations, "
                       f"best {config.objective}: {summary.best_result.metrics.get(config.objective) if summary.best_result else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            summary.status = ExperimentStatus.FAILED
            summary.end_time = datetime.now()
            
        finally:
            await backend.cleanup()
            self._save_experiment(summary)
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
        
        return summary

    def _validate_parameters(self, config: ExperimentConfig, parameters: Dict[str, Any]) -> bool:
        """Validate parameter values against configuration."""
        param_dict = {p.name: p for p in config.parameters}
        
        for name, value in parameters.items():
            if name not in param_dict:
                return False
            if not param_dict[name].validate_value(value):
                return False
        
        return True

    def _is_better_result(
        self,
        result: ExperimentResult,
        current_best: Optional[ExperimentResult],
        maximize: bool
    ) -> bool:
        """Check if a result is better than the current best."""
        if current_best is None:
            return result.objective_value is not None
        
        if result.objective_value is None:
            return False
        
        if current_best.objective_value is None:
            return True
        
        if maximize:
            return result.objective_value > current_best.objective_value
        else:
            return result.objective_value < current_best.objective_value

    def _save_experiment(self, summary: ExperimentSummary) -> None:
        """Save experiment results to disk."""
        filename = f"{summary.experiment_id}_{summary.name.replace(' ', '_')}.json"
        filepath = self.results_dir / filename
        
        # Save summary
        with open(filepath, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        # Save detailed results
        results_filename = f"{summary.experiment_id}_results.json"
        results_filepath = self.results_dir / results_filename
        
        results_data = [asdict(result) for result in summary.results]
        with open(results_filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Experiment saved: {filepath}")

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Load experiment from disk."""
        # Find the experiment file
        pattern = f"{experiment_id}_*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            return None
        
        # Load summary
        with open(files[0]) as f:
            data = json.load(f)
        
        # Load detailed results
        results_file = self.results_dir / f"{experiment_id}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results_data = json.load(f)
            # Note: Would need to reconstruct ExperimentResult objects here
        
        # Note: This is a simplified version - full implementation would
        # reconstruct all objects properly
        return None

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        experiments = []
        for filepath in self.results_dir.glob("*.json"):
            if not filepath.name.endswith("_results.json"):
                experiment_id = filepath.stem.split("_")[0]
                experiments.append(experiment_id)
        return experiments

    def get_running_experiments(self) -> Dict[str, ExperimentSummary]:
        """Get currently running experiments."""
        return self.running_experiments.copy()

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        if experiment_id in self.running_experiments:
            self.running_experiments[experiment_id].status = ExperimentStatus.CANCELLED
            return True
        return False