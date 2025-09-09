"""
Grid search optimization strategy for hyperparameter tuning.

This module implements comprehensive grid search with support for:
- Full factorial parameter sweeps
- Partial grid search with sampling
- Smart parameter space reduction
- Early stopping based on performance
"""

import itertools
import random
from typing import Any, Dict, List, Optional, Set
import numpy as np
import logging

from ..core.base import (
    ExperimentConfig, ExperimentResult, ExperimentOptimizer,
    Parameter, ParameterType
)

logger = logging.getLogger(__name__)


class GridSearchOptimizer(ExperimentOptimizer):
    """
    Grid search optimizer that exhaustively searches parameter space.
    
    Features:
    - Full factorial design
    - Random sampling for large spaces
    - Smart space reduction based on early results
    - Resume capability for interrupted experiments
    """

    def __init__(
        self,
        max_combinations: Optional[int] = None,
        random_sample: bool = False,
        early_stopping: bool = False,
        early_stopping_rounds: int = 10,
        early_stopping_threshold: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            max_combinations: Maximum number of parameter combinations to try
            random_sample: If True, randomly sample from grid instead of exhaustive search
            early_stopping: Enable early stopping based on performance plateau
            early_stopping_rounds: Number of rounds without improvement before stopping
            early_stopping_threshold: Minimum improvement threshold for early stopping
            seed: Random seed for reproducibility
        """
        self.max_combinations = max_combinations
        self.random_sample = random_sample
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_threshold = early_stopping_threshold
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._parameter_grid: List[Dict[str, Any]] = []
        self._current_index = 0
        self._tried_combinations: Set[str] = set()
        self._rounds_without_improvement = 0
        self._best_objective_value: Optional[float] = None

    def _generate_parameter_grid(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if self._parameter_grid:
            return self._parameter_grid
        
        param_values = {}
        
        for param in config.parameters:
            if param.type == ParameterType.CONTINUOUS:
                # For continuous parameters, create a discrete grid
                if param.bounds:
                    min_val, max_val = param.bounds
                    # Default to 10 points if not specified
                    num_points = getattr(param, 'grid_points', 10)
                    values = np.linspace(min_val, max_val, num_points).tolist()
                else:
                    values = [param.default] if param.default is not None else [0.0]
                
            elif param.type == ParameterType.DISCRETE:
                if param.bounds:
                    min_val, max_val = param.bounds
                    values = list(range(int(min_val), int(max_val) + 1))
                elif param.choices:
                    values = param.choices
                else:
                    values = [param.default] if param.default is not None else [0]
                    
            elif param.type == ParameterType.CATEGORICAL:
                values = param.choices if param.choices else [param.default]
                
            elif param.type == ParameterType.BOOLEAN:
                values = [True, False]
            
            param_values[param.name] = values
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        self._parameter_grid = [
            dict(zip(param_names, combination)) 
            for combination in param_combinations
        ]
        
        logger.info(f"Generated grid with {len(self._parameter_grid)} parameter combinations")
        
        # Apply sampling if needed
        if self.max_combinations and len(self._parameter_grid) > self.max_combinations:
            if self.random_sample:
                self._parameter_grid = random.sample(self._parameter_grid, self.max_combinations)
                logger.info(f"Randomly sampled {self.max_combinations} combinations")
            else:
                self._parameter_grid = self._parameter_grid[:self.max_combinations]
                logger.info(f"Truncated to first {self.max_combinations} combinations")
        
        return self._parameter_grid

    def _combination_key(self, parameters: Dict[str, Any]) -> str:
        """Generate a unique key for a parameter combination."""
        # Sort to ensure consistent keys
        items = sorted(parameters.items())
        return str(items)

    def suggest_parameters(
        self,
        config: ExperimentConfig,
        previous_results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Suggest next set of parameters to evaluate."""
        # Generate grid if not already done
        grid = self._generate_parameter_grid(config)
        
        # Update tried combinations from previous results
        for result in previous_results:
            key = self._combination_key(result.parameters)
            self._tried_combinations.add(key)
        
        # Check for early stopping
        if self.early_stopping and self._should_stop_early(config, previous_results):
            raise StopIteration("Early stopping criteria met")
        
        # Find next untried combination
        while self._current_index < len(grid):
            combination = grid[self._current_index]
            key = self._combination_key(combination)
            
            self._current_index += 1
            
            if key not in self._tried_combinations:
                self._tried_combinations.add(key)
                return combination
        
        # If we've tried all combinations, we're done
        raise StopIteration("All parameter combinations have been evaluated")

    def _should_stop_early(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult]
    ) -> bool:
        """Check if early stopping criteria are met."""
        if not results:
            return False
        
        # Get recent results with valid objective values
        recent_results = [
            r for r in results[-self.early_stopping_rounds:] 
            if r.objective_value is not None
        ]
        
        if len(recent_results) < self.early_stopping_rounds:
            return False
        
        # Check if we've seen significant improvement
        recent_values = [r.objective_value for r in recent_results]
        
        if config.maximize:
            best_recent = max(recent_values)
            if self._best_objective_value is None:
                self._best_objective_value = best_recent
                return False
            
            improvement = (best_recent - self._best_objective_value) / abs(self._best_objective_value)
            if improvement > self.early_stopping_threshold:
                self._best_objective_value = best_recent
                self._rounds_without_improvement = 0
                return False
        else:
            best_recent = min(recent_values)
            if self._best_objective_value is None:
                self._best_objective_value = best_recent
                return False
            
            improvement = (self._best_objective_value - best_recent) / abs(self._best_objective_value)
            if improvement > self.early_stopping_threshold:
                self._best_objective_value = best_recent
                self._rounds_without_improvement = 0
                return False
        
        self._rounds_without_improvement += 1
        return self._rounds_without_improvement >= self.early_stopping_rounds

    def is_finished(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult]
    ) -> bool:
        """Check if optimization should stop."""
        # Check if we've hit max evaluations
        if len(results) >= config.max_evaluations:
            return True
        
        # Check if we've tried all combinations
        grid = self._generate_parameter_grid(config)
        tried_count = len([r for r in results if r.error is None])
        
        return tried_count >= len(grid)

    def get_progress(self, config: ExperimentConfig, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Get current optimization progress."""
        grid = self._generate_parameter_grid(config)
        tried_count = len([r for r in results if r.error is None])
        
        return {
            'total_combinations': len(grid),
            'completed_combinations': tried_count,
            'remaining_combinations': len(grid) - tried_count,
            'progress_percent': (tried_count / len(grid)) * 100 if grid else 0,
            'current_index': self._current_index,
            'early_stopping_active': self.early_stopping,
            'rounds_without_improvement': self._rounds_without_improvement
        }

    def get_parameter_importance(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult]
    ) -> Dict[str, float]:
        """Analyze parameter importance based on results."""
        if not results:
            return {}
        
        # Filter valid results
        valid_results = [r for r in results if r.objective_value is not None and not r.error]
        if len(valid_results) < 2:
            return {}
        
        importance = {}
        
        for param in config.parameters:
            param_name = param.name
            
            # Group results by parameter value
            value_groups = {}
            for result in valid_results:
                if param_name in result.parameters:
                    value = result.parameters[param_name]
                    if value not in value_groups:
                        value_groups[value] = []
                    value_groups[value].append(result.objective_value)
            
            # Calculate variance between groups vs within groups
            if len(value_groups) > 1:
                all_values = [r.objective_value for r in valid_results]
                overall_mean = np.mean(all_values)
                
                # Between-group variance
                between_var = 0
                total_count = 0
                for value, group_values in value_groups.items():
                    group_mean = np.mean(group_values)
                    group_count = len(group_values)
                    between_var += group_count * (group_mean - overall_mean) ** 2
                    total_count += group_count
                
                between_var /= (len(value_groups) - 1) if len(value_groups) > 1 else 1
                
                # Within-group variance
                within_var = 0
                for value, group_values in value_groups.items():
                    if len(group_values) > 1:
                        group_var = np.var(group_values)
                        within_var += (len(group_values) - 1) * group_var
                
                within_var /= max(1, total_count - len(value_groups))
                
                # F-statistic as importance measure
                if within_var > 0:
                    f_stat = between_var / within_var
                    importance[param_name] = f_stat
                else:
                    importance[param_name] = float('inf')
            else:
                importance[param_name] = 0.0
        
        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}
        
        return importance


class AdaptiveGridSearchOptimizer(GridSearchOptimizer):
    """
    Advanced grid search that adapts based on early results.
    
    Features:
    - Focuses search on promising regions
    - Dynamically adjusts grid resolution
    - Prunes unpromising parameter ranges
    """

    def __init__(
        self,
        initial_resolution: int = 5,
        refinement_factor: int = 2,
        top_percent: float = 0.2,
        **kwargs
    ):
        """
        Initialize adaptive grid search.
        
        Args:
            initial_resolution: Initial number of points per parameter
            refinement_factor: Factor to increase resolution in promising regions
            top_percent: Percentage of top results to focus refinement on
        """
        super().__init__(**kwargs)
        self.initial_resolution = initial_resolution
        self.refinement_factor = refinement_factor
        self.top_percent = top_percent
        self._refinement_stage = 0
        self._promising_regions: Dict[str, List[Any]] = {}

    def _generate_parameter_grid(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate adaptive parameter grid based on current stage."""
        if self._refinement_stage == 0:
            # Initial coarse grid
            return self._generate_initial_grid(config)
        else:
            # Refined grid focusing on promising regions
            return self._generate_refined_grid(config)

    def _generate_initial_grid(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate initial coarse grid."""
        param_values = {}
        
        for param in config.parameters:
            if param.type == ParameterType.CONTINUOUS:
                if param.bounds:
                    min_val, max_val = param.bounds
                    values = np.linspace(min_val, max_val, self.initial_resolution).tolist()
                else:
                    values = [param.default] if param.default is not None else [0.0]
                    
            elif param.type == ParameterType.DISCRETE:
                if param.bounds:
                    min_val, max_val = param.bounds
                    step = max(1, (max_val - min_val) // self.initial_resolution)
                    values = list(range(int(min_val), int(max_val) + 1, step))
                elif param.choices:
                    # Sample from choices if too many
                    if len(param.choices) > self.initial_resolution:
                        values = random.sample(param.choices, self.initial_resolution)
                    else:
                        values = param.choices
                else:
                    values = [param.default] if param.default is not None else [0]
                    
            else:
                # Categorical and boolean parameters use all values
                if param.type == ParameterType.CATEGORICAL:
                    values = param.choices if param.choices else [param.default]
                else:  # Boolean
                    values = [True, False]
            
            param_values[param.name] = values
        
        # Generate combinations
        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        grid = [dict(zip(param_names, combination)) for combination in combinations]
        
        if self.max_combinations and len(grid) > self.max_combinations:
            grid = random.sample(grid, self.max_combinations)
        
        return grid

    def _generate_refined_grid(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate refined grid focusing on promising regions."""
        if not self._promising_regions:
            return []
        
        param_values = {}
        
        for param in config.parameters:
            param_name = param.name
            
            if param_name in self._promising_regions:
                # Focus on promising values
                promising_values = self._promising_regions[param_name]
                
                if param.type == ParameterType.CONTINUOUS:
                    # Create refined grid around promising values
                    refined_values = []
                    for value in promising_values:
                        if param.bounds:
                            min_val, max_val = param.bounds
                            range_size = (max_val - min_val) / (self.initial_resolution * self.refinement_factor)
                            
                            refined_min = max(min_val, value - range_size / 2)
                            refined_max = min(max_val, value + range_size / 2)
                            
                            refined_values.extend(
                                np.linspace(refined_min, refined_max, self.refinement_factor).tolist()
                            )
                    
                    param_values[param_name] = list(set(refined_values))
                
                else:
                    # For discrete/categorical, use promising values directly
                    param_values[param_name] = promising_values
            
            else:
                # Use original parameter space for non-promising parameters
                if param.type == ParameterType.CONTINUOUS and param.bounds:
                    min_val, max_val = param.bounds
                    param_values[param_name] = np.linspace(min_val, max_val, self.initial_resolution).tolist()
                elif param.type == ParameterType.DISCRETE and param.bounds:
                    min_val, max_val = param.bounds
                    param_values[param_name] = list(range(int(min_val), int(max_val) + 1))
                elif param.choices:
                    param_values[param_name] = param.choices
                else:
                    param_values[param_name] = [param.default] if param.default is not None else [0]
        
        # Generate refined combinations
        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        return [dict(zip(param_names, combination)) for combination in combinations]

    def suggest_parameters(
        self,
        config: ExperimentConfig,
        previous_results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Suggest parameters with adaptive refinement."""
        # Check if we should move to refinement stage
        if (self._refinement_stage == 0 and 
            len(previous_results) >= len(self._generate_initial_grid(config)) * 0.8):
            
            self._identify_promising_regions(config, previous_results)
            self._refinement_stage += 1
            self._current_index = 0  # Reset for refined grid
            self._tried_combinations.clear()
            
            logger.info(f"Moving to refinement stage {self._refinement_stage}")
        
        return super().suggest_parameters(config, previous_results)

    def _identify_promising_regions(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult]
    ) -> None:
        """Identify promising parameter regions based on top results."""
        valid_results = [r for r in results if r.objective_value is not None and not r.error]
        if not valid_results:
            return
        
        # Sort by objective value
        sorted_results = sorted(
            valid_results,
            key=lambda x: x.objective_value,
            reverse=config.maximize
        )
        
        # Take top percentage of results
        top_count = max(1, int(len(sorted_results) * self.top_percent))
        top_results = sorted_results[:top_count]
        
        # Extract promising parameter values
        for param in config.parameters:
            param_name = param.name
            promising_values = [r.parameters[param_name] for r in top_results if param_name in r.parameters]
            
            if promising_values:
                self._promising_regions[param_name] = list(set(promising_values))
        
        logger.info(f"Identified promising regions: {self._promising_regions}")