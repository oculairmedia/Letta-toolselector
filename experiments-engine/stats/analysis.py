"""
Statistical analysis engine for experiment results.

This module provides comprehensive statistical analysis capabilities including:
- Descriptive statistics and confidence intervals
- Hypothesis testing and significance analysis
- Effect size calculations
- Performance comparisons and rankings
- Bayesian analysis and posterior distributions
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import logging

from ..core.base import ExperimentResult, ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a metric."""
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'iqr': self.iqr,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistic."""
    lower: float
    upper: float
    confidence_level: float
    method: str
    
    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower
    
    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'lower': self.lower,
            'upper': self.upper,
            'width': self.width,
            'confidence_level': self.confidence_level,
            'method': self.method
        }


@dataclass
class HypothesisTest:
    """Result of a hypothesis test."""
    statistic: float
    p_value: float
    critical_value: Optional[float]
    reject_null: bool
    alpha: float
    test_name: str
    effect_size: Optional[float] = None
    power: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'statistic': self.statistic,
            'p_value': self.p_value,
            'critical_value': self.critical_value,
            'reject_null': self.reject_null,
            'alpha': self.alpha,
            'test_name': self.test_name,
            'effect_size': self.effect_size,
            'power': self.power,
            'significance_level': 'significant' if self.reject_null else 'not significant'
        }


@dataclass
class ComparisonResult:
    """Result of comparing two groups."""
    group1_stats: DescriptiveStats
    group2_stats: DescriptiveStats
    difference: float
    difference_ci: ConfidenceInterval
    hypothesis_test: HypothesisTest
    effect_size_cohen_d: Optional[float] = None
    effect_size_hedges_g: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'group1_stats': self.group1_stats.to_dict(),
            'group2_stats': self.group2_stats.to_dict(),
            'difference': self.difference,
            'difference_ci': self.difference_ci.to_dict(),
            'hypothesis_test': self.hypothesis_test.to_dict(),
            'effect_size_cohen_d': self.effect_size_cohen_d,
            'effect_size_hedges_g': self.effect_size_hedges_g
        }


class StatisticalAnalyzer:
    """
    Main statistical analysis engine.
    
    Provides methods for analyzing experiment results with proper
    statistical rigor including confidence intervals, hypothesis tests,
    and effect size calculations.
    """

    def __init__(self, confidence_level: float = 0.95, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.confidence_level = confidence_level
        self.alpha = alpha

    def compute_descriptive_stats(self, values: List[float]) -> DescriptiveStats:
        """Compute comprehensive descriptive statistics."""
        if not values:
            raise ValueError("Cannot compute statistics for empty list")
        
        arr = np.array(values)
        
        return DescriptiveStats(
            count=len(values),
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
            iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            skewness=float(stats.skew(arr)) if len(values) > 2 else 0.0,
            kurtosis=float(stats.kurtosis(arr)) if len(values) > 3 else 0.0
        )

    def compute_confidence_interval(
        self,
        values: List[float],
        confidence_level: Optional[float] = None,
        method: str = "bootstrap"
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for the mean.
        
        Args:
            values: List of values
            confidence_level: Confidence level (uses instance default if None)
            method: Method to use ("bootstrap", "t", "normal")
        """
        if not values:
            raise ValueError("Cannot compute CI for empty list")
        
        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level
        
        arr = np.array(values)
        
        if method == "bootstrap":
            return self._bootstrap_ci(arr, conf_level)
        elif method == "t":
            return self._t_ci(arr, conf_level)
        elif method == "normal":
            return self._normal_ci(arr, conf_level)
        else:
            raise ValueError(f"Unknown CI method: {method}")

    def _bootstrap_ci(self, values: np.ndarray, conf_level: float) -> ConfidenceInterval:
        """Compute bootstrap confidence interval."""
        def mean_stat(x):
            return np.mean(x)
        
        try:
            # Use scipy's bootstrap method
            res = bootstrap(
                (values,),
                mean_stat,
                n_resamples=10000,
                confidence_level=conf_level,
                random_state=42
            )
            
            return ConfidenceInterval(
                lower=float(res.confidence_interval.low),
                upper=float(res.confidence_interval.high),
                confidence_level=conf_level,
                method="bootstrap"
            )
        except Exception as e:
            logger.warning(f"Bootstrap CI failed, falling back to t-distribution: {e}")
            return self._t_ci(values, conf_level)

    def _t_ci(self, values: np.ndarray, conf_level: float) -> ConfidenceInterval:
        """Compute t-distribution confidence interval."""
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of mean
        
        if len(values) <= 1:
            # Can't compute CI with single value
            return ConfidenceInterval(
                lower=mean,
                upper=mean,
                confidence_level=conf_level,
                method="t"
            )
        
        alpha = 1 - conf_level
        df = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin = t_critical * sem
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=conf_level,
            method="t"
        )

    def _normal_ci(self, values: np.ndarray, conf_level: float) -> ConfidenceInterval:
        """Compute normal distribution confidence interval."""
        mean = np.mean(values)
        sem = stats.sem(values)
        
        alpha = 1 - conf_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        margin = z_critical * sem
        
        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=conf_level,
            method="normal"
        )

    def t_test_one_sample(
        self,
        values: List[float],
        expected_mean: float,
        alternative: str = "two-sided"
    ) -> HypothesisTest:
        """
        Perform one-sample t-test.
        
        Args:
            values: Sample values
            expected_mean: Expected population mean under null hypothesis
            alternative: "two-sided", "greater", or "less"
        """
        if len(values) < 2:
            raise ValueError("Need at least 2 values for t-test")
        
        statistic, p_value = stats.ttest_1samp(values, expected_mean, alternative=alternative)
        
        # Calculate critical value
        df = len(values) - 1
        if alternative == "two-sided":
            critical_value = stats.t.ppf(1 - self.alpha/2, df)
        else:
            critical_value = stats.t.ppf(1 - self.alpha, df)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(values) - expected_mean) / np.std(values, ddof=1)
        
        return HypothesisTest(
            statistic=float(statistic),
            p_value=float(p_value),
            critical_value=float(critical_value),
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            test_name="One-sample t-test",
            effect_size=float(effect_size)
        )

    def t_test_independent(
        self,
        group1: List[float],
        group2: List[float],
        equal_var: bool = True,
        alternative: str = "two-sided"
    ) -> HypothesisTest:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group values
            group2: Second group values
            equal_var: Assume equal variances (Welch's t-test if False)
            alternative: "two-sided", "greater", or "less"
        """
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Need at least 2 values in each group for t-test")
        
        statistic, p_value = stats.ttest_ind(
            group1, group2, 
            equal_var=equal_var, 
            alternative=alternative
        )
        
        # Calculate degrees of freedom and critical value
        if equal_var:
            df = len(group1) + len(group2) - 2
        else:
            # Welch's t-test degrees of freedom
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        if alternative == "two-sided":
            critical_value = stats.t.ppf(1 - self.alpha/2, df)
        else:
            critical_value = stats.t.ppf(1 - self.alpha, df)
        
        # Effect size (Cohen's d)
        if equal_var:
            pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                                 (len(group2)-1)*np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
        else:
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return HypothesisTest(
            statistic=float(statistic),
            p_value=float(p_value),
            critical_value=float(critical_value),
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            test_name=f"Independent t-test ({'equal var' if equal_var else 'Welch'})",
            effect_size=float(effect_size)
        )

    def mann_whitney_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = "two-sided"
    ) -> HypothesisTest:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            group1: First group values
            group2: Second group values
            alternative: "two-sided", "greater", or "less"
        """
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, 
            alternative=alternative
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return HypothesisTest(
            statistic=float(statistic),
            p_value=float(p_value),
            critical_value=None,  # Not typically used for Mann-Whitney
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            test_name="Mann-Whitney U test",
            effect_size=float(effect_size)
        )

    def compare_groups(
        self,
        group1: List[float],
        group2: List[float],
        test_type: str = "auto",
        equal_var: bool = True
    ) -> ComparisonResult:
        """
        Comprehensive comparison of two groups.
        
        Args:
            group1: First group values
            group2: Second group values
            test_type: "auto", "t_test", "mann_whitney"
            equal_var: Assume equal variances for t-test
        """
        # Compute descriptive statistics
        stats1 = self.compute_descriptive_stats(group1)
        stats2 = self.compute_descriptive_stats(group2)
        
        # Calculate difference
        difference = stats1.mean - stats2.mean
        
        # Compute confidence interval for difference
        # Using bootstrap for difference in means
        def diff_stat(x, y):
            return np.mean(x) - np.mean(y)
        
        try:
            res = bootstrap(
                (np.array(group1), np.array(group2)),
                diff_stat,
                paired=False,
                n_resamples=10000,
                confidence_level=self.confidence_level,
                random_state=42
            )
            
            difference_ci = ConfidenceInterval(
                lower=float(res.confidence_interval.low),
                upper=float(res.confidence_interval.high),
                confidence_level=self.confidence_level,
                method="bootstrap"
            )
        except Exception as e:
            logger.warning(f"Bootstrap CI for difference failed: {e}")
            # Fallback to normal approximation
            pooled_se = np.sqrt(stats1.std**2/stats1.count + stats2.std**2/stats2.count)
            z_critical = stats.norm.ppf(1 - (1-self.confidence_level)/2)
            margin = z_critical * pooled_se
            
            difference_ci = ConfidenceInterval(
                lower=difference - margin,
                upper=difference + margin,
                confidence_level=self.confidence_level,
                method="normal approximation"
            )
        
        # Choose and perform hypothesis test
        if test_type == "auto":
            # Use Shapiro-Wilk test for normality (if sample size allows)
            if len(group1) >= 3 and len(group2) >= 3:
                _, p1 = stats.shapiro(group1) if len(group1) <= 5000 else (0, 0.05)
                _, p2 = stats.shapiro(group2) if len(group2) <= 5000 else (0, 0.05)
                
                # If both groups appear normal (p > 0.05), use t-test
                if p1 > 0.05 and p2 > 0.05:
                    test_type = "t_test"
                else:
                    test_type = "mann_whitney"
            else:
                test_type = "t_test"  # Default for small samples
        
        if test_type == "t_test":
            hypothesis_test = self.t_test_independent(group1, group2, equal_var=equal_var)
        elif test_type == "mann_whitney":
            hypothesis_test = self.mann_whitney_test(group1, group2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect sizes
        cohens_d = None
        hedges_g = None
        
        if len(group1) > 1 and len(group2) > 1:
            # Cohen's d
            if equal_var:
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                                     (len(group2)-1)*np.var(group2, ddof=1)) / 
                                    (len(group1) + len(group2) - 2))
            else:
                pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
            
            cohens_d = difference / pooled_std
            
            # Hedges' g (bias-corrected)
            df = len(group1) + len(group2) - 2
            correction = 1 - (3 / (4 * df - 1))
            hedges_g = cohens_d * correction
        
        return ComparisonResult(
            group1_stats=stats1,
            group2_stats=stats2,
            difference=difference,
            difference_ci=difference_ci,
            hypothesis_test=hypothesis_test,
            effect_size_cohen_d=cohens_d,
            effect_size_hedges_g=hedges_g
        )

    def analyze_experiment_results(
        self,
        results: List[ExperimentResult],
        metric_name: str,
        group_by_param: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of experiment results.
        
        Args:
            results: List of experiment results
            metric_name: Name of metric to analyze
            group_by_param: Parameter name to group results by
        """
        # Extract metric values
        valid_results = [
            r for r in results 
            if r.metrics and metric_name in r.metrics and r.error is None
        ]
        
        if not valid_results:
            return {'error': f'No valid results found for metric {metric_name}'}
        
        values = [r.metrics[metric_name] for r in valid_results]
        
        analysis = {
            'metric': metric_name,
            'total_results': len(results),
            'valid_results': len(valid_results),
            'descriptive_stats': self.compute_descriptive_stats(values).to_dict(),
            'confidence_interval': self.compute_confidence_interval(values).to_dict()
        }
        
        # Group analysis if requested
        if group_by_param:
            groups = {}
            for result in valid_results:
                if group_by_param in result.parameters:
                    param_value = str(result.parameters[group_by_param])
                    if param_value not in groups:
                        groups[param_value] = []
                    groups[param_value].append(result.metrics[metric_name])
            
            if len(groups) > 1:
                group_analysis = {}
                
                # Descriptive statistics for each group
                for group_name, group_values in groups.items():
                    if group_values:
                        group_analysis[group_name] = {
                            'descriptive_stats': self.compute_descriptive_stats(group_values).to_dict(),
                            'confidence_interval': self.compute_confidence_interval(group_values).to_dict(),
                            'count': len(group_values)
                        }
                
                # Pairwise comparisons
                group_names = list(groups.keys())
                comparisons = {}
                
                for i, name1 in enumerate(group_names):
                    for name2 in group_names[i+1:]:
                        if groups[name1] and groups[name2]:
                            comparison_key = f"{name1}_vs_{name2}"
                            comparison = self.compare_groups(groups[name1], groups[name2])
                            comparisons[comparison_key] = comparison.to_dict()
                
                analysis['group_analysis'] = {
                    'parameter': group_by_param,
                    'groups': group_analysis,
                    'comparisons': comparisons
                }
        
        return analysis

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: Optional[float] = None,
        test_type: str = "two_sample"
    ) -> Dict[str, float]:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level (uses instance alpha if None)
            test_type: "one_sample" or "two_sample"
        """
        alpha_val = alpha or self.alpha
        
        if test_type == "two_sample":
            # Two-sample t-test power
            # Using normal approximation for large samples
            df = 2 * sample_size - 2
            t_critical = stats.t.ppf(1 - alpha_val/2, df)
            
            # Non-centrality parameter
            delta = effect_size * np.sqrt(sample_size / 2)
            
            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_critical, df, delta) + stats.nct.cdf(-t_critical, df, delta)
            
        elif test_type == "one_sample":
            # One-sample t-test power
            df = sample_size - 1
            t_critical = stats.t.ppf(1 - alpha_val/2, df)
            
            delta = effect_size * np.sqrt(sample_size)
            power = 1 - stats.nct.cdf(t_critical, df, delta) + stats.nct.cdf(-t_critical, df, delta)
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'power': float(power),
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha_val,
            'test_type': test_type
        }

    def sample_size_calculation(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: Optional[float] = None,
        test_type: str = "two_sample"
    ) -> Dict[str, Any]:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            test_type: "one_sample" or "two_sample"
        """
        alpha_val = alpha or self.alpha
        
        # Binary search for sample size
        min_n = 2
        max_n = 10000
        
        target_power = power
        tolerance = 0.01
        
        while max_n - min_n > 1:
            mid_n = (min_n + max_n) // 2
            
            power_result = self.power_analysis(
                effect_size=effect_size,
                sample_size=mid_n,
                alpha=alpha_val,
                test_type=test_type
            )
            
            if power_result['power'] < target_power:
                min_n = mid_n
            else:
                max_n = mid_n
        
        final_power = self.power_analysis(
            effect_size=effect_size,
            sample_size=max_n,
            alpha=alpha_val,
            test_type=test_type
        )
        
        return {
            'required_sample_size': max_n,
            'achieved_power': final_power['power'],
            'target_power': target_power,
            'effect_size': effect_size,
            'alpha': alpha_val,
            'test_type': test_type
        }