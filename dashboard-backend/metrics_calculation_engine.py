"""
LDTS-41: Metrics Calculation Engine
Implements standard information retrieval metrics including Precision@K, NDCG, MRR, MAP
"""

import math
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from collections import defaultdict

from automated_batch_evaluation import BatchEvaluationResult, EvaluationQuery

class MetricType(Enum):
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k" 
    F1_AT_K = "f1_at_k"
    NDCG = "ndcg"
    NDCG_AT_K = "ndcg_at_k"
    MRR = "mrr"
    MAP = "map"
    HITS_AT_K = "hits_at_k"
    DCG = "dcg"
    DCG_AT_K = "dcg_at_k"
    RECIPROCAL_RANK = "reciprocal_rank"
    SUCCESS_AT_K = "success_at_k"

@dataclass
class MetricResult:
    """Single metric calculation result"""
    metric_type: MetricType
    value: float
    k: Optional[int] = None  # For @K metrics
    query_id: Optional[str] = None
    configuration_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregateMetricResult:
    """Aggregated metric results across multiple queries"""
    metric_type: MetricType
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    count: int
    k: Optional[int] = None
    individual_results: List[MetricResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricConfiguration:
    """Configuration for metric calculation"""
    metrics_to_calculate: List[MetricType]
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    relevance_threshold: float = 2.0  # Minimum relevance score for binary relevance
    use_graded_relevance: bool = True  # Use graded vs binary relevance
    normalize_scores: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCalculationEngine:
    """Main engine for calculating information retrieval metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # Core Metric Calculations
    def precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate Precision@K"""
        if k <= 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    def recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate Recall@K"""
        if not relevant_docs or k <= 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def f1_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate F1@K"""
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def dcg_at_k(
        self,
        retrieved_docs: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """Calculate Discounted Cumulative Gain at K"""
        if k <= 0:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            # DCG formula: rel_i / log2(i + 2)
            dcg += relevance / math.log2(i + 2)
        
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if k <= 0:
            return 0.0
        
        # Calculate DCG
        dcg = self.dcg_at_k(retrieved_docs, relevance_scores, k)
        
        # Calculate IDCG (Ideal DCG)
        # Sort by relevance score in descending order
        sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        ideal_docs = [doc_id for doc_id, _ in sorted_docs[:k]]
        idcg = self.dcg_at_k(ideal_docs, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def reciprocal_rank(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate Reciprocal Rank"""
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def average_precision(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate Average Precision"""
        if not relevant_docs:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant_docs)
    
    def hits_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int
    ) -> float:
        """Calculate Hits@K (binary: 1 if any relevant doc in top-k, 0 otherwise)"""
        if k <= 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0
    
    def success_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int,
        threshold: float = 1.0
    ) -> float:
        """Calculate Success@K (alias for Hits@K)"""
        return self.hits_at_k(retrieved_docs, relevant_docs, k)
    
    # High-level metric calculation methods
    def calculate_single_query_metrics(
        self,
        query: EvaluationQuery,
        result: BatchEvaluationResult,
        config: MetricConfiguration
    ) -> List[MetricResult]:
        """Calculate metrics for a single query-result pair"""
        
        metrics = []
        retrieved_docs = result.retrieved_docs
        relevance_scores = query.relevance_scores
        
        # Determine relevant documents based on threshold
        if config.use_graded_relevance:
            relevant_docs = [doc for doc, score in relevance_scores.items() 
                           if score >= config.relevance_threshold]
        else:
            relevant_docs = list(relevance_scores.keys())
        
        for metric_type in config.metrics_to_calculate:
            try:
                if metric_type == MetricType.PRECISION_AT_K:
                    for k in config.k_values:
                        value = self.precision_at_k(retrieved_docs, relevant_docs, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.RECALL_AT_K:
                    for k in config.k_values:
                        value = self.recall_at_k(retrieved_docs, relevant_docs, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.F1_AT_K:
                    for k in config.k_values:
                        value = self.f1_at_k(retrieved_docs, relevant_docs, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.NDCG_AT_K:
                    for k in config.k_values:
                        value = self.ndcg_at_k(retrieved_docs, relevance_scores, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.DCG_AT_K:
                    for k in config.k_values:
                        value = self.dcg_at_k(retrieved_docs, relevance_scores, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.HITS_AT_K:
                    for k in config.k_values:
                        value = self.hits_at_k(retrieved_docs, relevant_docs, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.SUCCESS_AT_K:
                    for k in config.k_values:
                        value = self.success_at_k(retrieved_docs, relevant_docs, k)
                        metrics.append(MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.id,
                            configuration_id=result.configuration_id
                        ))
                
                elif metric_type == MetricType.MRR:
                    value = self.reciprocal_rank(retrieved_docs, relevant_docs)
                    metrics.append(MetricResult(
                        metric_type=metric_type,
                        value=value,
                        query_id=query.id,
                        configuration_id=result.configuration_id
                    ))
                
                elif metric_type == MetricType.MAP:
                    value = self.average_precision(retrieved_docs, relevant_docs)
                    metrics.append(MetricResult(
                        metric_type=metric_type,
                        value=value,
                        query_id=query.id,
                        configuration_id=result.configuration_id
                    ))
                
                elif metric_type == MetricType.RECIPROCAL_RANK:
                    value = self.reciprocal_rank(retrieved_docs, relevant_docs)
                    metrics.append(MetricResult(
                        metric_type=metric_type,
                        value=value,
                        query_id=query.id,
                        configuration_id=result.configuration_id
                    ))
                
                elif metric_type == MetricType.NDCG:
                    # Full NDCG (no k limit)
                    value = self.ndcg_at_k(retrieved_docs, relevance_scores, len(retrieved_docs))
                    metrics.append(MetricResult(
                        metric_type=metric_type,
                        value=value,
                        query_id=query.id,
                        configuration_id=result.configuration_id
                    ))
                
                elif metric_type == MetricType.DCG:
                    # Full DCG (no k limit)
                    value = self.dcg_at_k(retrieved_docs, relevance_scores, len(retrieved_docs))
                    metrics.append(MetricResult(
                        metric_type=metric_type,
                        value=value,
                        query_id=query.id,
                        configuration_id=result.configuration_id
                    ))
                    
            except Exception as e:
                self.logger.error(f"Failed to calculate {metric_type.value} for query {query.id}: {e}")
                continue
        
        return metrics
    
    def aggregate_metrics(
        self,
        metric_results: List[MetricResult],
        group_by_config: bool = True
    ) -> List[AggregateMetricResult]:
        """Aggregate metric results across queries"""
        
        # Group metrics by type, k value, and optionally configuration
        grouped_metrics = defaultdict(list)
        
        for result in metric_results:
            if group_by_config:
                key = (result.metric_type, result.k, result.configuration_id)
            else:
                key = (result.metric_type, result.k)
            grouped_metrics[key].append(result)
        
        aggregated_results = []
        
        for key, results in grouped_metrics.items():
            if group_by_config:
                metric_type, k, config_id = key
            else:
                metric_type, k = key
                config_id = None
            
            values = [r.value for r in results]
            
            if not values:
                continue
            
            aggregated = AggregateMetricResult(
                metric_type=metric_type,
                mean=np.mean(values),
                median=np.median(values),
                std_dev=np.std(values),
                min_value=np.min(values),
                max_value=np.max(values),
                count=len(values),
                k=k,
                individual_results=results,
                metadata={
                    'configuration_id': config_id if group_by_config else None
                }
            )
            
            aggregated_results.append(aggregated)
        
        return aggregated_results
    
    def calculate_batch_metrics(
        self,
        queries: List[EvaluationQuery],
        results: List[BatchEvaluationResult],
        config: MetricConfiguration
    ) -> Tuple[List[MetricResult], List[AggregateMetricResult]]:
        """Calculate metrics for a batch of query-result pairs"""
        
        all_metrics = []
        
        # Create lookup for results by query_id and config_id
        result_lookup = {}
        for result in results:
            key = (result.query_id, result.configuration_id)
            result_lookup[key] = result
        
        # Calculate metrics for each query-result pair
        for query in queries:
            for result in results:
                if result.query_id == query.id and not result.error_message:
                    metrics = self.calculate_single_query_metrics(query, result, config)
                    all_metrics.extend(metrics)
        
        # Aggregate metrics
        aggregated_metrics = self.aggregate_metrics(all_metrics, group_by_config=True)
        
        return all_metrics, aggregated_metrics
    
    def compare_configurations(
        self,
        queries: List[EvaluationQuery],
        results: List[BatchEvaluationResult],
        config: MetricConfiguration
    ) -> Dict[str, List[AggregateMetricResult]]:
        """Compare metrics across different configurations"""
        
        # Group results by configuration
        config_results = defaultdict(list)
        for result in results:
            if not result.error_message:
                config_results[result.configuration_id].append(result)
        
        comparison_results = {}
        
        for config_id, config_results_list in config_results.items():
            # Filter queries to match available results
            relevant_query_ids = {r.query_id for r in config_results_list}
            relevant_queries = [q for q in queries if q.id in relevant_query_ids]
            
            # Calculate metrics for this configuration
            all_metrics, aggregated_metrics = self.calculate_batch_metrics(
                relevant_queries, config_results_list, config
            )
            
            comparison_results[config_id] = aggregated_metrics
        
        return comparison_results
    
    def calculate_statistical_significance(
        self,
        metrics_a: List[MetricResult],
        metrics_b: List[MetricResult],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Calculate statistical significance between two sets of metrics"""
        
        from scipy import stats
        
        # Extract values for the same metric type and k
        values_a = [m.value for m in metrics_a]
        values_b = [m.value for m in metrics_b]
        
        if len(values_a) == 0 or len(values_b) == 0:
            return {
                'test': 'insufficient_data',
                'p_value': None,
                'is_significant': False,
                'confidence_level': 1 - alpha
            }
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            return {
                'test': 'two_sample_t_test',
                'statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < alpha,
                'confidence_level': 1 - alpha,
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'std_a': np.std(values_a),
                'std_b': np.std(values_b),
                'n_a': len(values_a),
                'n_b': len(values_b)
            }
        except Exception as e:
            return {
                'test': 'failed',
                'error': str(e),
                'is_significant': False,
                'confidence_level': 1 - alpha
            }
    
    def generate_metrics_report(
        self,
        aggregated_metrics: List[AggregateMetricResult],
        include_individual: bool = False
    ) -> Dict[str, Any]:
        """Generate a comprehensive metrics report"""
        
        report = {
            'summary': {
                'total_metrics': len(aggregated_metrics),
                'metric_types': list(set(m.metric_type.value for m in aggregated_metrics)),
                'configurations': list(set(m.metadata.get('configuration_id') for m in aggregated_metrics if m.metadata.get('configuration_id'))),
                'generated_at': datetime.utcnow().isoformat()
            },
            'metrics_by_type': {},
            'top_performers': {},
            'detailed_results': []
        }
        
        # Group by metric type
        by_type = defaultdict(list)
        for metric in aggregated_metrics:
            by_type[metric.metric_type.value].append(metric)
        
        # Process each metric type
        for metric_type, metrics in by_type.items():
            type_summary = {
                'count': len(metrics),
                'best_mean': max(m.mean for m in metrics),
                'worst_mean': min(m.mean for m in metrics),
                'overall_mean': np.mean([m.mean for m in metrics]),
                'by_k_value': {}
            }
            
            # Group by k value
            by_k = defaultdict(list)
            for m in metrics:
                k_key = f"@{m.k}" if m.k else "overall"
                by_k[k_key].append(m)
            
            for k_value, k_metrics in by_k.items():
                type_summary['by_k_value'][k_value] = {
                    'best_mean': max(m.mean for m in k_metrics),
                    'best_configuration': max(k_metrics, key=lambda x: x.mean).metadata.get('configuration_id'),
                    'mean': np.mean([m.mean for m in k_metrics]),
                    'configurations': len(k_metrics)
                }
            
            report['metrics_by_type'][metric_type] = type_summary
        
        # Add detailed results
        for metric in aggregated_metrics:
            detail = {
                'metric_type': metric.metric_type.value,
                'k': metric.k,
                'configuration_id': metric.metadata.get('configuration_id'),
                'mean': metric.mean,
                'median': metric.median,
                'std_dev': metric.std_dev,
                'min_value': metric.min_value,
                'max_value': metric.max_value,
                'count': metric.count
            }
            
            if include_individual:
                detail['individual_results'] = [asdict(r) for r in metric.individual_results]
            
            report['detailed_results'].append(detail)
        
        return report

# Global instance
metrics_engine = MetricsCalculationEngine()

# Example usage and testing functions
async def test_metrics_calculation():
    """Test the metrics calculation engine"""
    
    # Create example data
    query = EvaluationQuery(
        id="test_query_1",
        query="test query",
        expected_relevant_docs=["doc_1", "doc_2", "doc_3"],
        relevance_scores={
            "doc_1": 4.0,
            "doc_2": 3.0,
            "doc_3": 2.0,
            "doc_4": 1.0,
            "doc_5": 0.0
        }
    )
    
    result = BatchEvaluationResult(
        query_id="test_query_1",
        configuration_id="test_config",
        retrieved_docs=["doc_2", "doc_1", "doc_4", "doc_3", "doc_5"],
        relevance_scores={
            "doc_1": 0.9,
            "doc_2": 0.95,
            "doc_3": 0.7,
            "doc_4": 0.6,
            "doc_5": 0.1
        },
        execution_time_ms=100.0
    )
    
    # Configure metrics to calculate
    config = MetricConfiguration(
        metrics_to_calculate=[
            MetricType.PRECISION_AT_K,
            MetricType.RECALL_AT_K,
            MetricType.NDCG_AT_K,
            MetricType.MRR,
            MetricType.MAP
        ],
        k_values=[1, 3, 5],
        relevance_threshold=2.0
    )
    
    # Calculate metrics
    metrics = metrics_engine.calculate_single_query_metrics(query, result, config)
    
    # Display results
    print("Calculated Metrics:")
    for metric in metrics:
        k_str = f"@{metric.k}" if metric.k else ""
        print(f"  {metric.metric_type.value}{k_str}: {metric.value:.4f}")
    
    # Aggregate metrics (with just one query, this will be the same)
    aggregated = metrics_engine.aggregate_metrics(metrics, group_by_config=False)
    
    print("\nAggregated Metrics:")
    for agg in aggregated:
        k_str = f"@{agg.k}" if agg.k else ""
        print(f"  {agg.metric_type.value}{k_str}: mean={agg.mean:.4f}, std={agg.std_dev:.4f}")
    
    # Generate report
    report = metrics_engine.generate_metrics_report(aggregated)
    print(f"\nMetrics Report Generated: {len(report['detailed_results'])} results")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_metrics_calculation())