"""
LDTS-80: Implement metrics computation service (P@K, NDCG, MRR, MAP)

Comprehensive evaluation metrics computation service for search and ranking evaluation.
Implements industry-standard metrics for information retrieval systems.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Supported evaluation metrics"""
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    F1_AT_K = "f1_at_k"
    NDCG_AT_K = "ndcg_at_k"
    MRR = "mrr"
    MAP = "map"
    DCG_AT_K = "dcg_at_k"
    IDCG_AT_K = "idcg_at_k"
    HIT_RATE_AT_K = "hit_rate_at_k"
    RECIPROCAL_RANK = "reciprocal_rank"

@dataclass
class EvaluationQuery:
    """Single query for evaluation with relevant documents"""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]
    relevant_scores: Optional[List[float]] = None  # Relevance scores (for NDCG)
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResult:
    """Results from ranking system for a query"""
    query_id: str
    ranked_doc_ids: List[str]
    scores: Optional[List[float]] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricResult:
    """Single metric computation result"""
    metric_type: MetricType
    value: float
    k: Optional[int] = None
    query_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationSummary:
    """Summary of evaluation metrics across all queries"""
    total_queries: int
    individual_results: List[Dict[str, MetricResult]]
    aggregate_metrics: Dict[MetricType, float]
    k_values: List[int]
    processing_time: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class EvaluationMetricsService:
    """
    Comprehensive evaluation metrics computation service for search and ranking systems.
    
    Implements standard information retrieval metrics:
    - Precision@K: Fraction of retrieved documents that are relevant
    - Recall@K: Fraction of relevant documents that are retrieved
    - F1@K: Harmonic mean of Precision and Recall
    - NDCG@K: Normalized Discounted Cumulative Gain
    - MRR: Mean Reciprocal Rank
    - MAP: Mean Average Precision
    - Hit Rate@K: Binary indicator of relevant document presence
    """
    
    def __init__(self, default_k_values: List[int] = None):
        """
        Initialize evaluation metrics service
        
        Args:
            default_k_values: Default K values for @K metrics (default: [1, 3, 5, 10, 20])
        """
        self.default_k_values = default_k_values or [1, 3, 5, 10, 20]
        self.computation_cache = {}
        self.statistics = {
            "total_evaluations": 0,
            "total_queries_evaluated": 0,
            "total_computation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Initialized EvaluationMetricsService with K values: {self.default_k_values}")
    
    def evaluate_ranking(
        self,
        queries: List[EvaluationQuery],
        results: List[EvaluationResult],
        k_values: Optional[List[int]] = None,
        metrics: Optional[List[MetricType]] = None,
        cache_results: bool = True
    ) -> EvaluationSummary:
        """
        Evaluate ranking results against ground truth queries
        
        Args:
            queries: List of evaluation queries with ground truth
            results: List of ranking results to evaluate
            k_values: K values for @K metrics (default: self.default_k_values)
            metrics: Specific metrics to compute (default: all)
            cache_results: Whether to cache computation results
            
        Returns:
            EvaluationSummary with individual and aggregate results
        """
        start_time = time.time()
        k_values = k_values or self.default_k_values
        metrics = metrics or list(MetricType)
        
        # Validate input
        self._validate_evaluation_input(queries, results)
        
        # Create query lookup
        query_lookup = {q.query_id: q for q in queries}
        
        individual_results = []
        metric_aggregators = defaultdict(list)
        
        for result in results:
            if result.query_id not in query_lookup:
                logger.warning(f"No ground truth found for query: {result.query_id}")
                continue
            
            query = query_lookup[result.query_id]
            query_metrics = self._compute_query_metrics(
                query, result, k_values, metrics, cache_results
            )
            
            individual_results.append({
                "query_id": result.query_id,
                "metrics": query_metrics
            })
            
            # Aggregate metrics
            for metric_result in query_metrics.values():
                if isinstance(metric_result, dict):
                    for k, mr in metric_result.items():
                        metric_aggregators[f"{mr.metric_type.value}@{k}"].append(mr.value)
                else:
                    metric_aggregators[metric_result.metric_type.value].append(metric_result.value)
        
        # Compute aggregate metrics
        aggregate_metrics = {}
        for metric_name, values in metric_aggregators.items():
            if values:
                aggregate_metrics[metric_name] = np.mean(values)
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.statistics["total_evaluations"] += 1
        self.statistics["total_queries_evaluated"] += len(individual_results)
        self.statistics["total_computation_time"] += processing_time
        
        return EvaluationSummary(
            total_queries=len(individual_results),
            individual_results=individual_results,
            aggregate_metrics=aggregate_metrics,
            k_values=k_values,
            processing_time=processing_time,
            timestamp=time.time(),
            metadata={
                "metrics_computed": [m.value for m in metrics],
                "total_ground_truth_queries": len(queries),
                "total_results": len(results)
            }
        )
    
    def _compute_query_metrics(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        k_values: List[int],
        metrics: List[MetricType],
        cache_results: bool = True
    ) -> Dict[str, Union[MetricResult, Dict[int, MetricResult]]]:
        """Compute all metrics for a single query"""
        
        cache_key = f"{query.query_id}:{hash(tuple(result.ranked_doc_ids))}:{hash(tuple(k_values))}"
        
        if cache_results and cache_key in self.computation_cache:
            self.statistics["cache_hits"] += 1
            return self.computation_cache[cache_key]
        
        self.statistics["cache_misses"] += 1
        
        relevant_set = set(query.relevant_doc_ids)
        ranked_docs = result.ranked_doc_ids
        
        query_metrics = {}
        
        # Compute @K metrics
        if any(m in metrics for m in [MetricType.PRECISION_AT_K, MetricType.RECALL_AT_K, 
                                     MetricType.F1_AT_K, MetricType.NDCG_AT_K, 
                                     MetricType.DCG_AT_K, MetricType.IDCG_AT_K, 
                                     MetricType.HIT_RATE_AT_K]):
            
            for metric_type in [MetricType.PRECISION_AT_K, MetricType.RECALL_AT_K, 
                               MetricType.F1_AT_K, MetricType.NDCG_AT_K, 
                               MetricType.HIT_RATE_AT_K]:
                if metric_type in metrics:
                    query_metrics[metric_type.value] = {}
                    
                    for k in k_values:
                        if metric_type == MetricType.PRECISION_AT_K:
                            value = self._compute_precision_at_k(ranked_docs, relevant_set, k)
                        elif metric_type == MetricType.RECALL_AT_K:
                            value = self._compute_recall_at_k(ranked_docs, relevant_set, k)
                        elif metric_type == MetricType.F1_AT_K:
                            p_k = self._compute_precision_at_k(ranked_docs, relevant_set, k)
                            r_k = self._compute_recall_at_k(ranked_docs, relevant_set, k)
                            value = self._compute_f1_score(p_k, r_k)
                        elif metric_type == MetricType.NDCG_AT_K:
                            value = self._compute_ndcg_at_k(
                                ranked_docs, query.relevant_doc_ids, 
                                query.relevant_scores, k
                            )
                        elif metric_type == MetricType.HIT_RATE_AT_K:
                            value = self._compute_hit_rate_at_k(ranked_docs, relevant_set, k)
                        
                        query_metrics[metric_type.value][k] = MetricResult(
                            metric_type=metric_type,
                            value=value,
                            k=k,
                            query_id=query.query_id
                        )
        
        # Compute non-@K metrics
        if MetricType.MRR in metrics:
            mrr_value = self._compute_mrr(ranked_docs, relevant_set)
            query_metrics[MetricType.MRR.value] = MetricResult(
                metric_type=MetricType.MRR,
                value=mrr_value,
                query_id=query.query_id
            )
        
        if MetricType.MAP in metrics:
            map_value = self._compute_map(ranked_docs, relevant_set)
            query_metrics[MetricType.MAP.value] = MetricResult(
                metric_type=MetricType.MAP,
                value=map_value,
                query_id=query.query_id
            )
        
        if MetricType.RECIPROCAL_RANK in metrics:
            rr_value = self._compute_reciprocal_rank(ranked_docs, relevant_set)
            query_metrics[MetricType.RECIPROCAL_RANK.value] = MetricResult(
                metric_type=MetricType.RECIPROCAL_RANK,
                value=rr_value,
                query_id=query.query_id
            )
        
        if cache_results:
            self.computation_cache[cache_key] = query_metrics
        
        return query_metrics
    
    def _compute_precision_at_k(self, ranked_docs: List[str], relevant_set: Set[str], k: int) -> float:
        """Compute Precision@K"""
        if k <= 0 or not ranked_docs:
            return 0.0
        
        top_k_docs = ranked_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_set)
        
        return relevant_in_top_k / min(k, len(top_k_docs))
    
    def _compute_recall_at_k(self, ranked_docs: List[str], relevant_set: Set[str], k: int) -> float:
        """Compute Recall@K"""
        if k <= 0 or not relevant_set or not ranked_docs:
            return 0.0
        
        top_k_docs = ranked_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_set)
        
        return relevant_in_top_k / len(relevant_set)
    
    def _compute_f1_score(self, precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _compute_hit_rate_at_k(self, ranked_docs: List[str], relevant_set: Set[str], k: int) -> float:
        """Compute Hit Rate@K (binary indicator)"""
        if k <= 0 or not relevant_set or not ranked_docs:
            return 0.0
        
        top_k_docs = ranked_docs[:k]
        return 1.0 if any(doc in relevant_set for doc in top_k_docs) else 0.0
    
    def _compute_dcg_at_k(self, ranked_docs: List[str], relevant_docs: List[str], 
                         relevant_scores: Optional[List[float]], k: int) -> float:
        """Compute Discounted Cumulative Gain@K"""
        if k <= 0 or not ranked_docs:
            return 0.0
        
        # Create relevance score lookup
        if relevant_scores:
            relevance_lookup = {doc: score for doc, score in zip(relevant_docs, relevant_scores)}
        else:
            relevance_lookup = {doc: 1.0 for doc in relevant_docs}
        
        dcg = 0.0
        for i, doc in enumerate(ranked_docs[:k]):
            relevance = relevance_lookup.get(doc, 0.0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 1)
        
        return dcg
    
    def _compute_idcg_at_k(self, relevant_docs: List[str], 
                          relevant_scores: Optional[List[float]], k: int) -> float:
        """Compute Ideal Discounted Cumulative Gain@K"""
        if k <= 0 or not relevant_docs:
            return 0.0
        
        if relevant_scores:
            # Sort by relevance scores (descending)
            sorted_pairs = sorted(zip(relevant_docs, relevant_scores), 
                                key=lambda x: x[1], reverse=True)
            ideal_docs = [doc for doc, _ in sorted_pairs[:k]]
            ideal_scores = [score for _, score in sorted_pairs[:k]]
        else:
            ideal_docs = relevant_docs[:k]
            ideal_scores = [1.0] * len(ideal_docs)
        
        return self._compute_dcg_at_k(ideal_docs, ideal_docs, ideal_scores, k)
    
    def _compute_ndcg_at_k(self, ranked_docs: List[str], relevant_docs: List[str],
                          relevant_scores: Optional[List[float]], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain@K"""
        dcg = self._compute_dcg_at_k(ranked_docs, relevant_docs, relevant_scores, k)
        idcg = self._compute_idcg_at_k(relevant_docs, relevant_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_reciprocal_rank(self, ranked_docs: List[str], relevant_set: Set[str]) -> float:
        """Compute Reciprocal Rank (for single query)"""
        for i, doc in enumerate(ranked_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _compute_mrr(self, ranked_docs: List[str], relevant_set: Set[str]) -> float:
        """Compute Mean Reciprocal Rank (same as RR for single query)"""
        return self._compute_reciprocal_rank(ranked_docs, relevant_set)
    
    def _compute_map(self, ranked_docs: List[str], relevant_set: Set[str]) -> float:
        """Compute Mean Average Precision"""
        if not relevant_set:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(ranked_docs):
            if doc in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_set) if relevant_set else 0.0
    
    def _validate_evaluation_input(self, queries: List[EvaluationQuery], 
                                  results: List[EvaluationResult]):
        """Validate evaluation input parameters"""
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        if not results:
            raise ValueError("Results list cannot be empty")
        
        query_ids = {q.query_id for q in queries}
        result_ids = {r.query_id for r in results}
        
        if not query_ids.intersection(result_ids):
            raise ValueError("No matching query IDs between queries and results")
        
        # Validate individual queries
        for query in queries:
            if not query.query_id:
                raise ValueError("Query ID cannot be empty")
            if not query.relevant_doc_ids:
                raise ValueError(f"Query {query.query_id} must have at least one relevant document")
        
        # Validate individual results
        for result in results:
            if not result.query_id:
                raise ValueError("Result query ID cannot be empty")
            if not result.ranked_doc_ids:
                raise ValueError(f"Result for query {result.query_id} must have ranked documents")
    
    def get_supported_metrics(self) -> Dict[str, str]:
        """Get list of supported metrics with descriptions"""
        return {
            MetricType.PRECISION_AT_K.value: "Fraction of retrieved documents that are relevant",
            MetricType.RECALL_AT_K.value: "Fraction of relevant documents that are retrieved",
            MetricType.F1_AT_K.value: "Harmonic mean of Precision@K and Recall@K",
            MetricType.NDCG_AT_K.value: "Normalized Discounted Cumulative Gain at rank K",
            MetricType.MRR.value: "Mean Reciprocal Rank - average of reciprocal ranks",
            MetricType.MAP.value: "Mean Average Precision across all relevant documents",
            MetricType.HIT_RATE_AT_K.value: "Binary indicator if any relevant document in top K",
            MetricType.RECIPROCAL_RANK.value: "Reciprocal of rank of first relevant document"
        }
    
    def clear_cache(self):
        """Clear computation cache"""
        cache_size = len(self.computation_cache)
        self.computation_cache.clear()
        logger.info(f"Cleared computation cache ({cache_size} entries)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.statistics,
            "cache_size": len(self.computation_cache),
            "cache_hit_rate": (
                self.statistics["cache_hits"] / 
                (self.statistics["cache_hits"] + self.statistics["cache_misses"])
                if (self.statistics["cache_hits"] + self.statistics["cache_misses"]) > 0
                else 0.0
            ),
            "default_k_values": self.default_k_values,
            "supported_metrics": list(self.get_supported_metrics().keys())
        }

# Global evaluation metrics service instance
evaluation_metrics_service = EvaluationMetricsService()