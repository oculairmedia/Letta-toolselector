"""
Evaluation Blueprint

Provides endpoints for evaluation, analytics, and A/B comparison testing.

Routes:
- POST /api/v1/evaluations - Submit an evaluation
- GET /api/v1/evaluations - Get evaluations
- GET /api/v1/analytics - Get analytics data
- POST /api/v1/rerank/compare - Compare reranker configurations
- POST /api/v1/ab-comparison/run - Run A/B comparison
- GET /api/v1/ab-comparison/results - Get A/B comparison results
- GET /api/v1/ab-comparison/results/<id> - Get specific A/B comparison result
"""

from quart import Blueprint, request, jsonify
import logging
import os
import json
import uuid
import math
import time
from typing import Optional, Callable, List, Dict, Any

logger = logging.getLogger(__name__)

# Create the blueprint
evaluation_bp = Blueprint('evaluation', __name__)

# Module state - injected via configure()
_search_tools_func: Optional[Callable] = None
_bm25_vector_override_service: Optional[Callable] = None
_cache_dir: Optional[str] = None

# Optional aiofiles for async file operations
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    logger.warning("aiofiles not available, using synchronous file operations")


def configure(
    search_tools_func: Optional[Callable] = None,
    bm25_vector_override_service: Optional[Callable] = None,
    cache_dir: Optional[str] = None
):
    """
    Configure the evaluation blueprint with required dependencies.
    
    Args:
        search_tools_func: Function to search tools (from api_server)
        bm25_vector_override_service: BM25/vector override service function
        cache_dir: Directory for storing A/B comparison results
    """
    global _search_tools_func, _bm25_vector_override_service, _cache_dir
    
    _search_tools_func = search_tools_func
    _bm25_vector_override_service = bm25_vector_override_service
    _cache_dir = cache_dir
    
    logger.info("Evaluation blueprint configured")


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_overlap(list_a: List[str], list_b: List[str]) -> float:
    """Calculate the overlap percentage between two lists (Jaccard similarity)."""
    if not list_a or not list_b:
        return 0.0
    set_a = set(list_a)
    set_b = set(list_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / union) * 100 if union > 0 else 0.0


def calculate_rank_correlation(results_a: List[Dict], results_b: List[Dict]) -> float:
    """Calculate Spearman rank correlation between two result sets."""
    try:
        # Create rank mappings
        rank_map_a = {r['tool']['id']: r['rank'] for r in results_a}
        rank_map_b = {r['tool']['id']: r['rank'] for r in results_b}
        
        # Find common tools
        common_tools = set(rank_map_a.keys()) & set(rank_map_b.keys())
        if len(common_tools) < 2:
            return 0.0
        
        # Calculate Spearman correlation
        ranks_a = [rank_map_a[tool_id] for tool_id in common_tools]
        ranks_b = [rank_map_b[tool_id] for tool_id in common_tools]
        
        n = len(ranks_a)
        sum_d_squared = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        correlation = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
        
        return round(correlation, 3)
    except Exception:
        return 0.0


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_median(values: List[float]) -> float:
    """Calculate median of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def calculate_search_metrics(search_results: List[Dict], expected_tools: List[str], k_values: List[int]) -> Dict[str, Any]:
    """Calculate search metrics for a single query."""
    if not search_results:
        return {
            "precision_at_k": {str(k): 0.0 for k in k_values},
            "recall_at_k": {str(k): 0.0 for k in k_values},
            "mrr": 0.0,
            "ndcg": 0.0,
            "total_results": 0
        }
    
    # Extract tool IDs from results
    result_tool_ids = []
    for result in search_results:
        if isinstance(result, dict):
            tool_id = result.get('tool', {}).get('id') or result.get('id')
            if tool_id:
                result_tool_ids.append(tool_id)
    
    expected_set = set(expected_tools)
    metrics = {"total_results": len(result_tool_ids)}
    
    # Calculate precision@K and recall@K
    precision_at_k = {}
    recall_at_k = {}
    
    for k in k_values:
        top_k_results = result_tool_ids[:k]
        relevant_in_top_k = len(set(top_k_results) & expected_set)
        
        precision_at_k[str(k)] = relevant_in_top_k / k if k > 0 else 0.0
        recall_at_k[str(k)] = relevant_in_top_k / len(expected_set) if expected_set else 0.0
    
    metrics["precision_at_k"] = precision_at_k
    metrics["recall_at_k"] = recall_at_k
    
    # Calculate MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, tool_id in enumerate(result_tool_ids):
        if tool_id in expected_set:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr"] = mrr
    
    # Calculate NDCG@10 (simplified)
    dcg = 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_set), 10)))
    
    for i, tool_id in enumerate(result_tool_ids[:10]):
        if tool_id in expected_set:
            dcg += 1.0 / math.log2(i + 2)
    
    metrics["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    
    return metrics


def simplified_significance_test(values_a: List[float], values_b: List[float], confidence_level: float) -> Dict[str, Any]:
    """Simplified significance test when scipy is not available."""
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test_type": "insufficient_data",
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "confidence_interval": [0.0, 0.0]
        }
    
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    
    std_a = calculate_std(values_a)
    std_b = calculate_std(values_b)
    
    # Simple two-sample z-test approximation
    pooled_se = math.sqrt(std_a ** 2 / len(values_a) + std_b ** 2 / len(values_b))
    z_score = abs(mean_b - mean_a) / pooled_se if pooled_se > 0 else 0.0
    
    # Approximate p-value using normal distribution
    p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))))
    
    # Effect size
    pooled_std = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
    effect_size = abs(mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0
    
    alpha = 1 - confidence_level
    significant = p_value < alpha
    
    return {
        "test_type": "simplified_z_test",
        "z_score": z_score,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
        "mean_difference": mean_b - mean_a
    }


def perform_significance_test(values_a: List[float], values_b: List[float], confidence_level: float, min_effect_size: float) -> Dict[str, Any]:
    """Perform statistical significance test between two sets of values."""
    try:
        import scipy.stats as stats
    except ImportError:
        logger.warning("scipy not available, using simplified significance testing")
        return simplified_significance_test(values_a, values_b, confidence_level)
    
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test_type": "insufficient_data",
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "confidence_interval": [0.0, 0.0],
            "power": 0.0
        }
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = math.sqrt(((len(values_a) - 1) * calculate_std(values_a) ** 2 + 
                           (len(values_b) - 1) * calculate_std(values_b) ** 2) / 
                          (len(values_a) + len(values_b) - 2))
    
    effect_size = abs(sum(values_b) / len(values_b) - sum(values_a) / len(values_a)) / pooled_std if pooled_std > 0 else 0.0
    
    # Determine significance
    alpha = 1 - confidence_level
    significant = p_value < alpha and effect_size >= min_effect_size
    
    # Calculate confidence interval for difference in means
    mean_diff = sum(values_b) / len(values_b) - sum(values_a) / len(values_a)
    se_diff = math.sqrt(calculate_std(values_a) ** 2 / len(values_a) + calculate_std(values_b) ** 2 / len(values_b))
    t_critical = stats.t.ppf(1 - alpha / 2, len(values_a) + len(values_b) - 2)
    
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Simple power calculation (approximate)
    power = 1 - stats.t.cdf(t_critical - effect_size * math.sqrt(len(values_a) * len(values_b) / (len(values_a) + len(values_b)) / 2), 
                           len(values_a) + len(values_b) - 2)
    
    return {
        "test_type": "t_test",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
        "confidence_interval": [ci_lower, ci_upper],
        "power": min(power, 1.0),
        "sample_size_a": len(values_a),
        "sample_size_b": len(values_b),
        "mean_difference": mean_diff
    }


def calculate_overall_comparison_stats(results_a: List[Dict], results_b: List[Dict], metrics: List[str], k_values: List[int], confidence_level: float) -> Dict[str, Any]:
    """Calculate overall comparison statistics across all metrics."""
    overall_improvements = 0
    overall_degradations = 0
    total_comparisons = 0
    
    for metric in metrics:
        for k in k_values if metric in ['precision_at_k', 'recall_at_k'] else [None]:
            values_a = []
            values_b = []
            
            for metrics_a, metrics_b in zip(results_a, results_b):
                if k:
                    val_a = metrics_a.get(metric, {}).get(str(k), 0.0)
                    val_b = metrics_b.get(metric, {}).get(str(k), 0.0)
                else:
                    val_a = metrics_a.get(metric, 0.0)
                    val_b = metrics_b.get(metric, 0.0)
                
                values_a.append(val_a)
                values_b.append(val_b)
            
            if values_a and values_b:
                mean_a = sum(values_a) / len(values_a)
                mean_b = sum(values_b) / len(values_b)
                
                total_comparisons += 1
                
                if mean_b > mean_a:
                    overall_improvements += 1
                elif mean_b < mean_a:
                    overall_degradations += 1
    
    return {
        "total_metric_comparisons": total_comparisons,
        "improvements": overall_improvements,
        "degradations": overall_degradations,
        "no_change": total_comparisons - overall_improvements - overall_degradations,
        "improvement_rate": overall_improvements / total_comparisons if total_comparisons > 0 else 0.0,
        "degradation_rate": overall_degradations / total_comparisons if total_comparisons > 0 else 0.0
    }


def count_significant_improvements(statistical_results: Dict[str, Dict]) -> int:
    """Count number of statistically significant improvements."""
    count = 0
    for metric_key, stats in statistical_results.items():
        if stats.get('significant', False) and stats.get('mean_difference', 0) > 0:
            count += 1
    return count


def count_significant_degradations(statistical_results: Dict[str, Dict]) -> int:
    """Count number of statistically significant degradations.""" 
    count = 0
    for metric_key, stats in statistical_results.items():
        if stats.get('significant', False) and stats.get('mean_difference', 0) < 0:
            count += 1
    return count


# =============================================================================
# Async Helper Functions
# =============================================================================

async def execute_search_with_config(query: str, config: Dict[str, Any], limit: int) -> List[Dict]:
    """Execute search with specific configuration."""
    try:
        # Build search parameters from config
        weaviate_overrides = config.get('weaviate', {})
        reranker_config = config.get('reranker', {})
        
        # Use the existing search infrastructure
        if _bm25_vector_override_service and (weaviate_overrides or reranker_config.get('enabled', False)):
            results = _bm25_vector_override_service(
                query,
                limit=limit,
                overrides=weaviate_overrides,
                reranker_config=reranker_config if reranker_config.get('enabled') else None
            )
        elif _search_tools_func:
            # Standard search
            results = _search_tools_func(query, limit=limit)
        else:
            logger.warning("No search function configured")
            return []
        
        # Ensure consistent format
        if isinstance(results, dict) and 'tools' in results:
            return results['tools']
        return results if isinstance(results, list) else []
        
    except Exception as e:
        logger.error(f"Error executing search with config: {str(e)}")
        return []


async def execute_ab_comparison(
    comparison_id: str,
    queries: List[Any],
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    limit: int,
    metrics: List[str],
    k_values: List[int],
    confidence_level: float,
    min_effect_size: float
) -> Dict[str, Any]:
    """Execute A/B comparison with statistical significance testing."""
    results_a = []
    results_b = []
    query_results = []
    
    logger.info(f"Starting A/B comparison {comparison_id} with {len(queries)} queries")
    
    for i, query_data in enumerate(queries):
        try:
            # Handle different query formats
            if isinstance(query_data, str):
                query_text = query_data
                expected_tools = []
            elif isinstance(query_data, dict):
                query_text = query_data.get('query', '')
                expected_tools = query_data.get('expected_tools', [])
            else:
                logger.warning(f"Invalid query format at index {i}: {query_data}")
                continue
            
            if not query_text.strip():
                continue
            
            logger.debug(f"Processing query {i+1}/{len(queries)}: {query_text}")
            
            # Execute search with config A
            search_a = await execute_search_with_config(query_text, config_a, limit)
            
            # Execute search with config B  
            search_b = await execute_search_with_config(query_text, config_b, limit)
            
            # Calculate metrics for this query
            metrics_a = calculate_search_metrics(search_a, expected_tools, k_values)
            metrics_b = calculate_search_metrics(search_b, expected_tools, k_values)
            
            query_result = {
                "query": query_text,
                "expected_tools": expected_tools,
                "results_a": search_a,
                "results_b": search_b,
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "timestamp": time.time()
            }
            
            query_results.append(query_result)
            results_a.append(metrics_a)
            results_b.append(metrics_b)
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            continue
    
    if not results_a or not results_b:
        raise ValueError("No valid query results obtained for comparison")
    
    # Calculate aggregate metrics and statistical significance
    statistical_results = {}
    aggregate_metrics_a = {}
    aggregate_metrics_b = {}
    
    for metric in metrics:
        for k in k_values:
            metric_key = f"{metric}@{k}" if metric in ['precision_at_k', 'recall_at_k'] else metric
            
            # Extract metric values from all queries
            values_a = []
            values_b = []
            
            for m_a, m_b in zip(results_a, results_b):
                if metric in ['precision_at_k', 'recall_at_k']:
                    val_a = m_a.get(metric, {}).get(str(k), 0.0)
                    val_b = m_b.get(metric, {}).get(str(k), 0.0)
                else:
                    val_a = m_a.get(metric, 0.0)
                    val_b = m_b.get(metric, 0.0)
                
                values_a.append(val_a)
                values_b.append(val_b)
            
            # Calculate aggregate statistics
            avg_a = sum(values_a) / len(values_a) if values_a else 0.0
            avg_b = sum(values_b) / len(values_b) if values_b else 0.0
            
            aggregate_metrics_a[metric_key] = {
                "mean": avg_a,
                "std": calculate_std(values_a),
                "min": min(values_a) if values_a else 0.0,
                "max": max(values_a) if values_a else 0.0,
                "median": calculate_median(values_a)
            }
            
            aggregate_metrics_b[metric_key] = {
                "mean": avg_b,
                "std": calculate_std(values_b),
                "min": min(values_b) if values_b else 0.0,
                "max": max(values_b) if values_b else 0.0,
                "median": calculate_median(values_b)
            }
            
            # Perform statistical significance testing
            stat_result = perform_significance_test(
                values_a, values_b, confidence_level, min_effect_size
            )
            statistical_results[metric_key] = stat_result
    
    # Calculate overall effect sizes and power analysis
    overall_stats = calculate_overall_comparison_stats(
        results_a, results_b, metrics, k_values, confidence_level
    )
    
    return {
        "comparison_id": comparison_id,
        "config_a": config_a,
        "config_b": config_b,
        "query_results": query_results,
        "aggregate_metrics_a": aggregate_metrics_a,
        "aggregate_metrics_b": aggregate_metrics_b,
        "statistical_results": statistical_results,
        "overall_statistics": overall_stats,
        "options": {
            "limit": limit,
            "metrics": metrics,
            "k_values": k_values,
            "confidence_level": confidence_level,
            "min_effect_size": min_effect_size
        },
        "summary": {
            "total_queries": len(queries),
            "successful_queries": len(query_results),
            "significant_improvements": count_significant_improvements(statistical_results),
            "significant_degradations": count_significant_degradations(statistical_results)
        },
        "timestamp": time.time()
    }


async def read_json_file(filepath: str) -> Dict[str, Any]:
    """Read JSON file, using aiofiles if available."""
    if HAS_AIOFILES:
        async with aiofiles.open(filepath, 'r') as f:
            return json.loads(await f.read())
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


async def write_json_file(filepath: str, data: Dict[str, Any]) -> None:
    """Write JSON file, using aiofiles if available."""
    if HAS_AIOFILES:
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Route Handlers
# =============================================================================

@evaluation_bp.route('/api/v1/evaluations', methods=['POST'])
async def submit_evaluation():
    """Submit an evaluation rating."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['query', 'result_id', 'rating']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Create evaluation record
        evaluation = {
            "id": "eval_" + str(int(time.time())),
            "query": data.get("query"),
            "result_id": data.get("result_id"),
            "rating": data.get("rating"),
            "feedback": data.get("feedback", ""),
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
        
        # For now, just log the evaluation (can be extended to store in database)
        logger.info(f"Evaluation submitted: {evaluation}")
        
        return jsonify({
            "success": True,
            "data": evaluation
        })
    except Exception as e:
        logger.error(f"Error submitting evaluation: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/evaluations', methods=['GET'])
async def get_evaluations():
    """Get evaluations with optional query and limit parameters."""
    try:
        query = request.args.get('query')
        limit = request.args.get('limit', 50, type=int)
        
        # For now, return empty array since we don't have persistent storage
        # This can be extended to query from database
        evaluations = []
        
        # If we had stored evaluations, we would filter and limit them here
        logger.info(f"Evaluations requested - query: {query}, limit: {limit}")
        
        return jsonify({
            "success": True,
            "data": evaluations,
            "total": len(evaluations)
        })
    except Exception as e:
        logger.error(f"Error getting evaluations: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/analytics', methods=['GET'])
async def get_analytics():
    """Get analytics with optional date range parameters."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # For now, return mock analytics data
        # This can be extended to calculate real metrics
        analytics_data = {
            "search_count": 0,
            "tool_usage": {},
            "avg_rating": 0.0,
            "total_evaluations": 0,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "top_tools": [],
            "recent_searches": []
        }
        
        logger.info(f"Analytics requested - start: {start_date}, end: {end_date}")
        
        return jsonify({
            "success": True,
            "data": analytics_data
        })
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/rerank/compare', methods=['POST'])
async def compare_rerank():
    """Compare two reranker configurations side-by-side."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Validate required fields
        query = data.get('query')
        config_a = data.get('config_a')
        config_b = data.get('config_b')
        limit = data.get('limit', 10)
        
        if not query or not config_a or not config_b:
            return jsonify({"success": False, "error": "Missing required fields: query, config_a, config_b"}), 400
        
        if not _search_tools_func:
            return jsonify({"success": False, "error": "Search function not configured"}), 503
        
        # Perform search with both configurations
        results_a = []
        results_b = []
        
        try:
            # Search with configuration A
            search_results_a = _search_tools_func(query, limit=limit, reranker_config=config_a)
            for i, result in enumerate(search_results_a):
                formatted_result = {
                    "tool": {
                        "id": result.get('id', ''),
                        "name": result.get('name', ''),
                        "description": result.get('description', ''),
                        "source": result.get('source', 'unknown'),
                        "category": result.get('category'),
                        "tags": result.get('tags', [])
                    },
                    "score": result.get('score', 0),
                    "rank": i + 1,
                    "reasoning": result.get('reasoning', ''),
                    "config": "A"
                }
                results_a.append(formatted_result)
        except Exception as e:
            logger.error(f"Error with configuration A: {str(e)}")
            results_a = []
            
        try:
            # Search with configuration B
            search_results_b = _search_tools_func(query, limit=limit, reranker_config=config_b)
            for i, result in enumerate(search_results_b):
                formatted_result = {
                    "tool": {
                        "id": result.get('id', ''),
                        "name": result.get('name', ''),
                        "description": result.get('description', ''),
                        "source": result.get('source', 'unknown'),
                        "category": result.get('category'),
                        "tags": result.get('tags', [])
                    },
                    "score": result.get('score', 0),
                    "rank": i + 1,
                    "reasoning": result.get('reasoning', ''),
                    "config": "B"
                }
                results_b.append(formatted_result)
        except Exception as e:
            logger.error(f"Error with configuration B: {str(e)}")
            results_b = []
        
        # Calculate comparison metrics
        comparison_metrics = {
            "total_results_a": len(results_a),
            "total_results_b": len(results_b),
            "avg_score_a": sum(r.get('score', 0) for r in results_a) / max(len(results_a), 1),
            "avg_score_b": sum(r.get('score', 0) for r in results_b) / max(len(results_b), 1),
            "top_5_overlap": calculate_overlap([r['tool']['id'] for r in results_a[:5]], 
                                               [r['tool']['id'] for r in results_b[:5]]),
            "rank_correlation": calculate_rank_correlation(results_a, results_b)
        }
        
        response_data = {
            "query": query,
            "results_a": results_a,
            "results_b": results_b,
            "comparison_metrics": comparison_metrics,
            "config_a_name": config_a.get('name', 'Configuration A'),
            "config_b_name": config_b.get('name', 'Configuration B'),
            "timestamp": time.time()
        }
        
        logger.info(f"Rerank comparison completed for query: {query}")
        return jsonify({"success": True, "data": response_data})
        
    except Exception as e:
        logger.error(f"Error during rerank comparison: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/ab-comparison/run', methods=['POST'])
async def run_ab_comparison():
    """Run A/B comparison tests with statistical significance analysis."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['queries', 'config_a', 'config_b']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        queries = data['queries']
        config_a = data['config_a']
        config_b = data['config_b']
        options = data.get('options', {})
        
        # Validate queries format
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({"success": False, "error": "queries must be a non-empty array"}), 400
        
        # Options with defaults
        limit = options.get('limit', 10)
        metrics_to_compare = options.get('metrics', ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg'])
        k_values = options.get('k_values', [1, 3, 5, 10])
        confidence_level = options.get('confidence_level', 0.95)
        min_effect_size = options.get('min_effect_size', 0.1)
        
        # Run comparison
        comparison_id = str(uuid.uuid4())
        results = await execute_ab_comparison(
            comparison_id, queries, config_a, config_b, 
            limit, metrics_to_compare, k_values, confidence_level, min_effect_size
        )
        
        # Store results if cache_dir is configured
        if _cache_dir:
            ab_comparison_dir = os.path.join(_cache_dir, 'ab_comparisons')
            os.makedirs(ab_comparison_dir, exist_ok=True)
            
            results_file = os.path.join(ab_comparison_dir, f'{comparison_id}.json')
            await write_json_file(results_file, results)
        
        logger.info(f"A/B comparison completed: {comparison_id}")
        return jsonify({"success": True, "data": results})
        
    except Exception as e:
        logger.error(f"Error running A/B comparison: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/ab-comparison/results', methods=['GET'])
async def get_ab_results():
    """Get A/B comparison results with filtering."""
    try:
        # Query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        config_name = request.args.get('config_name')
        min_significance = request.args.get('min_significance', type=float)
        
        if not _cache_dir:
            return jsonify({"success": True, "data": {"results": [], "total": 0}})
        
        ab_comparison_dir = os.path.join(_cache_dir, 'ab_comparisons')
        if not os.path.exists(ab_comparison_dir):
            return jsonify({"success": True, "data": {"results": [], "total": 0}})
        
        # Load all comparison results
        all_results = []
        for filename in os.listdir(ab_comparison_dir):
            if filename.endswith('.json'):
                try:
                    result = await read_json_file(os.path.join(ab_comparison_dir, filename))
                    
                    # Apply filters
                    if config_name:
                        if (config_name.lower() not in result.get('config_a', {}).get('name', '').lower() and 
                            config_name.lower() not in result.get('config_b', {}).get('name', '').lower()):
                            continue
                    
                    if min_significance is not None:
                        max_p_value = max([stat.get('p_value', 1.0) 
                                         for stat in result.get('statistical_results', {}).values()])
                        if max_p_value > (1 - min_significance):
                            continue
                    
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Error loading comparison result {filename}: {e}")
        
        # Sort by timestamp (newest first)
        all_results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Apply pagination
        total = len(all_results)
        paginated_results = all_results[offset:offset + limit]
        
        return jsonify({
            "success": True,
            "data": {
                "results": paginated_results,
                "total": total,
                "offset": offset,
                "limit": limit
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving A/B comparison results: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@evaluation_bp.route('/api/v1/ab-comparison/results/<comparison_id>', methods=['GET'])
async def get_ab_result_by_id(comparison_id):
    """Get specific A/B comparison result."""
    try:
        if not _cache_dir:
            return jsonify({"success": False, "error": "Cache directory not configured"}), 503
        
        ab_comparison_dir = os.path.join(_cache_dir, 'ab_comparisons')
        result_file = os.path.join(ab_comparison_dir, f'{comparison_id}.json')
        
        if not os.path.exists(result_file):
            return jsonify({"success": False, "error": "Comparison result not found"}), 404
        
        result = await read_json_file(result_file)
        
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        logger.error(f"Error retrieving A/B comparison result: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
