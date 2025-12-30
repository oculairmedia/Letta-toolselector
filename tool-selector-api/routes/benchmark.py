"""
Benchmark Routes Blueprint

Handles benchmark query set CRUD and run operations:
- /api/v1/benchmark/query-sets - List, create query sets
- /api/v1/benchmark/query-sets/<id> - Get, update, delete query sets
- /api/v1/benchmark/query-sets/<id>/run - Run benchmark
- /api/v1/benchmark/runs - Get run history
"""

from quart import Blueprint, request, jsonify
import logging
import os
import json
import time
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
benchmark_bp = Blueprint('benchmark', __name__, url_prefix='/api/v1/benchmark')

# Module state - to be configured
_cache_dir = None
_search_tools = None


def configure(cache_dir=None, search_tools=None):
    """Configure the benchmark routes with required dependencies."""
    global _cache_dir, _search_tools
    _cache_dir = cache_dir
    _search_tools = search_tools


def _get_benchmark_path():
    """Get path to benchmark query sets file."""
    return os.path.join(_cache_dir, 'benchmark_query_sets.json') if _cache_dir else None


def _get_run_history_path():
    """Get path to benchmark run history file."""
    return os.path.join(_cache_dir, 'benchmark_run_history.json') if _cache_dir else None


def _load_benchmark_data():
    """Load benchmark data from file."""
    benchmark_path = _get_benchmark_path()
    if benchmark_path and os.path.exists(benchmark_path):
        with open(benchmark_path, 'r') as f:
            return json.load(f)
    return {"query_sets": [], "last_updated": None}


def _save_benchmark_data(data):
    """Save benchmark data to file."""
    benchmark_path = _get_benchmark_path()
    if benchmark_path:
        os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
        with open(benchmark_path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Query Set CRUD Endpoints
# =============================================================================

@benchmark_bp.route('/query-sets', methods=['GET'])
async def list_benchmark_query_sets():
    """Get all benchmark query sets."""
    try:
        benchmark_data = _load_benchmark_data()
        
        return jsonify({
            "success": True,
            "data": {
                "query_sets": benchmark_data.get('query_sets', []),
                "total": len(benchmark_data.get('query_sets', [])),
                "last_updated": benchmark_data.get('last_updated')
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing benchmark query sets: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@benchmark_bp.route('/query-sets', methods=['POST'])
async def create_benchmark_query_set():
    """Create a new benchmark query set."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'queries']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Validate queries format
        queries = data.get('queries', [])
        for i, query in enumerate(queries):
            if not isinstance(query, dict) or 'query' not in query:
                return jsonify({"success": False, "error": f"Invalid query format at index {i}"}), 400
        
        # Load existing benchmark sets
        benchmark_data = _load_benchmark_data()
        
        # Create new query set
        new_query_set = {
            "id": f"qs_{int(time.time())}_{len(benchmark_data['query_sets'])}",
            "name": data['name'],
            "description": data['description'],
            "queries": queries,
            "metadata": data.get('metadata', {}),
            "created_at": datetime.utcnow().isoformat(),
            "created_by": data.get('created_by', 'system'),
            "tags": data.get('tags', []),
            "query_count": len(queries),
            "last_run": None,
            "run_count": 0,
            "active": True
        }
        
        # Add to benchmark data
        benchmark_data['query_sets'].append(new_query_set)
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        _save_benchmark_data(benchmark_data)
        
        logger.info(f"Created benchmark query set: {new_query_set['id']}")
        
        return jsonify({
            "success": True,
            "data": new_query_set,
            "message": f"Query set '{data['name']}' created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@benchmark_bp.route('/query-sets/<query_set_id>', methods=['GET'])
async def get_benchmark_query_set(query_set_id):
    """Get a specific benchmark query set."""
    try:
        benchmark_data = _load_benchmark_data()
        
        if not benchmark_data.get('query_sets'):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        # Find the query set
        query_set = None
        for qs in benchmark_data.get('query_sets', []):
            if qs['id'] == query_set_id:
                query_set = qs
                break
        
        if not query_set:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        return jsonify({
            "success": True,
            "data": query_set
        })
        
    except Exception as e:
        logger.error(f"Error getting benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@benchmark_bp.route('/query-sets/<query_set_id>', methods=['PUT'])
async def update_benchmark_query_set(query_set_id):
    """Update a benchmark query set."""
    try:
        data = await request.get_json()
        
        benchmark_data = _load_benchmark_data()
        
        if not benchmark_data.get('query_sets'):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        # Find and update the query set
        query_set_index = None
        for i, qs in enumerate(benchmark_data.get('query_sets', [])):
            if qs['id'] == query_set_id:
                query_set_index = i
                break
        
        if query_set_index is None:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        # Update allowed fields
        updatable_fields = ['name', 'description', 'queries', 'metadata', 'tags', 'active']
        query_set = benchmark_data['query_sets'][query_set_index]
        
        for field in updatable_fields:
            if field in data:
                query_set[field] = data[field]
                if field == 'queries':
                    query_set['query_count'] = len(data[field])
        
        query_set['updated_at'] = datetime.utcnow().isoformat()
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        _save_benchmark_data(benchmark_data)
        
        logger.info(f"Updated benchmark query set: {query_set_id}")
        
        return jsonify({
            "success": True,
            "data": query_set,
            "message": f"Query set {query_set_id} updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@benchmark_bp.route('/query-sets/<query_set_id>', methods=['DELETE'])
async def delete_benchmark_query_set(query_set_id):
    """Delete a benchmark query set."""
    try:
        benchmark_data = _load_benchmark_data()
        
        if not benchmark_data.get('query_sets'):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        # Find and remove the query set
        original_count = len(benchmark_data.get('query_sets', []))
        benchmark_data['query_sets'] = [
            qs for qs in benchmark_data.get('query_sets', []) 
            if qs['id'] != query_set_id
        ]
        
        if len(benchmark_data['query_sets']) == original_count:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save to cache
        _save_benchmark_data(benchmark_data)
        
        logger.info(f"Deleted benchmark query set: {query_set_id}")
        
        return jsonify({
            "success": True,
            "message": f"Query set {query_set_id} deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error deleting benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Benchmark Run Endpoints
# =============================================================================

@benchmark_bp.route('/query-sets/<query_set_id>/run', methods=['POST'])
async def run_benchmark_query_set(query_set_id):
    """Run a benchmark query set against current configuration."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        
        benchmark_data = _load_benchmark_data()
        
        if not benchmark_data.get('query_sets'):
            return jsonify({"success": False, "error": "No benchmark query sets found"}), 404
        
        # Find the query set
        query_set = None
        query_set_index = None
        for i, qs in enumerate(benchmark_data.get('query_sets', [])):
            if qs['id'] == query_set_id:
                query_set = qs
                query_set_index = i
                break
        
        if not query_set:
            return jsonify({"success": False, "error": f"Query set {query_set_id} not found"}), 404
        
        if not query_set.get('active', True):
            return jsonify({"success": False, "error": f"Query set {query_set_id} is inactive"}), 400
        
        # Get configuration to test with
        config = data.get('config')  # Optional configuration override
        limit = data.get('limit', 10)
        include_details = data.get('include_details', False)
        
        logger.info(f"Running benchmark query set: {query_set_id} with {len(query_set['queries'])} queries")
        
        # Run each query in the set
        results = []
        start_time = time.time()
        
        for i, query_item in enumerate(query_set['queries']):
            query_text = query_item['query']
            expected_results = query_item.get('expected_results', [])
            
            try:
                # Run the search query
                if _search_tools:
                    if config:
                        search_results = await asyncio.to_thread(
                            _search_tools, 
                            query=query_text, 
                            limit=limit, 
                            reranker_config=config
                        )
                    else:
                        search_results = await asyncio.to_thread(
                            _search_tools, 
                            query=query_text, 
                            limit=limit
                        )
                else:
                    search_results = []
                
                # Calculate metrics if expected results are provided
                metrics = {}
                if expected_results:
                    actual_tool_ids = [r.get('id') or r.get('tool_id') for r in search_results[:limit]]
                    expected_tool_ids = [exp.get('tool_id') for exp in expected_results if exp.get('tool_id')]
                    
                    if expected_tool_ids:
                        relevant_found = len(set(actual_tool_ids) & set(expected_tool_ids))
                        precision_at_k = relevant_found / len(actual_tool_ids) if actual_tool_ids else 0
                        recall_at_k = relevant_found / len(expected_tool_ids) if expected_tool_ids else 0
                        
                        metrics = {
                            "precision_at_k": precision_at_k,
                            "recall_at_k": recall_at_k,
                            "relevant_found": relevant_found,
                            "total_relevant": len(expected_tool_ids),
                            "total_retrieved": len(actual_tool_ids)
                        }
                
                query_result = {
                    "query_index": i,
                    "query": query_text,
                    "results_count": len(search_results),
                    "metrics": metrics,
                    "execution_time": time.time() - start_time
                }
                
                if include_details:
                    query_result["results"] = search_results[:limit]
                    query_result["expected_results"] = expected_results
                
                results.append(query_result)
                
            except Exception as query_error:
                logger.error(f"Error running query {i} in benchmark {query_set_id}: {str(query_error)}")
                results.append({
                    "query_index": i,
                    "query": query_text,
                    "error": str(query_error),
                    "metrics": {},
                    "execution_time": 0
                })
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        successful_queries = [r for r in results if 'error' not in r]
        avg_precision = sum(r['metrics'].get('precision_at_k', 0) for r in successful_queries) / max(len(successful_queries), 1)
        avg_recall = sum(r['metrics'].get('recall_at_k', 0) for r in successful_queries) / max(len(successful_queries), 1)
        
        run_summary = {
            "run_id": f"run_{int(time.time())}",
            "query_set_id": query_set_id,
            "query_set_name": query_set['name'],
            "timestamp": datetime.utcnow().isoformat(),
            "total_queries": len(query_set['queries']),
            "successful_queries": len(successful_queries),
            "failed_queries": len(results) - len(successful_queries),
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(results) if results else 0,
            "aggregate_metrics": {
                "avg_precision_at_k": avg_precision,
                "avg_recall_at_k": avg_recall,
                "success_rate": len(successful_queries) / len(results) if results else 0
            },
            "config_used": config,
            "results": results
        }
        
        # Update query set with run information
        query_set['last_run'] = datetime.utcnow().isoformat()
        query_set['run_count'] = query_set.get('run_count', 0) + 1
        benchmark_data['query_sets'][query_set_index] = query_set
        benchmark_data['last_updated'] = datetime.utcnow().isoformat()
        
        # Save updated benchmark data
        _save_benchmark_data(benchmark_data)
        
        # Save run to history
        run_history_path = _get_run_history_path()
        if run_history_path:
            if os.path.exists(run_history_path):
                with open(run_history_path, 'r') as f:
                    run_history = json.load(f)
            else:
                run_history = {"runs": []}
            
            run_history['runs'].append(run_summary)
            run_history['runs'] = run_history['runs'][-100:]  # Keep last 100
            
            with open(run_history_path, 'w') as f:
                json.dump(run_history, f, indent=2)
        
        logger.info(f"Completed benchmark run for {query_set_id}: {len(successful_queries)}/{len(results)} queries successful")
        
        return jsonify({
            "success": True,
            "data": run_summary,
            "message": f"Benchmark run completed: {len(successful_queries)}/{len(results)} queries successful"
        })
        
    except Exception as e:
        logger.error(f"Error running benchmark query set: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@benchmark_bp.route('/runs', methods=['GET'])
async def get_benchmark_run_history():
    """Get benchmark run history."""
    try:
        limit = request.args.get('limit', 20, type=int)
        query_set_id = request.args.get('query_set_id')
        
        run_history_path = _get_run_history_path()
        
        if run_history_path and os.path.exists(run_history_path):
            with open(run_history_path, 'r') as f:
                run_history = json.load(f)
        else:
            run_history = {"runs": []}
        
        runs = run_history.get('runs', [])
        
        # Filter by query_set_id if provided
        if query_set_id:
            runs = [run for run in runs if run.get('query_set_id') == query_set_id]
        
        # Sort by timestamp (newest first) and limit
        runs = sorted(runs, key=lambda r: r.get('timestamp', ''), reverse=True)[:limit]
        
        return jsonify({
            "success": True,
            "data": {
                "runs": runs,
                "total": len(runs),
                "filtered_by_query_set": query_set_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting benchmark run history: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
