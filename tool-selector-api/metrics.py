"""
Prometheus metrics for the Tool Selector API.

Provides observability into:
- Request latency and throughput
- Tool search and attachment operations
- Weaviate connection pool health
- Pruning operations
- Error rates
"""

import time
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

# Try to import prometheus_client, gracefully degrade if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - metrics disabled")

# Only define metrics if prometheus is available
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'tool_selector_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    REQUEST_LATENCY = Histogram(
        'tool_selector_request_duration_seconds',
        'Request latency in seconds',
        ['method', 'endpoint'],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    # Tool search metrics
    SEARCH_COUNT = Counter(
        'tool_selector_search_total',
        'Total tool search requests',
        ['search_type']  # hybrid, vector, bm25
    )
    
    SEARCH_LATENCY = Histogram(
        'tool_selector_search_duration_seconds',
        'Tool search latency in seconds',
        ['search_type'],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    SEARCH_RESULTS = Histogram(
        'tool_selector_search_results_count',
        'Number of tools returned per search',
        buckets=[0, 1, 2, 5, 10, 20, 50, 100]
    )
    
    # Tool attachment metrics
    ATTACH_COUNT = Counter(
        'tool_selector_attach_total',
        'Total tool attachment operations',
        ['status']  # success, failed, skipped
    )
    
    ATTACH_TOOLS = Histogram(
        'tool_selector_attach_tools_count',
        'Number of tools attached per operation',
        buckets=[0, 1, 2, 5, 10, 20, 50]
    )
    
    # Detach metrics
    DETACH_COUNT = Counter(
        'tool_selector_detach_total',
        'Total tool detachment operations',
        ['status']  # success, failed, protected
    )
    
    # Pruning metrics
    PRUNE_COUNT = Counter(
        'tool_selector_prune_total',
        'Total pruning operations',
        ['mode', 'status']  # mode: scheduled/manual, status: success/failed
    )
    
    PRUNE_TOOLS = Histogram(
        'tool_selector_prune_tools_count',
        'Number of tools pruned per operation',
        buckets=[0, 1, 2, 5, 10, 20, 50]
    )
    
    PRUNE_AGENTS = Histogram(
        'tool_selector_prune_agents_count',
        'Number of agents processed per prune run',
        buckets=[1, 5, 10, 25, 50, 100]
    )
    
    # Weaviate connection pool metrics
    WEAVIATE_POOL_SIZE = Gauge(
        'tool_selector_weaviate_pool_size',
        'Current Weaviate connection pool size'
    )
    
    WEAVIATE_POOL_ACTIVE = Gauge(
        'tool_selector_weaviate_pool_active',
        'Active Weaviate connections'
    )
    
    WEAVIATE_POOL_WAITS = Counter(
        'tool_selector_weaviate_pool_waits_total',
        'Total times requests waited for a connection'
    )
    
    WEAVIATE_POOL_TIMEOUTS = Counter(
        'tool_selector_weaviate_pool_timeouts_total',
        'Total connection pool timeout errors'
    )
    
    # Enrichment metrics
    ENRICHMENT_COUNT = Counter(
        'tool_selector_enrichment_total',
        'Total enrichment operations',
        ['operation', 'status']  # operation: tool/server/sync, status: success/failed/cached
    )
    
    ENRICHMENT_LATENCY = Histogram(
        'tool_selector_enrichment_duration_seconds',
        'Enrichment operation latency',
        ['operation'],
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
    )
    
    # Error metrics
    ERROR_COUNT = Counter(
        'tool_selector_errors_total',
        'Total errors by type',
        ['error_type']  # weaviate, letta_api, validation, internal
    )
    
    # API info
    API_INFO = Info(
        'tool_selector',
        'Tool Selector API information'
    )


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    return generate_latest()


def get_content_type() -> str:
    """Get the content type for metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


def set_api_info(version: str = "1.0.0", environment: str = "production"):
    """Set API info metrics."""
    if PROMETHEUS_AVAILABLE:
        API_INFO.info({
            'version': version,
            'environment': environment
        })


# Decorator for timing requests
def track_request(endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)
            
            start = time.time()
            status = "200"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start
                REQUEST_LATENCY.labels(method="POST", endpoint=endpoint).observe(duration)
                REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=status).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            start = time.time()
            status = "200"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start
                REQUEST_LATENCY.labels(method="POST", endpoint=endpoint).observe(duration)
                REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=status).inc()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# Helper functions for recording metrics
def record_search(search_type: str, duration: float, result_count: int):
    """Record search metrics."""
    if PROMETHEUS_AVAILABLE:
        SEARCH_COUNT.labels(search_type=search_type).inc()
        SEARCH_LATENCY.labels(search_type=search_type).observe(duration)
        SEARCH_RESULTS.observe(result_count)


def record_attach(status: str, tool_count: int = 0):
    """Record tool attachment metrics."""
    if PROMETHEUS_AVAILABLE:
        ATTACH_COUNT.labels(status=status).inc()
        if tool_count > 0:
            ATTACH_TOOLS.observe(tool_count)


def record_detach(status: str):
    """Record tool detachment metrics."""
    if PROMETHEUS_AVAILABLE:
        DETACH_COUNT.labels(status=status).inc()


def record_prune(mode: str, status: str, tool_count: int = 0, agent_count: int = 0):
    """Record pruning metrics."""
    if PROMETHEUS_AVAILABLE:
        PRUNE_COUNT.labels(mode=mode, status=status).inc()
        if tool_count > 0:
            PRUNE_TOOLS.observe(tool_count)
        if agent_count > 0:
            PRUNE_AGENTS.observe(agent_count)


def record_enrichment(operation: str, status: str, duration: float = 0):
    """Record enrichment metrics."""
    if PROMETHEUS_AVAILABLE:
        ENRICHMENT_COUNT.labels(operation=operation, status=status).inc()
        if duration > 0:
            ENRICHMENT_LATENCY.labels(operation=operation).observe(duration)


def record_error(error_type: str):
    """Record error metrics."""
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(error_type=error_type).inc()


def update_weaviate_pool(pool_size: int, active: int):
    """Update Weaviate pool metrics."""
    if PROMETHEUS_AVAILABLE:
        WEAVIATE_POOL_SIZE.set(pool_size)
        WEAVIATE_POOL_ACTIVE.set(active)


def record_weaviate_wait():
    """Record a Weaviate pool wait event."""
    if PROMETHEUS_AVAILABLE:
        WEAVIATE_POOL_WAITS.inc()


def record_weaviate_timeout():
    """Record a Weaviate pool timeout."""
    if PROMETHEUS_AVAILABLE:
        WEAVIATE_POOL_TIMEOUTS.inc()
