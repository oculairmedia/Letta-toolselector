"""
Performance test: Tool attachment under load (vly)

Simulates concurrent webhook/attachment requests to test scalability.
Run with: python -m pytest tests/performance/test_attachment_load.py -v -s
"""

import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass
from typing import List
import pytest

# Configuration
API_BASE_URL = "http://localhost:8020"
SEARCH_ENDPOINT = f"{API_BASE_URL}/api/v1/tools/search"


@dataclass
class RequestResult:
    """Result of a single request."""
    duration_ms: float
    success: bool
    status_code: int
    error: str = ""


@dataclass 
class LoadTestResults:
    """Aggregated load test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float


async def make_search_request(
    session: aiohttp.ClientSession,
    query: str,
    limit: int = 5
) -> RequestResult:
    """Make a single search request and measure latency."""
    start = time.perf_counter()
    try:
        async with session.post(
            SEARCH_ENDPOINT,
            json={"query": query, "limit": limit},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            await response.json()
            duration_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                duration_ms=duration_ms,
                success=response.status == 200,
                status_code=response.status
            )
    except asyncio.TimeoutError:
        duration_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            duration_ms=duration_ms,
            success=False,
            status_code=0,
            error="Timeout"
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            duration_ms=duration_ms,
            success=False,
            status_code=0,
            error=str(e)
        )


async def run_concurrent_requests(
    num_requests: int,
    queries: List[str]
) -> List[RequestResult]:
    """Run multiple concurrent requests."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            query = queries[i % len(queries)]
            tasks.append(make_search_request(session, query))
        
        results = await asyncio.gather(*tasks)
        return list(results)


def calculate_percentile(latencies: List[float], percentile: float) -> float:
    """Calculate percentile from list of latencies."""
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    index = int(len(sorted_latencies) * percentile / 100)
    index = min(index, len(sorted_latencies) - 1)
    return sorted_latencies[index]


def analyze_results(results: List[RequestResult], total_duration_ms: float) -> LoadTestResults:
    """Analyze load test results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.duration_ms for r in results]
    
    return LoadTestResults(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_duration_ms=total_duration_ms,
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=calculate_percentile(latencies, 50),
        p95_latency_ms=calculate_percentile(latencies, 95),
        p99_latency_ms=calculate_percentile(latencies, 99),
        requests_per_second=len(results) / (total_duration_ms / 1000) if total_duration_ms > 0 else 0
    )


def print_results(results: LoadTestResults, test_name: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Load Test Results: {test_name}")
    print(f"{'='*60}")
    print(f"Total Requests:     {results.total_requests}")
    print(f"Successful:         {results.successful_requests}")
    print(f"Failed:             {results.failed_requests}")
    print(f"Total Duration:     {results.total_duration_ms:.2f}ms")
    print(f"Requests/Second:    {results.requests_per_second:.2f}")
    print(f"\nLatency (ms):")
    print(f"  Min:              {results.min_latency_ms:.2f}")
    print(f"  Avg:              {results.avg_latency_ms:.2f}")
    print(f"  P50:              {results.p50_latency_ms:.2f}")
    print(f"  P95:              {results.p95_latency_ms:.2f}")
    print(f"  P99:              {results.p99_latency_ms:.2f}")
    print(f"  Max:              {results.max_latency_ms:.2f}")
    print(f"{'='*60}\n")


# Test queries simulating different webhook scenarios
TEST_QUERIES = [
    "create a new issue",
    "send a message to user",
    "read file contents",
    "search the knowledge graph",
    "manage project tasks",
    "schedule a meeting",
    "analyze data trends",
    "generate report",
    "update configuration",
    "deploy application"
]


class TestAttachmentLoad:
    """Performance tests for tool attachment under load."""

    @pytest.mark.performance
    def test_10_concurrent_requests(self):
        """Test with 10 concurrent requests - baseline."""
        start = time.perf_counter()
        results = asyncio.run(run_concurrent_requests(10, TEST_QUERIES))
        total_duration_ms = (time.perf_counter() - start) * 1000
        
        analysis = analyze_results(results, total_duration_ms)
        print_results(analysis, "10 Concurrent Requests")
        
        # Assertions
        assert analysis.failed_requests == 0, f"Expected 0 failures, got {analysis.failed_requests}"
        assert analysis.p95_latency_ms < 5000, f"P95 latency {analysis.p95_latency_ms}ms exceeds 5000ms threshold"

    @pytest.mark.performance
    def test_25_concurrent_requests(self):
        """Test with 25 concurrent requests - moderate load."""
        start = time.perf_counter()
        results = asyncio.run(run_concurrent_requests(25, TEST_QUERIES))
        total_duration_ms = (time.perf_counter() - start) * 1000
        
        analysis = analyze_results(results, total_duration_ms)
        print_results(analysis, "25 Concurrent Requests")
        
        # Assertions - slightly relaxed for higher load
        assert analysis.failed_requests == 0, f"Expected 0 failures, got {analysis.failed_requests}"
        assert analysis.p95_latency_ms < 8000, f"P95 latency {analysis.p95_latency_ms}ms exceeds 8000ms threshold"

    @pytest.mark.performance
    def test_50_concurrent_requests(self):
        """Test with 50 concurrent requests - high load."""
        start = time.perf_counter()
        results = asyncio.run(run_concurrent_requests(50, TEST_QUERIES))
        total_duration_ms = (time.perf_counter() - start) * 1000
        
        analysis = analyze_results(results, total_duration_ms)
        print_results(analysis, "50 Concurrent Requests")
        
        # Assertions - more relaxed for stress test
        failure_rate = analysis.failed_requests / analysis.total_requests
        assert failure_rate < 0.05, f"Failure rate {failure_rate*100:.1f}% exceeds 5% threshold"
        assert analysis.p95_latency_ms < 15000, f"P95 latency {analysis.p95_latency_ms}ms exceeds 15000ms threshold"

    @pytest.mark.performance
    def test_sequential_baseline(self):
        """Sequential requests for baseline comparison."""
        results = []
        start = time.perf_counter()
        
        for i in range(10):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            result = asyncio.run(run_concurrent_requests(1, [query]))
            results.extend(result)
        
        total_duration_ms = (time.perf_counter() - start) * 1000
        
        analysis = analyze_results(results, total_duration_ms)
        print_results(analysis, "10 Sequential Requests (Baseline)")
        
        assert analysis.failed_requests == 0, f"Expected 0 failures, got {analysis.failed_requests}"


if __name__ == "__main__":
    # Run tests directly
    print("Running performance tests...")
    
    test = TestAttachmentLoad()
    
    print("\n1. Sequential baseline...")
    test.test_sequential_baseline()
    
    print("\n2. 10 concurrent requests...")
    test.test_10_concurrent_requests()
    
    print("\n3. 25 concurrent requests...")
    test.test_25_concurrent_requests()
    
    print("\n4. 50 concurrent requests...")
    test.test_50_concurrent_requests()
    
    print("\nAll performance tests completed!")
