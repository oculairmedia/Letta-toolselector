#!/usr/bin/env python3
"""
Reranking Performance Validation Tests

This script validates the performance and accuracy of the reranking system
with enhanced descriptions, comparing search quality metrics across different
configurations and measuring response times.
"""

import time
import statistics
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchResult:
    """Search result with metadata"""
    name: str
    score: float
    rank: int
    reasoning: str = ""

@dataclass
class PerformanceMetrics:
    """Performance metrics for a test"""
    query: str
    response_time_ms: float
    total_results: int
    relevant_results: int
    precision_at_5: float
    ndcg_at_5: float
    with_reranking: bool

class RerankingPerformanceValidator:
    """Validates reranking performance and accuracy"""
    
    def __init__(self, api_base_url: str = "http://localhost:8020"):
        self.api_base_url = api_base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_test_queries(self) -> List[Dict[str, Any]]:
        """Define test queries with expected relevant tools"""
        return [
            {
                "query": "create and manage agent memory blocks",
                "expected_tools": ["create_memory_block", "list_memory_blocks", "update_memory_block", "read_memory_block"],
                "category": "memory_management"
            },
            {
                "query": "search for information in archives and conversations",
                "expected_tools": ["archival_memory_search", "conversation_search", "search_memory", "search_memory_facts"],
                "category": "information_retrieval"
            },
            {
                "query": "project issue creation and tracking",
                "expected_tools": ["huly_create_issue", "huly_create_subissue", "huly_update_issue", "huly_search_issues"],
                "category": "project_management"
            },
            {
                "query": "file system operations and document management",
                "expected_tools": ["read_file", "write_file", "create_directory", "list_directory", "search_files"],
                "category": "file_operations"
            },
            {
                "query": "agent lifecycle and tool management",
                "expected_tools": ["create_agent", "modify_agent", "list_agents", "attach_tool", "list_agent_tools"],
                "category": "agent_management"
            },
            {
                "query": "content creation and knowledge base management",
                "expected_tools": ["create_book", "create_page", "update_page", "read_book", "list_books"],
                "category": "content_management"
            },
            {
                "query": "data analysis and research workflows",
                "expected_tools": ["fetch_webpage", "web_search", "extract", "analyze_trace"],
                "category": "research_tools"
            },
            {
                "query": "social media and communication management",
                "expected_tools": ["create-post", "matrix_send_message", "matrix_create_room"],
                "category": "communication"
            }
        ]
    
    async def search_with_reranking(self, query: str, limit: int = 10) -> Tuple[List[SearchResult], float]:
        """Perform search with reranking and measure response time"""
        start_time = time.time()
        
        payload = {
            "query": query,
            "limit": limit,
            "reranker_config": {
                "enabled": True,
                "model": "dengcao/Qwen3-Reranker-4B:Q5_K_M",
                "provider": "ollama"
            }
        }
        
        async with self.session.post(
            f"{self.api_base_url}/api/v1/tools/search/rerank",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            response_time = (time.time() - start_time) * 1000
            
            if response.status == 200:
                data = await response.json()
                if data.get("success"):
                    results = []
                    for item in data.get("data", {}).get("results", []):
                        results.append(SearchResult(
                            name=item["tool"]["name"],
                            score=item["score"],
                            rank=item["rank"],
                            reasoning=item.get("reasoning", "")
                        ))
                    return results, response_time
            
            return [], response_time
    
    async def search_without_reranking(self, query: str, limit: int = 10) -> Tuple[List[SearchResult], float]:
        """Perform search without reranking for comparison"""
        start_time = time.time()
        
        payload = {
            "query": query,
            "limit": limit
        }
        
        async with self.session.post(
            f"{self.api_base_url}/api/v1/tools/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            response_time = (time.time() - start_time) * 1000
            
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list):
                    results = []
                    for i, item in enumerate(data[:limit]):
                        results.append(SearchResult(
                            name=item["name"],
                            score=item.get("score", 0),
                            rank=i + 1
                        ))
                    return results, response_time
            
            return [], response_time
    
    def calculate_precision_at_k(self, results: List[SearchResult], expected_tools: List[str], k: int = 5) -> float:
        """Calculate precision@k metric"""
        if not results:
            return 0.0
        
        top_k = results[:k]
        relevant_count = sum(1 for result in top_k if result.name in expected_tools)
        return relevant_count / min(len(top_k), k)
    
    def calculate_ndcg_at_k(self, results: List[SearchResult], expected_tools: List[str], k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k"""
        if not results:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            if result.name in expected_tools:
                dcg += 1.0 / (1 + i)**0.5  # Using log2(i+2) simplified
        
        # IDCG calculation (ideal ranking)
        ideal_relevant = min(len(expected_tools), k)
        idcg = sum(1.0 / (1 + i)**0.5 for i in range(ideal_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    async def run_performance_test(self, test_case: Dict[str, Any]) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
        """Run performance test for a single query with and without reranking"""
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]
        
        # Test with reranking
        reranked_results, rerank_time = await self.search_with_reranking(query)
        rerank_precision = self.calculate_precision_at_k(reranked_results, expected_tools)
        rerank_ndcg = self.calculate_ndcg_at_k(reranked_results, expected_tools)
        rerank_relevant = sum(1 for r in reranked_results if r.name in expected_tools)
        
        rerank_metrics = PerformanceMetrics(
            query=query,
            response_time_ms=rerank_time,
            total_results=len(reranked_results),
            relevant_results=rerank_relevant,
            precision_at_5=rerank_precision,
            ndcg_at_5=rerank_ndcg,
            with_reranking=True
        )
        
        # Test without reranking
        regular_results, regular_time = await self.search_without_reranking(query)
        regular_precision = self.calculate_precision_at_k(regular_results, expected_tools)
        regular_ndcg = self.calculate_ndcg_at_k(regular_results, expected_tools)
        regular_relevant = sum(1 for r in regular_results if r.name in expected_tools)
        
        regular_metrics = PerformanceMetrics(
            query=query,
            response_time_ms=regular_time,
            total_results=len(regular_results),
            relevant_results=regular_relevant,
            precision_at_5=regular_precision,
            ndcg_at_5=regular_ndcg,
            with_reranking=False
        )
        
        return rerank_metrics, regular_metrics
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance validation"""
        print("🚀 Starting Reranking Performance Validation Tests")
        print("=" * 60)
        
        test_queries = self.get_test_queries()
        all_rerank_metrics = []
        all_regular_metrics = []
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n📊 Test {i}/{len(test_queries)}: {test_case['category']}")
            print(f"Query: '{test_case['query']}'")
            
            try:
                rerank_metrics, regular_metrics = await self.run_performance_test(test_case)
                all_rerank_metrics.append(rerank_metrics)
                all_regular_metrics.append(regular_metrics)
                
                # Print individual results
                print(f"  With Reranking:")
                print(f"    Response Time: {rerank_metrics.response_time_ms:.1f}ms")
                print(f"    Precision@5: {rerank_metrics.precision_at_5:.3f}")
                print(f"    NDCG@5: {rerank_metrics.ndcg_at_5:.3f}")
                print(f"    Relevant Results: {rerank_metrics.relevant_results}/{rerank_metrics.total_results}")
                
                print(f"  Without Reranking:")
                print(f"    Response Time: {regular_metrics.response_time_ms:.1f}ms")
                print(f"    Precision@5: {regular_metrics.precision_at_5:.3f}")
                print(f"    NDCG@5: {regular_metrics.ndcg_at_5:.3f}")
                print(f"    Relevant Results: {regular_metrics.relevant_results}/{regular_metrics.total_results}")
                
                # Show improvement
                precision_improvement = ((rerank_metrics.precision_at_5 - regular_metrics.precision_at_5) / max(regular_metrics.precision_at_5, 0.001)) * 100
                print(f"  📈 Precision Improvement: {precision_improvement:+.1f}%")
                
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
        
        # Calculate aggregate metrics
        if all_rerank_metrics and all_regular_metrics:
            return self.calculate_aggregate_results(all_rerank_metrics, all_regular_metrics)
        
        return {}
    
    def calculate_aggregate_results(self, rerank_metrics: List[PerformanceMetrics], regular_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate aggregate performance results"""
        print("\n" + "=" * 60)
        print("📊 AGGREGATE PERFORMANCE RESULTS")
        print("=" * 60)
        
        # Response time analysis
        rerank_times = [m.response_time_ms for m in rerank_metrics]
        regular_times = [m.response_time_ms for m in regular_metrics]
        
        avg_rerank_time = statistics.mean(rerank_times)
        avg_regular_time = statistics.mean(regular_times)
        
        # Accuracy analysis
        avg_rerank_precision = statistics.mean([m.precision_at_5 for m in rerank_metrics])
        avg_regular_precision = statistics.mean([m.precision_at_5 for m in regular_metrics])
        
        avg_rerank_ndcg = statistics.mean([m.ndcg_at_5 for m in rerank_metrics])
        avg_regular_ndcg = statistics.mean([m.ndcg_at_5 for m in regular_metrics])
        
        # Calculate improvements
        precision_improvement = ((avg_rerank_precision - avg_regular_precision) / max(avg_regular_precision, 0.001)) * 100
        ndcg_improvement = ((avg_rerank_ndcg - avg_regular_ndcg) / max(avg_regular_ndcg, 0.001)) * 100
        time_overhead = ((avg_rerank_time - avg_regular_time) / max(avg_regular_time, 1)) * 100
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(rerank_metrics),
            "performance": {
                "average_response_time": {
                    "with_reranking_ms": avg_rerank_time,
                    "without_reranking_ms": avg_regular_time,
                    "overhead_percent": time_overhead
                },
                "accuracy_metrics": {
                    "precision_at_5": {
                        "with_reranking": avg_rerank_precision,
                        "without_reranking": avg_regular_precision,
                        "improvement_percent": precision_improvement
                    },
                    "ndcg_at_5": {
                        "with_reranking": avg_rerank_ndcg,
                        "without_reranking": avg_regular_ndcg,
                        "improvement_percent": ndcg_improvement
                    }
                }
            }
        }
        
        # Print summary
        print(f"⏱️  Response Time:")
        print(f"   With Reranking: {avg_rerank_time:.1f}ms")
        print(f"   Without Reranking: {avg_regular_time:.1f}ms")
        print(f"   Time Overhead: {time_overhead:+.1f}%")
        
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Precision@5:")
        print(f"     With Reranking: {avg_rerank_precision:.3f}")
        print(f"     Without Reranking: {avg_regular_precision:.3f}")
        print(f"     Improvement: {precision_improvement:+.1f}%")
        
        print(f"   NDCG@5:")
        print(f"     With Reranking: {avg_rerank_ndcg:.3f}")
        print(f"     Without Reranking: {avg_regular_ndcg:.3f}")
        print(f"     Improvement: {ndcg_improvement:+.1f}%")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if precision_improvement > 10:
            print(f"   ✅ Excellent precision improvement: {precision_improvement:+.1f}%")
        elif precision_improvement > 5:
            print(f"   ✅ Good precision improvement: {precision_improvement:+.1f}%")
        else:
            print(f"   ⚠️  Modest precision improvement: {precision_improvement:+.1f}%")
        
        if time_overhead < 50:
            print(f"   ✅ Acceptable time overhead: {time_overhead:+.1f}%")
        else:
            print(f"   ⚠️  High time overhead: {time_overhead:+.1f}%")
        
        return results


async def main():
    """Main test execution"""
    print("🔧 Reranking Performance Validation Test Suite")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async with RerankingPerformanceValidator() as validator:
        results = await validator.run_all_tests()
        
        if results:
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"/opt/stacks/lettatoolsselector/tests/performance/reranking_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
            
            # Performance recommendation
            precision_improvement = results["performance"]["accuracy_metrics"]["precision_at_5"]["improvement_percent"]
            time_overhead = results["performance"]["average_response_time"]["overhead_percent"]
            
            print(f"\n🎯 RECOMMENDATION:")
            if precision_improvement > 10 and time_overhead < 100:
                print("   ✅ Reranking provides excellent value - recommend enabling in production")
            elif precision_improvement > 5:
                print("   ✅ Reranking provides good improvements - recommend enabling for accuracy-critical workflows")
            else:
                print("   ⚠️  Reranking improvements are modest - evaluate based on specific use cases")


if __name__ == "__main__":
    asyncio.run(main())