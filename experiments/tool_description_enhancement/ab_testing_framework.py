#!/usr/bin/env python3
"""
A/B Testing Framework for Tool Description Enhancement

This framework evaluates the impact of LLM-enhanced tool descriptions on search accuracy
by comparing search results between original and enhanced descriptions using statistical
analysis with confidence intervals.

Key Features:
- Split-test search results with/without enhanced descriptions
- Statistical significance testing
- Search relevance metrics (NDCG, MRR, Hit Rate)
- Performance tracking and comprehensive reporting
- Integration with existing reranking system
"""

import asyncio
import json
import os
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from scipy import stats
import numpy as np

import requests
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestQuery:
    """Represents a test query with expected relevant tools"""
    query: str
    expected_tools: List[str]  # Tool names that should be relevant
    category: str  # Query category for analysis
    priority: float = 1.0  # Relative importance of this query

@dataclass
class SearchResult:
    """Represents a search result with metrics"""
    tool_name: str
    score: float
    rank: int
    is_relevant: bool = False

@dataclass
class TestResult:
    """Results from A/B test comparison"""
    query: str
    category: str
    enhanced_results: List[SearchResult]
    baseline_results: List[SearchResult]
    enhanced_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    performance_improvement: Dict[str, float]
    search_time_enhanced: float
    search_time_baseline: float

class SearchMetrics:
    """Calculate search relevance metrics"""

    @staticmethod
    def hit_rate_at_k(results: List[SearchResult], k: int = 5) -> float:
        """Calculate hit rate at position k"""
        top_k = results[:k]
        hits = sum(1 for r in top_k if r.is_relevant)
        return hits / min(k, len(results)) if results else 0.0

    @staticmethod
    def mean_reciprocal_rank(results: List[SearchResult]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, result in enumerate(results):
            if result.is_relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(results: List[SearchResult], k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        def dcg(relevance_scores: List[float]) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(relevance_scores))

        # Get relevance scores for top-k results
        top_k = results[:k]
        actual_scores = [1.0 if r.is_relevant else 0.0 for r in top_k]

        if not any(actual_scores):
            return 0.0

        # Calculate DCG
        actual_dcg = dcg(actual_scores)

        # Calculate ideal DCG (perfect ranking)
        ideal_scores = sorted(actual_scores, reverse=True)
        ideal_dcg = dcg(ideal_scores)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @classmethod
    def calculate_all_metrics(cls, results: List[SearchResult]) -> Dict[str, float]:
        """Calculate all search metrics"""
        return {
            "hit_rate_at_1": cls.hit_rate_at_k(results, 1),
            "hit_rate_at_3": cls.hit_rate_at_k(results, 3),
            "hit_rate_at_5": cls.hit_rate_at_k(results, 5),
            "mrr": cls.mean_reciprocal_rank(results),
            "ndcg_at_3": cls.ndcg_at_k(results, 3),
            "ndcg_at_5": cls.ndcg_at_k(results, 5),
            "total_relevant": sum(1 for r in results if r.is_relevant),
            "total_results": len(results)
        }

class ABTestFramework:
    """A/B Testing framework for tool description enhancement evaluation"""

    def __init__(self,
                 api_base_url: str = "http://localhost:8020",
                 weaviate_collection_enhanced: str = "ToolEnhanced",
                 weaviate_collection_baseline: str = "Tool"):
        self.api_base_url = api_base_url
        self.weaviate_collection_enhanced = weaviate_collection_enhanced
        self.weaviate_collection_baseline = weaviate_collection_baseline
        self.test_results: List[TestResult] = []

        # Load test queries
        self.test_queries = self._load_test_queries()

    def _load_test_queries(self) -> List[TestQuery]:
        """Load test queries from configuration or create default set"""
        # Default test queries covering various tool categories
        default_queries = [
            TestQuery(
                query="create issue project management",
                expected_tools=["huly_create_issue", "huly_create_subissue"],
                category="project_management"
            ),
            TestQuery(
                query="file operations read write",
                expected_tools=["Read", "Write", "Edit"],
                category="file_operations"
            ),
            TestQuery(
                query="memory management agent",
                expected_tools=["create_memory_block", "update_memory_block", "list_memory_blocks"],
                category="memory_management"
            ),
            TestQuery(
                query="search tools vector database",
                expected_tools=["find_tools", "Grep", "Glob"],
                category="search_discovery"
            ),
            TestQuery(
                query="run commands terminal bash",
                expected_tools=["Bash", "BashOutput"],
                category="system_operations"
            ),
            TestQuery(
                query="web search fetch content",
                expected_tools=["WebSearch", "WebFetch"],
                category="web_operations"
            ),
            TestQuery(
                query="agent creation management",
                expected_tools=["create_agent", "list_agents", "modify_agent"],
                category="agent_management"
            ),
            TestQuery(
                query="git version control",
                expected_tools=["Bash"],  # git commands through bash
                category="version_control"
            ),
            TestQuery(
                query="documentation bookstack wiki",
                expected_tools=["create_page", "create_book", "list_pages"],
                category="documentation"
            ),
            TestQuery(
                query="data analysis metrics",
                expected_tools=["find_tools"],  # analytics tools
                category="analytics"
            )
        ]

        # Try to load from file if it exists
        test_queries_file = Path("ab_test_queries.json")
        if test_queries_file.exists():
            try:
                with open(test_queries_file, 'r') as f:
                    data = json.load(f)
                    return [TestQuery(**q) for q in data]
            except Exception as e:
                logger.warning(f"Failed to load test queries from file: {e}. Using defaults.")

        return default_queries

    async def search_with_collection(self,
                                   query: str,
                                   collection_name: str,
                                   limit: int = 10) -> Tuple[List[Dict], float]:
        """Search using specific Weaviate collection"""
        start_time = time.time()

        # Use the search endpoint with collection parameter
        url = f"{self.api_base_url}/api/v1/tools/search"
        payload = {
            "query": query,
            "limit": limit,
            "weaviate_collection": collection_name
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        results = await response.json()
                        search_time = time.time() - start_time
                        return results, search_time
                    else:
                        logger.error(f"Search failed with status {response.status}")
                        return [], time.time() - start_time
            except Exception as e:
                logger.error(f"Search request failed: {e}")
                return [], time.time() - start_time

    def _mark_relevant_results(self,
                              results: List[Dict],
                              expected_tools: List[str]) -> List[SearchResult]:
        """Convert search results to SearchResult objects with relevance marking"""
        search_results = []

        for i, result in enumerate(results):
            tool_name = result.get('name', '')
            score = result.get('score', 0.0)
            is_relevant = tool_name in expected_tools

            search_results.append(SearchResult(
                tool_name=tool_name,
                score=score,
                rank=i + 1,
                is_relevant=is_relevant
            ))

        return search_results

    async def run_single_test(self, test_query: TestQuery) -> TestResult:
        """Run A/B test for a single query"""
        logger.info(f"Testing query: '{test_query.query}' (category: {test_query.category})")

        # Search with enhanced descriptions
        enhanced_results_raw, enhanced_time = await self.search_with_collection(
            test_query.query,
            self.weaviate_collection_enhanced
        )

        # Search with baseline descriptions
        baseline_results_raw, baseline_time = await self.search_with_collection(
            test_query.query,
            self.weaviate_collection_baseline
        )

        # Mark relevant results
        enhanced_results = self._mark_relevant_results(enhanced_results_raw, test_query.expected_tools)
        baseline_results = self._mark_relevant_results(baseline_results_raw, test_query.expected_tools)

        # Calculate metrics
        enhanced_metrics = SearchMetrics.calculate_all_metrics(enhanced_results)
        baseline_metrics = SearchMetrics.calculate_all_metrics(baseline_results)

        # Calculate performance improvements
        performance_improvement = {}
        for metric in enhanced_metrics:
            if metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                if baseline_val > 0:
                    improvement = (enhanced_val - baseline_val) / baseline_val * 100
                else:
                    improvement = float('inf') if enhanced_val > 0 else 0.0
                performance_improvement[metric] = improvement

        return TestResult(
            query=test_query.query,
            category=test_query.category,
            enhanced_results=enhanced_results,
            baseline_results=baseline_results,
            enhanced_metrics=enhanced_metrics,
            baseline_metrics=baseline_metrics,
            performance_improvement=performance_improvement,
            search_time_enhanced=enhanced_time,
            search_time_baseline=baseline_time
        )

    async def run_full_test_suite(self) -> List[TestResult]:
        """Run complete A/B test suite"""
        logger.info(f"Starting A/B test suite with {len(self.test_queries)} queries")

        results = []
        for i, test_query in enumerate(self.test_queries, 1):
            logger.info(f"Progress: {i}/{len(self.test_queries)}")

            result = await self.run_single_test(test_query)
            results.append(result)
            self.test_results.append(result)

            # Small delay between tests
            await asyncio.sleep(0.5)

        logger.info("✅ A/B test suite completed")
        return results

    def calculate_statistical_significance(self, metric_name: str) -> Dict[str, float]:
        """Calculate statistical significance for a metric across all tests"""
        enhanced_values = []
        baseline_values = []

        for result in self.test_results:
            if metric_name in result.enhanced_metrics and metric_name in result.baseline_metrics:
                enhanced_values.append(result.enhanced_metrics[metric_name])
                baseline_values.append(result.baseline_metrics[metric_name])

        if len(enhanced_values) < 2:
            return {"p_value": 1.0, "significant": False, "effect_size": 0.0}

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(enhanced_values, baseline_values)

        # Calculate effect size (Cohen's d)
        diff = np.array(enhanced_values) - np.array(baseline_values)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

        # Determine significance (p < 0.05)
        significant = p_value < 0.05

        return {
            "p_value": p_value,
            "t_statistic": t_stat,
            "significant": significant,
            "effect_size": effect_size,
            "mean_improvement": np.mean(diff),
            "confidence_interval": stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive A/B test report with statistical analysis"""
        if not self.test_results:
            return {"error": "No test results available"}

        # Overall metrics aggregation
        all_metrics = ["hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "mrr", "ndcg_at_3", "ndcg_at_5"]

        overall_summary = {}
        significance_results = {}

        for metric in all_metrics:
            enhanced_vals = [r.enhanced_metrics.get(metric, 0) for r in self.test_results]
            baseline_vals = [r.baseline_metrics.get(metric, 0) for r in self.test_results]

            overall_summary[metric] = {
                "enhanced_mean": statistics.mean(enhanced_vals),
                "baseline_mean": statistics.mean(baseline_vals),
                "relative_improvement": ((statistics.mean(enhanced_vals) - statistics.mean(baseline_vals)) / statistics.mean(baseline_vals) * 100) if statistics.mean(baseline_vals) > 0 else 0
            }

            significance_results[metric] = self.calculate_statistical_significance(metric)

        # Category breakdown
        category_breakdown = {}
        for result in self.test_results:
            cat = result.category
            if cat not in category_breakdown:
                category_breakdown[cat] = {
                    "queries": 0,
                    "enhanced_metrics": {},
                    "baseline_metrics": {},
                    "improvements": {}
                }

            category_breakdown[cat]["queries"] += 1
            for metric in all_metrics:
                if metric not in category_breakdown[cat]["enhanced_metrics"]:
                    category_breakdown[cat]["enhanced_metrics"][metric] = []
                    category_breakdown[cat]["baseline_metrics"][metric] = []
                    category_breakdown[cat]["improvements"][metric] = []

                category_breakdown[cat]["enhanced_metrics"][metric].append(result.enhanced_metrics.get(metric, 0))
                category_breakdown[cat]["baseline_metrics"][metric].append(result.baseline_metrics.get(metric, 0))
                category_breakdown[cat]["improvements"][metric].append(result.performance_improvement.get(metric, 0))

        # Calculate category averages
        for cat in category_breakdown:
            for metric in all_metrics:
                enhanced_vals = category_breakdown[cat]["enhanced_metrics"][metric]
                baseline_vals = category_breakdown[cat]["baseline_metrics"][metric]
                improvement_vals = category_breakdown[cat]["improvements"][metric]

                category_breakdown[cat]["enhanced_metrics"][metric] = statistics.mean(enhanced_vals) if enhanced_vals else 0
                category_breakdown[cat]["baseline_metrics"][metric] = statistics.mean(baseline_vals) if baseline_vals else 0
                category_breakdown[cat]["improvements"][metric] = statistics.mean(improvement_vals) if improvement_vals else 0

        # Performance analysis
        search_time_enhanced = [r.search_time_enhanced for r in self.test_results]
        search_time_baseline = [r.search_time_baseline for r in self.test_results]

        return {
            "test_summary": {
                "total_queries": len(self.test_results),
                "categories": len(category_breakdown),
                "test_date": datetime.now().isoformat()
            },
            "overall_metrics": overall_summary,
            "statistical_significance": significance_results,
            "category_breakdown": category_breakdown,
            "performance_analysis": {
                "enhanced_search_time": {
                    "mean": statistics.mean(search_time_enhanced),
                    "median": statistics.median(search_time_enhanced),
                    "std": statistics.stdev(search_time_enhanced) if len(search_time_enhanced) > 1 else 0
                },
                "baseline_search_time": {
                    "mean": statistics.mean(search_time_baseline),
                    "median": statistics.median(search_time_baseline),
                    "std": statistics.stdev(search_time_baseline) if len(search_time_baseline) > 1 else 0
                }
            },
            "detailed_results": [asdict(r) for r in self.test_results]
        }

    def save_report(self, output_dir: str = "ab_test_results") -> str:
        """Save comprehensive report to files"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate report
        report = self.generate_comprehensive_report()

        # Save JSON report
        json_file = f"{output_dir}/ab_test_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save human-readable report
        md_file = f"{output_dir}/ab_test_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            self._write_markdown_report(f, report)

        logger.info(f"A/B test report saved to {json_file} and {md_file}")
        return json_file

    def _write_markdown_report(self, file, report: Dict[str, Any]):
        """Write human-readable markdown report"""
        file.write("# A/B Test Report: Tool Description Enhancement\n\n")
        file.write(f"**Generated**: {report['test_summary']['test_date']}\n")
        file.write(f"**Total Queries**: {report['test_summary']['total_queries']}\n")
        file.write(f"**Categories Tested**: {report['test_summary']['categories']}\n\n")

        # Executive Summary
        file.write("## Executive Summary\n\n")
        overall = report['overall_metrics']
        sig = report['statistical_significance']

        file.write("### Key Findings\n\n")
        for metric in ["hit_rate_at_5", "mrr", "ndcg_at_5"]:
            if metric in overall:
                improvement = overall[metric]['relative_improvement']
                p_value = sig[metric]['p_value']
                significant = "✅ Significant" if sig[metric]['significant'] else "❌ Not Significant"

                file.write(f"- **{metric.upper()}**: {improvement:+.1f}% improvement, p={p_value:.4f} ({significant})\n")

        # Detailed Metrics
        file.write("\n## Detailed Metrics\n\n")
        file.write("| Metric | Enhanced | Baseline | Improvement | P-Value | Significant |\n")
        file.write("|--------|----------|----------|-------------|---------|-------------|\n")

        for metric in overall:
            enhanced = overall[metric]['enhanced_mean']
            baseline = overall[metric]['baseline_mean']
            improvement = overall[metric]['relative_improvement']
            p_val = sig[metric]['p_value']
            significant = "Yes" if sig[metric]['significant'] else "No"

            file.write(f"| {metric} | {enhanced:.3f} | {baseline:.3f} | {improvement:+.1f}% | {p_val:.4f} | {significant} |\n")

        # Category Breakdown
        file.write("\n## Category Analysis\n\n")
        for category, data in report['category_breakdown'].items():
            file.write(f"### {category.replace('_', ' ').title()}\n\n")
            file.write(f"**Queries**: {data['queries']}\n\n")

            file.write("| Metric | Enhanced | Baseline | Improvement |\n")
            file.write("|--------|----------|----------|-------------|\n")

            for metric in ["hit_rate_at_5", "mrr", "ndcg_at_5"]:
                if metric in data['enhanced_metrics']:
                    enhanced = data['enhanced_metrics'][metric]
                    baseline = data['baseline_metrics'][metric]
                    improvement = data['improvements'][metric]
                    file.write(f"| {metric} | {enhanced:.3f} | {baseline:.3f} | {improvement:+.1f}% |\n")
            file.write("\n")

        # Performance Analysis
        perf = report['performance_analysis']
        file.write("## Performance Analysis\n\n")
        file.write(f"- **Enhanced Search Time**: {perf['enhanced_search_time']['mean']:.3f}s ± {perf['enhanced_search_time']['std']:.3f}s\n")
        file.write(f"- **Baseline Search Time**: {perf['baseline_search_time']['mean']:.3f}s ± {perf['baseline_search_time']['std']:.3f}s\n")

        time_diff = perf['enhanced_search_time']['mean'] - perf['baseline_search_time']['mean']
        file.write(f"- **Time Difference**: {time_diff:+.3f}s\n\n")

    def print_summary(self):
        """Print quick summary of results"""
        if not self.test_results:
            print("No test results available")
            return

        report = self.generate_comprehensive_report()

        print(f"\n{'='*70}")
        print("A/B TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total Queries: {len(self.test_results)}")
        print(f"Categories: {len(report['category_breakdown'])}")

        print(f"\nKEY METRICS IMPROVEMENT:")
        for metric in ["hit_rate_at_5", "mrr", "ndcg_at_5"]:
            if metric in report['overall_metrics']:
                improvement = report['overall_metrics'][metric]['relative_improvement']
                p_value = report['statistical_significance'][metric]['p_value']
                significant = "✅" if report['statistical_significance'][metric]['significant'] else "❌"
                print(f"  {metric:15}: {improvement:+6.1f}% (p={p_value:.4f}) {significant}")


async def main():
    """Main function to run A/B test framework"""
    logger.info("Starting A/B Test Framework for Tool Description Enhancement")

    # Initialize framework
    framework = ABTestFramework()

    # Check if we have both collections (would need to be set up)
    logger.info("Note: This framework requires two Weaviate collections:")
    logger.info("  - 'Tool' (baseline with original descriptions)")
    logger.info("  - 'ToolEnhanced' (with LLM-enhanced descriptions)")
    logger.info("  Make sure both collections are populated before running tests.")

    # Run test suite
    results = await framework.run_full_test_suite()

    # Generate and save report
    report_file = framework.save_report()

    # Print summary
    framework.print_summary()

    logger.info(f"✅ A/B testing completed. Report saved to {report_file}")


if __name__ == "__main__":
    asyncio.run(main())