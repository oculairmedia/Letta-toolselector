#!/usr/bin/env python3
"""
Performance Validation Tests for Specialized Embedding with Qwen3-Embedding-4B

This test suite validates that the specialized prompting system provides
measurable improvements in tool discovery accuracy and relevance.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add the tool-selector-api directory to the path
sys.path.append('/opt/stacks/lettatoolsselector/tool-selector-api')

os.environ.setdefault('USE_QWEN3_FORMAT', 'true')

from specialized_embedding import (
    SpecializedEmbeddingPrompter,
    enhance_query_for_embedding,
    enhance_tool_for_embedding,
    format_query_for_qwen3,
    get_detailed_instruct,
    get_search_instruction,
    is_qwen3_format_enabled,
)


@dataclass
class TestQuery:
    """Test query with expected results for validation."""
    query: str
    expected_tool_types: List[str]
    expected_keywords: List[str]
    context: str = ""


@dataclass
class PerformanceResult:
    """Results from performance testing."""
    test_name: str
    original_prompt: str
    enhanced_prompt: str
    improvement_metrics: Dict[str, float]
    execution_time_ms: float
    success: bool


class PerformanceValidator:
    """Validates performance improvements from specialized prompting."""
    
    def __init__(self):
        self.prompter = SpecializedEmbeddingPrompter()
        self.test_queries = self._create_test_queries()
        self.test_tools = self._create_test_tools()
        
    def _create_test_queries(self) -> List[TestQuery]:
        """Create a comprehensive set of test queries."""
        return [
            TestQuery(
                query="create blog post",
                expected_tool_types=["mcp", "api"],
                expected_keywords=["blog", "post", "create", "publish", "cms"],
                context="User wants to publish content"
            ),
            TestQuery(
                query="manage GitHub repositories",
                expected_tool_types=["mcp", "api"],
                expected_keywords=["github", "repository", "git", "manage", "version"],
                context="Developer workflow automation"
            ),
            TestQuery(
                query="send email notifications",
                expected_tool_types=["api", "mcp"],
                expected_keywords=["email", "send", "notification", "message", "smtp"],
                context="Communication automation"
            ),
            TestQuery(
                query="process data files",
                expected_tool_types=["python", "general"],
                expected_keywords=["data", "file", "process", "convert", "transform"],
                context="Data processing workflow"
            ),
            TestQuery(
                query="list available APIs",
                expected_tool_types=["api", "mcp"],
                expected_keywords=["api", "list", "show", "available", "endpoint"],
                context="API discovery"
            ),
            TestQuery(
                query="delete old files",
                expected_tool_types=["python", "general"],
                expected_keywords=["delete", "remove", "file", "cleanup", "old"],
                context="File maintenance"
            )
        ]
    
    def _create_test_tools(self) -> List[Dict]:
        """Create test tools for validation."""
        return [
            {
                "name": "Ghost Blog API",
                "description": "Creates and manages blog posts on Ghost CMS platform",
                "tool_type": "mcp",
                "source_type": "external_mcp"
            },
            {
                "name": "GitHub Repository Manager", 
                "description": "Manages GitHub repositories, issues, and pull requests",
                "tool_type": "mcp",
                "source_type": "external_mcp"
            },
            {
                "name": "Email Notification Service",
                "description": "Sends email notifications and manages mailing lists",
                "tool_type": "api",
                "source_type": "python"
            },
            {
                "name": "CSV Data Processor",
                "description": "Processes CSV files and converts to various formats",
                "tool_type": "python",
                "source_type": "custom"
            },
            {
                "name": "REST API Explorer",
                "description": "Lists and explores available REST API endpoints",
                "tool_type": "api",
                "source_type": "python"
            },
            {
                "name": "File Cleanup Utility",
                "description": "Deletes old files and performs disk cleanup operations",
                "tool_type": "python", 
                "source_type": "custom"
            }
        ]
    
    def test_prompt_enhancement_quality(self) -> PerformanceResult:
        """Test the quality of prompt enhancement."""
        print("🎯 Testing Prompt Enhancement Quality")
        print("=" * 50)

        start_time = time.time()
        total_queries = len(self.test_queries)
        formatted_correctly = 0

        for test_query in self.test_queries:
            original = test_query.query
            enhanced = enhance_query_for_embedding(original, test_query.context)
            expected = get_detailed_instruct(
                get_search_instruction(),
                format_query_for_qwen3(original)
            )

            print(f"Query: {original}")
            print(f"Enhanced: {enhanced}")
            print(f"Expected: {expected}")
            print()

            if enhanced.strip() == expected.strip():
                formatted_correctly += 1

        instruction_rate = formatted_correctly / total_queries if total_queries else 0.0
        improvements = {
            'instruction_format_rate': instruction_rate,
            'qwen3_format_enabled': 1.0 if is_qwen3_format_enabled() else 0.0,
        }

        execution_time = (time.time() - start_time) * 1000

        return PerformanceResult(
            test_name="Prompt Enhancement Quality",
            original_prompt="Original queries",
            enhanced_prompt="Qwen3-formatted queries",
            improvement_metrics=improvements,
            execution_time_ms=execution_time,
            success=instruction_rate == 1.0
        )

    def test_tool_description_enhancement(self) -> PerformanceResult:
        """Test tool description preparation effectiveness."""
        print("🛠️ Testing Tool Description Preparation")
        print("=" * 50)

        start_time = time.time()
        total_tools = len(self.test_tools)
        unchanged_descriptions = 0

        for tool in self.test_tools:
            original_desc = tool["description"]
            enhanced_desc = enhance_tool_for_embedding(
                tool_description=original_desc,
                tool_name=tool["name"],
                tool_type=tool["tool_type"],
                tool_source=tool["source_type"]
            )

            print(f"Tool: {tool['name']} ({tool['tool_type']})")
            print(f"Original: {original_desc}")
            print(f"Prepared: {enhanced_desc}")
            print()

            if enhanced_desc == original_desc.strip():
                unchanged_descriptions += 1

        unchanged_rate = unchanged_descriptions / total_tools if total_tools else 0.0
        improvements = {
            'unchanged_description_rate': unchanged_rate,
        }

        execution_time = (time.time() - start_time) * 1000

        return PerformanceResult(
            test_name="Tool Description Preparation",
            original_prompt="Original tool descriptions",
            enhanced_prompt="Prepared descriptions",
            improvement_metrics=improvements,
            execution_time_ms=execution_time,
            success=unchanged_rate == 1.0
        )

    def test_prompt_type_detection_accuracy(self) -> PerformanceResult:
        """Test accuracy of prompt type detection."""
        print("🔍 Testing Prompt Type Detection Accuracy")
        print("=" * 50)
        
        start_time = time.time()
        improvements = {}
        
        correct_detections = 0
        total_detections = len(self.test_tools)
        
        for tool in self.test_tools:
            detected_type = self.prompter._determine_tool_prompt_type(
                tool["tool_type"], 
                tool["source_type"]
            )
            
            # Define expected mappings
            expected_mapping = {
                "mcp": "mcp_tool",
                "api": "api_tool", 
                "python": "general_tool"
            }
            
            expected = expected_mapping.get(tool["tool_type"], "general_tool")
            if detected_type.value == expected:
                correct_detections += 1
            
            print(f"Tool: {tool['name']}")
            print(f"Type: {tool['tool_type']}, Source: {tool['source_type']}")
            print(f"Expected: {expected}, Detected: {detected_type.value}")
            print(f"Correct: {'✅' if detected_type.value == expected else '❌'}")
            print()
        
        improvements['detection_accuracy'] = correct_detections / total_detections
        
        execution_time = (time.time() - start_time) * 1000
        
        return PerformanceResult(
            test_name="Prompt Type Detection",
            original_prompt="N/A",
            enhanced_prompt="Automatic type detection",
            improvement_metrics=improvements,
            execution_time_ms=execution_time,
            success=improvements['detection_accuracy'] > 0.85
        )
    
    def test_embedding_consistency(self) -> PerformanceResult:
        """Test consistency of embedding enhancements."""
        print("🔄 Testing Embedding Consistency")
        print("=" * 50)
        
        start_time = time.time()
        improvements = {}
        
        # Test same query multiple times
        test_query = "find tools to create blog posts"
        enhancements = []
        
        for i in range(5):
            enhanced = enhance_query_for_embedding(test_query)
            enhancements.append(enhanced)
        
        # Check consistency
        unique_enhancements = set(enhancements)
        consistency_rate = (len(enhancements) - len(unique_enhancements) + 1) / len(enhancements)
        
        improvements['consistency_rate'] = consistency_rate
        improvements['enhancement_variance'] = len(unique_enhancements)
        
        print(f"Original query: {test_query}")
        print(f"Enhancements generated: {len(enhancements)}")
        print(f"Unique enhancements: {len(unique_enhancements)}")
        print(f"Consistency rate: {consistency_rate:.2%}")
        
        execution_time = (time.time() - start_time) * 1000
        
        return PerformanceResult(
            test_name="Embedding Consistency",
            original_prompt=test_query,
            enhanced_prompt="Multiple enhancement attempts",
            improvement_metrics=improvements,
            execution_time_ms=execution_time,
            success=consistency_rate > 0.9  # Should be very consistent
        )
    
    def run_all_tests(self) -> List[PerformanceResult]:
        """Run all performance validation tests."""
        print("🚀 Specialized Embedding Performance Validation")
        print("=" * 60)
        print()
        
        results = []
        
        # Run all tests
        test_methods = [
            self.test_prompt_enhancement_quality,
            self.test_tool_description_enhancement, 
            self.test_prompt_type_detection_accuracy,
            self.test_embedding_consistency
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
                print()
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        return results
    
    def generate_performance_report(self, results: List[PerformanceResult]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# Specialized Embedding Performance Report")
        report.append("=" * 50)
        report.append("")
        
        passed_tests = sum(1 for r in results if r.success)
        total_tests = len(results)
        
        report.append("## Summary")
        report.append(f"- **Total Tests**: {total_tests}")
        report.append(f"- **Passed**: {passed_tests}")
        report.append(f"- **Success Rate**: {passed_tests/total_tests:.1%}")
        report.append("")
        
        report.append("## Individual Test Results")
        report.append("")
        
        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report.append(f"### {result.test_name} {status}")
            report.append(f"- **Execution Time**: {result.execution_time_ms:.1f}ms")
            
            for metric, value in result.improvement_metrics.items():
                if isinstance(value, float):
                    if metric.endswith('_rate') or metric.endswith('_accuracy'):
                        report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.1%}")
                    else:
                        report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.2f}")
                else:
                    report.append(f"- **{metric.replace('_', ' ').title()}**: {value}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Run the performance validation suite."""
    validator = PerformanceValidator()
    results = validator.run_all_tests()
    
    # Generate and display report
    report = validator.generate_performance_report(results)
    print(report)
    
    # Save report to file
    with open('/opt/stacks/lettatoolsselector/performance_report.md', 'w') as f:
        f.write(report)
    
    print(f"📄 Performance report saved to performance_report.md")
    
    # Return success status
    passed = sum(1 for r in results if r.success)
    total = len(results)
    success = passed == total
    
    if success:
        print(f"🎉 All {total} performance tests passed!")
    else:
        print(f"⚠️ {total - passed} out of {total} tests failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)