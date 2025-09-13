#!/usr/bin/env python3
"""
Test Enhancement Script

This script demonstrates the tool description enhancement functionality by:
1. Loading sample tools from sample_tools.json
2. Using enhancement prompts to create enhanced descriptions
3. Comparing original vs enhanced descriptions
4. Testing different prompting strategies
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from enhancement_prompts import EnhancementPrompts, ToolContext
from ollama_client import OllamaClient, OllamaResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancementTester:
    """
    Tool description enhancement tester with comparison and analysis capabilities.
    """
    
    def __init__(self, 
                 ollama_base_url: str = "http://100.81.139.20:11434/v1",
                 ollama_model: str = "gemma3:12b"):
        self.client = OllamaClient(base_url=ollama_base_url, model=ollama_model)
        self.results = []
        
    async def load_sample_tools(self, file_path: str = "sample_tools.json") -> List[Dict[str, Any]]:
        """Load sample tools from JSON file"""
        try:
            with open(file_path, 'r') as f:
                tools = json.load(f)
            logger.info(f"Loaded {len(tools)} sample tools")
            return tools
        except FileNotFoundError:
            logger.error(f"Sample tools file {file_path} not found")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return []
    
    def tool_to_context(self, tool: Dict[str, Any]) -> ToolContext:
        """Convert tool dictionary to ToolContext object"""
        return ToolContext(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            tool_type=tool.get("tool_type", ""),
            mcp_server_name=tool.get("mcp_server_name"),
            parameters=tool.get("json_schema", {}).get("parameters"),
            category=tool.get("category"),
            tags=tool.get("tags", [])
        )
    
    async def enhance_single_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance description for a single tool"""
        logger.info(f"Enhancing tool: {tool.get('name', 'unknown')}")
        
        # Convert to ToolContext
        tool_context = self.tool_to_context(tool)
        
        # Build enhancement prompt
        system_prompt, user_prompt = EnhancementPrompts.build_prompt(tool_context)
        
        # Get enhancement from Ollama
        start_time = time.time()
        response = await self.client.enhance_description(user_prompt, system_prompt)
        
        # Build result
        result = {
            "tool_id": tool.get("id", "unknown"),
            "tool_name": tool.get("name", "unknown"),
            "tool_type": tool.get("tool_type", "unknown"),
            "mcp_server": tool.get("mcp_server_name"),
            "category": tool.get("category"),
            "original_description": tool.get("description", ""),
            "enhanced_description": response.content if response.success else None,
            "enhancement_success": response.success,
            "processing_time": response.processing_time,
            "error_message": response.error_message,
            "prompt_category": EnhancementPrompts.categorize_tool(tool_context).value,
            "enhancement_time": time.time() - start_time
        }
        
        if response.success:
            logger.info(f"✅ Enhanced {tool.get('name')} in {response.processing_time:.2f}s")
        else:
            logger.error(f"❌ Failed to enhance {tool.get('name')}: {response.error_message}")
        
        return result
    
    async def enhance_all_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance descriptions for all tools"""
        logger.info(f"Starting enhancement of {len(tools)} tools")
        
        results = []
        for tool in tools:
            result = await self.enhance_single_tool(tool)
            results.append(result)
            
            # Small delay between requests to be respectful
            await asyncio.sleep(0.5)
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze enhancement results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        total_tools = len(self.results)
        successful = sum(1 for r in self.results if r["enhancement_success"])
        failed = total_tools - successful
        
        # Calculate metrics
        total_time = sum(r["processing_time"] for r in self.results if r["processing_time"])
        avg_time = total_time / total_tools if total_tools > 0 else 0
        
        # Description length analysis
        original_lengths = [len(r["original_description"]) for r in self.results]
        enhanced_lengths = [len(r["enhanced_description"]) for r in self.results if r["enhanced_description"]]
        
        avg_original_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0
        avg_enhanced_length = sum(enhanced_lengths) / len(enhanced_lengths) if enhanced_lengths else 0
        
        # Category breakdown
        category_breakdown = {}
        for result in self.results:
            cat = result["prompt_category"]
            if cat not in category_breakdown:
                category_breakdown[cat] = {"total": 0, "successful": 0}
            category_breakdown[cat]["total"] += 1
            if result["enhancement_success"]:
                category_breakdown[cat]["successful"] += 1
        
        return {
            "summary": {
                "total_tools": total_tools,
                "successful_enhancements": successful,
                "failed_enhancements": failed,
                "success_rate": successful / total_tools if total_tools > 0 else 0,
                "total_processing_time": total_time,
                "average_processing_time": avg_time
            },
            "description_analysis": {
                "average_original_length": avg_original_length,
                "average_enhanced_length": avg_enhanced_length,
                "length_increase_factor": avg_enhanced_length / avg_original_length if avg_original_length > 0 else 0
            },
            "category_breakdown": category_breakdown
        }
    
    def save_results(self, output_dir: str = "results") -> str:
        """Save results to files"""
        # Create results directory
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{output_dir}/enhancement_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "results": self.results,
                "analysis": self.analyze_results(),
                "client_stats": self.client.get_stats()
            }, f, indent=2)
        
        # Save human-readable comparison
        comparison_file = f"{output_dir}/comparison_{timestamp}.md"
        with open(comparison_file, 'w') as f:
            f.write("# Tool Description Enhancement Results\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            analysis = self.analyze_results()
            f.write("## Summary\n\n")
            f.write(f"- **Total Tools**: {analysis['summary']['total_tools']}\n")
            f.write(f"- **Success Rate**: {analysis['summary']['success_rate']:.1%}\n")
            f.write(f"- **Avg Processing Time**: {analysis['summary']['average_processing_time']:.2f}s\n")
            f.write(f"- **Description Length Increase**: {analysis['description_analysis']['length_increase_factor']:.1f}x\n\n")
            
            # Individual comparisons
            f.write("## Tool Comparisons\n\n")
            for result in self.results:
                if result["enhancement_success"]:
                    f.write(f"### {result['tool_name']}\n\n")
                    f.write(f"**Type**: {result['tool_type']} | **Category**: {result['category']}\n\n")
                    if result['mcp_server']:
                        f.write(f"**MCP Server**: {result['mcp_server']}\n\n")
                    
                    f.write("**Original Description:**\n")
                    f.write(f"> {result['original_description']}\n\n")
                    
                    f.write("**Enhanced Description:**\n")
                    f.write(f"> {result['enhanced_description']}\n\n")
                    
                    f.write(f"**Processing Time**: {result['processing_time']:.2f}s\n\n")
                    f.write("---\n\n")
        
        logger.info(f"Results saved to {results_file} and {comparison_file}")
        return results_file
    
    def print_comparison(self, tool_name: Optional[str] = None):
        """Print comparison for a specific tool or all tools"""
        if not self.results:
            print("No results available")
            return
        
        results_to_show = self.results
        if tool_name:
            results_to_show = [r for r in self.results if r["tool_name"] == tool_name]
        
        for result in results_to_show:
            if result["enhancement_success"]:
                print(f"\n{'='*60}")
                print(f"TOOL: {result['tool_name']}")
                print(f"Type: {result['tool_type']} | Category: {result['category']}")
                if result['mcp_server']:
                    print(f"MCP Server: {result['mcp_server']}")
                print(f"Processing: {result['processing_time']:.2f}s")
                print(f"{'='*60}")
                
                print(f"\nORIGINAL ({len(result['original_description'])} chars):")
                print(f"{result['original_description']}")
                
                print(f"\nENHANCED ({len(result['enhanced_description'])} chars):")
                print(f"{result['enhanced_description']}")
                
                print(f"\nLENGTH INCREASE: {len(result['enhanced_description']) / len(result['original_description']):.1f}x")
            else:
                print(f"\n❌ FAILED: {result['tool_name']} - {result['error_message']}")


async def main():
    """Main test function"""
    logger.info("Starting Tool Description Enhancement Test")
    
    # Initialize tester
    tester = EnhancementTester()
    
    # Test connection first
    logger.info("Testing Ollama connection...")
    if not await tester.client.test_connection():
        logger.error("❌ Cannot connect to Ollama. Check configuration and try again.")
        return
    
    logger.info("✅ Ollama connection successful")
    
    # Load sample tools
    tools = await tester.load_sample_tools()
    if not tools:
        logger.error("❌ No tools loaded. Cannot continue.")
        return
    
    # Enhance all tools
    logger.info("Starting enhancement process...")
    results = await tester.enhance_all_tools(tools)
    
    # Analyze and display results
    analysis = tester.analyze_results()
    
    print(f"\n{'='*80}")
    print("ENHANCEMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Success Rate: {analysis['summary']['success_rate']:.1%} ({analysis['summary']['successful_enhancements']}/{analysis['summary']['total_tools']})")
    print(f"Average Processing Time: {analysis['summary']['average_processing_time']:.2f}s")
    print(f"Description Length Increase: {analysis['description_analysis']['length_increase_factor']:.1f}x")
    print(f"Total Processing Time: {analysis['summary']['total_processing_time']:.2f}s")
    
    # Show category breakdown
    print(f"\nCATEGORY BREAKDOWN:")
    for category, stats in analysis['category_breakdown'].items():
        success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {category}: {success_rate:.1%} ({stats['successful']}/{stats['total']})")
    
    # Print detailed comparisons
    print(f"\n{'='*80}")
    print("DETAILED COMPARISONS")
    print(f"{'='*80}")
    tester.print_comparison()
    
    # Save results
    results_file = tester.save_results()
    
    # Client statistics
    client_stats = tester.client.get_stats()
    print(f"\n{'='*80}")
    print("CLIENT STATISTICS")
    print(f"{'='*80}")
    print(json.dumps(client_stats, indent=2))
    
    logger.info(f"✅ Enhancement test completed. Results saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())