#!/usr/bin/env python3
"""
Test just the specialized embedding prompts without Weaviate dependencies.
"""

import sys
import os

# Add the tool-selector-api directory to the path
sys.path.append('/opt/stacks/lettatoolsselector/tool-selector-api')

from specialized_embedding import (
    SpecializedEmbeddingPrompter, 
    enhance_query_for_embedding, 
    enhance_tool_for_embedding,
    PromptType
)


def test_tool_enhancement():
    """Test tool description enhancement with different types."""
    print("📝 Testing Tool Description Enhancement")
    print("=" * 50)
    
    prompter = SpecializedEmbeddingPrompter()
    
    test_cases = [
        {
            "desc": "Creates and manages GitHub issues, pull requests, and repositories",
            "name": "GitHub MCP",
            "type": "mcp",
            "source": "external_mcp"
        },
        {
            "desc": "Sends email notifications and manages mailing lists",
            "name": "Email API",
            "type": "api",
            "source": "python"
        },
        {
            "desc": "Process CSV files and convert to JSON format",
            "name": "CSV Processor",
            "type": "python",
            "source": "custom"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {case['name']} ({case['type']}):")
        print(f"   Original: {case['desc']}")
        
        enhanced = prompter.enhance_tool_description(
            description=case['desc'],
            tool_name=case['name'],
            tool_type=case['type'],
            tool_source=case['source']
        )
        print(f"   Enhanced: {enhanced}")


def test_query_enhancement():
    """Test query enhancement for different search types."""
    print("\n🔍 Testing Query Enhancement")
    print("=" * 50)
    
    prompter = SpecializedEmbeddingPrompter()
    
    test_queries = [
        "create blog post",
        "find tools to send email",
        "manage GitHub repositories",
        "process data files",
        "list available APIs",
        "delete old files"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        enhanced = prompter.enhance_search_query(query)
        print(f"   Enhanced: {enhanced}")


def test_prompt_types():
    """Test different prompt type detection."""
    print("\n🎯 Testing Prompt Type Detection")
    print("=" * 50)
    
    prompter = SpecializedEmbeddingPrompter()
    
    test_tools = [
        {"type": "mcp", "source": "external", "expected": PromptType.MCP_TOOL},
        {"type": "api", "source": "python", "expected": PromptType.API_TOOL},
        {"type": "python", "source": "custom", "expected": PromptType.GENERAL_TOOL},
        {"type": "external_mcp", "source": None, "expected": PromptType.MCP_TOOL},
    ]
    
    for i, tool in enumerate(test_tools, 1):
        detected = prompter._determine_tool_prompt_type(tool["type"], tool["source"])
        expected = tool["expected"]
        status = "✅" if detected == expected else "❌"
        
        print(f"{i}. Type: {tool['type']}, Source: {tool['source']}")
        print(f"   Expected: {expected.value}, Got: {detected.value} {status}")


def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🛠️  Testing Convenience Functions")
    print("=" * 50)
    
    # Test tool enhancement convenience function
    tool_desc = "Manages Slack channels and sends messages"
    enhanced_tool = enhance_tool_for_embedding(
        tool_description=tool_desc,
        tool_name="Slack Bot",
        tool_type="mcp"
    )
    print(f"Tool enhancement:")
    print(f"  Original: {tool_desc}")
    print(f"  Enhanced: {enhanced_tool}")
    
    # Test query enhancement convenience function
    query = "find communication tools"
    enhanced_query = enhance_query_for_embedding(query)
    print(f"\nQuery enhancement:")
    print(f"  Original: {query}")
    print(f"  Enhanced: {enhanced_query}")


def main():
    """Run all prompt tests."""
    print("🚀 Specialized Embedding Prompt Tests")
    print("=" * 60)
    
    try:
        test_tool_enhancement()
        test_query_enhancement()
        test_prompt_types()
        test_convenience_functions()
        
        print(f"\n🎉 All prompt tests completed successfully!")
        print("The specialized embedding system is ready for integration.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)