import os
from dotenv import load_dotenv
from weaviate_tool_search import search_tools
import json
import textwrap

def format_tool_result(tool, index):
    """Format a single tool result for display"""
    match_score = round((1 - tool.get('distance', 1)) * 100, 2)
    header = f"\n{index}. {tool['name']} (Match: {match_score}%)"
    if "explain_score" in tool:
        header += f"\nScore explanation: {tool['explain_score']}"
    header = f"{header}\n{'-' * 80}"
    
    basic_info = [
        f"ID: {tool.get('tool_id', 'N/A')}",
        f"Type: {tool.get('source_type', 'N/A')}",
        f"Tags: {', '.join(tool.get('tags', []))}",
        f"\nDescription:",
        textwrap.fill(tool['description'], width=80, initial_indent='  ', subsequent_indent='  ')
    ]
    return "\n".join([header] + basic_info)

def print_results(result):
    """Print search results, handling both grouped and flat results."""
    if result["grouped"]:
        for group_name, tools in result["groups"].items():
            print(f"\nGroup: {group_name}")
            print("-" * 40)
            
            if not tools:
                print("No tools in this group")
                continue
                
            tools.sort(key=lambda x: x.get('distance', 1), reverse=False)
            for idx, tool in enumerate(tools, 1):
                print(format_tool_result(tool, idx))
    else:
        if not result["tools"]:
            print("\nNo tools found")
            return
            
        print(f"\nFound {len(result['tools'])} matching tools:")
        result["tools"].sort(key=lambda x: x.get('distance', 1), reverse=False)
        for idx, tool in enumerate(result["tools"], 1):
            print(format_tool_result(tool, idx))

def test_advanced_search():
    """Test advanced search features."""
    test_cases = [
        {
            "name": "Basic Query with Query Expansion",
            "query": "create a post",
            "params": {"limit": 3}
        },
        {
            "name": "Filter by Tags",
            "query": "list or display content",
            "params": {
                "limit": 3,
                "filter_tags": ["mcp:postiz", "mcp:ghost"]
            }
        },
        {
            "name": "Group by Source Type",
            "query": "manage content",
            "params": {
                "limit": 5,
                "group_by": "source_type"
            }
        },
        {
            "name": "Vector-Heavy Search (alpha=0.9)",
            "query": "tools for content management",
            "params": {
                "limit": 3,
                "alpha": 0.9
            }
        },
        {
            "name": "Keyword-Heavy Search (alpha=0.3)",
            "query": "tools for content management",
            "params": {
                "limit": 3,
                "alpha": 0.3
            }
        }
    ]
    
    print("\nTesting Advanced Search Features")
    print("=" * 80)
    
    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        print(f"Query: {case['query']}")
        print(f"Parameters: {case['params']}")
        print("=" * 80)
        
        try:
            result = search_tools(query=case['query'], **case['params'])
            print_results(result)
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "#" * 100 + "\n")

if __name__ == "__main__":
    load_dotenv()
    required_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        exit(1)
        
    test_advanced_search()