import os
import json
from dotenv import load_dotenv
from weaviate_tool_search import search_tools
import textwrap
import time

def format_tool_result(tool, index):
    """Format a single tool result for display"""
    # Create header with tool name and match score
    match_score = round((1 - tool.get('distance', 1)) * 100, 2)
    header = f"{index}. {tool['name']} (Match: {match_score}%)"
    header = f"\n{header}\n{'-' * len(header)}"
    
    # Format basic info
    basic_info = [
        f"ID: {tool.get('tool_id', 'N/A')}",
        f"Type: {tool.get('source_type', 'N/A')}",
        f"Tags: {', '.join(tool.get('tags', []))}",
        f"\nDescription:",
        textwrap.fill(tool['description'], width=80, initial_indent='  ', subsequent_indent='  ')
    ]
    
    # Format schema if available
    schema_info = []
    if tool.get('json_schema'):
        schema = tool['json_schema']
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except:
                pass
                
        if isinstance(schema, dict):
            schema_info.extend([
                "\nInput Parameters:"
            ])
            
            # Extract input parameters
            if 'properties' in schema:
                for param_name, param_info in schema['properties'].items():
                    param_desc = param_info.get('description', 'No description')
                    param_type = param_info.get('type', 'any')
                    param_info = textwrap.fill(
                        f"{param_name} ({param_type}): {param_desc}",
                        width=76,
                        initial_indent='  - ',
                        subsequent_indent='    '
                    )
                    schema_info.append(param_info)

    return "\n".join([header] + basic_info + schema_info)

def test_comprehensive_search():
    """Test the Weaviate tool search functionality with a comprehensive set of queries"""
    
    # Comprehensive test queries covering various tool categories
    test_queries = [
        # Content and social media
        "create or publish a post",
        "manage social media content",
        
        # API and integration
        "api integration tools",
        "connect to external services",
        
        # Data and analytics
        "data analysis tools",
        "generate reports or analytics",
        
        # File operations
        "file upload and download",
        "document management",
        
        # Search and retrieval
        "search tools",
        "find information in databases",
        
        # Communication
        "send messages or notifications",
        "email integration",
        
        # Utility functions
        "date and time utilities",
        "format conversion tools"
    ]
    
    print("\nComprehensive Weaviate Tool Search Test")
    print("=" * 80)
    
    results_summary = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("=" * 80)
        
        try:
            start_time = time.time()
            tools = search_tools(query=query, limit=3)
            search_time = time.time() - start_time
            
            print(f"\nFound {len(tools)} matching tools (search took {search_time:.2f}s):")
            
            # Sort tools by match score (inverse of distance)
            tools.sort(key=lambda x: x.get('distance', 1), reverse=False)
            
            for idx, tool in enumerate(tools, 1):
                print(format_tool_result(tool, idx))
                print()
                
            # Add to summary
            if tools:
                top_match = tools[0]['name'] if tools else "No results"
                top_score = round((1 - tools[0].get('distance', 1)) * 100, 2) if tools else 0
            else:
                top_match = "No results"
                top_score = 0
                
            results_summary.append({
                "query": query,
                "results_count": len(tools),
                "top_match": top_match,
                "top_score": top_score,
                "search_time": search_time
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results_summary.append({
                "query": query,
                "error": str(e)
            })
        
        print("=" * 80)
    
    # Print summary table
    print("\n\nSearch Results Summary")
    print("=" * 80)
    print(f"{'Query':<30} | {'Results':<8} | {'Top Match':<30} | {'Score':<8} | {'Time (s)':<8}")
    print("-" * 90)
    
    for result in results_summary:
        if "error" in result:
            print(f"{result['query']:<30} | ERROR: {result['error']}")
        else:
            print(f"{result['query'][:30]:<30} | {result['results_count']:<8} | {result['top_match'][:30]:<30} | {result['top_score']:<8.2f} | {result['search_time']:<8.2f}")

if __name__ == "__main__":
    # Verify required environment variables
    load_dotenv()
    # Only OpenAI API key is required for local connection
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        exit(1)
        
    test_comprehensive_search()