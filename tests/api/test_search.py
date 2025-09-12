import os
from dotenv import load_dotenv
from weaviate_tool_search import search_tools
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

def test_search():
    """Test semantic search with query expansion."""
    test_queries = [
        "create new post",
        "list social posts",
        "search content",
        "manage api integrations",
        "view blog articles",
        "update existing content"
    ]
    
    print("\nTesting Semantic Search with Query Expansion")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("=" * 80)
        
        try:
            tools = search_tools(query=query, limit=3)
            print(f"\nFound {len(tools)} matching tools:")
            
            tools.sort(key=lambda x: x.get('distance', 1), reverse=False)
            for idx, tool in enumerate(tools, 1):
                print(format_tool_result(tool, idx))
                
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
        
    test_search()