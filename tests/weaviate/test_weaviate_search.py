import os
import json
from dotenv import load_dotenv
from weaviate_tool_search import search_tools
import textwrap

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

def test_tool_search():
    """Test the Weaviate tool search functionality"""
    
    # Test queries focused on Postizz tools
    test_queries = [
        "create or publish a post",
        "social media posting tools",
        "tools for managing posts and content",
        "postizz api tools",
        "post creation and management"
    ]
    
    print("\nTesting Weaviate Tool Search")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("=" * 80)
        
        try:
            tools = search_tools(query=query, limit=3)
            print(f"\nFound {len(tools)} matching tools:")
            
            # Sort tools by match score (inverse of distance)
            tools.sort(key=lambda x: x.get('distance', 1), reverse=False)
            
            for idx, tool in enumerate(tools, 1):
                print(format_tool_result(tool, idx))
                print()
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("=" * 80)
        
        # input("\nPress Enter to continue to next query...") # Removed for automated testing

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
        
    test_tool_search()