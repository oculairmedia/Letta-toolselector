import os
import json
from dotenv import load_dotenv
from weaviate_tool_search import search_tools
import textwrap

def format_tool_result(tool, index):
    """Format a single tool result for display"""
    # Create header with tool name and match score
    match_score = round((1 - tool.get('distance', 1)) * 100, 2)
    header = f"\n{index}. {tool['name']} (Match: {match_score}%)"
    header = f"{header}\n{'-' * len(header)}"
    
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
            schema_info.extend(["\nInput Parameters:"])
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

def interactive_search():
    """Interactive tool for testing Weaviate search"""
    print("\nWeaviate Tool Search Interface")
    print("Enter 'quit' to exit, 'help' for example queries")
    print("=" * 80)
    
    example_queries = [
        "tools for social media and content management",
        "search and research tools",
        "tools for managing posts and scheduling",
        "API integration tools",
        "tools for managing time and schedules"
    ]
    
    while True:
        try:
            print("\nOptions:")
            print("1. Enter a search query")
            print("2. See example queries")
            print("3. Quit")
            
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == "3" or choice.lower() == "quit":
                break
                
            if choice == "2":
                print("\nExample queries:")
                for i, query in enumerate(example_queries, 1):
                    print(f"{i}. {query}")
                continue
            
            query = input("\nEnter search query: ").strip()
            if query.lower() in ["quit", "exit"]:
                break
            
            if not query:
                print("Please enter a valid query")
                continue
                
            print("\nSearching...")
            print("=" * 80)
            
            try:
                tools = search_tools(query=query, limit=5)
                print(f"\nFound {len(tools)} matching tools:")
                
                # Sort tools by match score
                tools.sort(key=lambda x: x.get('distance', 1), reverse=False)
                
                for idx, tool in enumerate(tools, 1):
                    print(format_tool_result(tool, idx))
                    print()
                    
            except Exception as e:
                print(f"Error performing search: {str(e)}")
            
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Verify required environment variables
    load_dotenv()
    required_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        exit(1)
        
    interactive_search()