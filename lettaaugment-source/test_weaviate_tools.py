import os
import json
from dotenv import load_dotenv
from weaviate_tool_finder import tool_implementations

# Load environment variables
load_dotenv()

def test_upload_and_search():
    """Test uploading tools to Weaviate and searching them"""
    try:
        # First upload tools
        print("\nUploading tools to Weaviate...")
        result = tool_implementations["upload_tools_to_weaviate"]("all_letta_tools_20250414_130502.json")
        print("Upload result:", result)
        
        if result.get("status") == "success":
            # Test various search queries
            queries = [
                "find tools for web research",
                "tools for managing social media",
                "tools for data analysis"
            ]
            
            print("\nTesting tool searches...")
            for query in queries:
                print(f"\nSearching for: {query}")
                search_result = tool_implementations["find_tools_weaviate"](query, 3)
                
                if search_result.get("status") == "success":
                    tools = search_result.get("tools", [])
                    print(f"Found {len(tools)} tools:")
                    for i, tool in enumerate(tools, 1):
                        print(f"\n{i}. {tool['name']}")
                        print(f"   Description: {tool['description'][:200]}...")
                else:
                    print(f"Search failed: {search_result.get('message')}")
        else:
            print(f"Upload failed: {result.get('message')}")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    # Verify environment variables
    required_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        exit(1)
        
    test_upload_and_search()