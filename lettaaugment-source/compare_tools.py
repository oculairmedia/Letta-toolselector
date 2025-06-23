import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from fetch_all_tools import fetch_all_tools
from dotenv import load_dotenv
import os
from pprint import pprint

def get_weaviate_tools():
    """Fetch all tools from Weaviate."""
    load_dotenv()
    
    # Initialize Weaviate client
    print("\nConnecting to Weaviate...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv('WEAVIATE_URL'),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        },
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=60)
        )
    )
    
    try:
        # Get Tool collection
        collection = client.collections.get("Tool")
        
        # Query all tools
        result = collection.query.fetch_objects(
            limit=1000,  # Adjust if you have more tools
        )
        
        return {obj.properties["name"]: obj.properties for obj in result.objects}
    finally:
        client.close()

def compare_tools():
    """Compare tools between Letta and Weaviate."""
    print("Fetching tools from Letta...")
    letta_tools = fetch_all_tools()
    letta_tools_dict = {tool["name"]: tool for tool in letta_tools}
    
    print(f"Found {len(letta_tools)} tools in Letta")
    
    print("\nFetching tools from Weaviate...")
    weaviate_tools = get_weaviate_tools()
    print(f"Found {len(weaviate_tools)} tools in Weaviate")
    
    # Find tools in Weaviate but not in Letta (obsolete)
    obsolete_tools = set(weaviate_tools.keys()) - set(letta_tools_dict.keys())
    
    # Find tools in Letta but not in Weaviate (new)
    new_tools = set(letta_tools_dict.keys()) - set(weaviate_tools.keys())
    
    # Find tools that exist in both
    common_tools = set(weaviate_tools.keys()) & set(letta_tools_dict.keys())
    
    print("\nSummary:")
    print(f"- Tools in both systems: {len(common_tools)}")
    print(f"- New tools (in Letta but not Weaviate): {len(new_tools)}")
    print(f"- Obsolete tools (in Weaviate but not Letta): {len(obsolete_tools)}")
    
    if obsolete_tools:
        print("\nObsolete tools that should be removed from Weaviate:")
        for name in sorted(obsolete_tools):
            tool = weaviate_tools[name]
            print(f"- {name} ({tool.get('tool_id', 'no id')}) [{tool.get('tool_type', 'unknown type')}]")
    
    if new_tools:
        print("\nNew tools that should be added to Weaviate:")
        for name in sorted(new_tools):
            tool = letta_tools_dict[name]
            print(f"- {name} ({tool.get('id', 'no id')}) [{tool.get('tool_type', 'unknown type')}]")

if __name__ == "__main__":
    compare_tools()