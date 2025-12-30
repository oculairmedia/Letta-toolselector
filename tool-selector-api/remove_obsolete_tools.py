import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from dotenv import load_dotenv
import os
import weaviate.classes.query as wq
from fetch_all_tools import fetch_all_tools
from compare_tools import get_weaviate_tools # Assuming compare_tools.py is in the same directory

def remove_obsolete_tools():
    """Remove tools from Weaviate that no longer exist in Letta."""
    
    print("Fetching tools from Letta...")
    letta_tools = fetch_all_tools()
    letta_tool_names = {tool["name"] for tool in letta_tools}
    print(f"Found {len(letta_tool_names)} tools in Letta.")

    print("\nFetching tools from Weaviate...")
    weaviate_tools_dict = get_weaviate_tools()
    weaviate_tool_names = set(weaviate_tools_dict.keys())
    print(f"Found {len(weaviate_tool_names)} tools in Weaviate.")

    # Determine obsolete tools (in Weaviate but not Letta)
    obsolete_tool_names = weaviate_tool_names - letta_tool_names
    
    if not obsolete_tool_names:
        print("\nNo obsolete tools found in Weaviate.")
        return
        
    print(f"\nFound {len(obsolete_tool_names)} obsolete tools to remove:")
    for name in sorted(obsolete_tool_names):
        print(f"- {name}")
    
    load_dotenv()
    
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
        deleted_count = 0
        
        print("\nAttempting to remove obsolete tools...")
        for tool_name in obsolete_tool_names:
            try:
                # Find the tool by name
                name_filter = wq.Filter.by_property("name").equal(tool_name)
                result = collection.query.fetch_objects(
                    limit=1,
                    filters=name_filter
                )
                
                if result.objects:
                    # Delete the tool by name
                    result = collection.data.delete_many(
                        where=name_filter,
                        verbose=True
                    )
                    # The delete_many operation returns a DeleteManyReturn object
                    # with successful and failed counts
                    if hasattr(result, 'successful') and result.successful > 0:
                        deleted_count += 1
                        print(f"- Successfully deleted tool: {tool_name}")
                    else:
                        print(f"- Failed to delete tool: {tool_name}")
                    deleted_count += 1
                    print(f"- Deleted tool: {tool_name}")
                else:
                    print(f"- Tool not found: {tool_name}")
            
            except Exception as e:
                print(f"Error deleting tool {tool_name}: {str(e)}")
        
        print(f"\nOperation complete. Removed {deleted_count} obsolete tools.")
    
    finally:
        client.close()

if __name__ == "__main__":
    remove_obsolete_tools()