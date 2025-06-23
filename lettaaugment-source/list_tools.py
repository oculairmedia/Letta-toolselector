from weaviate_tool_search import init_client

def main():
    all_tools = []
    client = None
    
    try:
        client = init_client()
        collection = client.collections.get("Tool")
        
        # First get total count
        count_result = collection.aggregate.over_all(total_count=True)
        total_count = count_result.total_count
        print(f"\nTotal tools in Weaviate: {total_count}\n")

        # Get all tools with pagination since we have 139 tools
        result = collection.query.fetch_objects(limit=200)  # Set limit higher than total count
        
        if hasattr(result, 'objects'):
            all_tools = [
                {
                    'name': obj.properties.get('name', 'Unknown Name'),
                    'description': obj.properties.get('description', 'No description'),
                    'source_type': obj.properties.get('source_type', 'No type'),
                    'tags': obj.properties.get('tags', [])
                }
                for obj in result.objects
            ]
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if client:
            client.close()
    
    if all_tools:
        print(f"Found {len(all_tools)} tools in Weaviate:\n")
        
        for i, tool in enumerate(all_tools, 1):
            print(f"{i}. {tool['name']}")
            print(f"   Description: {tool['description']}")
            print(f"   Type: {tool['source_type']}")
            print(f"   Tags: {', '.join(tool['tags'])}")
            print()

if __name__ == "__main__":
    main()