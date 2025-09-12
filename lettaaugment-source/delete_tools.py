import requests
import json
# import time  # Currently unused

def list_tools() -> None:
    """List all tools and their full details with pagination."""
    base_url = "https://letta2.oculair.ca/v1/tools"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-BARE-PASSWORD": "password lettaSecurePass123"
    }
    
    page = 1
    limit = 20
    total_tools = 0
    
    try:
        while True:
            url = f"{base_url}?limit={limit}&after={((page-1)*limit) if page > 1 else ''}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                tools = response.json()
                if not tools:  # No more tools to fetch
                    break
                    
                print(f"\nPage {page} Tools:")
                print("===============")
                for tool in tools:
                    total_tools += 1
                    print(f"ID: {tool.get('id')}")
                    print(f"Name: {tool.get('name')}")
                    print(f"Type: {tool.get('tool_type')}")
                    print(f"Description: {tool.get('description')}")
                    if tool.get('metadata_'):
                        print(f"Metadata: {json.dumps(tool.get('metadata_'), indent=2)}")
                    if tool.get('tags'):
                        print(f"Tags: {', '.join(tool.get('tags'))}")
                    print("Source type:", tool.get('source_type'))
                    print("Organization ID:", tool.get('organization_id'))
                    print("Created by:", tool.get('created_by_id'))
                    print("Last updated by:", tool.get('last_updated_by_id'))
                    print("Return char limit:", tool.get('return_char_limit'))
                    print("===============")
                    print()
                
                # Ask whether to continue to next page
                if len(tools) == limit:
                    print(f"\nShowing tools {(page-1)*limit + 1} to {page*limit}")
                    print("Press Enter to see more tools, or type 'q' to stop listing: ")
                    if input().lower() == 'q':
                        break
                else:
                    break
                    
                page += 1
                
            else:
                print(f"Failed to get tools list: {response.status_code}")
                break
                
        print(f"\nTotal tools listed: {total_tools}")
        
    except Exception as e:
        print(f"Error getting tools: {str(e)}")

def delete_tool(tool_id: str) -> bool:
    """Delete a tool from Letta."""
    url = f"https://letta2.oculair.ca/v1/tools/{tool_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-BARE-PASSWORD": "password lettaSecurePass123"
    }
    
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code == 200:
            print(f"Successfully deleted tool {tool_id}")
            return True
        else:
            print(f"Failed to delete tool {tool_id}: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error deleting tool {tool_id}: {str(e)}")
        return False

def main():
    # List all tools first with pagination
    list_tools()
    
    # Ask for tool ID
    tool_id = input("\nEnter the tool ID to delete: ")
    
    if tool_id:
        if delete_tool(tool_id):
            print("\nTool successfully deleted")
        else:
            print("\nFailed to delete tool")
    else:
        print("\nNo tool ID provided")

if __name__ == "__main__":
    main()