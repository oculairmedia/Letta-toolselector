import requests
import json
from typing import Dict, Optional

def detach_mcp_tools(
    agent_id: str,
    debug_level: str = "INFO"
) -> Dict[str, Optional[str]]:
    """
    Detach all external MCP tools from the specified agent.

    Args:
        agent_id (str): 
            The unique identifier of the agent to detach tools from (e.g., "agent-123e4567-e89b-12d3-a456-426614174000").
            This is the UUID of the agent in the Letta system.
        debug_level (str, optional): 
            Controls the verbosity of logging. Valid values are:
            - "DEBUG": Most verbose, includes detailed debugging information
            - "INFO": Standard information messages (default)
            - "WARNING": Only warning and error messages
            - "ERROR": Only error messages
            Default is "INFO".

    Returns:
        Dict[str, Optional[str]]: A dictionary containing:
            - success: "true" if all tools were detached, "false" if any failed
            - response: JSON string containing lists of detached_tools and failed_tools
            - error: Error message if operation failed, None if successful
            - metadata: JSON string with operation statistics:
                - total_mcp_tools: Number of MCP tools found
                - detached_count: Number of tools successfully detached
                - failed_count: Number of tools that failed to detach
                - agent_id: ID of the agent processed

    Example:
        >>> result = detach_mcp_tools("agent-123e4567-e89b-12d3-a456-426614174000")
        >>> if result["success"] == "true":
        ...     print(f"Successfully detached {json.loads(result['metadata'])['detached_count']} tools")
        ... else:
        ...     print(f"Error: {result['error']}")
    """
    # Tool info for schema generation
    if agent_id == "__tool_info__":
        info = {
            "name": "detach_mcp_tools",
            "description": "Detach all external MCP tools from the specified agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "UUID of the agent to detach tools from (e.g., 'agent-123e4567-e89b-12d3-a456-426614174000')"
                    },
                    "debug_level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "description": "Controls logging verbosity. Options: DEBUG, INFO (default), WARNING, ERROR",
                        "default": "INFO"
                    }
                },
                "required": ["agent_id"]
            }
        }
        return {
            "success": "true",
            "response": json.dumps(info),
            "error": None,
            "metadata": None
        }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-BARE-PASSWORD": "password lettaSecurePass123"
    }

    base_url = "https://letta2.oculair.ca/v1"
    
    try:
        # First, get the agent's tools
        agent_url = f"{base_url}/agents/{agent_id}"
        agent_response = requests.get(agent_url, headers=headers)
        agent_response.raise_for_status()
        agent_data = agent_response.json()
        
        # Filter for external MCP tools
        mcp_tools = [tool for tool in agent_data.get("tools", []) 
                    if tool.get("tool_type") == "external_mcp"]
        
        results = {
            "detached_tools": [],
            "failed_tools": []
        }
        
        # Detach each MCP tool
        for tool in mcp_tools:
            try:
                detach_url = f"{base_url}/agents/{agent_id}/tools/detach/{tool['id']}"
                detach_response = requests.patch(detach_url, headers=headers)
                detach_response.raise_for_status()
                
                results["detached_tools"].append({
                    "tool_id": tool["id"],
                    "name": tool["name"],
                    "type": tool["tool_type"]
                })
            except Exception as e:
                results["failed_tools"].append({
                    "tool_id": tool["id"],
                    "name": tool["name"],
                    "error": f"{type(e).__name__}: {str(e)}"
                })

        # Create operation metadata
        metadata = {
            "total_mcp_tools": len(mcp_tools),
            "detached_count": len(results["detached_tools"]),
            "failed_count": len(results["failed_tools"]),
            "agent_id": agent_id
        }

        # Determine overall success
        success = len(results["failed_tools"]) == 0
        
        return {
            "success": "true" if success else "false",
            "response": json.dumps(results, indent=2),
            "error": None if success else f"Failed to detach {len(results['failed_tools'])} tools",
            "metadata": json.dumps(metadata)
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {
            "success": "false",
            "response": None,
            "error": error_msg,
            "metadata": json.dumps({"agent_id": agent_id})
        }

if __name__ == "__main__":
    # Test the tool
    test_agent_id = "agent-d5d91a6a-cc16-47dd-97be-07101cdbd49d"
    print(f"\nTesting with agent ID: {test_agent_id}")
    
    result = detach_mcp_tools(test_agent_id)
    
    if result["success"] == "true":
        response_data = json.loads(result["response"])
        metadata = json.loads(result["metadata"])
        
        print("\nOperation successful!")
        print(f"\nDetached {metadata['detached_count']} tools:")
        for tool in response_data["detached_tools"]:
            print(f"- {tool['name']} ({tool['tool_id']})")
            
        if response_data["failed_tools"]:
            print(f"\nFailed to detach {metadata['failed_count']} tools:")
            for tool in response_data["failed_tools"]:
                print(f"- {tool['name']}: {tool['error']}")
    else:
        print(f"\nOperation failed: {result['error']}")