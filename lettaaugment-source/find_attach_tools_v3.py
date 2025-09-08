import json
import requests
import sys

def find_attach_tools(query: str = None, agent_id: str = None, keep_tools: str = None, 
                        limit: int = 10, min_score: float = 50.0, request_heartbeat: bool = False) -> str:
    """
    Silently manage tools for the agent.
    
    Args:
        query (str): Your search query - what kind of tool are you looking for?
        agent_id (str): Your agent ID
        keep_tools (str): Comma-separated list of tool IDs to preserve
        limit (int): Maximum number of tools to find (default: 10)
        min_score (float): Minimum match score 0-100 (default: 75.0)
        request_heartbeat (bool): Whether to request an immediate heartbeat (default: False)
    
    Returns:
        str: Success message or error details if something goes wrong
    """
    url = "http://100.81.139.20/api/v1/tools/attach"
    headers = {"Content-Type": "application/json"}
    
    # Convert keep_tools string to list
    keep_tool_ids = []
    if keep_tools:
        keep_tool_ids = [t.strip() for t in keep_tools.split(',')]
    
    # Build payload with only non-None values
    payload = {
        "limit": limit if limit is not None else 3,
        "min_score": min_score if min_score is not None else 75.0,
        "keep_tools": keep_tool_ids,
        "request_heartbeat": request_heartbeat
    }
    
    # Only add optional parameters if they are provided
    if query is not None:
        payload["query"] = query
    if agent_id is not None and agent_id != "":
        payload["agent_id"] = agent_id

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        
        # Print full details to stdout for logging
        print("Details:", file=sys.stdout)
        print(json.dumps(result, indent=2), file=sys.stdout)
        
        # Return minimal response
        if response.status_code == 200 and result.get("success"):
            return "Tools updated successfully."
        else:
            error = result.get("error", f"HTTP {response.status_code}")
            return f"Error: {error}"
            
    except Exception as e:
        print(f"Error details: {str(e)}", file=sys.stdout)
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Get args from sys.argv without using argparse
    args = {}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            key = sys.argv[i][2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i+1].startswith('--'):
                args[key] = sys.argv[i+1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    # Convert types
    limit = int(args.get('limit', '10'))
    min_score = float(args.get('min_score', '50.0'))
    request_heartbeat = args.get('request_heartbeat', 'false').lower() == 'true'
    
    result = find_attach_tools(
        query=args.get('query'),
        agent_id=args.get('agent_id'),
        keep_tools=args.get('keep_tools'),
        limit=limit,
        min_score=min_score,
        request_heartbeat=request_heartbeat
    )
    print(result)
