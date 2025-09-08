import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, List

def find_attach_tools_v2(
    query: str,
    target_agent_id: Optional[str] = None,
    limit: int = 5,
    min_score: float = 75.0,
    request_heartbeat: bool = False,
    debug_level: str = "INFO"
) -> Optional[str]:
    """
    Find and attach relevant tools to an agent using semantic search.
    First detaches any existing MCP tools, then searches for and attaches new tools.
    
    Queries should be expanded to include relevant keywords that will match tool descriptions.
    Think about different ways the same functionality might be described:

    Example Expansions:
    - "What's happening in Toronto this weekend?"
      → "Toronto events activities entertainment shows concerts festivals exhibitions calendar schedule upcoming weekend search find discover local city"
    
    - "Send a message to the team about project updates"
      → "message communication notify alert team group chat email send broadcast update project status report inform"
    
    - "Help me organize my files"
      → "file document organize sort categorize manage folder directory storage system archive search find move copy"

    - "Track my daily tasks"
      → "task todo list track manage organize schedule planner calendar reminder project timeline progress monitor"

    The key is to include:
    1. Different ways to describe the action (e.g., search/find/discover/lookup)
    2. Related concepts and use cases
    3. Common tool naming patterns
    4. Associated functionality users might need

    Args:
        query (str):
            The expanded search query with relevant keywords for tool matching.
        target_agent_id (Optional[str]):
            ID of the agent to attach tools to. Defaults to the server's default agent.
        limit (int):
            Maximum number of tools to return and attach (default: 5).
        min_score (float):
            Minimum similarity score (0-100) for tools to be included (default: 75.0).
        request_heartbeat (bool):
            Request an immediate heartbeat after function execution.
        debug_level (str):
            Logging verbosity level (DEBUG, INFO, WARNING, ERROR). Default: INFO

    Returns:
        Optional[str]: A JSON string with operation results or None on failure
    """
    # Provide tool info if requested
    if query == "__tool_info__":
        info = {
            "name": "find_attach_tools_v2",
            "description": "Find and attach relevant tools using semantic search. Queries should be expanded with synonyms and related terms for better matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Expanded search query with relevant keywords. Example: 'Toronto events' → 'Toronto events activities entertainment shows concerts festivals calendar local city search find discover'"
                    },
                    "target_agent_id": {
                        "type": "string",
                        "description": "ID of the agent to attach tools to (defaults to server's default agent if not provided)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tools to return and attach (1-20)",
                        "default": 5
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (0-100) for tools to be included in results",
                        "default": 75.0
                    },
                    "request_heartbeat": {
                        "type": "boolean",
                        "description": "Request an immediate heartbeat after function execution",
                        "default": False
                    },
                    "debug_level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "description": "Controls logging verbosity",
                        "default": "INFO"
                    }
                },
                "required": ["query"]
            }
        }
        import json
        return json.dumps(info)

    if not query:
        return None

    # API endpoints
    letta_base = "https://letta2.oculair.ca/v1"
    weaviate_host = "100.81.139.20"
    weaviate_port = 8020
    weaviate_base = f"http://{weaviate_host}:{weaviate_port}"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-BARE-PASSWORD": "password lettaSecurePass123"
    }

    # Set up session with retries
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PATCH"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        # 1. First get current agent tools and detach MCP tools
        detach_results = {
            "detached_tools": [],
            "failed_tools": []
        }

        if target_agent_id:
            agent_response = session.get(f"{letta_base}/agents/{target_agent_id}", headers=headers)
            if agent_response.status_code == 200:
                agent_data = agent_response.json()
                mcp_tools = [tool for tool in agent_data.get("tools", []) 
                            if tool.get("tool_type") == "external_mcp"]

                for tool in mcp_tools:
                    try:
                        detach_response = session.patch(
                            f"{letta_base}/agents/{target_agent_id}/tools/detach/{tool['id']}", 
                            headers=headers
                        )
                        if detach_response.status_code == 200:
                            detach_results["detached_tools"].append({
                                "tool_id": tool["id"],
                                "name": tool["name"],
                                "type": tool["tool_type"]
                            })
                    except:
                        detach_results["failed_tools"].append({
                            "tool_id": tool["id"],
                            "name": tool["name"]
                        })

        # 2. Search and attach new tools
        search_payload = {
            "q": query,
            "size": limit,
            "threshold": min_score,
            "agent": target_agent_id
        }

        search_response = session.post(f"{weaviate_base}/api/v1/tools/attach", json=search_payload)
        if search_response.status_code != 200:
            return None
            
        search_result = search_response.json()

        # 3. Combine results
        final_result = {
            "detached_tools": detach_results["detached_tools"],
            "failed_detachments": detach_results["failed_tools"],
            "search_results": search_result.get("matches", []),
            "attached_tools": search_result.get("attached_tools", []),
            "failed_attachments": search_result.get("failed_attachments", [])
        }

        import json
        return json.dumps(final_result, indent=2)

    except:
        return None
    finally:
        session.close()

if __name__ == "__main__":
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description="Find and attach tools using semantic search.")
    parser.add_argument("--query", required=True, help="Search query for finding relevant tools.")
    parser.add_argument("--target_agent_id", dest="target_agent_id", default=None, help="ID of the target agent (optional).")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of tools to find (default: 5).")
    parser.add_argument("--min_score", type=float, default=75.0, help="Minimum similarity score (0-100) (default: 75.0).")
    parser.add_argument("--request_heartbeat", action="store_true", help="Request immediate heartbeat after execution.")
    parser.add_argument("--debug_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging verbosity level (default: INFO).")

    args = parser.parse_args()

    try:
        result_json = find_attach_tools_v2(
            query=args.query,
            target_agent_id=args.target_agent_id,
            limit=args.limit,
            min_score=args.min_score,
            request_heartbeat=args.request_heartbeat,
            debug_level=args.debug_level
        )
        print(result_json if result_json is not None else "None")

    except:
        print("None")
        sys.exit(1)
