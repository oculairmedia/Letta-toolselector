import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional
import argparse
import logging

# Setup logging based on debug level
logging.basicConfig()
logger = logging.getLogger(__name__)

def find_attach_tools(
    query: str,
    target_agent_id: Optional[str] = None,
    limit: int = 5,
    min_score: float = 75.0,
    request_heartbeat: bool = False,
    debug_level: str = "INFO",
    weaviate_base: str = "http://100.81.139.20:8020"
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
        weaviate_base (str):
            Base URL for Weaviate API server. Default: http://100.81.139.20:8020

    Returns:
        Optional[str]: A JSON string with operation results or None on failure
    """

    # Set logging level
    log_level = getattr(logging, debug_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    logger.info(f"Starting find_attach_tools with query: '{query}', agent_id: {target_agent_id}")
    logger.debug(f"Parameters - limit: {limit}, min_score: {min_score}, heartbeat: {request_heartbeat}, url: {weaviate_base}")

    if not query:
        logger.error("Query cannot be empty.")
        return None
    if not target_agent_id:
        logger.error("Target agent ID is required.")
        return json.dumps({"error": "Target agent ID is required."})


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
        logger.info(f"Fetching current tools for agent {target_agent_id} from {letta_base}")
        detach_results = {
            "detached_tools": [],
            "failed_tools": []
        }

        agent_response = session.get(f"{letta_base}/agents/{target_agent_id}", headers=headers)
        logger.debug(f"Letta agent GET response status: {agent_response.status_code}")
        agent_response.raise_for_status()
        agent_data = agent_response.json()
        logger.debug(f"Agent data received: {json.dumps(agent_data, indent=2)}")

        mcp_tools = [tool for tool in agent_data.get("tools", [])
                    if tool.get("tool_type") == "external_mcp"]
        logger.info(f"Found {len(mcp_tools)} existing MCP tools to detach.")

        for tool in mcp_tools:
            tool_id = tool.get('id')
            tool_name = tool.get('name', 'N/A')
            logger.info(f"Attempting to detach tool: ID={tool_id}, Name={tool_name}")
            try:
                detach_url = f"{letta_base}/agents/{target_agent_id}/tools/detach/{tool_id}"
                logger.debug(f"Sending PATCH to {detach_url}")
                detach_response = session.patch(detach_url, headers=headers)
                logger.debug(f"Letta detach response status: {detach_response.status_code}")
                detach_response.raise_for_status()

                detach_results["detached_tools"].append({
                    "tool_id": tool_id,
                    "name": tool_name,
                    "type": tool.get("tool_type")
                })
                logger.info(f"Successfully detached tool {tool_name} ({tool_id})")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to detach tool {tool_name} ({tool_id}): {error_msg}")
                detach_results["failed_tools"].append({
                    "tool_id": tool_id,
                    "name": tool_name,
                    "error": error_msg
                })

        # 2. Search and attach new tools via the Weaviate API server
        logger.info(f"Searching and attaching tools via {weaviate_base} with query: '{query}'")
        search_payload = {
            "query": query,
            "limit": limit,
            "agent_id": target_agent_id # Use 'agent_id' as expected by the API
            # min_score is handled server-side by attach_tools_from_query
        }
        logger.debug(f"Sending attach payload: {json.dumps(search_payload)}")

        attach_url = f"{weaviate_base}/api/v1/tools/attach"
        search_response = session.post(attach_url, json=search_payload)
        logger.debug(f"Attach API response status: {search_response.status_code}")
        search_response.raise_for_status() # Will raise HTTPError for 4xx/5xx
        search_result = search_response.json() # This is the response from the API
        logger.debug(f"Attach API response body: {json.dumps(search_result, indent=2)}")

        # 3. Combine results
        final_result = {
            "detached_tools": detach_results["detached_tools"],
            "failed_detachments": detach_results["failed_tools"],
            "search_results": search_result.get("matches", []),
            "attached_tools": search_result.get("attached_tools", []),
            "failed_attachments": search_result.get("failed_attachments", [])
        }
        logger.info("Find and attach process completed.")

        return json.dumps(final_result, indent=2)

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {type(e).__name__} - {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            try:
                logger.error(f"Response body: {e.response.text}")
            except Exception:
                logger.error("Could not read response body.")
        return json.dumps({"error": f"HTTP Request failed: {str(e)}"})
    except Exception as e:
        logger.error(f"An unexpected error occurred: {type(e).__name__} - {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and attach tools to a Letta agent via Weaviate search API.')
    parser.add_argument('--weaviate_url', default='http://100.81.139.20:8020', help='Weaviate tool search API base URL')
    parser.add_argument('--query', required=True, help='Search query for finding relevant tools')
    parser.add_argument('--agent_id', required=True, help='ID of the target Letta agent')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of tools to find and attach')
    parser.add_argument('--min_score', type=float, default=75.0, help='Minimum similarity score (0-100) - Used server-side')
    parser.add_argument('--request_heartbeat', action='store_true', help='(Currently unused in this script) Request immediate heartbeat')
    parser.add_argument('--debug_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging verbosity')

    args = parser.parse_args()

    result = find_attach_tools(
        query=args.query,
        target_agent_id=args.agent_id,
        limit=args.limit,
        min_score=args.min_score, # Passed to function but used server-side
        request_heartbeat=args.request_heartbeat,
        debug_level=args.debug_level,
        weaviate_base=args.weaviate_url
    )

    if result:
        print(result)
    else:
        print("Script execution failed. Check logs for details.")