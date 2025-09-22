import sys
from typing import Optional

from tool_selector_client import DEFAULT_LIMIT, DEFAULT_MIN_SCORE, attach_tools_as_json


def _log(message: str) -> None:
    print(f"[find_tools] {message}", file=sys.stderr)
def Find_tools(
    query: str = None,
    agent_id: str = None,
    keep_tools: str = None,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    request_heartbeat: bool = False,
) -> str:
    """
    Silently manage tools for the agent.

    Args:
        query (str): Your search query - what kind of tool are you looking for?
        agent_id (str): Your agent ID
        keep_tools (str): Comma-separated list of tool IDs to preserve
        limit (int): Maximum number of tools to find (default: 10)
        min_score (float): Minimum match score 0-100 (default: 50.0)
        request_heartbeat (bool): Whether to request an immediate heartbeat (default: False)

    Returns:
        str: JSON string describing the outcome of the tool update
    """

    return attach_tools_as_json(
        query=query,
        agent_id=agent_id,
        keep_tools=keep_tools,
        limit=limit,
        min_score=min_score,
        request_heartbeat=bool(request_heartbeat),
        logger=_log,
    )

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
    limit_value = args.get('limit')
    min_score_value = args.get('min_score')
    request_heartbeat = args.get('request_heartbeat', 'false').lower() == 'true'

    result = Find_tools(
        query=args.get('query'),
        agent_id=args.get('agent_id'),
        keep_tools=args.get('keep_tools'),
        limit=limit_value,
        min_score=min_score_value,
        request_heartbeat=request_heartbeat
    )
    print(result)
