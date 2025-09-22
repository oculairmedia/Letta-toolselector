import os
import requests
import json
from typing import Optional, Dict
import sys

# Environment configuration
LETTA_URL = os.environ.get('LETTA_API_URL', 'https://letta.oculair.ca/v1')
# Ensure it's always HTTPS
if LETTA_URL.startswith('http://'):
    LETTA_URL = LETTA_URL.replace('http://', 'https://', 1)
if not LETTA_URL.endswith('/v1'):
    LETTA_URL = LETTA_URL.rstrip('/') + '/v1'

LETTA_API_KEY = os.environ.get('LETTA_PASSWORD')

_TOOL_SELECTOR_BASE_URL: Optional[str] = None
_TOOL_SELECTOR_WARNING_EMITTED = False


def _emit_tool_selector_warning(message: str) -> None:
    """Log a warning to stderr only once to avoid spamming."""
    global _TOOL_SELECTOR_WARNING_EMITTED
    if not _TOOL_SELECTOR_WARNING_EMITTED:
        print(f"[TOOL_SELECTOR] {message}", file=sys.stderr)
        _TOOL_SELECTOR_WARNING_EMITTED = True


def get_tool_selector_base_url() -> str:
    """Resolve the Tool Selector API base URL with sensible defaults."""
    global _TOOL_SELECTOR_BASE_URL
    if _TOOL_SELECTOR_BASE_URL is not None:
        return _TOOL_SELECTOR_BASE_URL

    candidates = [
        os.getenv('TOOLS_API_BASE_URL'),
        os.getenv('TOOLS_API_URL'),
        os.getenv('TOOL_SELECTOR_API_URL'),
        os.getenv('LETTA_TOOL_SELECTOR_URL'),
        os.getenv('WEAVIATE_TOOLS_API_URL'),
        os.getenv('TOOLS_API_DEFAULT_URL'),
    ]

    base = next((c.strip() for c in candidates if c and c.strip()), None)

    if not base:
        base = 'https://tool-selector-api:8020'
        _emit_tool_selector_warning(
            "TOOLS_API_BASE_URL not set; defaulting to https://tool-selector-api:8020. "
            "Set TOOLS_API_BASE_URL to the fully-qualified Tool Selector API endpoint."
        )

    if base.startswith('http://'):
        allow_insecure = os.getenv('TOOLS_API_ALLOW_INSECURE_HTTP', '').lower() in {'1', 'true', 'yes', 'on'}
        if not allow_insecure:
            secure_base = 'https://' + base.split('://', 1)[1]
            _emit_tool_selector_warning(
                f"Promoting Tool Selector URL from '{base}' to '{secure_base}'. "
                "Set TOOLS_API_ALLOW_INSECURE_HTTP=true to keep HTTP."
            )
            base = secure_base

    _TOOL_SELECTOR_BASE_URL = base.rstrip('/')
    return _TOOL_SELECTOR_BASE_URL


def build_tool_selector_headers() -> Dict[str, str]:
    """Create standard headers for Tool Selector API requests."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    password = os.getenv('TOOL_SELECTOR_PASSWORD') or LETTA_API_KEY
    if password:
        headers["X-BARE-PASSWORD"] = f"password {password}"

    bearer = os.getenv('TOOL_SELECTOR_BEARER_TOKEN')
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    return headers


def get_tool_selector_timeout(default: float = 15.0) -> float:
    """Return the timeout (seconds) for Tool Selector API requests."""
    timeout_value = os.getenv('TOOLS_API_TIMEOUT')
    if not timeout_value:
        return default

    try:
        timeout = float(timeout_value)
        if timeout <= 0:
            raise ValueError
        return timeout
    except ValueError:
        _emit_tool_selector_warning(
            f"Invalid TOOLS_API_TIMEOUT '{timeout_value}'; falling back to {default} seconds."
        )
        return default

def get_find_tools_id(agent_id: Optional[str] = None) -> Optional[str]:
    """
    Dynamically query the Letta API to find the tool ID for find_tools.
    If agent_id is provided, checks the agent's tools first.
    Otherwise checks regular tools, then MCP servers.
    
    Args:
        agent_id: Optional agent ID to check for attached tools
        
    Returns:
        str: The tool ID for find_tools, or None if not found
    """
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    if LETTA_API_KEY:
        headers["Authorization"] = f"Bearer {LETTA_API_KEY}"
    
    try:
        # If agent_id is provided, check agent's tools first
        if agent_id:
            agent_tools_url = f"{LETTA_URL}/agents/{agent_id}/tools"
            print(f"[TOOL_LOOKUP] Checking agent {agent_id} tools: {agent_tools_url}", file=sys.stderr)
            
            try:
                agent_response = requests.get(agent_tools_url, headers=headers, timeout=10)
                if agent_response.status_code == 200:
                    agent_tools = agent_response.json()
                    
                    # Look for find_tools in agent's tools
                    for tool in agent_tools:
                        tool_name = tool.get('name', '').lower()
                        tool_id = tool.get('id', '')
                        
                        if tool_name == 'find_tools':
                            print(f"[TOOL_LOOKUP] Found in agent's tools: {tool_id} (name: {tool_name})", file=sys.stderr)
                            return tool_id
                        
                        # Also check for variations
                        if 'find' in tool_name and 'tool' in tool_name:
                            print(f"[TOOL_LOOKUP] Found potential match in agent's tools: {tool_id} (name: {tool_name})", file=sys.stderr)
                            # Store as candidate but continue searching for exact match
                            candidate_id = tool_id
                    
                    # Return candidate if no exact match found
                    if 'candidate_id' in locals():
                        return candidate_id
                        
            except Exception as e:
                print(f"[TOOL_LOOKUP] Error checking agent tools: {e}", file=sys.stderr)
        
        # Fall back to checking all tools
        url = f"{LETTA_URL}/tools"
        print(f"[TOOL_LOOKUP] Querying all tools: {url}", file=sys.stderr)
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tools = response.json()
            
            # Search for find_tools by name
            for tool in tools:
                tool_name = tool.get('name', '').lower()
                tool_id = tool.get('id', '')
                
                if tool_name == 'find_tools':
                    print(f"[TOOL_LOOKUP] Found in tools: {tool_id} (name: {tool_name})", file=sys.stderr)
                    return tool_id
        
        # Now try MCP servers
        mcp_url = f"{LETTA_URL}/tools/mcp/servers"
        print(f"[TOOL_LOOKUP] Querying MCP servers: {mcp_url}", file=sys.stderr)
        mcp_response = requests.get(mcp_url, headers=headers, timeout=10)
        
        if mcp_response.status_code == 200:
            mcp_servers = mcp_response.json()
            
            # Check each MCP server for find_tools
            for server_name, server_info in mcp_servers.items():
                print(f"[TOOL_LOOKUP] Checking MCP server: {server_name}", file=sys.stderr)
                
                # Get tools from this MCP server
                server_tools_url = f"{LETTA_URL}/tools/mcp/servers/{server_name}/tools"
                try:
                    tools_response = requests.get(server_tools_url, headers=headers, timeout=10)
                    if tools_response.status_code == 200:
                        server_tools = tools_response.json()
                        
                        for tool in server_tools:
                            tool_name = tool.get('name', '').lower()
                            
                            if tool_name == 'find_tools':
                                # MCP tools need to be looked up by their registered ID
                                print(f"[TOOL_LOOKUP] Found find_tools in MCP server: {server_name}", file=sys.stderr)
                                print(f"[TOOL_LOOKUP] Note: Need to find actual tool ID from agent or tools list", file=sys.stderr)
                                # Don't return a constructed ID, it won't work
                                
                except Exception as e:
                    print(f"[TOOL_LOOKUP] Error checking server {server_name}: {e}", file=sys.stderr)
        
        print("[TOOL_LOOKUP] Warning: Could not find find_tools tool in Letta API", file=sys.stderr)
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"[TOOL_LOOKUP] Error querying Letta API: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[TOOL_LOOKUP] Unexpected error: {e}", file=sys.stderr)
        return None

def get_find_tools_id_with_fallback(agent_id: Optional[str] = None, fallback_id: Optional[str] = None) -> str:
    """
    Get the find_tools ID with a fallback if dynamic lookup fails.
    
    Args:
        agent_id: Optional agent ID to check for attached tools
        fallback_id: The ID to use if dynamic lookup fails (if not provided, uses hardcoded fallback)
        
    Returns:
        str: The tool ID (either dynamically found or the fallback)
    """
    # Use the original hardcoded IDs as ultimate fallbacks
    if fallback_id is None:
        # These are the known tool IDs from the conversation history
        fallback_id = "tool-e34b5c60-5bd5-4288-a97f-2167ddf3062b"  # Original ID
        
    dynamic_id = get_find_tools_id(agent_id)
    if dynamic_id:
        return dynamic_id
    else:
        print(f"[TOOL_LOOKUP] Using fallback ID: {fallback_id}", file=sys.stderr)
        return fallback_id
