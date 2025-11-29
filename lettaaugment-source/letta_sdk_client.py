"""
Letta SDK Client Wrapper

Provides a centralized, async-compatible wrapper around the letta-client SDK.
Handles authentication, error handling, retries, and logging for all Letta API calls.

This module replaces direct aiohttp HTTP calls with SDK method calls while maintaining
the same async interface expected by the rest of the application.
"""

import os
import logging
import asyncio
from functools import wraps
from typing import Optional, Dict, Any, List, TypeVar, Callable
from concurrent.futures import ThreadPoolExecutor

from letta_client import Letta

logger = logging.getLogger(__name__)

# Thread pool for running sync SDK calls in async context
_executor = ThreadPoolExecutor(max_workers=10)

# Type var for generic return types
T = TypeVar('T')


class LettaSDKClient:
    """
    Async wrapper for the Letta SDK client.
    
    The SDK is synchronous, so we use a thread pool executor to run SDK calls
    in a non-blocking manner for async compatibility.
    """
    
    _instance: Optional['LettaSDKClient'] = None
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        Initialize the Letta SDK client.
        
        Args:
            base_url: Letta API base URL (defaults to LETTA_API_URL env var)
            api_key: API key/password (defaults to LETTA_PASSWORD env var)
            max_retries: Number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self._base_url = base_url or os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1')
        self._api_key = api_key or os.getenv('LETTA_PASSWORD', '')
        self._max_retries = max_retries
        self._timeout = timeout
        
        # Normalize base URL (remove /v1 suffix as SDK adds it)
        if self._base_url.endswith('/v1'):
            self._base_url = self._base_url[:-3]
        
        # Ensure HTTPS
        self._base_url = self._base_url.replace('http://', 'https://')
        
        # Initialize SDK client with custom auth header
        # The Letta server uses X-BARE-PASSWORD header format
        custom_headers = {
            'X-BARE-PASSWORD': f'password {self._api_key}'
        }
        
        self._client: Letta = Letta(
            base_url=self._base_url,
            default_headers=custom_headers,
            max_retries=max_retries,
            timeout=timeout
        )
        
        logger.info(f"Letta SDK client initialized for {self._base_url}")
    
    @classmethod
    def get_instance(cls) -> 'LettaSDKClient':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        if cls._instance and cls._instance._client:
            cls._instance._client.close()
        cls._instance = None
    
    async def _run_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))
    
    # =========================================================================
    # Agent Operations
    # =========================================================================
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch agent information.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            Agent data dict or None if not found
        """
        try:
            agent = await self._run_sync(
                self._client.agents.retrieve,
                agent_id=agent_id
            )
            # Convert to dict for backward compatibility
            return agent.model_dump() if hasattr(agent, 'model_dump') else dict(agent)
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            raise
    
    async def get_agent_name(self, agent_id: str) -> str:
        """
        Fetch agent name.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            Agent name or "Unknown Agent" if not found
        """
        try:
            agent_data = await self.get_agent(agent_id)
            if agent_data:
                return agent_data.get("name", "Unknown Agent")
            return "Unknown Agent"
        except Exception as e:
            logger.error(f"Failed to get agent name for {agent_id}: {e}")
            return "Unknown Agent"
    
    # =========================================================================
    # Tool Operations
    # =========================================================================
    
    async def list_agent_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        List all tools attached to an agent.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            List of tool dicts
        """
        try:
            tools = await self._run_sync(
                self._client.agents.tools.list,
                agent_id=agent_id
            )
            # Convert to list of dicts for backward compatibility
            return [t.model_dump() if hasattr(t, 'model_dump') else dict(t) for t in tools]
        except Exception as e:
            logger.error(f"Failed to list tools for agent {agent_id}: {e}")
            raise
    
    async def attach_tool(
        self, 
        agent_id: str, 
        tool_id: str,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Attach a tool to an agent.
        
        Args:
            agent_id: The agent's unique identifier
            tool_id: The tool's unique identifier
            tool_name: Optional tool name for logging
            
        Returns:
            Result dict with success status and details
        """
        tool_name = tool_name or tool_id
        try:
            result = await self._run_sync(
                self._client.agents.tools.attach,
                agent_id=agent_id,
                tool_id=tool_id
            )
            logger.debug(f"Successfully attached tool {tool_name} ({tool_id}) to agent {agent_id}")
            return {
                "success": True,
                "tool_id": tool_id,
                "name": tool_name,
                "match_score": 100  # Default score for direct attachment
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to attach tool {tool_name} ({tool_id}) to agent {agent_id}: {error_msg}")
            return {
                "success": False,
                "tool_id": tool_id,
                "name": tool_name,
                "error": error_msg
            }
    
    async def detach_tool(
        self, 
        agent_id: str, 
        tool_id: str,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detach a tool from an agent.
        
        Args:
            agent_id: The agent's unique identifier
            tool_id: The tool's unique identifier
            tool_name: Optional tool name for logging
            
        Returns:
            Result dict with success status and details
        """
        tool_name = tool_name or tool_id
        try:
            await self._run_sync(
                self._client.agents.tools.detach,
                agent_id=agent_id,
                tool_id=tool_id
            )
            logger.debug(f"Successfully detached tool {tool_name} ({tool_id}) from agent {agent_id}")
            return {
                "success": True,
                "tool_id": tool_id,
                "name": tool_name
            }
        except Exception as e:
            error_msg = str(e)
            # Handle 404 as success (tool already detached)
            if "404" in error_msg or "not found" in error_msg.lower():
                logger.warning(f"Tool {tool_name} ({tool_id}) not found or already detached (404)")
                return {
                    "success": True,
                    "tool_id": tool_id,
                    "name": tool_name,
                    "warning": "Tool not found or already detached"
                }
            logger.error(f"Failed to detach tool {tool_name} ({tool_id}) from agent {agent_id}: {error_msg}")
            return {
                "success": False,
                "tool_id": tool_id,
                "name": tool_name,
                "error": error_msg
            }
    
    async def register_mcp_tool(
        self, 
        tool_name: str, 
        server_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Register a tool from an MCP server.
        
        Args:
            tool_name: Name of the tool to register
            server_name: Name of the MCP server
            
        Returns:
            Registered tool dict with normalized ID fields, or None if failed
        """
        try:
            # The SDK's add method registers MCP tools
            # We use the low-level _post method as the SDK may not have direct MCP support
            result = await self._run_sync(
                self._client.tools._post,
                f"/tools/mcp/servers/{server_name}/{tool_name}",
                cast_to=dict
            )
            
            # Normalize ID fields
            if result:
                tool_id = result.get('id') or result.get('tool_id')
                if tool_id:
                    result['id'] = tool_id
                    result['tool_id'] = tool_id
                    logger.info(f"Successfully registered MCP tool '{tool_name}' from server '{server_name}'")
                    return result
            
            logger.warning(f"Failed to register MCP tool '{tool_name}' - no ID returned")
            return None
            
        except Exception as e:
            logger.error(f"Error registering MCP tool '{tool_name}' from server '{server_name}': {e}")
            return None
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    async def batch_attach_tools(
        self,
        agent_id: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """
        Attach multiple tools to an agent in parallel.
        
        Args:
            agent_id: The agent's unique identifier
            tools: List of tool dicts with 'id' or 'tool_id' and optional 'name'
            
        Returns:
            Dict with 'successful' and 'failed' lists
        """
        tasks = []
        for tool in tools:
            tool_id = tool.get('id') or tool.get('tool_id')
            tool_name = tool.get('name', tool_id)
            if tool_id:
                tasks.append(self.attach_tool(agent_id, tool_id, tool_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                failed.append({"error": str(result)})
            elif isinstance(result, dict) and result.get('success'):
                successful.append(result)
            elif isinstance(result, dict):
                failed.append(result)
        
        return {"successful": successful, "failed": failed}
    
    async def batch_detach_tools(
        self,
        agent_id: str,
        tool_ids: List[str],
        tool_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, List]:
        """
        Detach multiple tools from an agent in parallel.
        
        Args:
            agent_id: The agent's unique identifier
            tool_ids: List of tool IDs to detach
            tool_names: Optional mapping of tool_id -> name for logging
            
        Returns:
            Dict with 'successful' and 'failed' lists
        """
        tool_names = tool_names or {}
        tasks = [
            self.detach_tool(agent_id, tid, tool_names.get(tid))
            for tid in tool_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                failed.append({"error": str(result)})
            elif isinstance(result, dict) and result.get('success'):
                successful.append(result)
            elif isinstance(result, dict):
                failed.append(result)
        
        return {"successful": successful, "failed": failed}
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Letta API health.
        
        Returns:
            Health status dict
        """
        try:
            result = await self._run_sync(self._client.health)
            return {
                "status": result.status,
                "version": result.version,
                "healthy": result.status == "ok"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "healthy": False
            }
    
    def close(self):
        """Close the SDK client."""
        if self._client:
            self._client.close()
            logger.info("Letta SDK client closed")


# =========================================================================
# Module-level convenience functions
# =========================================================================

def get_client() -> LettaSDKClient:
    """Get the singleton SDK client instance."""
    return LettaSDKClient.get_instance()


async def init_client() -> LettaSDKClient:
    """Initialize and return the SDK client (async-compatible)."""
    return get_client()


async def close_client():
    """Close the SDK client and reset singleton."""
    LettaSDKClient.reset_instance()
