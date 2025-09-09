import aiohttp
import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
import time

from config.settings import settings

logger = logging.getLogger(__name__)

class LDTSClient:
    """Client for interacting with LDTS (Letta Dynamic Tool Selector) services."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_url = settings.LDTS_API_URL
        self.mcp_url = settings.LDTS_MCP_URL
        self.weaviate_url = settings.WEAVIATE_URL
        
    async def initialize(self):
        """Initialize the HTTP client session."""
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        logger.info("LDTS client initialized")
    
    async def close(self):
        """Close the HTTP client session."""
        if self.session:
            await self.session.close()
            logger.info("LDTS client closed")
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of all LDTS services."""
        if not self.session:
            raise RuntimeError("LDTS client not initialized")
        
        health_status = {
            "api_server": "unknown",
            "mcp_server": "unknown", 
            "weaviate": "unknown",
            "timestamp": time.time()
        }
        
        # Check API server
        try:
            async with self.session.get(f"{self.api_url}/health") as response:
                if response.status == 200:
                    health_status["api_server"] = "healthy"
                else:
                    health_status["api_server"] = f"unhealthy_{response.status}"
        except Exception as e:
            health_status["api_server"] = f"error: {str(e)}"
            logger.warning(f"API server health check failed: {e}")
        
        # Check MCP server
        try:
            async with self.session.get(f"{self.mcp_url}/health") as response:
                if response.status == 200:
                    health_status["mcp_server"] = "healthy"
                else:
                    health_status["mcp_server"] = f"unhealthy_{response.status}"
        except Exception as e:
            health_status["mcp_server"] = f"error: {str(e)}"
            logger.warning(f"MCP server health check failed: {e}")
        
        # Check Weaviate
        try:
            async with self.session.get(f"{self.weaviate_url}/v1/meta") as response:
                if response.status == 200:
                    health_status["weaviate"] = "healthy"
                else:
                    health_status["weaviate"] = f"unhealthy_{response.status}"
        except Exception as e:
            health_status["weaviate"] = f"error: {str(e)}"
            logger.warning(f"Weaviate health check failed: {e}")
        
        return health_status
    
    async def search_tools(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        enable_reranking: bool = True,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """Search for tools using LDTS search functionality."""
        if not self.session:
            raise RuntimeError("LDTS client not initialized")
        
        # Use MCP endpoint for tool search
        mcp_payload = {
            "jsonrpc": "2.0",
            "id": int(time.time()),
            "method": "tools/call",
            "params": {
                "name": "find_tools",
                "arguments": {
                    "query": query,
                    "agent_id": agent_id,
                    "limit": limit,
                    "enable_reranking": enable_reranking,
                    "min_score": min_score,
                    "detailed_response": True
                }
            }
        }
        
        try:
            async with self.session.post(self.mcp_url, json=mcp_payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "error" in result:
                    raise RuntimeError(f"MCP error: {result['error']}")
                
                return result.get("result", {})
                
        except Exception as e:
            logger.error(f"Tool search failed: {e}", exc_info=True)
            raise
    
    async def rerank_tools(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rerank tools using the reranker service."""
        if not self.session:
            raise RuntimeError("LDTS client not initialized")
        
        if not settings.ENABLE_RERANKING or not settings.RERANKER_URL:
            logger.warning("Reranking disabled or reranker URL not configured")
            return tools
        
        rerank_payload = {
            "query": query,
            "passages": [
                {
                    "id": tool.get("id", str(i)),
                    "text": f"{tool.get('name', '')} - {tool.get('description', '')}"
                }
                for i, tool in enumerate(tools)
            ],
            "model": model or settings.RERANKER_MODEL
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=settings.RERANKER_TIMEOUT)
            async with self.session.post(
                f"{settings.RERANKER_URL}/rerank",
                json=rerank_payload,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                rerank_result = await response.json()
                
                # Reorder tools based on reranking scores
                if "results" in rerank_result:
                    reranked_indices = [
                        int(result["passage"]["id"]) 
                        for result in rerank_result["results"]
                    ]
                    reranked_tools = [tools[i] for i in reranked_indices if i < len(tools)]
                    
                    # Add rerank scores to tools
                    for i, tool in enumerate(reranked_tools):
                        if i < len(rerank_result["results"]):
                            tool["rerank_score"] = rerank_result["results"][i]["score"]
                    
                    return reranked_tools
                
                return tools
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Return original tools if reranking fails
            return tools
    
    async def get_agent_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get current tools attached to an agent."""
        if not self.session:
            raise RuntimeError("LDTS client not initialized")
        
        # This would typically call the Letta API
        # For now, return empty list as placeholder
        logger.info(f"Getting tools for agent {agent_id}")
        return []
    
    async def attach_tools(
        self, 
        agent_id: str, 
        tool_ids: List[str],
        enable_pruning: bool = False
    ) -> Dict[str, Any]:
        """Attach tools to an agent (requires ENABLE_DANGEROUS_OPERATIONS)."""
        if not self.session:
            raise RuntimeError("LDTS client not initialized")
        
        if not settings.ENABLE_DANGEROUS_OPERATIONS:
            raise RuntimeError("Tool attachment requires ENABLE_DANGEROUS_OPERATIONS=true")
        
        payload = {
            "agent_id": agent_id,
            "tool_ids": tool_ids,
            "enable_pruning": enable_pruning
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/v1/tools/attach",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"Tool attachment failed: {e}", exc_info=True)
            raise