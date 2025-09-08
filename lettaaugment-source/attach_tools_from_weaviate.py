import json
import os
import logging
import asyncio
import aiohttp
import warnings
from typing import List, Dict, Any, Optional
from weaviate_tool_search import search_tools # Assuming this works correctly
from dotenv import load_dotenv
from contextlib import AsyncExitStack
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Set higher level for noisy libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


# Constants
BASE_URL = "https://letta2.oculair.ca"
API_KEY = os.getenv("LETTA_API_KEY", "lettaSecurePass123") # Load from env if available
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-BARE-PASSWORD": f"password {API_KEY}" if API_KEY else ""
}

# Global client instance - Consider a more robust way to manage this in a real app
_weaviate_client = None

def get_weaviate_client():
    """Get or initialize the Weaviate client."""
    global _weaviate_client
    if _weaviate_client is None:
        load_dotenv() # Ensure env vars are loaded
        logger.info("Initializing Weaviate client...")
        try:
            _weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv('WEAVIATE_URL'),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                },
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=60, query=60)
                )
            )
            logger.info("Weaviate client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise # Re-raise the exception
    return _weaviate_client

async def search_and_attach_tools(
    tools: List[Dict[str, Any]],
    target_agent_id: str,
    session: aiohttp.ClientSession,
    min_score: float = 75.0
) -> Dict[str, Any]:
    """Combined search and attach operation for better performance."""
    logger.info(f"Starting search_and_attach_tools for agent {target_agent_id} with {len(tools)} candidate tools.")
    logger.debug(f"Min score threshold: {min_score}")
    try:
        # Prepare tools
        tools_to_attach_info = []
        tool_ids_to_attach = []
        for tool in tools:
            tool_id = tool.get("tool_id")
            tool_name = tool.get("name", "N/A")
            distance = tool.get("distance")

            if distance is None:
                logger.warning(f"Tool {tool_name} ({tool_id}) missing distance, skipping.")
                continue

            # Calculate score (higher is better)
            match_score = round((1 - distance) * 100, 2)
            logger.debug(f"Tool: {tool_name} ({tool_id}), Distance: {distance:.4f}, Calculated Score: {match_score:.2f}%")

            if match_score >= min_score:
                logger.info(f"Tool {tool_name} ({tool_id}) passed threshold ({match_score:.2f}% >= {min_score:.2f}%). Adding for attachment.")
                tool_ids_to_attach.append(tool_id)
                tools_to_attach_info.append({
                    "tool_id": tool_id,
                    "name": tool_name,
                    "description": tool.get("description", ""),
                    "source_type": tool.get("source_type", ""),
                    "tags": tool.get("tags", []),
                    "match_score": match_score
                })
            else:
                 logger.debug(f"Tool {tool_name} ({tool_id}) failed threshold ({match_score:.2f}% < {min_score:.2f}%). Skipping.")


        if not tool_ids_to_attach:
            logger.warning(f"No tools met the minimum score threshold of {min_score}%. No attachments will be attempted.")
            return {
                "success": True, # Operation succeeded, just found nothing to attach
                "message": f"No tools found meeting minimum score threshold of {min_score}%",
                "details": {
                    "target_agent": target_agent_id,
                    "processed_count": len(tools),
                    "passed_filter_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "successful_attachments": [],
                    "failed_attachments": []
                }
            }

        logger.info(f"Attempting to attach {len(tool_ids_to_attach)} tools to agent {target_agent_id}...")
        # Attach tools concurrently
        tasks = []
        for tool_id in tool_ids_to_attach:
            url = f"{BASE_URL}/v1/agents/{target_agent_id}/tools/attach/{tool_id}"
            logger.debug(f"Creating PATCH task for URL: {url}")
            tasks.append(session.patch(url, headers=HEADERS))

        # Execute tasks and gather results
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"Received {len(responses)} responses/exceptions from attachment tasks.")

        # Process results
        successful = []
        failed = []

        for i, (response, tool_info) in enumerate(zip(responses, tools_to_attach_info)):
            tool_id = tool_info["tool_id"]
            tool_name = tool_info["name"]
            if isinstance(response, Exception):
                error_msg = f"Exception during attachment: {type(response).__name__}: {str(response)}"
                logger.error(f"Failed to attach tool {tool_name} ({tool_id}): {error_msg}")
                failed.append({"tool_id": tool_id, "name": tool_name, "error": error_msg})
            elif response.status >= 400:
                error_msg = f"HTTP {response.status}"
                try:
                    # Attempt to get more detail from the response body
                    error_body = await response.text()
                    error_msg += f" - Body: {error_body[:200]}" # Log first 200 chars
                except Exception as read_err:
                    error_msg += f" (Failed to read response body: {read_err})"
                logger.error(f"Failed to attach tool {tool_name} ({tool_id}): {error_msg}")
                failed.append({"tool_id": tool_id, "name": tool_name, "error": error_msg})
            else:
                logger.info(f"Successfully attached tool {tool_name} ({tool_id}) to agent {target_agent_id}.")
                successful.append({
                    "tool_id": tool_id,
                    "name": tool_name,
                    "match_score": tool_info["match_score"]
                })

        # Prepare result
        result = {
            "success": len(failed) == 0,
            "message": "",
            "details": {
                "target_agent": target_agent_id,
                "processed_count": len(tools),
                "passed_filter_count": len(tool_ids_to_attach),
                "success_count": len(successful),
                "failure_count": len(failed),
                "successful_attachments": successful,
                "failed_attachments": failed
            }
        }

        if result["success"]:
            result["message"] = f"Successfully processed {len(tools)} candidates, attached {len(successful)} tool(s) to agent {target_agent_id}"
        else:
            result["message"] = f"Processed {len(tools)} candidates, attempted {len(tool_ids_to_attach)} attachments. {len(successful)} succeeded, {len(failed)} failed."
        logger.info(result["message"])

        return result

    except Exception as e:
        logger.error(f"Unexpected error in search_and_attach_tools: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Internal server error: {str(e)}",
            "details": { "target_agent": target_agent_id, "error": str(e) }
        }

async def _run_async(query: str, target_agent_id: str, limit: int, min_score: float) -> Dict[str, Any]:
    """Run the async operations with proper resource management."""
    logger.info(f"Running async task for query: '{query}', agent: {target_agent_id}")
    async with AsyncExitStack() as stack:
        # Ensure Weaviate client is ready (though search_tools might handle this)
        # get_weaviate_client() # Uncomment if search_tools doesn't initialize client

        session = await stack.enter_async_context(aiohttp.ClientSession())
        logger.info(f"Calling search_tools with query='{query}', limit={limit}")
        try:
            # Assuming search_tools is synchronous for now based on previous context
            # If search_tools becomes async, use 'await'
            candidate_tools = search_tools(query=query, limit=limit)
            logger.info(f"search_tools returned {len(candidate_tools)} candidate tools.")
            logger.debug(f"Candidate tools raw result: {json.dumps(candidate_tools, indent=2)}")
        except Exception as search_err:
            logger.error(f"Error calling search_tools: {search_err}", exc_info=True)
            raise # Propagate the error

        return await search_and_attach_tools(
            tools=candidate_tools,
            target_agent_id=target_agent_id,
            session=session,
            min_score=min_score
        )

def attach_tools_from_query(
    query: str,
    target_agent_id: str, # Made non-optional based on API server logic
    limit: int = 5,
    min_score: float = 75.0,
    request_id: Optional[str] = None # Keep for potential future use
) -> Dict[str, Any]:
    """
    Find relevant tools using Weaviate search and attach them to the specified agent.
    Uses optimized concurrent operations for better performance.
    """
    logger.info(f"Received request for attach_tools_from_query. Request ID: {request_id}, Agent: {target_agent_id}, Query: '{query}'")
    if not target_agent_id:
         logger.error("attach_tools_from_query called without target_agent_id")
         return {"success": False, "message": "agent_id is required", "request_id": request_id}

    try:
        # Create and run event loop
        # Check if an event loop is already running (e.g., in Flask with async views)
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Using existing asyncio event loop.")
            # If running within an existing loop, just await the coroutine
            # This might require the Flask endpoint to be async def
            # For simplicity assuming standalone execution or sync Flask for now:
            result = asyncio.run(_run_async(
                 query=query, target_agent_id=target_agent_id, limit=limit, min_score=min_score
            ))

        except RuntimeError: # No running event loop
            logger.debug("No existing event loop found, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_run_async(
                    query=query,
                    target_agent_id=target_agent_id,
                    limit=limit,
                    min_score=min_score
                ))
            finally:
                logger.debug("Closing the created event loop.")
                loop.close()
                asyncio.set_event_loop(None) # Clean up loop association

        # Add request ID if provided
        if request_id:
            result["request_id"] = request_id

        logger.info(f"attach_tools_from_query completed. Success: {result.get('success')}")
        return result

    except Exception as e:
        logger.error(f"Error in attach_tools_from_query: {type(e).__name__}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Server error during tool attachment: {str(e)}",
            "request_id": request_id
        }

if __name__ == "__main__":
    # This part remains the same for standalone testing if needed
    import argparse

    # Suppress ResourceWarnings if necessary
    warnings.filterwarnings("ignore", category=ResourceWarning)

    parser = argparse.ArgumentParser(description='Find and attach relevant tools to an agent')
    parser.add_argument('query', help='Natural language description of desired tool functionality')
    parser.add_argument('--target-agent', required=True, # Make required for standalone test
                      help='ID of the agent to attach tools to')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of tools to return')
    parser.add_argument('--min-score', type=float, default=75.0,
                       help='Minimum similarity score (0-100) for tools to be included')
    parser.add_argument('--request-id', help='Optional request ID for tracking')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set logging level')

    args = parser.parse_args()

    # Set log level for the main script execution
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Run the tool search and attachment
    result = attach_tools_from_query(
        query=args.query,
        target_agent_id=args.target_agent,
        limit=args.limit,
        min_score=args.min_score,
        request_id=args.request_id
    )

    # Print results
    print("\nOperation Result:")
    print(json.dumps(result, indent=2))