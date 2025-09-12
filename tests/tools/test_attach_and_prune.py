import requests
import json
import os
from dotenv import load_dotenv
import logging

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# URL of your running tool server (api_server.py)
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://localhost:8020") # Changed to localhost
# Agent ID to test with
AGENT_ID = os.getenv("TEST_AGENT_ID", "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444")

# Endpoint for attaching tools
ATTACH_ENDPOINT = "/api/v1/tools/attach"

def test_attach_and_trigger_prune():
    """
    Tests the /api/v1/tools/attach endpoint, which should subsequently trigger
    the tool pruning logic if attachments are successful and a query is provided.
    """
    if not AGENT_ID or AGENT_ID == "your_agent_id_here":
        logger.error("AGENT_ID is not set or is set to the placeholder. Please set TEST_AGENT_ID in your .env file or directly in the script.")
        return

    if not TOOL_SERVER_URL or TOOL_SERVER_URL == "http://localhost:5001": # Check against a common default if .env is not specific
        logger.warning(f"TOOL_SERVER_URL is using a default or common value: {TOOL_SERVER_URL}. Ensure this is correct for your local setup.")
        # Removed the specific IP check as localhost is now the primary default


    attach_url = f"{TOOL_SERVER_URL.rstrip('/')}{ATTACH_ENDPOINT}"

    # Sample query to find and attach tools
    # This query will also be used by the pruning logic if attachments are successful
    user_query = "tools for webresearch"

    payload = {
        "agent_id": AGENT_ID,
        "query": user_query,
        "limit": 20,  # How many tools to try and match/attach
        "keep_tools": [] # Optionally, specify tool IDs to always keep attached
    }

    logger.info(f"Attempting to attach tools for agent {AGENT_ID} with query: '{user_query}'")
    logger.info(f"Sending POST request to: {attach_url}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(attach_url, json=payload, timeout=60) # Increased timeout

        logger.info(f"Response Status Code: {response.status_code}")
        
        response_data = {}
        try:
            response_data = response.json()
            logger.info(f"Response JSON: {json.dumps(response_data, indent=2)}")
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Failed to decode JSON response. Response Text: {response.text}")

        if response.status_code == 200 and response_data.get("success"):
            logger.info("Tool attachment request was successful.")
            details = response_data.get("details", {})
            successful_attachments = details.get("successful_attachments", [])
            logger.info(f"Number of tools successfully attached: {len(successful_attachments)}")
            if successful_attachments:
                logger.info("Successfully attached tools:")
                for tool in successful_attachments:
                    logger.info(f"  - ID: {tool.get('tool_id')}, Name: {tool.get('name')}, Score: {tool.get('match_score')}")
            
            logger.info("Check the logs of your api_server.py to see details of the attachment and subsequent pruning process.")
        else:
            logger.error(f"Tool attachment request failed. Status: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during the request to {attach_url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    logger.info("Starting test for tool attachment and pruning...")
    # For this test to be meaningful:
    # 1. Your api_server.py (Tool Server) must be running and accessible at TOOL_SERVER_URL.
    # 2. Your Weaviate instance must be running and populated with tools.
    # 3. The AGENT_ID must be valid for your Letta instance (which the Tool Server communicates with).
    # 4. The Tool Server should have its LETTA_API_URL and LETTA_PASSWORD configured to talk to your Letta instance.
    
    test_attach_and_trigger_prune()
    logger.info("Test finished. Review api_server.py logs for pruning details.")