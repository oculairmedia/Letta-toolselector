import requests
import json
# import os  # Currently unused
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for your API server
API_SERVER_BASE_URL = "http://192.168.50.90:8020" # Your remote API server
AGENT_ID = "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444"

API_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def get_agent_tools_from_api_server(agent_id_to_check):
    """
    Fetches the tools for a given agent by querying the API server's
    prune endpoint with a drop_rate of 0. This indirectly shows
    the 'tools_on_agent_before' count from the server's perspective.
    """
    
    prune_url = f"{API_SERVER_BASE_URL}/api/v1/tools/prune"
    payload = {
        "user_prompt": "diagnostic prompt to check agent tools",
        "agent_id": agent_id_to_check,
        "drop_rate": 0.0 # Keep all tools, just to trigger the fetch_agent_tools call
    }
    
    print(f"Attempting to get agent tool info via prune endpoint: {prune_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(prune_url, json=payload, headers=API_HEADERS, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nPrune Response Data:")
            print(json.dumps(data, indent=2))
            
            details = data.get("details", {})
            tools_on_agent_before = details.get("tools_on_agent_before")
            
            if tools_on_agent_before is not None:
                print(f"\n✅ API Server reports agent '{agent_id_to_check}' had {tools_on_agent_before} tools before pruning attempt.")
                if tools_on_agent_before > 0:
                    print("This indicates fetch_agent_tools is likely working correctly on the server.")
                else:
                    print("⚠️ API Server reports 0 tools on agent. This might be the issue if tools are expected.")
            else:
                print("⚠️ 'tools_on_agent_before' not found in prune response details.")
        else:
            print(f"\n❌ Error response from prune endpoint: {response.text}")

    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print(f"Ensure the API server is running and accessible at {API_SERVER_BASE_URL}")
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out connecting to {prune_url}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Checking tools attached to an agent via the API server...")
    get_agent_tools_from_api_server(AGENT_ID)
    print("\nCheck complete.")