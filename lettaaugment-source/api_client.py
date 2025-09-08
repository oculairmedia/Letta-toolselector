#!/usr/bin/env python
# api_client.py
import requests
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Client for interacting with the Weaviate Tool API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Attach tools command
    attach_parser = subparsers.add_parser("attach-tools", help="Find and attach tools based on a query")
    attach_parser.add_argument("--query", "-q", required=True, help="The query to search for tools")
    attach_parser.add_argument("--agent-id", "-a", default="agent-d5d91a6a-cc16-47dd-97be-07101cdbd49d", 
                             help="Agent ID to attach tools to")
    attach_parser.add_argument("--limit", "-l", type=int, default=5, 
                             help="Maximum number of tools to return")
    attach_parser.add_argument("--min-score", "-s", type=float, default=75.0,
                             help="Minimum similarity score (0-100) for tools to be included")
    attach_parser.add_argument("--request-id", "-r", help="Optional request ID for tracking")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check if the API server is running")
    
    # Server settings
    parser.add_argument("--host", default="localhost", help="API server hostname")
    parser.add_argument("--port", default=8000, type=int, help="API server port")
    
    args = parser.parse_args()
    
    # Build the base URL
    base_url = f"http://{args.host}:{args.port}"
    
    if args.command == "attach-tools":
        # Prepare the request payload
        payload = {
            "query": args.query,
            "agent_id": args.agent_id,
            "limit": args.limit,
            "min_score": args.min_score
        }
        
        if args.request_id:
            payload["request_id"] = args.request_id
        
        # Print the request
        print(f"Sending request to {base_url}/api/v1/tools/attach")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API call
            response = requests.post(f"{base_url}/api/v1/tools/attach", json=payload)
            
            # Handle the response
            if response.status_code == 200:
                result = response.json()
                print("\nAPI Response:")
                print(json.dumps(result, indent=2))
                
                # Summary of results
                if result.get("success", False):
                    details = result.get("details", {})
                    print(f"\nSuccess! Attached {details.get('success_count', 0)} tools to agent.")
                    
                    if details.get("successful_attachments"):
                        print("\nAttached tools:")
                        for tool in details.get("successful_attachments", []):
                            print(f"  - {tool.get('name')} (Score: {tool.get('match_score')}%)")
                else:
                    print(f"\nOperation failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API server: {e}")
            sys.exit(1)
            
    elif args.command == "health":
        try:
            response = requests.get(f"{base_url}/api/health")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Server status: {result.get('status', 'unknown')}")
                print(f"Message: {result.get('message', '')}")
            else:
                print(f"Health check failed with status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API server: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()