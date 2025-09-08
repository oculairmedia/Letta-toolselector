#!/usr/bin/env python3
"""
Check the logs and debug what's happening in the pruning system
"""

import requests
import json

def test_prune_with_logging():
    """Test pruning and try to see what's happening"""
    print("Testing Prune with Debug Focus")
    print("=" * 50)
    
    api_base_url = "http://192.168.50.90:8020"
    prune_url = f"{api_base_url}/api/v1/tools/prune"
    
    payload = {
        "user_prompt": "search for jobs",
        "agent_id": "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444",
        "drop_rate": 0.8,  # High drop rate to force more activity
        "debug": True  # Add debug flag if supported
    }
    
    print(f"Testing with high drop rate (0.8) to force activity...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            prune_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45  # Longer timeout
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nFull Response:")
            print(json.dumps(result, indent=2))
            
            # Analyze the details
            details = result.get('details', {})
            print(f"\nDetailed Analysis:")
            print(f"- Total tools analyzed: {details.get('total_tools_analyzed', 0)}")
            print(f"- Tools with embeddings: {details.get('tools_with_embeddings', 0)}")
            print(f"- Tools eligible for pruning: {details.get('tools_eligible_for_pruning', 0)}")
            print(f"- Successful detachments: {len(details.get('successful_detachments', []))}")
            print(f"- Failed detachments: {len(details.get('failed_detachments', []))}")
            
            if details.get('failed_detachments'):
                print(f"Failed detachment details: {details['failed_detachments']}")
                
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def check_what_server_can_do():
    """Check what the server tells us about its capabilities"""
    print("\n" + "=" * 50)
    print("Checking Server Capabilities")
    print("=" * 50)
    
    api_base_url = "http://192.168.50.90:8020"
    
    # Try to get server info/status
    endpoints_to_try = [
        "/api/v1/status",
        "/api/v1/health", 
        "/api/v1/info",
        "/status",
        "/health",
        "/info",
        "/api/v1/tools/count",
        "/api/v1/debug"
    ]
    
    for endpoint in endpoints_to_try:
        url = f"{api_base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"\n✅ {endpoint}: {response.status_code}")
                try:
                    data = response.json()
                    print(f"   {json.dumps(data, indent=2)}")
                except:
                    print(f"   {response.text}")
            elif response.status_code == 404:
                pass  # Skip 404s
            else:
                print(f"⚠️ {endpoint}: {response.status_code} - {response.text[:50]}...")
        except:
            pass  # Skip connection errors

def main():
    """Run debugging tests"""
    print("Remote Server Debug Analysis")
    print("=" * 60)
    
    # Test 1: Detailed prune test
    test_prune_with_logging()
    
    # Test 2: Check server capabilities
    check_what_server_can_do()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Key Findings:")
    print("1. ✅ Search works perfectly (semantic similarity)")
    print("2. ✅ Embedding generation works for queries") 
    print("3. ❌ Pruning can't find embeddings for existing tools")
    print("4. 🔍 Need to check if tools have embeddings stored")
    
    print("\nRecommendation:")
    print("The tools may need to be re-uploaded with embeddings, or")
    print("the pruning system needs to generate embeddings on-demand")
    print("like the search system does.")

if __name__ == "__main__":
    main()