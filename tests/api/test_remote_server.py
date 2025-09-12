#!/usr/bin/env python3
"""
Test the embedding functionality using the remote server setup:
- API Server: http://192.168.50.90:8020/
- Weaviate: http://192.168.50.90:8080/v1
"""

import requests
import json
from dotenv import load_dotenv

def test_remote_api_server():
    """Test the remote API server's prune endpoint"""
    print("Testing Remote API Server")
    print("=" * 50)
    
    # API server URL
    api_base_url = "http://192.168.50.90:8020"
    
    # Test payload for the prune endpoint
    payload = {
        "user_prompt": "search for jobs",
        "agent_id": "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444",
        "drop_rate": 0.1
    }
    
    print(f"Testing API server at: {api_base_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Test the prune endpoint
        prune_url = f"{api_base_url}/api/v1/tools/prune"
        
        print(f"\nSending POST request to: {prune_url}")
        response = requests.post(
            prune_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ ERROR: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the server is running at http://192.168.50.90:8020/")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_remote_weaviate_direct():
    """Test direct connection to remote Weaviate"""
    print("\nTesting Remote Weaviate Direct Connection")
    print("=" * 50)
    
    # Weaviate URL
    weaviate_url = "http://192.168.50.90:8080/v1"
    
    try:
        # Test Weaviate health endpoint
        health_url = f"{weaviate_url}/.well-known/ready"
        print(f"Testing Weaviate health at: {health_url}")
        
        response = requests.get(health_url, timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Weaviate is healthy!")
        else:
            print(f"❌ Weaviate health check failed: {response.text}")
            return False
        
        # Test getting schema
        schema_url = f"{weaviate_url}/schema"
        print(f"\nTesting schema endpoint: {schema_url}")
        
        response = requests.get(schema_url, timeout=10)
        print(f"Schema check status: {response.status_code}")
        
        if response.status_code == 200:
            schema = response.json()
            classes = schema.get('classes', [])
            print(f"✅ Found {len(classes)} classes in schema")
            
            # Look for Tool class
            tool_class = None
            for cls in classes:
                if cls.get('class') == 'Tool':
                    tool_class = cls
                    break
            
            if tool_class:
                print("✅ Found Tool class in schema!")
                vectorizer = tool_class.get('vectorizer', 'None')
                print(f"Vectorizer: {vectorizer}")
                return True
            else:
                print("❌ Tool class not found in schema")
                return False
        else:
            print(f"❌ Schema check failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure Weaviate is running at http://192.168.50.90:8080/")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_remote_weaviate_with_client():
    """Test remote Weaviate using the Weaviate client"""
    print("\nTesting Remote Weaviate with Client")
    print("=" * 50)
    
    try:
        import weaviate
        import os
        load_dotenv()
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        print("Connecting to remote Weaviate with client...")
        
        # Connect to remote Weaviate
        client = weaviate.Client(
            url="http://192.168.50.90:8080",
            additional_headers={
                "X-OpenAI-Api-Key": openai_api_key
            }
        )
        
        # Test connection
        if client.is_ready():
            print("✅ Weaviate client connected successfully!")
            
            # Test getting some tools
            result = client.query.get("Tool").with_limit(3).do()
            
            if 'data' in result and 'Get' in result['data'] and 'Tool' in result['data']['Get']:
                tools = result['data']['Get']['Tool']
                print(f"✅ Found {len(tools)} tools in remote Weaviate")
                
                for i, tool in enumerate(tools):
                    name = tool.get('name', 'Unknown')
                    print(f"  {i+1}. {name}")
                
                return True
            else:
                print("❌ No tools found in remote Weaviate")
                return False
        else:
            print("❌ Weaviate client could not connect")
            return False
            
    except ImportError:
        print("❌ Weaviate client not available. Install with: pip install weaviate-client")
        return False
    except Exception as e:
        print(f"❌ Weaviate client error: {e}")
        return False

def test_api_endpoints():
    """Test various API endpoints on the remote server"""
    print("\nTesting Other API Endpoints")
    print("=" * 50)
    
    api_base_url = "http://192.168.50.90:8020"
    
    endpoints = [
        "/api/v1/tools",
        "/api/v1/tools/search?query=job&limit=5"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        url = f"{api_base_url}{endpoint}"
        print(f"\nTesting: {url}")
        
        try:
            response = requests.get(url, timeout=15)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        print(f"✅ Response keys: {list(data.keys())}")
                    elif isinstance(data, list):
                        print(f"✅ Response list length: {len(data)}")
                    results[endpoint] = True
                except json.JSONDecodeError:
                    print(f"✅ Response: {response.text[:100]}...")
                    results[endpoint] = True
            else:
                print(f"❌ Error: {response.text}")
                results[endpoint] = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[endpoint] = False
    
    return results

def main():
    """Run all remote server tests"""
    print("Remote Server Testing Suite")
    print("=" * 60)
    print("API Server: http://192.168.50.90:8020/")
    print("Weaviate: http://192.168.50.90:8080/v1")
    print("=" * 60)
    
    results = {}
    
    # Test 1: API Server prune endpoint
    results['api_prune'] = test_remote_api_server()
    
    # Test 2: Direct Weaviate connection
    results['weaviate_direct'] = test_remote_weaviate_direct()
    
    # Test 3: Weaviate client connection
    results['weaviate_client'] = test_remote_weaviate_with_client()
    
    # Test 4: Other API endpoints
    endpoint_results = test_api_endpoints()
    results.update(endpoint_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Remote server is working correctly.")
    elif results.get('api_prune', False):
        print("🎯 Main prune functionality is working!")
    else:
        print("⚠️ Some tests failed. Check the connection and server status.")

if __name__ == "__main__":
    main()