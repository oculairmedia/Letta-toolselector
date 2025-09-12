#!/usr/bin/env python3
"""
Test the pruning functionality in detail with the remote server
"""

import requests
import json

def test_prune_with_different_prompts():
    """Test pruning with various prompts to see if embedding generation works"""
    print("Testing Prune Functionality with Different Prompts")
    print("=" * 60)
    
    api_base_url = "http://192.168.50.90:8020"
    prune_url = f"{api_base_url}/api/v1/tools/prune"
    
    test_cases = [
        {
            "name": "Letta Job Tool Search",
            "prompt": "How can a letta agent help me find remote job opportunities using its tools?",
            "expected_relevance": "Should find job-related tools, possibly Letta-specific ones"
        },
        {
            "name": "Letta Agent Memory",
            "prompt": "How does a letta agent manage its memory and conversations?",
            "expected_relevance": "Should find memory-related tools for a Letta agent"
        },
        {
            "name": "Letta MCP Integration",
            "prompt": "I need to set up letta mcp server integrations and connect services for my agent",
            "expected_relevance": "Should find integration or MCP related tools for Letta"
        },
        {
            "name": "Letta Tool Info",
            "prompt": "help my letta agent search for information about its own mcp tools and servers",
            "expected_relevance": "Should find Letta-specific tools or general search tools related to agent capabilities"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Prompt: '{test_case['prompt']}'")
        print(f"Expected: {test_case['expected_relevance']}")
        
        payload = {
            "user_prompt": test_case['prompt'],
            "agent_id": "agent-0e99d1a5-d9ca-43b0-9df9-c09761d01444",
            "drop_rate": 0.1  # Higher drop rate to see more activity
        }
        
        try:
            response = requests.post(
                prune_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"Raw JSON Response from Prune Endpoint for '{test_case['name']}':")
                print(json.dumps(result, indent=2)) # Print the whole raw response
                
                details = result.get('details', {})
                
                # Use the new field names from the refactored _perform_tool_pruning
                tools_on_agent_before = details.get('tools_on_agent_before', 0)
                relevant_library_tools_count = details.get('relevant_library_tools_found_count', 0)
                detached_count = details.get('tools_detached_count', 0)
                
                print(f"   Tools on agent before: {tools_on_agent_before}")
                print(f"   Relevant library tools found: {relevant_library_tools_count}")
                print(f"   Tools detached: {detached_count}")
                
                # For consistency with previous "embedding_success" logic,
                # let's consider it a success if relevant library tools were found,
                # as this implies the prompt embedding worked.
                embedding_success = relevant_library_tools_count > 0
                results.append({
                    'test': test_case['name'],
                    'embedding_success': embedding_success,
                    'tools_analyzed': tools_on_agent_before, # This is more accurate for "analyzed"
                    'relevant_library_tools_found': relevant_library_tools_count
                })
                
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"   Error: {response.text}")
                results.append({
                    'test': test_case['name'],
                    'embedding_success': False,
                    'error': response.text
                })
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'test': test_case['name'],
                'embedding_success': False,
                'error': str(e)
            })
    
    return results

def test_search_endpoint():
    """Test the search endpoint to see if it works better"""
    print("\n" + "=" * 60)
    print("Testing Search Endpoint")
    print("=" * 60)
    
    api_base_url = "http://192.168.50.90:8020"
    
    # Try POST instead of GET for search
    search_url = f"{api_base_url}/api/v1/tools/search"
    
    test_queries = [
        "job search",
        "memory management", 
        "integration setup",
        "conversation search"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing search: '{query}'")
        
        # Try POST with JSON
        try:
            payload = {"query": query, "limit": 5}
            response = requests.post(
                search_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            print(f"POST Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    results = response.json()
                    if isinstance(results, list):
                        print(f"✅ Found {len(results)} results")
                        for i, tool in enumerate(results[:3]):
                            name = tool.get('name', 'Unknown')
                            score = tool.get('score', 'N/A')
                            print(f"  {i+1}. {name} (score: {score})")
                    else:
                        print(f"✅ Response: {results}")
                except json.JSONDecodeError:
                    print(f"✅ Response: {response.text[:100]}...")
            else:
                print(f"❌ Error: {response.text[:100]}...")
                
        except Exception as e:
            print(f"❌ POST Error: {e}")

def main():
    """Run detailed pruning tests"""
    print("Detailed Remote Server Pruning Tests")
    print("=" * 70)
    
    # Test 1: Prune with different prompts
    prune_results = test_prune_with_different_prompts()
    
    # Test 2: Search endpoint
    test_search_endpoint()
    
    # Summary
    print("\n" + "=" * 70)
    print("PRUNING TEST SUMMARY")
    print("=" * 70)
    
    for result in prune_results:
        test_name = result['test']
        embedding_success = result.get('embedding_success', False)
        tools_analyzed = result.get('tools_analyzed', 0)
        tools_with_embeddings = result.get('tools_with_embeddings', 0)
        
        status = "✅ PASS" if embedding_success else "❌ FAIL"
        print(f"{test_name:20} {status} (Tools: {tools_analyzed}, Embeddings: {tools_with_embeddings})")
    
    # Overall assessment
    successful_tests = sum(1 for r in prune_results if r.get('embedding_success', False))
    total_tests = len(prune_results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests showed embedding generation")
    
    if successful_tests > 0:
        print("🎉 Embedding generation is working in the pruning system!")
    else:
        print("⚠️ Embedding generation may need further investigation in the pruning system")

if __name__ == "__main__":
    main()