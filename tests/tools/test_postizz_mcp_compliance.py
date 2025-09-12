#!/usr/bin/env python3
"""
Test script to diagnose MCP protocol compliance issues with Postizz server
"""
import json
import requests
import time
from typing import Dict, Any, Optional

class MCPComplianceTester:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None
        self.session = requests.Session()
        
    def test_initialize(self) -> Dict[str, Any]:
        """Test MCP initialization sequence"""
        print("🔍 Testing MCP initialization...")
        
        # Test 1: Standard JSON-RPC initialization
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "compliance-tester", "version": "1.0.0"}
            }
        }
        
        results = {}
        
        # Test with standard headers (what Letta sends)
        print("  → Testing with standard headers...")
        try:
            response = self.session.post(
                self.server_url,
                json=init_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            results["standard_headers"] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "response_size": len(response.content),
                "has_json": self._is_json_response(response)
            }
            
            if results["standard_headers"]["has_json"]:
                results["standard_headers"]["response"] = response.json()
            else:
                results["standard_headers"]["raw_content"] = response.text[:200] + "..."
                
        except Exception as e:
            results["standard_headers"] = {"error": str(e)}
        
        # Test with SSE headers (what might work)
        print("  → Testing with SSE-compatible headers...")
        try:
            response = self.session.post(
                self.server_url,
                json=init_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                timeout=10
            )
            results["sse_headers"] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "response_size": len(response.content),
                "has_json": self._is_json_response(response),
                "mcp_session_id": response.headers.get("mcp-session-id")
            }
            
            if results["sse_headers"]["has_json"]:
                results["sse_headers"]["response"] = response.json()
            else:
                results["sse_headers"]["raw_content"] = response.text[:200] + "..."
                
            # Extract session ID if available
            if response.headers.get("mcp-session-id"):
                self.session_id = response.headers.get("mcp-session-id")
                
        except Exception as e:
            results["sse_headers"] = {"error": str(e)}
        
        return results
    
    def test_tools_list(self) -> Dict[str, Any]:
        """Test tools/list method"""
        print("🔍 Testing tools/list...")
        
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        results = {}
        
        # Test without session ID
        print("  → Testing without session ID...")
        try:
            response = self.session.post(
                self.server_url,
                json=tools_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            results["no_session"] = {
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "has_json": self._is_json_response(response)
            }
            
            if results["no_session"]["has_json"]:
                results["no_session"]["response"] = response.json()
            else:
                results["no_session"]["raw_content"] = response.text[:200] + "..."
                
        except Exception as e:
            results["no_session"] = {"error": str(e)}
        
        # Test with session ID if we have one
        if self.session_id:
            print(f"  → Testing with session ID: {self.session_id}")
            try:
                response = self.session.post(
                    self.server_url,
                    json=tools_request,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "Mcp-Session-Id": self.session_id
                    },
                    timeout=10
                )
                results["with_session"] = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type"),
                    "has_json": self._is_json_response(response)
                }
                
                if results["with_session"]["has_json"]:
                    response_data = response.json()
                    results["with_session"]["response"] = response_data
                    
                    # Count tools if available
                    if "result" in response_data and "tools" in response_data["result"]:
                        tool_count = len(response_data["result"]["tools"])
                        results["with_session"]["tool_count"] = tool_count
                        print(f"    ✅ Found {tool_count} tools!")
                        
                        # Show first few tools
                        for i, tool in enumerate(response_data["result"]["tools"][:3]):
                            print(f"       {i+1}. {tool.get('name', 'unknown')}")
                else:
                    results["with_session"]["raw_content"] = response.text[:200] + "..."
                    
            except Exception as e:
                results["with_session"] = {"error": str(e)}
        
        return results
    
    def _is_json_response(self, response) -> bool:
        """Check if response contains valid JSON"""
        try:
            response.json()
            return True
        except:
            return False
    
    def analyze_compliance(self, init_results: Dict, tools_results: Dict) -> Dict[str, Any]:
        """Analyze MCP compliance issues"""
        print("\n📋 MCP Compliance Analysis:")
        
        issues = []
        fixes = []
        
        # Check initialization compliance
        if "standard_headers" in init_results:
            std_result = init_results["standard_headers"]
            if "error" in std_result:
                issues.append("❌ Initialization fails with standard headers")
                fixes.append("✅ Accept requests without SSE-specific Accept headers")
            elif std_result.get("content_type") == "text/event-stream":
                issues.append("❌ Returns SSE stream instead of JSON for initialization")
                fixes.append("✅ Return JSON response for initialize method")
        
        # Check tools list compliance
        if "no_session" in tools_results:
            no_session_result = tools_results["no_session"]
            if "response" in no_session_result:
                response = no_session_result["response"]
                if "error" in response and "session" in response["error"].get("message", "").lower():
                    issues.append("❌ Requires session ID for tools/list")
                    fixes.append("✅ Make session management optional or return session in JSON response")
        
        # Check if tools are actually available
        tools_found = False
        if "with_session" in tools_results and "tool_count" in tools_results["with_session"]:
            tool_count = tools_results["with_session"]["tool_count"]
            if tool_count > 0:
                tools_found = True
                print(f"✅ Server has {tool_count} tools available")
            else:
                issues.append("❌ No tools returned from server")
        
        compliance_score = max(0, 100 - (len(issues) * 20))
        
        return {
            "compliance_score": compliance_score,
            "issues": issues,
            "fixes": fixes,
            "tools_found": tools_found
        }
    
    def run_full_test(self):
        """Run complete MCP compliance test"""
        print(f"🧪 Testing MCP Server: {self.server_url}")
        print("=" * 60)
        
        # Test initialization
        init_results = self.test_initialize()
        print()
        
        # Test tools list
        tools_results = self.test_tools_list()
        print()
        
        # Analyze compliance
        analysis = self.analyze_compliance(init_results, tools_results)
        
        print(f"📊 Compliance Score: {analysis['compliance_score']}/100")
        print()
        
        if analysis["issues"]:
            print("🚨 Issues Found:")
            for issue in analysis["issues"]:
                print(f"   {issue}")
            print()
            
            print("🔧 Required Fixes:")
            for fix in analysis["fixes"]:
                print(f"   {fix}")
            print()
        
        if analysis["tools_found"]:
            print("✅ Postizz tools are available but Letta can't access them due to protocol violations")
        else:
            print("❌ No tools found - server may have additional issues")
        
        return {
            "initialization": init_results,
            "tools_list": tools_results,
            "analysis": analysis
        }

def main():
    """Main test function"""
    postizz_url = "http://192.168.50.90:3457/mcp"
    
    tester = MCPComplianceTester(postizz_url)
    results = tester.run_full_test()
    
    # Save detailed results
    with open("/tmp/postizz_compliance_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed results saved to: /tmp/postizz_compliance_test.json")

if __name__ == "__main__":
    main()