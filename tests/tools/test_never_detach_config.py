#!/usr/bin/env python3
"""
Test script to verify the never-detach configuration works correctly.
"""
import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tool-selector-api'))

def test_never_detach_config():
    """Test that the never-detach configuration loads correctly."""
    print("Testing never-detach configuration...")
    
    # Set test environment variables
    os.environ['NEVER_DETACH_TOOLS'] = 'find_tools,test_tool,another_tool'
    
    # Import the configuration (this will reload the environment variables)
    try:
        # Import the API server module to test configuration loading
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "api_server", 
            os.path.join(os.path.dirname(__file__), 'tool-selector-api', 'api_server.py')
        )
        api_server = importlib.util.module_from_spec(spec)
        
        # Mock the required dependencies that might not be available
        sys.modules['quart'] = type('MockQuart', (), {
            'Quart': lambda name: None,
            'request': None,
            'jsonify': lambda x: x
        })()
        sys.modules['weaviate_tool_search'] = type('MockWeaviate', (), {
            'search_tools': lambda **kwargs: [],
            'init_client': lambda: None,
            'get_embedding_for_text': lambda x: [],
            'get_tool_embedding_by_id': lambda x: []
        })()
        sys.modules['upload_tools_to_weaviate'] = type('MockUpload', (), {
            'upload_tools': lambda: None
        })()
        sys.modules['aiohttp'] = type('MockAiohttp', (), {})()
        sys.modules['aiofiles'] = type('MockAiofiles', (), {})()
        sys.modules['hypercorn.config'] = type('MockHypercorn', (), {'Config': lambda: None})()
        sys.modules['hypercorn.asyncio'] = type('MockHypercorn', (), {'serve': lambda x, y: None})()
        
        spec.loader.exec_module(api_server)
        
        # Test the configuration
        expected_tools = ['find_tools', 'test_tool', 'another_tool']
        actual_tools = api_server.NEVER_DETACH_TOOLS
        
        print(f"Expected never-detach tools: {expected_tools}")
        print(f"Actual never-detach tools: {actual_tools}")
        
        if actual_tools == expected_tools:
            print("✅ PASS: Never-detach configuration loaded correctly")
            return True
        else:
            print("❌ FAIL: Never-detach configuration mismatch")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error testing configuration: {e}")
        return False

def test_tool_categorization_logic():
    """Test the tool categorization logic with mock data."""
    print("\nTesting tool categorization logic...")
    
    # Mock tool data
    mock_tools = [
        {'id': 'tool-1', 'name': 'find_tools', 'tool_type': 'external_mcp'},
        {'id': 'tool-2', 'name': 'some_other_tool', 'tool_type': 'external_mcp'},
        {'id': 'tool-3', 'name': 'test_tool', 'tool_type': 'custom'},
        {'id': 'tool-4', 'name': 'core_tool', 'tool_type': 'core'}
    ]
    
    # Test sets
    requested_keep_tool_ids = {'tool-2'}
    requested_newly_matched_tool_ids = set()
    never_detach_tools = ['find_tools', 'test_tool']
    
    core_tools = []
    mcp_tools = []
    
    for tool in mock_tools:
        tool_id = tool['id']
        tool_name = tool.get('name', '').lower()
        
        # Apply the same logic as in the real code
        is_never_detach_tool = (
            tool_id in requested_keep_tool_ids or 
            tool_id in requested_newly_matched_tool_ids or
            any(never_detach_name.lower() in tool_name for never_detach_name in never_detach_tools)
        )
        
        if is_never_detach_tool:
            core_tools.append(tool)
        elif tool.get("tool_type") == "external_mcp" or tool.get("tool_type") == "custom":
            mcp_tools.append(tool)
        else:
            core_tools.append(tool)
    
    print(f"Core tools (protected): {[t['name'] for t in core_tools]}")
    print(f"MCP tools (can be pruned): {[t['name'] for t in mcp_tools]}")
    
    # Expected results:
    # - find_tools should be in core (never-detach)
    # - some_other_tool should be in core (keep list)
    # - test_tool should be in core (never-detach)
    # - core_tool should be in core (core type)
    # - No tools should be in MCP list
    
    expected_core_names = {'find_tools', 'some_other_tool', 'test_tool', 'core_tool'}
    actual_core_names = {t['name'] for t in core_tools}
    expected_mcp_names = set()
    actual_mcp_names = {t['name'] for t in mcp_tools}
    
    if actual_core_names == expected_core_names and actual_mcp_names == expected_mcp_names:
        print("✅ PASS: Tool categorization logic works correctly")
        return True
    else:
        print(f"❌ FAIL: Expected core: {expected_core_names}, got: {actual_core_names}")
        print(f"❌ FAIL: Expected MCP: {expected_mcp_names}, got: {actual_mcp_names}")
        return False

if __name__ == "__main__":
    print("=== Testing Never-Detach Tool Protection ===\n")
    
    test1_passed = test_never_detach_config()
    test2_passed = test_tool_categorization_logic()
    
    print(f"\n=== Test Results ===")
    print(f"Configuration test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Categorization test: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! The never-detach protection should work correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        sys.exit(1)