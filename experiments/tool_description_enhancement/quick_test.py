#!/usr/bin/env python3
"""
Quick Enhancement Test

Tests enhancement on a single tool for immediate feedback.
"""

import asyncio
import json
from enhancement_prompts import EnhancementPrompts, ToolContext
from ollama_client import OllamaClient

async def quick_test():
    """Test enhancement on a single tool"""
    
    # Single test tool
    test_tool = {
        "id": "tool-sample-mcp-1",
        "name": "huly_create_issue",
        "description": "Create a new issue in a project",
        "tool_type": "external_mcp",
        "mcp_server_name": "huly",
        "category": "Project Management",
        "json_schema": {
            "parameters": {
                "type": "object",
                "properties": {
                    "project_identifier": {
                        "type": "string",
                        "description": "Project identifier (e.g., \"LMP\")"
                    },
                    "title": {
                        "type": "string", 
                        "description": "Issue title"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Issue priority"
                    }
                },
                "required": ["project_identifier", "title"]
            }
        }
    }
    
    print("=== QUICK ENHANCEMENT TEST ===")
    print(f"Tool: {test_tool['name']}")
    print(f"Original: {test_tool['description']}")
    print()
    
    # Convert to ToolContext
    tool_context = ToolContext(
        name=test_tool["name"],
        description=test_tool["description"],
        tool_type=test_tool["tool_type"],
        mcp_server_name=test_tool.get("mcp_server_name"),
        parameters=test_tool["json_schema"]["parameters"],
        category=test_tool.get("category"),
        tags=test_tool.get("tags", [])
    )
    
    # Build prompt
    system_prompt, user_prompt = EnhancementPrompts.build_prompt(tool_context)
    
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print()
    print("=== USER PROMPT ===")
    print(user_prompt)
    print()
    
    # Test with Ollama
    client = OllamaClient()
    print("Testing connection...")
    if await client.test_connection():
        print("✅ Connection successful!")
        
        print("\nEnhancing description...")
        response = await client.enhance_description(user_prompt, system_prompt)
        
        if response.success:
            print("✅ Enhancement successful!")
            print(f"Processing time: {response.processing_time:.2f}s")
            print()
            print("=== ENHANCED DESCRIPTION ===")
            print(response.content)
            
            # Compare lengths
            original_len = len(test_tool['description'])
            enhanced_len = len(response.content)
            print(f"\nLength comparison:")
            print(f"Original: {original_len} chars")
            print(f"Enhanced: {enhanced_len} chars") 
            print(f"Increase: {enhanced_len/original_len:.1f}x")
        else:
            print(f"❌ Enhancement failed: {response.error_message}")
    else:
        print("❌ Connection failed!")

if __name__ == "__main__":
    asyncio.run(quick_test())