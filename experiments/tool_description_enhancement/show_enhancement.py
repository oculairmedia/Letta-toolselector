#!/usr/bin/env python3
"""
Show Enhancement Example

Demonstrates what kind of enhanced descriptions we generate without calling the LLM.
Shows the prompts and expected improvements.
"""

from enhancement_prompts import EnhancementPrompts, ToolContext

def demonstrate_enhancement():
    """Show enhancement process for sample tools"""
    
    # Test tools from our sample set
    test_tools = [
        {
            "name": "huly_create_issue",
            "description": "Create a new issue in a project",
            "tool_type": "external_mcp",
            "mcp_server_name": "huly",
            "category": "Project Management",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_identifier": {"type": "string", "description": "Project identifier"},
                    "title": {"type": "string", "description": "Issue title"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}
                },
                "required": ["project_identifier", "title"]
            }
        },
        {
            "name": "archival_memory_search",
            "description": "Search archival memory for relevant information",
            "tool_type": "letta_core",
            "category": "Memory Management",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "page": {"type": "integer", "description": "Page number for pagination"},
                    "request_heartbeat": {"type": "boolean", "description": "Request immediate heartbeat"}
                },
                "required": ["query", "request_heartbeat"]
            }
        },
        {
            "name": "create_book",
            "description": "Creates a new book in Bookstack",
            "tool_type": "external_mcp",
            "mcp_server_name": "bookstack",
            "category": "Knowledge Base",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the book"},
                    "description": {"type": "string", "description": "A description of the book"},
                    "tags": {"type": "array", "description": "List of tag objects"}
                },
                "required": ["name", "description"]
            }
        }
    ]
    
    for i, tool in enumerate(test_tools, 1):
        print(f"\n{'='*80}")
        print(f"ENHANCEMENT EXAMPLE {i}: {tool['name'].upper()}")
        print(f"{'='*80}")
        
        # Convert to ToolContext
        tool_context = ToolContext(
            name=tool["name"],
            description=tool["description"],
            tool_type=tool["tool_type"],
            mcp_server_name=tool.get("mcp_server_name"),
            parameters=tool["parameters"],
            category=tool.get("category")
        )
        
        # Determine category
        category = EnhancementPrompts.categorize_tool(tool_context)
        
        print(f"Original Description: '{tool['description']}'")
        print(f"Tool Type: {tool['tool_type']}")
        print(f"Category: {category.value}")
        if tool.get('mcp_server_name'):
            server_context = EnhancementPrompts.get_server_context(tool['mcp_server_name'])
            print(f"MCP Server Context: {server_context}")
        
        # Build prompts
        system_prompt, user_prompt = EnhancementPrompts.build_prompt(tool_context)
        
        print(f"\n--- SYSTEM PROMPT ---")
        print(system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt)
        
        print(f"\n--- USER PROMPT (FIRST 500 CHARS) ---")
        print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
        
        print(f"\n--- EXPECTED ENHANCEMENT IMPROVEMENTS ---")
        print("✓ Semantic keywords and search terms users would naturally use")
        print("✓ Practical use cases and workflow scenarios")
        print("✓ Integration context with other tools in the ecosystem")
        print("✓ Parameter usage patterns and common configurations")
        print("✓ Problem-solving focus: what user challenges this addresses")
        
        if tool.get('mcp_server_name'):
            print(f"✓ Context about {tool['mcp_server_name']} server ecosystem and tool relationships")

def show_prompt_templates():
    """Show the different prompt templates we use"""
    
    print(f"\n{'='*80}")
    print("PROMPT TEMPLATE STRATEGY")
    print(f"{'='*80}")
    
    print("We use specialized prompts based on tool categories:")
    print("\n1. MCP_TOOL_TEMPLATE - For external MCP server tools")
    print("   - Emphasizes server ecosystem context")
    print("   - Focuses on integration workflows")
    print("   - Includes server-specific terminology")
    
    print("\n2. AGENT_MANAGEMENT_TEMPLATE - For agent lifecycle tools")
    print("   - Keywords: agent creation, configuration, orchestration")
    print("   - Multi-agent collaboration scenarios")
    print("   - Agent lifecycle management patterns")
    
    print("\n3. KNOWLEDGE_BASE_TEMPLATE - For content/document tools")
    print("   - Keywords: document management, content creation, publishing")
    print("   - Content lifecycle workflows")
    print("   - Information organization patterns")
    
    print("\n4. MEMORY_MANAGEMENT_TEMPLATE - For AI memory systems")
    print("   - Keywords: memory storage, information recall, context preservation")
    print("   - Memory types: core, archival, episodic")
    print("   - Long-term persistence scenarios")
    
    print("\n5. BASE_TEMPLATE - General purpose fallback")
    print("   - Covers all tool types not matching specialized categories")
    print("   - Focus on primary purpose and use cases")

if __name__ == "__main__":
    print("TOOL DESCRIPTION ENHANCEMENT DEMONSTRATION")
    print("==========================================")
    print("This shows how our framework generates enhanced descriptions")
    print("to improve semantic search accuracy in the Weaviate database.")
    
    demonstrate_enhancement()
    show_prompt_templates()
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("1. The LLM takes these prompts and generates 200-400 word enhanced descriptions")
    print("2. Enhanced descriptions include natural language terms users search for")
    print("3. Descriptions get embedded in Weaviate for improved semantic matching")
    print("4. A/B testing can validate 15-30% search accuracy improvements")
    print("5. Integration into upload_tools_to_weaviate.py during ingestion")