#!/usr/bin/env python3
"""
Enhancement Prompt Templates

This module contains carefully crafted prompt templates for different types of tools,
optimized for semantic search improvement through LLM enhancement.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ToolCategory(Enum):
    """Tool categories for specialized prompting"""
    MCP_TOOL = "mcp_tool"
    CORE_LETTA = "letta_core" 
    BUILTIN = "letta_builtin"
    AGENT_MANAGEMENT = "agent_management"
    KNOWLEDGE_BASE = "knowledge_base"
    MEMORY_MANAGEMENT = "memory_management"
    PROJECT_MANAGEMENT = "project_management"
    GENERAL = "general"


@dataclass
class ToolContext:
    """Context information for tool enhancement"""
    name: str
    description: str
    tool_type: str
    mcp_server_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    tags: Optional[list] = None


class EnhancementPrompts:
    """
    Collection of enhancement prompt templates optimized for different tool types.
    
    Based on research findings about co-construction strategies, semantic relevance,
    and prompt engineering best practices.
    """
    
    # System prompt for all enhancement requests
    SYSTEM_PROMPT = """You are a technical writing specialist focused on creating semantically rich tool descriptions for AI agent systems. Your enhanced descriptions dramatically improve search accuracy by including relevant keywords, use cases, and contextual information that users naturally search for.

Key principles:
- Focus on user intent and natural language queries
- Include practical use cases and scenarios  
- Add semantic keywords and related terminology
- Describe integration patterns and workflows
- Maintain technical accuracy while improving discoverability
- Keep descriptions concise but comprehensive (200-400 words)"""

    # Base template for all tool types
    BASE_TEMPLATE = """Please enhance this tool description to improve semantic search accuracy:

**Tool Information:**
- Name: {name}
- Current Description: {description}
- Tool Type: {tool_type}
- Category: {category}
{context_info}

**Parameters:**
{parameters_info}

**Enhancement Requirements:**
Create an enhanced description that includes:

1. **Primary Purpose**: Clear statement of what this tool does and why users need it
2. **Use Cases**: 3-5 specific scenarios where this tool would be used
3. **Keywords**: Natural language terms users might search for to find this tool
4. **Integration Context**: How this tool fits into larger workflows or processes
5. **User Benefits**: What problems this tool solves for users

**Output Format:**
Provide only the enhanced description as a single paragraph. Do not include headers, bullet points, or explanations - just the improved description that will replace the original."""

    # Specialized templates for different tool categories
    
    MCP_TOOL_TEMPLATE = """Please enhance this MCP (Model Context Protocol) tool description:

**Tool Information:**
- Name: {name}
- Current Description: {description}
- MCP Server: {mcp_server_name}
- Tool Type: {tool_type}
- Category: {category}

**Parameters:**
{parameters_info}

**MCP Server Context:**
This tool is provided by the "{mcp_server_name}" MCP server, which specializes in {server_context}.

**Enhancement Focus:**
Create an enhanced description emphasizing:
- The tool's role within the {mcp_server_name} ecosystem
- Integration with other {mcp_server_name} tools
- Specific workflows this enables
- Natural language terms users search for when needing this functionality

Provide only the enhanced description as a single comprehensive paragraph."""

    AGENT_MANAGEMENT_TEMPLATE = """Please enhance this agent management tool description:

**Tool Information:**
- Name: {name}
- Current Description: {description}
- Tool Type: {tool_type}
- Focus Area: Multi-agent systems and agent lifecycle management

**Parameters:**
{parameters_info}

**Enhancement Focus:**
Create an enhanced description emphasizing:
- Agent lifecycle stages (creation, configuration, management, communication)
- Multi-agent collaboration scenarios
- Agent orchestration and coordination patterns
- Common agent management workflows users search for
- Integration with agent frameworks and systems

Include keywords like: agent creation, agent configuration, multi-agent systems, agent communication, agent orchestration, AI agent management, agent lifecycle, agent coordination.

Provide only the enhanced description as a single comprehensive paragraph."""

    KNOWLEDGE_BASE_TEMPLATE = """Please enhance this knowledge base tool description:

**Tool Information:**
- Name: {name}
- Current Description: {description}
- Tool Type: {tool_type}
- Focus Area: Information storage, retrieval, and knowledge management

**Parameters:**
{parameters_info}

**Enhancement Focus:**
Create an enhanced description emphasizing:
- Information organization and retrieval patterns
- Content management workflows
- Document lifecycle management
- Search and discovery scenarios users need
- Integration with content creation and editing systems

Include keywords like: document management, content creation, information retrieval, knowledge organization, content storage, document search, information management, content publishing.

Provide only the enhanced description as a single comprehensive paragraph."""

    MEMORY_MANAGEMENT_TEMPLATE = """Please enhance this memory management tool description:

**Tool Information:**
- Name: {name}
- Current Description: {description}
- Tool Type: {tool_type}
- Focus Area: AI agent memory systems and information persistence

**Parameters:**
{parameters_info}

**Enhancement Focus:**
Create an enhanced description emphasizing:
- Memory types (core memory, archival memory, episodic memory)
- Memory operations (storage, retrieval, search, management)
- Memory-based reasoning and context maintenance
- Long-term information persistence scenarios
- Memory optimization and organization patterns

Include keywords like: memory storage, information recall, context preservation, long-term memory, episodic memory, memory search, information persistence, contextual memory.

Provide only the enhanced description as a single comprehensive paragraph."""

    @classmethod
    def get_server_context(cls, server_name: str) -> str:
        """Get contextual information about MCP servers"""
        server_contexts = {
            "huly": "project management, issue tracking, and team collaboration workflows",
            "bookstack": "documentation management, knowledge organization, and content publishing",
            "letta": "AI agent management, memory systems, and agent lifecycle operations",
            "ghost": "content management, blog publishing, and editorial workflows", 
            "graphiti": "knowledge graph management, entity relationships, and graph-based reasoning",
            "postiz": "social media management, content scheduling, and multi-platform publishing",
            "resumerx": "resume management, professional profile creation, and career document workflows",
            "matrix": "secure messaging, real-time communication, and federated chat systems",
            "context7": "code documentation, library integration, and developer tooling",
            "filesystem": "file operations, directory management, and system-level file interactions",
            "claude-code-mcp": "development tooling, code analysis, and programming assistance"
        }
        return server_contexts.get(server_name, "specialized functionality and integrations")

    @classmethod
    def categorize_tool(cls, tool_context: ToolContext) -> ToolCategory:
        """Determine the best category for a tool based on context"""
        
        # Check tool type first
        if tool_context.tool_type in ["letta_core", "letta_memory_core"]:
            return ToolCategory.CORE_LETTA
        elif tool_context.tool_type == "letta_builtin":
            return ToolCategory.BUILTIN
        
        # Check by name patterns for agent management
        agent_keywords = ["agent", "create_agent", "delete_agent", "list_agents", "modify_agent"]
        if any(keyword in tool_context.name.lower() for keyword in agent_keywords):
            return ToolCategory.AGENT_MANAGEMENT
            
        # Check by name patterns for memory management  
        memory_keywords = ["memory", "archival", "core_memory", "memory_block"]
        if any(keyword in tool_context.name.lower() for keyword in memory_keywords):
            return ToolCategory.MEMORY_MANAGEMENT
            
        # Check by MCP server for knowledge base
        knowledge_servers = ["bookstack", "ghost", "filesystem"]
        if tool_context.mcp_server_name in knowledge_servers:
            return ToolCategory.KNOWLEDGE_BASE
            
        # Check by category if available
        if tool_context.category:
            category_lower = tool_context.category.lower()
            if "knowledge" in category_lower or "document" in category_lower:
                return ToolCategory.KNOWLEDGE_BASE
            elif "agent" in category_lower:
                return ToolCategory.AGENT_MANAGEMENT
            elif "memory" in category_lower:
                return ToolCategory.MEMORY_MANAGEMENT
            elif "project" in category_lower or "issue" in category_lower:
                return ToolCategory.PROJECT_MANAGEMENT
        
        # Default to MCP tool if from MCP server, otherwise general
        if tool_context.mcp_server_name:
            return ToolCategory.MCP_TOOL
        else:
            return ToolCategory.GENERAL

    @classmethod  
    def build_prompt(cls, tool_context: ToolContext) -> tuple[str, str]:
        """
        Build the appropriate prompt for a tool based on its context.
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        category = cls.categorize_tool(tool_context)
        
        # Format parameters information
        parameters_info = "None specified"
        if tool_context.parameters and isinstance(tool_context.parameters, dict):
            param_lines = []
            properties = tool_context.parameters.get("properties", {})
            required = tool_context.parameters.get("required", [])
            
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "No description")
                is_required = param_name in required
                req_text = "required" if is_required else "optional"
                param_lines.append(f"- {param_name} ({param_type}, {req_text}): {param_desc}")
            
            if param_lines:
                parameters_info = "\n".join(param_lines)
        
        # Build context info
        context_info = ""
        if tool_context.mcp_server_name:
            context_info += f"- MCP Server: {tool_context.mcp_server_name}\n"
        if tool_context.tags:
            context_info += f"- Tags: {', '.join(tool_context.tags)}\n"
        
        # Select template based on category
        if category == ToolCategory.MCP_TOOL and tool_context.mcp_server_name:
            server_context = cls.get_server_context(tool_context.mcp_server_name)
            template = cls.MCP_TOOL_TEMPLATE.format(
                name=tool_context.name,
                description=tool_context.description,
                mcp_server_name=tool_context.mcp_server_name,
                tool_type=tool_context.tool_type,
                category=tool_context.category or "General",
                parameters_info=parameters_info,
                server_context=server_context
            )
        elif category == ToolCategory.AGENT_MANAGEMENT:
            template = cls.AGENT_MANAGEMENT_TEMPLATE.format(
                name=tool_context.name,
                description=tool_context.description,
                tool_type=tool_context.tool_type,
                parameters_info=parameters_info
            )
        elif category == ToolCategory.KNOWLEDGE_BASE:
            template = cls.KNOWLEDGE_BASE_TEMPLATE.format(
                name=tool_context.name,
                description=tool_context.description,
                tool_type=tool_context.tool_type,
                parameters_info=parameters_info
            )
        elif category == ToolCategory.MEMORY_MANAGEMENT:
            template = cls.MEMORY_MANAGEMENT_TEMPLATE.format(
                name=tool_context.name,
                description=tool_context.description,
                tool_type=tool_context.tool_type,
                parameters_info=parameters_info
            )
        else:
            # Use base template
            template = cls.BASE_TEMPLATE.format(
                name=tool_context.name,
                description=tool_context.description,
                tool_type=tool_context.tool_type,
                category=tool_context.category or "General",
                context_info=context_info,
                parameters_info=parameters_info
            )
        
        return cls.SYSTEM_PROMPT, template


# Example usage
def main():
    """Example usage of enhancement prompts"""
    
    # Test with sample tool
    tool_context = ToolContext(
        name="huly_create_issue",
        description="Create a new issue in a project", 
        tool_type="external_mcp",
        mcp_server_name="huly",
        category="Project Management",
        parameters={
            "properties": {
                "project_identifier": {
                    "type": "string",
                    "description": "Project identifier"
                },
                "title": {
                    "type": "string",
                    "description": "Issue title"  
                },
                "priority": {
                    "type": "string",
                    "description": "Issue priority level"
                }
            },
            "required": ["project_identifier", "title"]
        }
    )
    
    system_prompt, user_prompt = EnhancementPrompts.build_prompt(tool_context)
    
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===") 
    print(user_prompt)


if __name__ == "__main__":
    main()