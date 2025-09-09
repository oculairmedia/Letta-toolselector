#!/usr/bin/env python3
"""
Specialized Embedding Prompts for Qwen3-Embedding-4B Model

This module implements instruction-aware prompting optimized for the Qwen3-Embedding-4B
model to improve tool search performance. According to Qwen3 documentation, using
specialized instructions can improve embedding performance by 1-5%.

The module provides both query enhancement (for search requests) and document enhancement
(for tool descriptions) with task-specific instruction prompts.
"""

import os
import re
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class PromptType(Enum):
    """Types of embedding prompts available."""
    TOOL_DESCRIPTION = "tool_description"
    SEARCH_QUERY = "search_query"
    GENERAL_TOOL = "general_tool"
    MCP_TOOL = "mcp_tool"
    API_TOOL = "api_tool"


@dataclass
class PromptTemplate:
    """Template for generating specialized embedding prompts."""
    instruction: str
    context: Optional[str] = None
    suffix: Optional[str] = None


class SpecializedEmbeddingPrompter:
    """
    Generates instruction-aware prompts optimized for Qwen3-Embedding-4B model
    in tool search and discovery contexts.
    """
    
    # Default prompt templates optimized for tool search
    DEFAULT_TEMPLATES = {
        PromptType.TOOL_DESCRIPTION: PromptTemplate(
            instruction="Given a tool description, encode this tool's capabilities and purpose",
            context="This tool helps users accomplish specific tasks through its functionality.",
            suffix="Focus on the tool's core capabilities, use cases, and operational context."
        ),
        PromptType.SEARCH_QUERY: PromptTemplate(
            instruction="Given a tool search request, find tools that match the user's intent",
            context="The user is looking for tools to help accomplish a specific task or goal.",
            suffix="Focus on understanding the user's underlying need and matching it to relevant tool capabilities."
        ),
        PromptType.GENERAL_TOOL: PromptTemplate(
            instruction="Given a software tool description, encode its functionality and use cases",
            context="This is a general-purpose software tool with specific capabilities.",
            suffix="Emphasize the tool's primary functions and target use cases."
        ),
        PromptType.MCP_TOOL: PromptTemplate(
            instruction="Given an MCP tool description, encode its external service capabilities",
            context="This is an MCP (Model Context Protocol) tool that provides external service integration.",
            suffix="Focus on the external service functionality and integration capabilities it provides."
        ),
        PromptType.API_TOOL: PromptTemplate(
            instruction="Given an API tool description, encode its endpoint and integration capabilities",
            context="This tool provides API access to external services or data sources.",
            suffix="Emphasize the API functionality, data access, and integration possibilities."
        )
    }
    
    def __init__(self, custom_templates: Optional[Dict[PromptType, PromptTemplate]] = None):
        """
        Initialize the specialized embedding prompter.
        
        Args:
            custom_templates: Optional custom prompt templates to override defaults
        """
        self.templates = self.DEFAULT_TEMPLATES.copy()
        if custom_templates:
            self.templates.update(custom_templates)
        
        # Load configuration from environment
        self.use_context = os.getenv('EMBEDDING_PROMPT_USE_CONTEXT', 'true').lower() == 'true'
        self.use_suffix = os.getenv('EMBEDDING_PROMPT_USE_SUFFIX', 'true').lower() == 'true'
        self.max_prompt_length = int(os.getenv('EMBEDDING_PROMPT_MAX_LENGTH', '512'))
    
    def enhance_tool_description(
        self, 
        description: str, 
        tool_type: str = "general",
        tool_name: Optional[str] = None,
        tool_source: Optional[str] = None
    ) -> str:
        """
        Enhance a tool description with instruction-aware prompting.
        
        Args:
            description: Raw tool description
            tool_type: Type of tool (mcp, api, python, etc.)
            tool_name: Optional tool name for additional context
            tool_source: Optional tool source for classification
            
        Returns:
            Enhanced description with instruction prompting
        """
        # Determine prompt type based on tool characteristics
        prompt_type = self._determine_tool_prompt_type(tool_type, tool_source)
        
        # Get appropriate template
        template = self.templates.get(prompt_type, self.templates[PromptType.TOOL_DESCRIPTION])
        
        # Build the enhanced prompt
        enhanced_prompt = self._build_prompt(
            template=template,
            content=description,
            additional_context=f"Tool: {tool_name}" if tool_name else None
        )
        
        return self._truncate_prompt(enhanced_prompt)
    
    def enhance_search_query(
        self, 
        query: str, 
        context: Optional[str] = None,
        search_type: str = "general"
    ) -> str:
        """
        Enhance a search query with instruction-aware prompting.
        
        Args:
            query: Raw search query from user
            context: Optional additional context about the search
            search_type: Type of search being performed
            
        Returns:
            Enhanced query with instruction prompting
        """
        # Use search query template
        template = self.templates[PromptType.SEARCH_QUERY]
        
        # Build the enhanced prompt
        enhanced_prompt = self._build_prompt(
            template=template,
            content=query,
            additional_context=context
        )
        
        return self._truncate_prompt(enhanced_prompt)
    
    def _determine_tool_prompt_type(self, tool_type: str, tool_source: Optional[str] = None) -> PromptType:
        """
        Determine the most appropriate prompt type for a tool.
        
        Args:
            tool_type: Type classification of the tool
            tool_source: Source/origin of the tool
            
        Returns:
            Most appropriate PromptType for this tool
        """
        tool_type_lower = tool_type.lower() if tool_type else ""
        tool_source_lower = tool_source.lower() if tool_source else ""
        
        # Check for MCP tools
        if "mcp" in tool_type_lower or "external_mcp" in tool_type_lower:
            return PromptType.MCP_TOOL
        
        # Check for API tools
        if any(keyword in tool_type_lower for keyword in ["api", "rest", "http", "service"]):
            return PromptType.API_TOOL
        
        # Check tool source for additional context
        if tool_source and any(keyword in tool_source_lower for keyword in ["mcp", "external"]):
            return PromptType.MCP_TOOL
        
        # Default to general tool
        return PromptType.GENERAL_TOOL
    
    def _build_prompt(
        self, 
        template: PromptTemplate, 
        content: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Build a complete prompt from template and content.
        
        Args:
            template: Prompt template to use
            content: Main content to embed in the prompt
            additional_context: Optional additional context
            
        Returns:
            Complete formatted prompt
        """
        parts = []
        
        # Add instruction
        parts.append(template.instruction)
        
        # Add context if enabled and available
        if self.use_context and template.context:
            parts.append(template.context)
        
        # Add additional context if provided
        if additional_context:
            parts.append(additional_context)
        
        # Add main content
        parts.append(f"Content: {content}")
        
        # Add suffix if enabled and available
        if self.use_suffix and template.suffix:
            parts.append(template.suffix)
        
        return " ".join(parts)
    
    def _truncate_prompt(self, prompt: str) -> str:
        """
        Truncate prompt to maximum length while preserving meaning.
        
        Args:
            prompt: Full prompt to potentially truncate
            
        Returns:
            Truncated prompt if necessary
        """
        if len(prompt) <= self.max_prompt_length:
            return prompt
        
        # Try to truncate at sentence boundaries
        sentences = re.split(r'[.!?]\s+', prompt)
        truncated = ""
        
        for sentence in sentences:
            test_length = len(truncated) + len(sentence) + 1
            if test_length <= self.max_prompt_length:
                truncated += sentence + ". "
            else:
                break
        
        # If no complete sentences fit, truncate at word boundary
        if not truncated.strip():
            words = prompt.split()
            truncated = ""
            for word in words:
                test_length = len(truncated) + len(word) + 1
                if test_length <= self.max_prompt_length:
                    truncated += word + " "
                else:
                    break
        
        return truncated.strip()
    
    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types."""
        return [pt.value for pt in PromptType]
    
    def add_custom_template(self, prompt_type: PromptType, template: PromptTemplate):
        """Add or update a custom prompt template."""
        self.templates[prompt_type] = template
    
    def get_template(self, prompt_type: PromptType) -> Optional[PromptTemplate]:
        """Get a specific prompt template."""
        return self.templates.get(prompt_type)


# Convenience functions for backward compatibility and ease of use

def enhance_tool_for_embedding(
    tool_description: str,
    tool_name: Optional[str] = None,
    tool_type: str = "general",
    tool_source: Optional[str] = None
) -> str:
    """
    Convenience function to enhance a tool description for embedding.
    
    Args:
        tool_description: Raw tool description
        tool_name: Optional tool name
        tool_type: Tool type classification
        tool_source: Optional tool source
        
    Returns:
        Enhanced description ready for embedding
    """
    prompter = SpecializedEmbeddingPrompter()
    return prompter.enhance_tool_description(
        description=tool_description,
        tool_type=tool_type,
        tool_name=tool_name,
        tool_source=tool_source
    )


def enhance_query_for_embedding(
    search_query: str,
    context: Optional[str] = None
) -> str:
    """
    Convenience function to enhance a search query for embedding.
    
    Args:
        search_query: Raw search query
        context: Optional additional context
        
    Returns:
        Enhanced query ready for embedding
    """
    prompter = SpecializedEmbeddingPrompter()
    return prompter.enhance_search_query(query=search_query, context=context)


# Global instance for module-level usage
_global_prompter = None


def get_global_prompter() -> SpecializedEmbeddingPrompter:
    """Get or create the global prompter instance."""
    global _global_prompter
    if _global_prompter is None:
        _global_prompter = SpecializedEmbeddingPrompter()
    return _global_prompter


# Example usage and testing
if __name__ == "__main__":
    # Test the specialized prompting system
    prompter = SpecializedEmbeddingPrompter()
    
    # Test tool description enhancement
    tool_desc = "Creates and manages GitHub issues, pull requests, and repositories"
    enhanced_tool = prompter.enhance_tool_description(
        description=tool_desc,
        tool_name="GitHub MCP",
        tool_type="mcp",
        tool_source="external_mcp"
    )
    print("Original tool description:")
    print(tool_desc)
    print("\nEnhanced tool description:")
    print(enhanced_tool)
    print("\n" + "="*80 + "\n")
    
    # Test query enhancement
    query = "find tools to create blog posts"
    enhanced_query = prompter.enhance_search_query(
        query=query,
        context="User wants to publish content to a blog"
    )
    print("Original query:")
    print(query)
    print("\nEnhanced query:")
    print(enhanced_query)