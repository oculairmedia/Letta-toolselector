#!/usr/bin/env python3
"""
Specialized embedding helpers for Qwen3-Embedding-4B

This module centralizes the logic required to format search queries and tool descriptions
in a way that works optimally with the Qwen3-Embedding-4B model. It maintains
backward compatibility with the legacy instruction templating system so we can
gradually migrate existing embeddings.
"""

import os
import re
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str) -> str:
    """Read a text environment variable while providing a default."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


QWEN3_MIGRATION_MODE = os.getenv("QWEN3_MIGRATION_MODE", "gradual").strip().lower()
USE_QWEN3_FORMAT = _env_flag("USE_QWEN3_FORMAT", True)
QWEN3_LAST_TOKEN_POOLING = _env_flag("QWEN3_LAST_TOKEN_POOLING", True)
QWEN3_INSTRUCTION_MODE = os.getenv("QWEN3_INSTRUCTION_MODE", "search").strip().lower()

DEFAULT_SEARCH_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

QWEN3_SEARCH_INSTRUCTION = _env_text(
    "QWEN3_SEARCH_INSTRUCTION",
    DEFAULT_SEARCH_INSTRUCTION,
)


def is_qwen3_format_enabled() -> bool:
    """Determine if the Qwen3 instruction format should be applied."""
    if QWEN3_MIGRATION_MODE == "disabled":
        return False
    if QWEN3_MIGRATION_MODE == "immediate":
        return True
    return USE_QWEN3_FORMAT


def get_search_instruction() -> str:
    """Return the canonical Qwen3 search instruction."""
    return QWEN3_SEARCH_INSTRUCTION


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Generate the Instruct/Query format required by Qwen3."""
    return f"Instruct: {task_description}\nQuery: {query}"


def format_query_for_qwen3(query: str) -> str:
    """Format a query without contaminating the last token."""
    if query is None:
        return ""
    cleaned_query = " ".join(query.strip().split())
    cleaned_query = cleaned_query.rstrip("?!.,;: ")
    return cleaned_query


def format_tool_description_for_qwen(description: str) -> str:
    """Return a clean description for embedding."""
    if description is None:
        return ""
    return description.strip()


class PromptType(Enum):
    """Types of legacy embedding prompts available."""
    TOOL_DESCRIPTION = "tool_description"
    SEARCH_QUERY = "search_query"
    GENERAL_TOOL = "general_tool"
    MCP_TOOL = "mcp_tool"
    API_TOOL = "api_tool"


@dataclass
class PromptTemplate:
    """Template for generating legacy specialized embedding prompts."""
    instruction: str
    context: Optional[str] = None
    suffix: Optional[str] = None


class SpecializedEmbeddingPrompter:
    """Legacy instruction-aware prompter kept for backwards compatibility."""

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
        self.templates = self.DEFAULT_TEMPLATES.copy()
        if custom_templates:
            self.templates.update(custom_templates)

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
        prompt_type = self._determine_tool_prompt_type(tool_type, tool_source)
        template = self.templates.get(prompt_type, self.templates[PromptType.TOOL_DESCRIPTION])
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
        template = self.templates[PromptType.SEARCH_QUERY]
        enhanced_prompt = self._build_prompt(
            template=template,
            content=query,
            additional_context=context
        )
        return self._truncate_prompt(enhanced_prompt)

    def _determine_tool_prompt_type(self, tool_type: str, tool_source: Optional[str] = None) -> PromptType:
        tool_type_lower = tool_type.lower() if tool_type else ""
        tool_source_lower = tool_source.lower() if tool_source else ""

        if "mcp" in tool_type_lower or "external_mcp" in tool_type_lower:
            return PromptType.MCP_TOOL

        if any(keyword in tool_type_lower for keyword in ["api", "rest", "http", "service"]):
            return PromptType.API_TOOL

        if tool_source and any(keyword in tool_source_lower for keyword in ["mcp", "external"]):
            return PromptType.MCP_TOOL

        return PromptType.GENERAL_TOOL

    def _build_prompt(
        self,
        template: PromptTemplate,
        content: str,
        additional_context: Optional[str] = None
    ) -> str:
        parts = [template.instruction]

        if self.use_context and template.context:
            parts.append(template.context)

        if additional_context:
            parts.append(additional_context)

        parts.append(f"Content: {content}")

        if self.use_suffix and template.suffix:
            parts.append(template.suffix)

        return " ".join(parts)

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self.max_prompt_length:
            return prompt

        sentences = re.split(r'[.!?]\s+', prompt)
        truncated = ""

        for sentence in sentences:
            test_length = len(truncated) + len(sentence) + 1
            if test_length <= self.max_prompt_length:
                truncated += sentence + ". "
            else:
                break

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
        return [pt.value for pt in PromptType]

    def add_custom_template(self, prompt_type: PromptType, template: PromptTemplate):
        self.templates[prompt_type] = template

    def get_template(self, prompt_type: PromptType) -> Optional[PromptTemplate]:
        return self.templates.get(prompt_type)


def _build_qwen3_query(search_query: str, task_description: Optional[str] = None) -> str:
    cleaned_query = format_query_for_qwen3(search_query)
    instruction = task_description or get_search_instruction()
    return get_detailed_instruct(instruction, cleaned_query)


def enhance_tool_for_embedding(
    tool_description: str,
    tool_name: Optional[str] = None,
    tool_type: str = "general",
    tool_source: Optional[str] = None
) -> str:
    if is_qwen3_format_enabled():
        return format_tool_description_for_qwen(tool_description)

    prompter = SpecializedEmbeddingPrompter()
    return prompter.enhance_tool_description(
        description=tool_description,
        tool_type=tool_type,
        tool_name=tool_name,
        tool_source=tool_source
    )


def enhance_query_for_embedding(
    search_query: str,
    context: Optional[str] = None,
    task_description: Optional[str] = None
) -> str:
    if is_qwen3_format_enabled():
        return _build_qwen3_query(search_query, task_description)

    prompter = SpecializedEmbeddingPrompter()
    return prompter.enhance_search_query(query=search_query, context=context)


_global_prompter: Optional[SpecializedEmbeddingPrompter] = None


def get_global_prompter() -> SpecializedEmbeddingPrompter:
    global _global_prompter
    if _global_prompter is None:
        _global_prompter = SpecializedEmbeddingPrompter()
    return _global_prompter


if __name__ == "__main__":
    example_tool_description = "Creates and manages GitHub issues, pull requests, and repositories"
    example_query = "find tools to create blog posts"

    print("Qwen3 format enabled:", is_qwen3_format_enabled())
    print("Formatted tool description:")
    print(enhance_tool_for_embedding(example_tool_description, tool_name="GitHub MCP", tool_type="mcp"))
    print()
    print("Formatted search query:")
    print(enhance_query_for_embedding(example_query))
