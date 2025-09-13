# Tool Description Enhancement Experiments

This directory contains experimental code for enhancing tool descriptions using LLM analysis to improve semantic search accuracy.

## Overview

The goal is to take existing tool descriptions and use an LLM (Ollama Gemma3:12b) to create enhanced, semantically rich descriptions that improve search accuracy by 15-30%.

## Data Structure

Based on analysis of the current tool ingestion system, we have access to the following data for each tool:

- **Basic Info**: `id`, `name`, `description`, `tool_type`, `source_type`
- **Schema**: `json_schema` with detailed parameters and descriptions
- **Tags**: Categorization tags like `mcp:postiz`, `mcp:huly`, etc.
- **MCP Context**: `mcp_server_name`, `metadata_.mcp.server_name`
- **Categories**: Derived categories like "Knowledge Base", "Agent Management", etc.

## Files

- `test_enhancement.py` - Main experimental script
- `enhancement_prompts.py` - Prompt templates for different tool types
- `ollama_client.py` - Ollama integration client
- `sample_tools.json` - Sample tool data for testing
- `results/` - Enhancement results and comparisons

## Usage

```bash
# Set environment variables
export OLLAMA_BASE_URL="http://100.81.139.20:11434/v1"
export OLLAMA_MODEL="gemma3:12b"

# Run enhancement test
python test_enhancement.py
```

## Tool Categories Observed

Current tools fall into these main categories:
- **Agent Management** (17 tools) - Creating, managing, and communicating with agents
- **Knowledge Base** (26 tools) - Document management, file operations, reading
- **Memory Management** (9 tools) - Core memory, archival memory, memory blocks  
- **Ghost CMS** (10 tools) - Content management system operations
- **Other** (117 tools) - Various specialized tools from different MCP servers

## Enhancement Strategy

1. **Context-Aware Enhancement**: Use MCP server context and tool relationships
2. **Parameter Intelligence**: Analyze JSON schemas to understand usage patterns
3. **Use Case Generation**: Create realistic usage scenarios
4. **Semantic Keyword Expansion**: Add related terms and synonyms