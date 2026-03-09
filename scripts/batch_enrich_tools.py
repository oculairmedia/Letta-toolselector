#!/usr/bin/env python3
"""
Batch enrich all tools in Weaviate with enhanced descriptions.

Generates rich descriptions from tool names, descriptions, JSON schemas,
and server context — no LLM API required. Uses schema parameters and
server domain knowledge to produce search-friendly descriptions.
"""

import json
import re
import sys
import time
from typing import Any

import weaviate
from weaviate.classes.query import Filter


# Server domain context — maps MCP server names to domain descriptions
SERVER_CONTEXTS = {
    "letta": "AI agent management platform for creating, configuring, and managing AI agents with persistent memory, tools, conversations, and knowledge sources",
    "graphiti": "knowledge graph and episodic memory system for storing facts, entities, relationships, and temporal context",
    "huly": "project management and issue tracking platform for creating and managing issues, projects, sprints, and team workflows",
    "bookstack": "documentation and knowledge base platform for creating, organizing, and searching books, chapters, pages, and shelves",
    "ghost": "content management and blogging platform for publishing articles, managing posts, newsletters, and website content",
    "photoprism": "photo management and organization platform for browsing, searching, tagging, and managing photo libraries",
    "postiz": "social media management platform for creating posts, scheduling content, and publishing across multiple social media platforms",
    "komodo": "server and container infrastructure management for monitoring deployments, managing Docker containers, and server operations",
    "crawl4ai": "web crawling and scraping platform for extracting content, data, and information from websites",
    "openai_mcp": "OpenAI API integration for text generation, chat completion, embeddings, and AI model interactions",
    "tandoor": "recipe management and meal planning platform for storing, organizing, and searching recipes and ingredients",
    "penpot": "open-source design and prototyping platform for creating UI designs, wireframes, and visual assets",
    "payloadcms": "headless CMS for managing content, media, collections, and API-driven content delivery",
    "surefinance": "financial data and analysis platform for accessing market data, stock information, and financial metrics",
    "kitchen-orchestrator": "kitchen and cooking workflow orchestration for managing recipes, cooking steps, timers, and meal preparation",
    "Houdini_mcp": "3D modeling, VFX, and procedural generation with SideFX Houdini for creating geometry, materials, simulations, and animations",
    "hayhooks": "Haystack AI pipeline execution for running NLP pipelines, document processing, and retrieval-augmented generation",
    "opencode": "code editing and development environment management for working with files, projects, and development workflows",
    "Searxng": "meta-search engine for querying multiple search engines and aggregating web search results",
    "context7": "library and package documentation lookup for finding API references, code examples, and technical documentation",
    "matrix": "Matrix messaging and communication protocol for sending messages, managing rooms, and real-time chat",
    "switchboard": "communication routing and switching for managing message flows between different systems and services",
    "resume": "resume and CV management for creating, editing, and formatting professional resumes and career documents",
    "Reearcher": "research and information gathering for conducting web searches, analyzing sources, and synthesizing findings",
    "lettatoolsselector": "tool discovery and management for finding, attaching, and managing tools for AI agents based on semantic search",
    "agent_registry": "AI agent registry for discovering, listing, and managing available AI agents and their capabilities",
}


def extract_schema_info(json_schema_str: str) -> dict[str, Any]:
    """Extract useful info from a tool's JSON schema."""
    if not json_schema_str:
        return {"params": [], "actions": [], "entities": []}

    try:
        schema = json.loads(json_schema_str)
    except (json.JSONDecodeError, TypeError):
        return {"params": [], "actions": [], "entities": []}

    params = []
    properties = schema.get("parameters", {}).get("properties", {})
    for name, prop in properties.items():
        if name in ("request_heartbeat",):
            continue
        desc = prop.get("description", "")
        params.append({"name": name, "description": desc, "type": prop.get("type", "")})

    # Extract action words from parameter names and descriptions
    action_words = set()
    entity_words = set()
    action_patterns = r"\b(create|update|delete|search|find|list|get|set|add|remove|edit|manage|query|sync|upload|download|export|import|send|receive|start|stop|run|execute|attach|detach|browse|read|write|filter|sort|check|verify|monitor|track|schedule|publish|post|submit|cancel|close|open|connect|disconnect)\b"
    entity_patterns = r"\b(file|folder|page|book|chapter|issue|project|task|agent|tool|memory|block|message|room|user|post|image|photo|document|source|archive|conversation|job|server|pipeline|recipe|media|content|collection|template|design|report|comment|label|tag|status|config|setting|model|embed|vector)\b"

    all_text = " ".join([p["name"].replace("_", " ") + " " + p["description"] for p in params]).lower()
    action_words = set(re.findall(action_patterns, all_text))
    entity_words = set(re.findall(entity_patterns, all_text))

    return {"params": params, "actions": list(action_words), "entities": list(entity_words)}


def generate_enhanced_description(
    name: str,
    description: str,
    server_name: str,
    tags: list[str],
    json_schema_str: str,
) -> str:
    """Generate a rich enhanced description from tool metadata."""
    schema_info = extract_schema_info(json_schema_str)
    server_context = SERVER_CONTEXTS.get(server_name, "")

    parts = []

    # Core description
    if description:
        # Clean up the description - take first 2 sentences or 200 chars
        clean_desc = description.strip()
        if len(clean_desc) > 300:
            # Try to cut at sentence boundary
            sentences = re.split(r"(?<=[.!?])\s+", clean_desc)
            clean_desc = " ".join(sentences[:3])
            if len(clean_desc) > 400:
                clean_desc = clean_desc[:400] + "..."
        parts.append(clean_desc)

    # Server context
    if server_context:
        parts.append(f"Part of the {server_name} integration ({server_context}).")

    # Parameter-based capabilities
    if schema_info["params"]:
        param_descs = []
        for p in schema_info["params"][:8]:  # Top 8 params
            if p["description"]:
                param_descs.append(f"{p['name'].replace('_', ' ')}: {p['description'][:80]}")
            else:
                param_descs.append(p["name"].replace("_", " "))
        if param_descs:
            parts.append("Accepts parameters: " + "; ".join(param_descs) + ".")

    # Action-entity pairs for search
    if schema_info["actions"] or schema_info["entities"]:
        keywords = []
        for action in schema_info["actions"][:5]:
            for entity in schema_info["entities"][:5]:
                keywords.append(f"{action} {entity}")
        if keywords:
            parts.append("Related operations: " + ", ".join(keywords[:10]) + ".")

    # Tool name decomposition for keyword matching
    name_words = set(re.split(r"[_\-\s]+", name.lower()))
    name_words -= {"mcp", "tool", "the", "a", "an", "for", "and", "or", "to", "in", "of", "with"}
    if name_words:
        parts.append(f"Keywords: {', '.join(sorted(name_words))}.")

    return " ".join(parts)


def main():
    print("Connecting to Weaviate...")
    client = weaviate.connect_to_local(host="localhost", port=8091, grpc_port=50051)
    collection = client.collections.get("Tool")

    # Collect all unenriched tools
    unenriched = []
    already_enriched = 0
    for obj in collection.iterator(
        return_properties=["name", "description", "enhanced_description", "mcp_server_name", "tags", "json_schema"]
    ):
        ed = obj.properties.get("enhanced_description", "")
        if ed and ed.strip():
            already_enriched += 1
            continue
        unenriched.append(obj)

    print(f"Already enriched: {already_enriched}")
    print(f"To enrich: {len(unenriched)}")
    print()

    if not unenriched:
        print("All tools already enriched!")
        client.close()
        return

    # Enrich each tool
    success = 0
    errors = 0
    for i, obj in enumerate(unenriched):
        name = obj.properties.get("name", "")
        description = obj.properties.get("description", "")
        server = obj.properties.get("mcp_server_name", "")
        tags = obj.properties.get("tags", [])
        schema = obj.properties.get("json_schema", "")

        enhanced = generate_enhanced_description(name, description, server, tags, schema)

        if not enhanced.strip():
            print(f"  [{i + 1}/{len(unenriched)}] SKIP {name} (no data to enrich)")
            continue

        try:
            collection.data.update(uuid=obj.uuid, properties={"enhanced_description": enhanced})
            # Rate limit: Cohere trial API allows ~100 calls/min
            # Each Weaviate update triggers re-vectorization via Cohere
            time.sleep(0.7)
            success += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i + 1}/{len(unenriched)}] ✓ {name} ({len(enhanced)} chars)")
        except Exception as e:
            errors += 1
            print(f"  [{i + 1}/{len(unenriched)}] ✗ {name}: {e}")

    print()
    print(f"Done: {success} enriched, {errors} errors, {already_enriched} previously enriched")
    print(f"Total enriched: {success + already_enriched}/{success + errors + already_enriched}")

    client.close()


if __name__ == "__main__":
    main()
