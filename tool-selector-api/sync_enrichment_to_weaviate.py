#!/usr/bin/env python3
"""
Sync Enriched Tools to Weaviate

Updates the Weaviate Tool collection with enriched data from the semantic enrichment cache.
Adds action_entities, semantic_keywords, use_cases, and server_domain fields.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_enriched_tools(cache_path: str) -> Dict[str, Dict[str, Any]]:
    """Load enriched tools from cache file."""
    with open(cache_path) as f:
        return json.load(f)


def get_weaviate_client():
    """Create Weaviate client connection."""
    host = os.getenv("WEAVIATE_HTTP_HOST", "192.168.50.90")
    port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_host = os.getenv("WEAVIATE_GRPC_HOST", "192.168.50.90")
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    logger.info(f"Connecting to Weaviate at HTTP:{host}:{port}, gRPC:{grpc_host}:{grpc_port}")
    
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=False,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        )
    )
    return client


def ensure_schema_has_enrichment_fields(client) -> bool:
    """
    Ensure the Tool collection has the enrichment fields.
    Returns True if schema was updated, False if already had fields.
    """
    collection = client.collections.get("Tool")
    config = collection.config.get()
    
    existing_props = {p.name for p in config.properties}
    
    new_props = []
    
    if "action_entities" not in existing_props:
        new_props.append(Property(
            name="action_entities",
            data_type=DataType.TEXT_ARRAY,
            description="Action-entity pairs like 'create issue', 'delete file'"
        ))
    
    if "semantic_keywords" not in existing_props:
        new_props.append(Property(
            name="semantic_keywords",
            data_type=DataType.TEXT_ARRAY,
            description="Search terms from semantic enrichment"
        ))
    
    if "use_cases" not in existing_props:
        new_props.append(Property(
            name="use_cases",
            data_type=DataType.TEXT_ARRAY,
            description="Natural language use case scenarios"
        ))
    
    if "server_domain" not in existing_props:
        new_props.append(Property(
            name="server_domain",
            data_type=DataType.TEXT,
            description="MCP server's domain (e.g., 'project management')"
        ))
    
    if new_props:
        for prop in new_props:
            logger.info(f"Adding property: {prop.name}")
            collection.config.add_property(prop)
        return True
    
    logger.info("Schema already has enrichment fields")
    return False


def sync_enriched_tools(
    client,
    enriched_tools: Dict[str, Dict[str, Any]],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Sync enriched data to Weaviate.
    
    Returns:
        Stats dict with counts of updated, skipped, not_found
    """
    collection = client.collections.get("Tool")
    
    stats = {"updated": 0, "skipped": 0, "not_found": 0, "errors": 0}
    
    # Build lookup of existing tools by name -> (uuid, properties)
    # We need full properties because Weaviate's update() doesn't work for partial updates
    # We must use replace() with merged properties instead
    logger.info("Fetching existing tools from Weaviate...")
    existing_tools = {}
    
    for item in collection.iterator(include_vector=False):
        name = item.properties.get("name")
        if name:
            existing_tools[name] = (item.uuid, dict(item.properties))
    
    logger.info(f"Found {len(existing_tools)} tools in Weaviate")
    
    # Update each enriched tool
    for tool_id, enriched in enriched_tools.items():
        tool_name = enriched.get("name")
        
        if not tool_name:
            stats["skipped"] += 1
            continue
        
        tool_data = existing_tools.get(tool_name)
        if not tool_data:
            logger.debug(f"Tool not found in Weaviate: {tool_name}")
            stats["not_found"] += 1
            continue
        
        uuid, current_props = tool_data
        
        # Merge enrichment data into current properties
        # Using replace() instead of update() because Weaviate's update() 
        # doesn't properly handle partial property updates
        merged_props = dict(current_props)
        merged_props["enhanced_description"] = enriched.get("enhanced_description", "")
        merged_props["action_entities"] = enriched.get("action_entities", [])
        merged_props["semantic_keywords"] = enriched.get("semantic_keywords", [])
        merged_props["use_cases"] = enriched.get("use_cases", [])
        merged_props["server_domain"] = enriched.get("server_domain", "")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would update {tool_name}: {len(merged_props['action_entities'])} actions")
            stats["updated"] += 1
            continue
        
        try:
            collection.data.replace(
                uuid=uuid,
                properties=merged_props
            )
            stats["updated"] += 1
            logger.debug(f"Updated {tool_name}")
        except Exception as e:
            logger.error(f"Failed to update {tool_name}: {e}")
            stats["errors"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync enriched tools to Weaviate")
    parser.add_argument(
        "--enrichment-cache",
        default="/app/enrichment_cache/enriched_tools.json",
        help="Path to enriched tools JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema update check"
    )
    
    args = parser.parse_args()
    
    # Check cache exists
    if not os.path.exists(args.enrichment_cache):
        logger.error(f"Enrichment cache not found: {args.enrichment_cache}")
        sys.exit(1)
    
    # Load enriched tools
    logger.info(f"Loading enriched tools from {args.enrichment_cache}")
    enriched_tools = load_enriched_tools(args.enrichment_cache)
    logger.info(f"Loaded {len(enriched_tools)} enriched tools")
    
    # Connect to Weaviate
    client = get_weaviate_client()
    
    try:
        # Ensure schema has enrichment fields
        if not args.skip_schema and not args.dry_run:
            logger.info("Checking Weaviate schema...")
            ensure_schema_has_enrichment_fields(client)
        
        # Sync enriched data
        logger.info("Syncing enriched tools to Weaviate...")
        stats = sync_enriched_tools(client, enriched_tools, dry_run=args.dry_run)
        
        # Report
        logger.info("=" * 60)
        logger.info("SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Updated: {stats['updated']}")
        logger.info(f"Not found in Weaviate: {stats['not_found']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Errors: {stats['errors']}")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
