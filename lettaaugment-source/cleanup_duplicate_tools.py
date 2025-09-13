#!/usr/bin/env python3
"""
Cleanup Duplicate Tools in Weaviate

This script identifies and removes duplicate tools, keeping the enhanced versions
when available and preserving the most complete tool entries.
"""

import weaviate
import os
from dotenv import load_dotenv
from collections import defaultdict

def cleanup_duplicates():
    """Remove duplicate tools, keeping the best version of each"""
    load_dotenv()
    
    try:
        print("🔗 Connecting to Weaviate...")
        client = weaviate.connect_to_custom(
            http_host="192.168.50.90",
            http_port=8080,
            http_secure=False,
            grpc_host="192.168.50.90",
            grpc_port=50051,
            grpc_secure=False,
            skip_init_checks=True
        )
        
        collection = client.collections.get("Tool")
        
        # Get all tools
        print("📊 Analyzing tools...")
        all_tools = collection.query.fetch_objects(limit=1000)
        print(f"Total tools found: {len(all_tools.objects)}")
        
        # Group tools by name
        tool_groups = defaultdict(list)
        for tool in all_tools.objects:
            name = tool.properties.get('name')
            if name:
                tool_groups[name].append(tool)
        
        # Find duplicates
        duplicates = {name: tools for name, tools in tool_groups.items() if len(tools) > 1}
        print(f"Found {len(duplicates)} tools with duplicates")
        
        if not duplicates:
            print("✅ No duplicates found!")
            client.close()
            return
        
        # Process duplicates - keep the best version
        to_delete = []
        kept_count = 0
        deleted_count = 0
        
        for tool_name, duplicate_tools in duplicates.items():
            print(f"\n🔍 Processing '{tool_name}' ({len(duplicate_tools)} duplicates)")
            
            # Sort tools by preference:
            # 1. Enhanced descriptions first
            # 2. Longer descriptions
            # 3. More complete metadata
            def score_tool(tool):
                props = tool.properties
                score = 0
                
                # Enhanced description is much better
                enhanced = props.get('enhanced_description') or ''
                if enhanced and len(enhanced) > 50:
                    score += 1000
                
                # Longer original description is better
                original = props.get('description') or ''
                score += len(original)
                
                # Complete metadata is better
                if props.get('tool_id'):
                    score += 10
                if props.get('json_schema'):
                    score += 10
                if props.get('tags'):
                    score += 5
                
                return score
            
            # Sort by score descending (best first)
            sorted_tools = sorted(duplicate_tools, key=score_tool, reverse=True)
            best_tool = sorted_tools[0]
            tools_to_delete = sorted_tools[1:]
            
            # Show decision
            best_props = best_tool.properties
            best_enhanced = best_props.get('enhanced_description') or ''
            has_enhancement = "🤖" if best_enhanced else "📝"
            print(f"   ✅ Keeping: {has_enhancement} ID={best_tool.uuid} (score: {score_tool(best_tool)})")
            
            # Mark others for deletion
            for tool in tools_to_delete:
                props = tool.properties  
                enhanced = props.get('enhanced_description') or ''
                marker = "🤖" if enhanced else "📝"
                print(f"   ❌ Deleting: {marker} ID={tool.uuid} (score: {score_tool(tool)})")
                to_delete.append(tool.uuid)
                deleted_count += 1
            
            kept_count += 1
        
        # Perform deletions
        if to_delete:
            print(f"\n🗑️  Deleting {len(to_delete)} duplicate tools...")
            
            # Delete in batches
            batch_size = 50
            for i in range(0, len(to_delete), batch_size):
                batch = to_delete[i:i + batch_size]
                print(f"   Deleting batch {i//batch_size + 1}/{(len(to_delete) + batch_size - 1)//batch_size}")
                
                for uuid in batch:
                    try:
                        collection.data.delete_by_id(uuid)
                    except Exception as e:
                        print(f"   ⚠️  Failed to delete {uuid}: {e}")
        
        # Final verification
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"   ✅ Tools kept: {kept_count}")
        print(f"   ❌ Duplicates removed: {deleted_count}")
        
        # Verify final count
        final_tools = collection.query.fetch_objects(limit=1000)
        print(f"   📈 Final tool count: {len(final_tools.objects)}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    cleanup_duplicates()