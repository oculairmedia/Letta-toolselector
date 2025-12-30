#!/usr/bin/env python3
"""
Check Enhanced Tools in Weaviate

This script connects to Weaviate and examines the enhanced tool descriptions
to validate that the LLM enhancement process worked correctly.
"""

import weaviate
import os
import json
from dotenv import load_dotenv
import weaviate.classes.query as wq

def check_enhanced_tools():
    """Check the enhanced tools in Weaviate"""
    load_dotenv()
    
    try:
        # Connect to Weaviate
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
        
        # Get Tool collection
        collection = client.collections.get("Tool")
        
        # Get basic statistics
        print("📊 Getting collection statistics...")
        result = collection.query.fetch_objects(limit=5)  # Get first 5 tools as examples
        
        print(f"✅ Successfully connected to Weaviate")
        print(f"📈 Sample tools found: {len(result.objects)}")
        
        # Examine first few tools
        print("\n🔍 SAMPLE ENHANCED TOOL DESCRIPTIONS:")
        print("="*80)
        
        for i, obj in enumerate(result.objects, 1):
            props = obj.properties
            print(f"\n{i}. Tool: {props.get('name', 'Unknown')}")
            print(f"   Type: {props.get('tool_type', 'Unknown')}")
            print(f"   Enhancement Category: {props.get('enhancement_category', 'None')}")
            if props.get('mcp_server_name'):
                print(f"   MCP Server: {props.get('mcp_server_name')}")
            
            original = props.get('description', '')
            enhanced = props.get('enhanced_description') or ''
            
            print(f"\n   📝 Original ({len(original)} chars):")
            print(f"      {original}")
            
            print(f"\n   🤖 Enhanced ({len(enhanced)} chars):")
            if enhanced:
                print(f"      {enhanced[:200]}{'...' if len(enhanced) > 200 else ''}")
            else:
                print(f"      [No enhanced description - field is None or empty]")
            
            if enhanced and original:
                ratio = len(enhanced) / len(original) if len(original) > 0 else 0
                print(f"\n   📈 Length increase: {ratio:.1f}x")
            
            print("-" * 60)
        
        # Get enhancement category statistics
        print(f"\n📊 ENHANCEMENT CATEGORY BREAKDOWN:")
        print("="*50)
        
        # This is a simplified approach - in a real implementation you'd aggregate properly
        all_tools = collection.query.fetch_objects(limit=500)
        print(f"Total tools in database: {len(all_tools.objects)}")
        
        categories = {}
        enhanced_count = 0
        total_original_length = 0
        total_enhanced_length = 0
        
        for obj in all_tools.objects:
            props = obj.properties
            category = props.get('enhancement_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            original = props.get('description', '')
            enhanced = props.get('enhanced_description') or ''
            
            if enhanced and enhanced != original:
                enhanced_count += 1
                total_original_length += len(original)
                total_enhanced_length += len(enhanced)
        
        print(f"Tools with enhanced descriptions: {enhanced_count}/{len(all_tools.objects)}")
        
        if enhanced_count > 0:
            avg_original = total_original_length / enhanced_count
            avg_enhanced = total_enhanced_length / enhanced_count
            print(f"Average original length: {avg_original:.1f} chars")
            print(f"Average enhanced length: {avg_enhanced:.1f} chars")
            print(f"Average length increase: {avg_enhanced/avg_original:.1f}x")
        
        print(f"\nCategory distribution:")
        # Filter out None categories and sort
        filtered_categories = {k: v for k, v in categories.items() if k is not None}
        for category, count in sorted(filtered_categories.items()):
            percentage = (count / len(all_tools.objects)) * 100
            print(f"  {category}: {count} tools ({percentage:.1f}%)")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error checking enhanced tools: {e}")

def search_enhanced_tools(query: str, limit: int = 5):
    """Test search with enhanced descriptions"""
    load_dotenv()
    
    try:
        print(f"\n🔍 TESTING SEARCH: '{query}'")
        print("="*60)
        
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
        
        # Perform hybrid search (combines vector and keyword search)
        response = collection.query.hybrid(
            query=query,
            limit=limit,
            alpha=0.75  # 75% vector, 25% keyword - good for semantic search
        )
        
        print(f"Found {len(response.objects)} results:")
        
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            print(f"\n{i}. {props.get('name')} (Score: {obj.metadata.score:.3f})")
            print(f"   Type: {props.get('tool_type')}")
            if props.get('mcp_server_name'):
                print(f"   MCP: {props.get('mcp_server_name')}")
            print(f"   Description: {props.get('description', '')[:100]}...")
            enhanced = props.get('enhanced_description')
            if enhanced:
                print(f"   Enhanced: {enhanced[:100]}...")
            else:
                print(f"   Enhanced: [None - no enhanced description]")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Search error: {e}")

if __name__ == "__main__":
    # Check enhanced tools
    check_enhanced_tools()
    
    # Test some searches
    test_queries = [
        "create project issues",
        "search memory information", 
        "document management",
        "agent creation",
        "file operations"
    ]
    
    for query in test_queries:
        search_enhanced_tools(query)
        print()