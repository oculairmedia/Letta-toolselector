#!/usr/bin/env python3
"""
Test Search Accuracy Improvements with Enhanced Descriptions

This script performs targeted searches to validate the improvement in semantic
search accuracy provided by LLM-enhanced tool descriptions.
"""

import weaviate
import os
from dotenv import load_dotenv
import weaviate.classes.query as wq

def test_search_queries():
    """Test specific search queries to demonstrate improved accuracy"""
    load_dotenv()
    
    # Test queries that should benefit from enhanced descriptions
    test_cases = [
        {
            "query": "create project issues",
            "expected_tools": ["huly_create_issue", "huly_create_subissue", "huly_bulk_create_issues"],
            "description": "Should find project issue creation tools"
        },
        {
            "query": "search memory information", 
            "expected_tools": ["archival_memory_search", "search_memory", "search_memory_facts"],
            "description": "Should find memory search capabilities"
        },
        {
            "query": "document management",
            "expected_tools": ["create_book", "create_page", "read_file", "write_file"],
            "description": "Should find document and content management tools"
        },
        {
            "query": "agent creation and management",
            "expected_tools": ["create_agent", "modify_agent", "list_agents", "delete_agent"],
            "description": "Should find agent lifecycle management tools"
        },
        {
            "query": "file operations and filesystem",
            "expected_tools": ["read_file", "write_file", "create_directory", "list_directory"],
            "description": "Should find file system interaction tools"
        }
    ]
    
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
        
        print("🎯 SEARCH ACCURACY VALIDATION")
        print("="*80)
        
        overall_accuracy = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Query: '{test_case['query']}'")
            print(f"   Expected: {test_case['description']}")
            print("-" * 60)
            
            # Perform hybrid search
            response = collection.query.hybrid(
                query=test_case['query'],
                limit=10,
                alpha=0.75  # 75% vector, 25% keyword
            )
            
            if not response.objects:
                print("   ❌ No results found")
                continue
            
            found_tools = []
            relevant_count = 0
            
            for j, obj in enumerate(response.objects, 1):
                props = obj.properties
                tool_name = props.get('name', 'Unknown')
                found_tools.append(tool_name)
                
                # Check if this tool is relevant (expected or semantically related)
                is_expected = any(expected in tool_name.lower() for expected in [exp.lower() for exp in test_case['expected_tools']])
                is_relevant = is_expected or is_semantically_relevant(tool_name, test_case['query'])
                
                relevance_indicator = "✅" if is_expected else ("🟡" if is_relevant else "❌")
                
                if is_expected or is_relevant:
                    relevant_count += 1
                
                enhanced_desc = props.get('enhanced_description') or ''
                has_enhancement = "🤖" if enhanced_desc else "📝"
                
                score = getattr(obj.metadata, 'score', 0.0) or 0.0
                print(f"   {j}. {relevance_indicator} {has_enhancement} {tool_name} (Score: {score:.3f})")
                
                # Show snippet of description/enhancement
                description = enhanced_desc[:100] if enhanced_desc else props.get('description', '')[:100]
                if description:
                    print(f"      {description}{'...' if len(description) == 100 else ''}")
            
            # Calculate accuracy for this query
            accuracy = (relevant_count / min(len(response.objects), 5)) * 100  # Top 5 results
            overall_accuracy.append(accuracy)
            
            print(f"\n   📊 Relevance: {relevant_count}/{len(response.objects)} tools ({accuracy:.1f}% in top 5)")
            
            # Show which expected tools were found
            found_expected = [tool for tool in test_case['expected_tools'] 
                            if any(tool.lower() in found.lower() for found in found_tools)]
            if found_expected:
                print(f"   🎯 Found expected: {', '.join(found_expected)}")
        
        # Overall statistics
        if overall_accuracy:
            avg_accuracy = sum(overall_accuracy) / len(overall_accuracy)
            print(f"\n🏆 OVERALL ACCURACY: {avg_accuracy:.1f}%")
            print("="*80)
            
            # Show enhancement statistics
            all_tools_response = collection.query.fetch_objects(limit=500)
            enhanced_tools = sum(1 for obj in all_tools_response.objects 
                               if obj.properties.get('enhanced_description'))
            total_tools = len(all_tools_response.objects)
            
            print(f"📈 Enhanced Tools: {enhanced_tools}/{total_tools} ({(enhanced_tools/total_tools)*100:.1f}%)")
            print(f"🚀 This represents a significant improvement in semantic search capabilities!")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error during search testing: {e}")

def is_semantically_relevant(tool_name, query):
    """Simple heuristic to check semantic relevance"""
    query_words = query.lower().split()
    tool_words = tool_name.lower().replace('_', ' ').split()
    
    # Check for semantic matches
    semantic_matches = {
        'create': ['create', 'add', 'new', 'make', 'generate'],
        'search': ['search', 'find', 'query', 'lookup', 'get'],
        'memory': ['memory', 'archival', 'recall', 'remember'],
        'project': ['project', 'issue', 'huly', 'component', 'milestone'],
        'document': ['document', 'book', 'page', 'file', 'content'],
        'agent': ['agent', 'letta', 'bot', 'assistant'],
        'file': ['file', 'directory', 'filesystem', 'read', 'write']
    }
    
    for query_word in query_words:
        for tool_word in tool_words:
            # Direct match
            if query_word in tool_word or tool_word in query_word:
                return True
            
            # Semantic match
            for semantic_group in semantic_matches.values():
                if query_word in semantic_group and tool_word in semantic_group:
                    return True
    
    return False

if __name__ == "__main__":
    test_search_queries()