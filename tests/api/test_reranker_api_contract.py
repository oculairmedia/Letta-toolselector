#!/usr/bin/env python3
"""
Test script to validate Weaviate reranker API contract format
"""
import json
from typing import List, Dict, Any

def test_weaviate_reranker_format():
    """Test what API format Weaviate expects for reranker-transformers"""
    
    print("=== Testing Weaviate Reranker API Format ===")
    
    # This would be the format we need to implement in our adapter
    sample_request = {
        "query": "python csv processing tool", 
        "documents": [
            "CSVParser - Advanced CSV file processing with filtering",
            "DataProcessor - General purpose data manipulation tool",
            "FileManager - Basic file operations and management"
        ]
    }
    
    print("Expected request format:")
    print(json.dumps(sample_request, indent=2))
    
    # Expected response format
    sample_response = {
        "scores": [0.95, 0.23, 0.12]  # Scores in same order as input documents
    }
    
    print("\nExpected response format:")
    print(json.dumps(sample_response, indent=2))

def test_ollama_instruction_format():
    """Test instruction format for Qwen3-Reranker"""
    
    print("\n=== Testing Ollama Qwen3-Reranker Instruction Format ===")
    
    query = "python csv processing tool"
    document = "CSVParser - Advanced CSV file processing with filtering"
    
    # Test different instruction formats
    formats = [
        # Format 1: Simple relevance scoring
        f"Query: {query}\nDocument: {document}\nRelevance score (0.0-1.0):",
        
        # Format 2: More detailed instruction
        f"""Given a search query and a document, determine how relevant the document is to the query.
Output only a single number between 0.0 and 1.0, where:
- 0.0 means completely irrelevant
- 1.0 means perfectly relevant

Query: {query}
Document: {document}

Relevance score:""",

        # Format 3: Instruction-aware format for Qwen
        f"""<|im_start|>system
You are a relevance scoring system. Rate how relevant a document is to a query on a scale of 0.0 to 1.0.
<|im_end|>
<|im_start|>user
Query: {query}
Document: {document}
<|im_end|>
<|im_start|>assistant
Relevance score:"""
    ]
    
    for i, format_text in enumerate(formats, 1):
        print(f"\n--- Format {i} ---")
        print(format_text)
        print("---")

if __name__ == "__main__":
    test_weaviate_reranker_format()
    test_ollama_instruction_format()