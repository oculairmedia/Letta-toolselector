#!/usr/bin/env python3
"""
Test script for the minimal reranker adapter integration
"""
import requests
import json
import time
from typing import List, Tuple

def test_minimal_adapter():
    """Test the minimal reranker adapter"""
    
    print("=== Testing Minimal Ollama Reranker Adapter ===")
    
    # Adapter endpoint
    adapter_url = "http://localhost:8082"
    
    # Test data with varying relevance levels
    test_cases = [
        {
            "name": "CSV Processing Query",
            "query": "tool for parsing CSV files",
            "documents": [
                "CSVParser - Advanced tool for parsing and processing CSV files with filtering capabilities",
                "FileManager - Basic file operations including copy, move, and delete",
                "DataProcessor - General purpose data processing and transformation tool",
                "EmailSender - Tool for sending email notifications and managing mailing lists"
            ],
            "expected_top": "CSVParser"
        },
        {
            "name": "Email Query", 
            "query": "send email notifications",
            "documents": [
                "EmailSender - Tool for sending email notifications and managing mailing lists",
                "CSVParser - Advanced tool for parsing and processing CSV files",
                "FileManager - Basic file operations and management",
                "NotificationService - Push notifications and alerts system"
            ],
            "expected_top": "EmailSender"
        }
    ]
    
    # Test health endpoint first
    print("1. Testing health endpoint...")
    try:
        health_response = requests.get(f"{adapter_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check connection error: {e}")
        print("Make sure to run: python minimal_reranker_adapter.py &")
        return False
    
    # Run reranking tests
    print("\n2. Testing reranking functionality...")
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Query: {test_case['query']}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{adapter_url}/rerank", 
                json={
                    "query": test_case['query'],
                    "documents": test_case['documents']
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Reranking successful in {elapsed:.2f}s")
                
                # Sort documents by score
                doc_scores = list(zip(test_case['documents'], result['scores']))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                print("\nRanked results:")
                for i, (doc, score) in enumerate(doc_scores):
                    marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                    doc_short = doc[:50] + "..." if len(doc) > 50 else doc
                    print(f"{marker} [{score:.3f}] {doc_short}")
                
                # Check if expected document is ranked first
                top_doc = doc_scores[0][0]
                if test_case['expected_top'] in top_doc:
                    print(f"✅ Expected top result '{test_case['expected_top']}' ranked first")
                else:
                    print(f"❌ Expected '{test_case['expected_top']}' to be top, but got: {top_doc[:30]}...")
                    all_passed = False
                    
            else:
                print(f"❌ Reranking failed: {response.status_code}")
                print(f"Response: {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            all_passed = False
    
    return all_passed

def test_performance():
    """Test performance characteristics"""
    print("\n3. Testing performance...")
    
    adapter_url = "http://localhost:8082"
    
    # Performance test with multiple documents
    perf_test = {
        "query": "data processing and analysis tool",
        "documents": [
            f"Tool{i} - Description of tool number {i} for various data processing tasks"
            for i in range(1, 11)  # 10 documents
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{adapter_url}/rerank", json=perf_test, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Performance test passed: {len(result['scores'])} documents in {elapsed:.2f}s")
            print(f"   Average time per document: {elapsed/len(perf_test['documents']):.3f}s")
        else:
            print(f"❌ Performance test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")

if __name__ == "__main__":
    print("Starting reranker adapter integration test...")
    print("Note: Make sure minimal_reranker_adapter.py is running first!")
    
    success = test_minimal_adapter()
    test_performance()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 All tests passed! Reranker integration is working.")
    else:
        print("⚠️  Some tests failed. Review the output above.")
    print(f"{'='*50}")