#!/usr/bin/env python3
"""
Test script to verify Graphiti-compatible embedding configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_config import validate_embedding_config
import asyncio
from embedding_providers import EmbeddingProviderFactory, EmbeddingProviderContext

def test_config_validation():
    """Test the embedding configuration validation"""
    print("🧪 Testing Graphiti-compatible embedding configuration...")
    result = validate_embedding_config()
    print(f"✅ Configuration validation: {'PASS' if result else 'FAIL'}\n")
    return result

def test_provider_availability():
    """Test which providers are available"""
    print("🧪 Testing available embedding providers...")
    providers = EmbeddingProviderFactory.list_providers()
    print(f"Available providers: {providers}")
    
    for provider_name in providers:
        config = EmbeddingProviderFactory.get_provider_config(provider_name)
        print(f"{provider_name}: {config}")
    print()
    return True

async def test_openai_provider():
    """Test OpenAI provider with Graphiti-compatible settings"""
    print("🧪 Testing OpenAI provider...")
    try:
        async with EmbeddingProviderContext('openai') as provider:
            test_text = "This is a test query for Graphiti compatibility"
            embedding = await provider.get_single_embedding(test_text)
            
            print(f"✅ OpenAI provider test successful!")
            print(f"   Model: {provider.model}")
            print(f"   Dimensions: {len(embedding)} (expected: {provider.dimensions})")
            print(f"   Provider: {provider.provider_name}")
            return len(embedding) == provider.dimensions
            
    except Exception as e:
        print(f"❌ OpenAI provider test failed: {e}")
        return False

async def test_ollama_provider_config():
    """Test Ollama provider configuration (without actual connection)"""
    print("🧪 Testing Ollama provider configuration...")
    try:
        # Test configuration without making actual API calls
        provider = EmbeddingProviderFactory.create('ollama')
        
        print(f"✅ Ollama provider configuration successful!")
        print(f"   Model: {provider.model}")
        print(f"   Expected dimensions: {provider.dimensions}")
        print(f"   Base URL: {provider.base_url}")
        print(f"   Provider: {provider.provider_name}")
        
        # Validate Graphiti compatibility
        expected_model = "dengcao/Qwen3-Embedding-4B:Q4_K_M"
        expected_dimensions = 2560
        expected_host = "192.168.50.80"
        
        graphiti_compatible = (
            provider.model == expected_model and 
            provider.dimensions == expected_dimensions and
            expected_host in provider.base_url
        )
        
        print(f"✅ Graphiti compatibility: {'PASS' if graphiti_compatible else 'FAIL'}")
        if not graphiti_compatible:
            print(f"   Expected model: {expected_model}, got: {provider.model}")
            print(f"   Expected dimensions: {expected_dimensions}, got: {provider.dimensions}")
            print(f"   Expected host: {expected_host}, base_url: {provider.base_url}")
        
        await provider.close()
        return graphiti_compatible
        
    except Exception as e:
        print(f"❌ Ollama provider configuration test failed: {e}")
        return False

def test_environment_variable_support():
    """Test environment variable override support"""
    print("🧪 Testing environment variable support...")
    
    # Simulate Graphiti environment variables
    original_env = {}
    test_env_vars = {
        'EMBEDDING_PROVIDER': 'ollama',
        'OLLAMA_EMBEDDING_HOST': '192.168.50.80',
        'OLLAMA_EMBEDDING_MODEL': 'dengcao/Qwen3-Embedding-4B:Q4_K_M',
        'EMBEDDING_DIMENSION': '2560',
        'USE_OLLAMA_EMBEDDINGS': 'true'
    }
    
    # Save original values and set test values
    for key, value in test_env_vars.items():
        original_env[key] = os.getenv(key)
        os.environ[key] = value
    
    try:
        provider = EmbeddingProviderFactory.create_from_env()
        
        success = (
            provider.provider_name == 'ollama' and
            provider.model == 'dengcao/Qwen3-Embedding-4B:Q4_K_M' and
            provider.dimensions == 2560 and
            '192.168.50.80' in provider.base_url
        )
        
        print(f"✅ Environment variable support: {'PASS' if success else 'FAIL'}")
        if success:
            print(f"   Provider: {provider.provider_name}")
            print(f"   Model: {provider.model}")
            print(f"   Dimensions: {provider.dimensions}")
            print(f"   Base URL: {provider.base_url}")
        
        return success
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

def main():
    """Run all Graphiti compatibility tests"""
    print("🚀 Testing Graphiti-Compatible Embedding Configuration\n")
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Provider Availability", test_provider_availability),
        ("OpenAI Provider", lambda: asyncio.run(test_openai_provider())),
        ("Ollama Configuration", lambda: asyncio.run(test_ollama_provider_config())),
        ("Environment Variables", test_environment_variable_support)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
            print()
    
    print("📊 Graphiti Compatibility Test Results:")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All Graphiti compatibility tests passed!")
        print("Ready to use with Graphiti's Qwen3-Embedding-4B configuration!")
        return 0
    else:
        print("⚠️ Some tests failed - check configuration")
        return 1

if __name__ == "__main__":
    exit(main())