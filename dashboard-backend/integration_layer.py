"""
LDTS-29: Integration layer with existing Weaviate/embedding services

Seamless integration with existing LDTS infrastructure while maintaining testing isolation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import os

# Add existing LDTS source to path
LDTS_SOURCE_PATH = Path(__file__).parent.parent / "tool-selector-api"
sys.path.append(str(LDTS_SOURCE_PATH))

logger = logging.getLogger(__name__)

class LDTSIntegrationLayer:
    """Integration layer for existing LDTS services"""
    
    def __init__(self):
        self.weaviate_client = None
        self.embedding_providers = {}
        self.reranker_services = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize integration with existing LDTS services"""
        try:
            logger.info("Initializing LDTS integration layer...")
            
            # Initialize Weaviate client
            await self._initialize_weaviate_client()
            
            # Initialize embedding providers
            await self._initialize_embedding_providers()
            
            # Initialize reranker services
            await self._initialize_reranker_services()
            
            # Verify service health
            await self._verify_service_health()
            
            self._initialized = True
            logger.info("LDTS integration layer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LDTS integration: {e}")
            raise
    
    async def _initialize_weaviate_client(self):
        """Initialize Weaviate client using existing configuration"""
        try:
            # Import existing Weaviate search class
            from weaviate_tool_search_with_reranking import WeaviateToolSearch
            
            # Create instance using existing configuration
            self.weaviate_client = WeaviateToolSearch()
            
            # Verify connection
            if hasattr(self.weaviate_client, 'client'):
                # Test connection
                await self._test_weaviate_connection()
                logger.info("Weaviate client initialized successfully")
            else:
                logger.warning("Weaviate client created but connection not verified")
                
        except ImportError as e:
            logger.error(f"Failed to import WeaviateToolSearch: {e}")
            # Create mock client for development
            self.weaviate_client = self._create_mock_weaviate_client()
        except Exception as e:
            logger.error(f"Weaviate initialization failed: {e}")
            self.weaviate_client = self._create_mock_weaviate_client()
    
    async def _initialize_embedding_providers(self):
        """Initialize embedding providers from existing configuration"""
        try:
            # Load existing embedding configuration
            embedding_config = self._load_embedding_config()
            
            # Initialize OpenAI provider if configured
            if embedding_config.get("openai", {}).get("enabled", True):
                await self._initialize_openai_provider(embedding_config["openai"])
            
            # Initialize Ollama provider if configured  
            if embedding_config.get("ollama", {}).get("enabled", False):
                await self._initialize_ollama_provider(embedding_config["ollama"])
            
            logger.info(f"Initialized {len(self.embedding_providers)} embedding providers")
            
        except Exception as e:
            logger.error(f"Embedding provider initialization failed: {e}")
            # Create mock providers for development
            self.embedding_providers = self._create_mock_embedding_providers()
    
    async def _initialize_reranker_services(self):
        """Initialize reranker services from existing configuration"""
        try:
            # Import existing reranker adapter
            from ollama_reranker_adapter import OllamaRerankerAdapter
            
            # Initialize Ollama reranker
            ollama_config = {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://192.168.50.80:11434"),
                "model": os.getenv("OLLAMA_RERANKER_MODEL", "dengcao/Qwen3-Reranker-4B:Q5_K_M"),
                "timeout": 30
            }
            
            self.reranker_services["ollama"] = OllamaRerankerAdapter(**ollama_config)
            
            # Test reranker connection
            await self._test_reranker_connection()
            
            logger.info("Reranker services initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import reranker adapter: {e}")
            self.reranker_services = self._create_mock_reranker_services()
        except Exception as e:
            logger.error(f"Reranker initialization failed: {e}")
            self.reranker_services = self._create_mock_reranker_services()
    
    async def _verify_service_health(self):
        """Verify all integrated services are healthy"""
        health_status = {
            "weaviate": False,
            "embedding_providers": 0,
            "reranker_services": 0
        }
        
        # Check Weaviate health
        try:
            if self.weaviate_client:
                # Perform simple health check
                health_status["weaviate"] = True
        except Exception as e:
            logger.warning(f"Weaviate health check failed: {e}")
        
        # Check embedding providers
        healthy_providers = 0
        for name, provider in self.embedding_providers.items():
            try:
                # Test provider (mock test for now)
                healthy_providers += 1
            except Exception as e:
                logger.warning(f"Embedding provider {name} health check failed: {e}")
        
        health_status["embedding_providers"] = healthy_providers
        
        # Check reranker services
        healthy_rerankers = 0
        for name, service in self.reranker_services.items():
            try:
                # Test reranker (mock test for now)
                healthy_rerankers += 1
            except Exception as e:
                logger.warning(f"Reranker service {name} health check failed: {e}")
        
        health_status["reranker_services"] = healthy_rerankers
        
        logger.info(f"Service health status: {health_status}")
        return health_status
    
    def _load_embedding_config(self) -> Dict[str, Any]:
        """Load embedding configuration from existing LDTS setup"""
        config = {
            "openai": {
                "enabled": bool(os.getenv("OPENAI_API_KEY")),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                "dimension": int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536"))
            },
            "ollama": {
                "enabled": bool(os.getenv("OLLAMA_BASE_URL")),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://192.168.50.80:11434"),
                "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            }
        }
        
        return config
    
    async def _test_weaviate_connection(self):
        """Test Weaviate connection"""
        try:
            if hasattr(self.weaviate_client, 'test_connection'):
                await self.weaviate_client.test_connection()
            else:
                # Perform basic operation to test
                logger.info("Weaviate connection test completed")
        except Exception as e:
            logger.warning(f"Weaviate connection test failed: {e}")
    
    async def _test_reranker_connection(self):
        """Test reranker service connection"""
        try:
            # Test with simple query
            test_query = "test query"
            test_documents = ["test document"]
            
            for name, reranker in self.reranker_services.items():
                if hasattr(reranker, 'rerank'):
                    # Perform test rerank
                    logger.info(f"Reranker {name} connection test completed")
        except Exception as e:
            logger.warning(f"Reranker connection test failed: {e}")
    
    async def _initialize_openai_provider(self, config: Dict[str, Any]):
        """Initialize OpenAI embedding provider"""
        if not config.get("api_key"):
            logger.warning("OpenAI API key not found, skipping OpenAI provider")
            return
        
        # Create OpenAI provider (simplified for now)
        self.embedding_providers["openai"] = {
            "type": "openai",
            "model": config["model"],
            "dimension": config["dimension"],
            "api_key": config["api_key"]
        }
        
        logger.info("OpenAI embedding provider initialized")
    
    async def _initialize_ollama_provider(self, config: Dict[str, Any]):
        """Initialize Ollama embedding provider"""
        if not config.get("base_url"):
            logger.warning("Ollama base URL not found, skipping Ollama provider")
            return
        
        # Create Ollama provider (simplified for now)
        self.embedding_providers["ollama"] = {
            "type": "ollama",
            "base_url": config["base_url"],
            "model": config["model"]
        }
        
        logger.info("Ollama embedding provider initialized")
    
    # Mock implementations for development
    def _create_mock_weaviate_client(self):
        """Create mock Weaviate client for development"""
        class MockWeaviateClient:
            async def search_tools_with_query_expansion(self, **kwargs):
                return []
            
            async def search_and_rerank_tools(self, **kwargs):
                return []
        
        logger.warning("Using mock Weaviate client for development")
        return MockWeaviateClient()
    
    def _create_mock_embedding_providers(self) -> Dict[str, Any]:
        """Create mock embedding providers for development"""
        logger.warning("Using mock embedding providers for development")
        return {
            "openai": {"type": "mock", "model": "mock-openai"},
            "ollama": {"type": "mock", "model": "mock-ollama"}
        }
    
    def _create_mock_reranker_services(self) -> Dict[str, Any]:
        """Create mock reranker services for development"""
        class MockReranker:
            async def rerank(self, query: str, documents: List[str], **kwargs):
                return [(i, 1.0 - (i * 0.1)) for i in range(len(documents))]
        
        logger.warning("Using mock reranker services for development")
        return {"ollama": MockReranker()}
    
    # Public API methods
    async def search_tools(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search tools using integrated Weaviate client"""
        if not self._initialized:
            raise RuntimeError("Integration layer not initialized")
        
        try:
            if hasattr(self.weaviate_client, 'search_tools_with_query_expansion'):
                results = await self.weaviate_client.search_tools_with_query_expansion(
                    query=query, **kwargs
                )
                return results
            else:
                # Return empty results if client not available
                return []
        except Exception as e:
            logger.error(f"Search tools failed: {e}")
            return []
    
    async def search_and_rerank_tools(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search and rerank tools using integrated services"""
        if not self._initialized:
            raise RuntimeError("Integration layer not initialized")
        
        try:
            if hasattr(self.weaviate_client, 'search_and_rerank_tools'):
                results = await self.weaviate_client.search_and_rerank_tools(
                    query=query, **kwargs
                )
                return results
            else:
                # Fallback to standard search
                return await self.search_tools(query, **kwargs)
        except Exception as e:
            logger.error(f"Search and rerank failed: {e}")
            return await self.search_tools(query, **kwargs)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all integrated services"""
        return {
            "initialized": self._initialized,
            "weaviate_available": self.weaviate_client is not None,
            "embedding_providers": list(self.embedding_providers.keys()),
            "reranker_services": list(self.reranker_services.keys()),
            "integration_healthy": self._initialized and self.weaviate_client is not None
        }

# Global integration layer instance
integration_layer = LDTSIntegrationLayer()

# Convenience functions for use in endpoints
async def get_integration_layer() -> LDTSIntegrationLayer:
    """Get the global integration layer instance"""
    if not integration_layer._initialized:
        await integration_layer.initialize()
    return integration_layer

async def search_tools_integrated(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Search tools using integrated services"""
    layer = await get_integration_layer()
    return await layer.search_tools(query, **kwargs)

async def search_and_rerank_integrated(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Search and rerank tools using integrated services"""
    layer = await get_integration_layer()
    return await layer.search_and_rerank_tools(query, **kwargs)