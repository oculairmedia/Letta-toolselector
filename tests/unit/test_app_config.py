"""
Unit tests for the app_config module.

Tests cover:
- Configuration data classes
- Environment variable loading
- Test configuration creation
- Service container
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tool-selector-api"))


# ============================================================================
# Configuration Data Class Tests
# ============================================================================

class TestLettaConfig:
    """Tests for LettaConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from app_config import LettaConfig
        
        config = LettaConfig()
        
        assert "letta" in config.url.lower()
        assert config.use_sdk is False
        assert config.api_key is None
    
    def test_normalize_url(self):
        """Should normalize URLs to include /v1."""
        from app_config import LettaConfig
        
        config = LettaConfig(url="http://localhost:8283")
        
        # The url property doesn't auto-normalize, but message_base_urls does
        assert config._normalize_url("http://localhost:8283") == "http://localhost:8283/v1"
        assert config._normalize_url("http://localhost:8283/v1") == "http://localhost:8283/v1"
        assert config._normalize_url("http://localhost:8283/v1/") == "http://localhost:8283/v1"
    
    def test_message_base_urls(self):
        """Should build list of message URLs."""
        from app_config import LettaConfig
        
        config = LettaConfig(
            url="http://main:8283",
            direct_message_url="http://direct:8283"
        )
        
        urls = config.message_base_urls
        
        assert len(urls) == 2
        assert "direct" in urls[0]  # Direct URL comes first
        assert "main" in urls[1]


class TestWeaviateConfig:
    """Tests for WeaviateConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from app_config import WeaviateConfig
        
        config = WeaviateConfig()
        
        assert config.http_host == "weaviate"
        assert config.http_port == 8080
        assert config.grpc_port == 50051
    
    def test_http_url_property(self):
        """Should build HTTP URL from host and port."""
        from app_config import WeaviateConfig
        
        config = WeaviateConfig(http_host="localhost", http_port=9090)
        
        assert config.http_url == "http://localhost:9090"


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""
    
    def test_default_values(self):
        """Should default to vLLM provider."""
        from app_config import RerankerConfig
        
        config = RerankerConfig()
        
        assert config.enabled is True
        assert config.provider == "vllm"
        assert config.timeout == 30.0


class TestToolLimitsConfig:
    """Tests for ToolLimitsConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from app_config import ToolLimitsConfig
        
        config = ToolLimitsConfig()
        
        assert config.max_total_tools == 30
        assert config.max_mcp_tools == 20
        assert config.min_mcp_tools == 7
        assert config.default_drop_rate == 0.1
        assert "find_tools" in config.never_detach_tools


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        from app_config import CacheConfig
        
        config = CacheConfig()
        
        assert config.cache_dir == "/app/runtime_cache"
    
    def test_path_properties(self):
        """Should build file paths correctly."""
        from app_config import CacheConfig
        
        config = CacheConfig(cache_dir="/tmp/cache")
        
        assert config.tool_cache_path == "/tmp/cache/tool_cache.json"
        assert config.mcp_servers_cache_path == "/tmp/cache/mcp_servers_cache.json"


class TestAppConfig:
    """Tests for AppConfig master configuration."""
    
    def test_default_values(self):
        """Should create with all defaults."""
        from app_config import AppConfig
        
        config = AppConfig()
        
        assert config.letta is not None
        assert config.weaviate is not None
        assert config.search is not None
        assert config.tool_limits is not None
        assert config.cache is not None
        assert config.debug is False
        assert config.testing is False


# ============================================================================
# Configuration Loading Tests
# ============================================================================

class TestLoadConfigFromEnv:
    """Tests for load_config_from_env function."""
    
    def test_loads_defaults_without_env(self):
        """Should use defaults when no env vars set."""
        from app_config import load_config_from_env
        
        # Clear relevant env vars
        with patch.dict(os.environ, {}, clear=True):
            config = load_config_from_env()
        
        assert config is not None
        assert config.letta.use_sdk is False
    
    def test_loads_letta_config_from_env(self):
        """Should load Letta config from env vars."""
        from app_config import load_config_from_env
        
        env = {
            'LETTA_API_URL': 'http://test:8283/v1',
            'LETTA_PASSWORD': 'test-key',
            'USE_LETTA_SDK': 'true'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
        
        assert 'test:8283' in config.letta.url
        assert config.letta.api_key == 'test-key'
        assert config.letta.use_sdk is True
    
    def test_loads_weaviate_config_from_env(self):
        """Should load Weaviate config from env vars."""
        from app_config import load_config_from_env
        
        env = {
            'WEAVIATE_HTTP_HOST': 'custom-host',
            'WEAVIATE_HTTP_PORT': '9999',
            'WEAVIATE_GRPC_PORT': '50052'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
        
        assert config.weaviate.http_host == 'custom-host'
        assert config.weaviate.http_port == 9999
        assert config.weaviate.grpc_port == 50052
    
    def test_loads_reranker_config_from_env(self):
        """Should load reranker config from env vars."""
        from app_config import load_config_from_env
        
        env = {
            'ENABLE_RERANKING': 'false',
            'RERANKER_PROVIDER': 'ollama',
            'RERANKER_TIMEOUT': '60.0'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
        
        assert config.search.reranker.enabled is False
        assert config.search.reranker.provider == 'ollama'
        assert config.search.reranker.timeout == 60.0
    
    def test_loads_tool_limits_from_env(self):
        """Should load tool limits from env vars."""
        from app_config import load_config_from_env
        
        env = {
            'MAX_TOTAL_TOOLS': '50',
            'MAX_MCP_TOOLS': '40',
            'MIN_MCP_TOOLS': '10',
            'MANAGE_ONLY_MCP_TOOLS': 'true',
            'NEVER_DETACH_TOOLS': 'tool1,tool2,tool3'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
        
        assert config.tool_limits.max_total_tools == 50
        assert config.tool_limits.max_mcp_tools == 40
        assert config.tool_limits.min_mcp_tools == 10
        assert config.tool_limits.manage_only_mcp_tools is True
        assert config.tool_limits.never_detach_tools == ['tool1', 'tool2', 'tool3']
    
    def test_protected_tools_alias(self):
        """Should support PROTECTED_TOOLS as alias for NEVER_DETACH_TOOLS."""
        from app_config import load_config_from_env
        
        env = {
            'PROTECTED_TOOLS': 'protected1,protected2'
        }
        
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()
        
        assert 'protected1' in config.tool_limits.never_detach_tools
        assert 'protected2' in config.tool_limits.never_detach_tools


class TestCreateTestConfig:
    """Tests for create_test_config function."""
    
    def test_creates_test_config(self):
        """Should create config suitable for testing."""
        from app_config import create_test_config
        
        config = create_test_config()
        
        assert config.testing is True
        assert config.search.reranker.enabled is False
        assert "localhost" in config.letta.url
    
    def test_accepts_overrides(self):
        """Should accept override values."""
        from app_config import create_test_config
        
        config = create_test_config(debug=True)
        
        assert config.debug is True


# ============================================================================
# Service Container Tests
# ============================================================================

class TestServiceContainer:
    """Tests for ServiceContainer dataclass."""
    
    def test_default_values(self):
        """Should initialize with None values."""
        from app_config import ServiceContainer
        
        container = ServiceContainer()
        
        assert container.search_service is None
        assert container.agent_service is None
        assert container.tool_manager is None
        assert container.weaviate_client is None
    
    def test_is_configured(self):
        """Should report configuration status."""
        from app_config import ServiceContainer
        
        container = ServiceContainer()
        assert container.is_configured() is False
        
        container.weaviate_client = MagicMock()
        assert container.is_configured() is True
    
    def test_stores_config(self):
        """Should store AppConfig reference."""
        from app_config import ServiceContainer, AppConfig
        
        config = AppConfig()
        container = ServiceContainer(config=config)
        
        assert container.config is config


class TestGlobalServices:
    """Tests for global service functions."""
    
    def test_get_services_returns_container(self):
        """Should return a ServiceContainer."""
        from app_config import get_services, reset_services
        
        # Reset first
        reset_services()
        
        services = get_services()
        
        assert services is not None
        from app_config import ServiceContainer
        assert isinstance(services, ServiceContainer)
    
    def test_set_services_stores_container(self):
        """Should store provided container."""
        from app_config import get_services, set_services, reset_services, ServiceContainer
        
        reset_services()
        
        custom = ServiceContainer()
        custom.weaviate_client = "test_client"
        
        set_services(custom)
        
        retrieved = get_services()
        assert retrieved.weaviate_client == "test_client"
        
        # Clean up
        reset_services()
    
    def test_reset_services_clears_container(self):
        """Should clear the global container."""
        from app_config import get_services, set_services, reset_services, ServiceContainer
        
        set_services(ServiceContainer(weaviate_client="test"))
        reset_services()
        
        # After reset, get_services creates a new empty container
        services = get_services()
        assert services.weaviate_client is None
