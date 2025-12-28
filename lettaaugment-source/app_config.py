"""
Application Configuration

Centralized configuration management for the Tool Selector API.
Provides typed configuration classes and factory functions for creating
the application with different configurations (production, testing, etc.)

Usage:
    from app_config import AppConfig, load_config_from_env
    
    # Load from environment
    config = load_config_from_env()
    
    # Or create custom config for testing
    config = AppConfig(
        letta=LettaConfig(url="http://test:8283"),
        weaviate=WeaviateConfig(http_host="localhost"),
        ...
    )
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class LettaConfig:
    """Configuration for Letta API connection."""
    url: str = "https://letta2.oculair.ca/v1"
    api_key: Optional[str] = None
    use_sdk: bool = False
    direct_message_url: Optional[str] = None
    
    @property
    def message_base_urls(self) -> List[str]:
        """Get list of message endpoint URLs."""
        urls = []
        if self.direct_message_url:
            urls.append(self._normalize_url(self.direct_message_url))
        if self.url:
            urls.append(self._normalize_url(self.url))
        return urls
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL to include /v1 and no trailing slash."""
        if not url:
            return ""
        normalized = url.rstrip('/')
        if not normalized.endswith('/v1'):
            normalized = f"{normalized}/v1"
        return normalized


@dataclass
class WeaviateConfig:
    """Configuration for Weaviate connection."""
    http_host: str = "weaviate"
    http_port: int = 8080
    grpc_host: str = "weaviate"
    grpc_port: int = 50051
    
    @property
    def http_url(self) -> str:
        return f"http://{self.http_host}:{self.http_port}"


@dataclass
class RerankerConfig:
    """Configuration for the reranker service."""
    enabled: bool = True
    provider: str = "vllm"  # "vllm" or "ollama"
    url: str = "http://100.81.139.20:11435/rerank"
    model: str = "qwen3-reranker-4b"
    timeout: float = 30.0
    initial_limit: int = 30
    top_k: int = 10


@dataclass 
class QueryExpansionConfig:
    """Configuration for query expansion."""
    enabled: bool = True
    use_universal: bool = True


@dataclass
class ToolLimitsConfig:
    """Configuration for tool limits and management."""
    max_total_tools: int = 30
    max_mcp_tools: int = 20
    min_mcp_tools: int = 7
    default_drop_rate: float = 0.1
    manage_only_mcp_tools: bool = False
    exclude_letta_core_tools: bool = False
    exclude_official_tools: bool = False
    never_detach_tools: List[str] = field(default_factory=lambda: ["find_tools"])


@dataclass
class CacheConfig:
    """Configuration for caching."""
    cache_dir: str = "/app/runtime_cache"
    tool_cache_filename: str = "tool_cache.json"
    mcp_servers_cache_filename: str = "mcp_servers_cache.json"
    
    @property
    def tool_cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.tool_cache_filename)
    
    @property
    def mcp_servers_cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.mcp_servers_cache_filename)


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    provider: str = "weaviate"  # "weaviate", "letta", or "hybrid"
    default_min_score: float = 35.0
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)


@dataclass
class WebhookConfig:
    """Configuration for webhooks."""
    matrix_bridge_url: Optional[str] = None


@dataclass
class AppConfig:
    """
    Master application configuration.
    
    Holds all configuration settings for the Tool Selector API.
    Can be loaded from environment variables or created programmatically.
    """
    letta: LettaConfig = field(default_factory=LettaConfig)
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    tool_limits: ToolLimitsConfig = field(default_factory=ToolLimitsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    
    # Runtime flags
    debug: bool = False
    testing: bool = False


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config_from_env() -> AppConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        AppConfig populated from environment
    """
    # Parse NEVER_DETACH_TOOLS from comma-separated string
    protected_tools_env = os.getenv('PROTECTED_TOOLS') or os.getenv('NEVER_DETACH_TOOLS', 'find_tools')
    never_detach_tools = [name.strip() for name in protected_tools_env.split(',') if name.strip()]
    
    # Build Letta config
    raw_letta_url = os.getenv('LETTA_API_URL', 'https://letta2.oculair.ca/v1')
    direct_url = os.getenv('LETTA_DIRECT_MESSAGE_URL') or os.getenv('LETTA_DIRECT_URL')
    
    letta_config = LettaConfig(
        url=raw_letta_url,
        api_key=os.getenv('LETTA_PASSWORD'),
        use_sdk=os.getenv('USE_LETTA_SDK', 'false').lower() == 'true',
        direct_message_url=direct_url
    )
    
    # Build Weaviate config
    weaviate_config = WeaviateConfig(
        http_host=os.getenv('WEAVIATE_HTTP_HOST', 'weaviate'),
        http_port=int(os.getenv('WEAVIATE_HTTP_PORT', '8080')),
        grpc_host=os.getenv('WEAVIATE_GRPC_HOST', 'weaviate'),
        grpc_port=int(os.getenv('WEAVIATE_GRPC_PORT', '50051'))
    )
    
    # Build reranker config
    reranker_config = RerankerConfig(
        enabled=os.getenv('ENABLE_RERANKING', 'true').lower() == 'true',
        provider=os.getenv('RERANKER_PROVIDER', 'vllm'),
        url=os.getenv('RERANKER_URL', 'http://100.81.139.20:11435/rerank'),
        model=os.getenv('RERANKER_MODEL', 'qwen3-reranker-4b'),
        timeout=float(os.getenv('RERANKER_TIMEOUT', '30.0')),
        initial_limit=int(os.getenv('RERANK_INITIAL_LIMIT', '30')),
        top_k=int(os.getenv('RERANK_TOP_K', '10'))
    )
    
    # Build expansion config
    expansion_config = QueryExpansionConfig(
        enabled=os.getenv('ENABLE_QUERY_EXPANSION', 'true').lower() == 'true',
        use_universal=os.getenv('USE_UNIVERSAL_EXPANSION', 'true').lower() == 'true'
    )
    
    # Build search config
    search_config = SearchConfig(
        provider=os.getenv('TOOL_SEARCH_PROVIDER', 'weaviate').lower(),
        default_min_score=float(os.getenv('DEFAULT_MIN_SCORE', '35.0')),
        reranker=reranker_config,
        expansion=expansion_config
    )
    
    # Build tool limits config
    tool_limits_config = ToolLimitsConfig(
        max_total_tools=int(os.getenv('MAX_TOTAL_TOOLS', '30')),
        max_mcp_tools=int(os.getenv('MAX_MCP_TOOLS', '20')),
        min_mcp_tools=int(os.getenv('MIN_MCP_TOOLS', '7')),
        default_drop_rate=float(os.getenv('DEFAULT_DROP_RATE', '0.1')),
        manage_only_mcp_tools=os.getenv('MANAGE_ONLY_MCP_TOOLS', 'false').lower() == 'true',
        exclude_letta_core_tools=os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'false').lower() == 'true',
        exclude_official_tools=os.getenv('EXCLUDE_OFFICIAL_TOOLS', 'false').lower() == 'true',
        never_detach_tools=never_detach_tools
    )
    
    # Build cache config
    cache_config = CacheConfig(
        cache_dir=os.getenv('CACHE_DIR', '/app/runtime_cache')
    )
    
    # Build webhook config
    webhook_config = WebhookConfig(
        matrix_bridge_url=os.getenv('MATRIX_BRIDGE_WEBHOOK_URL')
    )
    
    return AppConfig(
        letta=letta_config,
        weaviate=weaviate_config,
        search=search_config,
        tool_limits=tool_limits_config,
        cache=cache_config,
        webhook=webhook_config,
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        testing=os.getenv('TESTING', 'false').lower() == 'true'
    )


def create_test_config(**overrides) -> AppConfig:
    """
    Create a configuration suitable for testing.
    
    Args:
        **overrides: Override specific config values
        
    Returns:
        AppConfig configured for testing
    """
    config = AppConfig(
        letta=LettaConfig(
            url="http://localhost:8283/v1",
            use_sdk=False
        ),
        weaviate=WeaviateConfig(
            http_host="localhost",
            http_port=8080
        ),
        search=SearchConfig(
            provider="weaviate",
            reranker=RerankerConfig(enabled=False)
        ),
        tool_limits=ToolLimitsConfig(
            max_total_tools=30,
            max_mcp_tools=20
        ),
        cache=CacheConfig(
            cache_dir="/tmp/test_cache"
        ),
        testing=True
    )
    
    # Apply overrides (simple top-level only for now)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# ============================================================================
# Service Container
# ============================================================================

@dataclass
class ServiceContainer:
    """
    Container holding all application services.
    
    Provides dependency injection for route handlers and other components.
    Services are initialized once during app startup and shared across requests.
    """
    # Core services (will be set during app initialization)
    search_service: Any = None
    agent_service: Any = None
    tool_manager: Any = None
    
    # Clients
    weaviate_client: Any = None
    http_session: Any = None
    letta_sdk_client_func: Optional[Callable] = None
    
    # Configuration
    config: Optional[AppConfig] = None
    
    def is_configured(self) -> bool:
        """Check if essential services are configured."""
        return (
            self.search_service is not None or
            self.weaviate_client is not None
        )


# Module-level service container (for backward compatibility)
_services: Optional[ServiceContainer] = None


def get_services() -> ServiceContainer:
    """
    Get the global service container.
    
    Returns:
        ServiceContainer with configured services
        
    Raises:
        RuntimeError: If services not initialized
    """
    global _services
    if _services is None:
        _services = ServiceContainer()
    return _services


def set_services(services: ServiceContainer):
    """
    Set the global service container.
    
    Args:
        services: Configured ServiceContainer
    """
    global _services
    _services = services


def reset_services():
    """Reset the global service container (for testing)."""
    global _services
    _services = None
