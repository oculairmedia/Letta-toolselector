"""
Application Factory

Creates and configures the Quart application with all dependencies.
Implements the factory pattern for testability and flexibility.

Usage:
    from app_factory import create_app
    
    # Create app with default config (from environment)
    app = create_app()
    
    # Create app with custom config (for testing)
    from app_config import create_test_config
    app = create_app(config=create_test_config())
    
    # Run the app
    app.run(host='0.0.0.0', port=8000)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from quart import Quart
import aiohttp

from app_config import (
    AppConfig,
    ServiceContainer,
    load_config_from_env,
    set_services,
    get_services
)

logger = logging.getLogger(__name__)


def create_app(config: Optional[AppConfig] = None) -> Quart:
    """
    Create and configure the Quart application.
    
    This is the main application factory. It:
    1. Creates the Quart app
    2. Loads configuration
    3. Initializes clients (Weaviate, HTTP session)
    4. Configures services (search, agent, tool_manager)
    5. Registers blueprints
    6. Sets up lifecycle hooks
    
    Args:
        config: Optional AppConfig. If not provided, loads from environment.
        
    Returns:
        Configured Quart application
    """
    # Create the app
    app = Quart(__name__)
    
    # Load configuration
    if config is None:
        config = load_config_from_env()
    
    # Store config in app
    app.config['APP_CONFIG'] = config
    
    # Create service container
    services = ServiceContainer(config=config)
    
    # Store services in app context
    app.services = services
    
    # Set global services (for backward compatibility)
    set_services(services)
    
    # Register lifecycle hooks
    _register_lifecycle_hooks(app, config, services)
    
    # Register blueprints (optional - can be enabled incrementally)
    if os.getenv('USE_BLUEPRINTS', 'false').lower() == 'true':
        _register_blueprints(app, services)
    
    logger.info(f"Application created with config: "
                f"search_provider={config.search.provider}, "
                f"reranking={'enabled' if config.search.reranker.enabled else 'disabled'}")
    
    return app


def _register_lifecycle_hooks(app: Quart, config: AppConfig, services: ServiceContainer):
    """Register before_serving and after_serving hooks."""
    
    @app.before_serving
    async def startup():
        """Initialize services on app startup."""
        logger.info("Application starting up...")
        
        # Initialize HTTP session
        services.http_session = aiohttp.ClientSession()
        logger.info("HTTP session created")
        
        # Initialize Weaviate client
        try:
            from weaviate_client_manager import get_client_manager
            client_manager = get_client_manager()
            health_status = client_manager.get_health_status()
            
            if health_status.get("status") in ["healthy", "warning"]:
                logger.info(f"Weaviate client manager initialized: {health_status['status']}")
            else:
                logger.warning(f"Weaviate client manager unhealthy: {health_status}")
                
            # Also try legacy client for compatibility
            from weaviate_tool_search_with_reranking import init_client as init_weaviate_client
            try:
                weaviate_client = init_weaviate_client()
                if weaviate_client and weaviate_client.is_connected():
                    services.weaviate_client = weaviate_client
                    logger.info("Weaviate client connected")
            except Exception as e:
                logger.warning(f"Legacy Weaviate client init failed: {e}")
                
        except Exception as e:
            logger.error(f"Weaviate initialization failed: {e}")
        
        # Initialize Letta SDK client function if enabled
        if config.letta.use_sdk:
            try:
                from letta_sdk_client import LettaSdkClient
                
                def get_letta_sdk_client():
                    return LettaSdkClient(
                        base_url=config.letta.url,
                        api_key=config.letta.api_key
                    )
                
                services.letta_sdk_client_func = get_letta_sdk_client
                logger.info("Letta SDK client configured")
            except ImportError as e:
                logger.warning(f"Letta SDK not available: {e}")
        
        # Configure services
        await _configure_services(config, services)
        
        # Ensure cache directory exists
        os.makedirs(config.cache.cache_dir, exist_ok=True)
        logger.info(f"Cache directory: {config.cache.cache_dir}")
        
        logger.info("Application startup complete")
    
    @app.after_serving
    async def shutdown():
        """Cleanup on app shutdown."""
        logger.info("Application shutting down...")
        
        # Close Weaviate client
        if services.weaviate_client:
            try:
                services.weaviate_client.close()
                logger.info("Weaviate client closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")
        
        # Close HTTP session
        if services.http_session:
            await services.http_session.close()
            logger.info("HTTP session closed")
        
        # Close Weaviate client manager
        try:
            from weaviate_client_manager import close_client_manager
            close_client_manager()
            logger.info("Weaviate client manager closed")
        except Exception as e:
            logger.error(f"Error closing client manager: {e}")
        
        logger.info("Application shutdown complete")


async def _configure_services(config: AppConfig, services: ServiceContainer):
    """Configure all services with their dependencies."""
    
    # Build headers for Letta API
    headers = {}
    if config.letta.api_key:
        headers["Authorization"] = f"Bearer {config.letta.api_key}"
    
    # Configure tool_manager
    try:
        import tool_manager
        from models import ToolLimitsConfig as ModelToolLimitsConfig
        
        tool_config = ModelToolLimitsConfig(
            max_total_tools=config.tool_limits.max_total_tools,
            max_mcp_tools=config.tool_limits.max_mcp_tools,
            min_mcp_tools=config.tool_limits.min_mcp_tools,
            manage_only_mcp_tools=config.tool_limits.manage_only_mcp_tools,
            never_detach_tools=config.tool_limits.never_detach_tools
        )
        
        tool_manager.configure(
            http_session=services.http_session,
            letta_url=config.letta.url,
            headers=headers,
            use_letta_sdk=config.letta.use_sdk,
            get_letta_sdk_client_func=services.letta_sdk_client_func,
            tool_config=tool_config
        )
        services.tool_manager = tool_manager
        logger.info("Tool manager configured")
    except Exception as e:
        logger.error(f"Failed to configure tool_manager: {e}")
    
    # Configure agent_service
    try:
        import agent_service
        
        agent_service.configure(
            http_session=services.http_session,
            letta_url=config.letta.url,
            headers=headers,
            use_letta_sdk=config.letta.use_sdk,
            get_letta_sdk_client_func=services.letta_sdk_client_func,
            letta_message_base_urls=config.letta.message_base_urls,
            matrix_bridge_webhook_url=config.webhook.matrix_bridge_url
        )
        services.agent_service = agent_service
        logger.info("Agent service configured")
    except Exception as e:
        logger.error(f"Failed to configure agent_service: {e}")
    
    # Configure search_service
    try:
        import search_service
        from search_service import (
            SearchConfig as SearchServiceConfig,
            RerankerConfig as SearchRerankerConfig,
            QueryExpansionConfig as SearchExpansionConfig
        )
        
        search_config = SearchServiceConfig(
            provider=config.search.provider,
            reranker=SearchRerankerConfig(
                enabled=config.search.reranker.enabled,
                provider=config.search.reranker.provider,
                url=config.search.reranker.url,
                model=config.search.reranker.model,
                timeout=config.search.reranker.timeout,
                initial_limit=config.search.reranker.initial_limit,
                top_k=config.search.reranker.top_k
            ),
            expansion=SearchExpansionConfig(
                enabled=config.search.expansion.enabled,
                use_universal=config.search.expansion.use_universal
            )
        )
        
        search_service.configure(
            weaviate_client=services.weaviate_client,
            letta_sdk_client_func=services.letta_sdk_client_func,
            config=search_config
        )
        services.search_service = search_service
        logger.info("Search service configured")
    except Exception as e:
        logger.error(f"Failed to configure search_service: {e}")


def _register_blueprints(app: Quart, services: ServiceContainer):
    """Register route blueprints."""
    try:
        from routes import tools_bp, health_bp
        from routes import tools as tools_routes, health as health_routes
        
        # Configure blueprints with service functions
        # (This wires up the delegation pattern we created in Phase 5)
        
        # For now, we don't wire up the handlers since routes are still in api_server.py
        # This will be done incrementally as routes are migrated
        
        # app.register_blueprint(tools_bp)
        # app.register_blueprint(health_bp)
        
        logger.info("Blueprints registered (disabled - routes still in api_server.py)")
    except ImportError as e:
        logger.warning(f"Could not import blueprints: {e}")


def get_app_services() -> ServiceContainer:
    """
    Get the service container.
    
    This is a convenience function that can be used in route handlers
    to access services.
    
    Returns:
        ServiceContainer with configured services
    """
    return get_services()
