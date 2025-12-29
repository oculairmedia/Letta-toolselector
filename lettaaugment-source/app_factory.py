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
    
    # Register blueprints is now done in startup hook after services are configured
    # The USE_BLUEPRINTS flag controls whether to use factory-based registration
    app.config['USE_FACTORY_BLUEPRINTS'] = os.getenv('USE_BLUEPRINTS', 'false').lower() == 'true'
    
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
        
        # Register blueprints if factory-based registration is enabled
        if app.config.get('USE_FACTORY_BLUEPRINTS', False):
            await _register_blueprints(app, config, services)
        
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


async def _register_blueprints(app: Quart, config: AppConfig, services: ServiceContainer):
    """
    Register all route blueprints with their dependencies.
    
    This mirrors the blueprint registration logic from api_server.py startup,
    allowing the app factory to create a fully configured application.
    """
    # Import shared dependencies
    from weaviate_tool_search_with_reranking import search_tools, init_client as init_weaviate_client
    from bm25_vector_overrides import bm25_vector_override_service
    from models import is_letta_core_tool as models_is_letta_core_tool
    from audit_logging import emit_batch_event, emit_pruning_event, AuditAction, AuditSource
    from services.tool_cache import get_tool_cache_service
    from services.tool_search import configure_search_service
    
    # Import utility functions that blueprints need
    # TODO: These should be extracted to dedicated service modules
    from api_server import (
        read_tool_cache, read_mcp_servers_cache, process_matching_tool,
        log_config_change, perform_configuration_validation, test_service_connection,
        test_letta_connection, test_weaviate_connection as api_test_weaviate,
        get_tool_count_from_cache, get_cache_size, get_last_sync_time,
        get_weaviate_index_status, get_memory_usage, get_disk_usage, get_cpu_info,
        get_log_file_size, get_recent_error_count, get_recent_warning_count,
        perform_cleanup_operation, perform_optimization,
        get_log_entries, perform_log_analysis, get_error_log_entries,
        clear_log_files, export_log_data, start_time
    )
    from routes.config import test_ollama_connection
    
    cache_dir = config.cache.cache_dir
    
    # Configure services layer
    configure_search_service(search_tools)
    tool_cache_service = get_tool_cache_service(cache_dir)
    
    # 1. Search routes
    try:
        from routes import search as search_routes, search_bp
        search_routes.configure(bm25_vector_override_service=bm25_vector_override_service)
        app.register_blueprint(search_bp)
        logger.info("Search routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register search blueprint: {e}")
    
    # 2. Config routes
    try:
        from routes import config as config_routes, config_bp
        config_routes.configure(
            http_session=services.http_session,
            log_config_change=log_config_change,
            tool_cache_service=tool_cache_service
        )
        app.register_blueprint(config_bp)
        logger.info("Config routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register config blueprint: {e}")
    
    # 3. Ollama routes
    try:
        from routes import ollama as ollama_routes, ollama_bp
        ollama_routes.configure()
        app.register_blueprint(ollama_bp)
        logger.info("Ollama routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register ollama blueprint: {e}")
    
    # 4. Backup routes
    try:
        from routes import backup as backup_routes, backup_bp
        backup_routes.configure(
            cache_dir=cache_dir,
            log_config_change=log_config_change,
            perform_configuration_validation=perform_configuration_validation,
            test_service_connection=test_service_connection
        )
        app.register_blueprint(backup_bp)
        logger.info("Backup routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register backup blueprint: {e}")
    
    # 5. Cost control routes
    try:
        from routes import cost_control as cost_control_routes, cost_control_bp
        try:
            from cost_control_manager import get_cost_manager, CostCategory, BudgetPeriod, AlertLevel
            cost_control_routes.configure(
                get_cost_manager=get_cost_manager,
                CostCategory=CostCategory,
                BudgetPeriod=BudgetPeriod,
                AlertLevel=AlertLevel
            )
        except ImportError:
            cost_control_routes.configure()
        app.register_blueprint(cost_control_bp)
        logger.info("Cost control routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register cost_control blueprint: {e}")
    
    # 6. Operations routes (maintenance, logs, environment)
    try:
        from routes import operations as operations_routes
        from routes.operations import maintenance_bp, logs_bp, environment_bp
        operations_routes.configure(
            start_time=start_time,
            cache_dir=cache_dir,
            log_config_change=log_config_change,
            test_weaviate_connection=api_test_weaviate,
            test_ollama_connection=test_ollama_connection,
            test_letta_connection=test_letta_connection,
            get_tool_count_from_cache=get_tool_count_from_cache,
            get_cache_size=get_cache_size,
            get_last_sync_time=get_last_sync_time,
            get_weaviate_index_status=get_weaviate_index_status,
            get_memory_usage=get_memory_usage,
            get_disk_usage=get_disk_usage,
            get_cpu_info=get_cpu_info,
            get_log_file_size=get_log_file_size,
            get_recent_error_count=get_recent_error_count,
            get_recent_warning_count=get_recent_warning_count,
            perform_cleanup_operation=perform_cleanup_operation,
            perform_optimization=perform_optimization,
            get_log_entries=get_log_entries,
            perform_log_analysis=perform_log_analysis,
            get_error_log_entries=get_error_log_entries,
            clear_log_files=clear_log_files,
            export_log_data=export_log_data
        )
        app.register_blueprint(maintenance_bp)
        app.register_blueprint(logs_bp)
        app.register_blueprint(environment_bp)
        logger.info("Operations routes blueprints registered")
    except Exception as e:
        logger.error(f"Failed to register operations blueprints: {e}")
    
    # 7. Benchmark routes
    try:
        from routes import benchmark as benchmark_routes
        from routes.benchmark import benchmark_bp
        benchmark_routes.configure(cache_dir=cache_dir, search_tools=search_tools)
        app.register_blueprint(benchmark_bp)
        logger.info("Benchmark routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register benchmark blueprint: {e}")
    
    # 8. Reranker routes
    try:
        from routes import reranker as reranker_routes
        from routes.reranker import reranker_bp
        reranker_routes.configure(cache_dir=cache_dir)
        app.register_blueprint(reranker_bp)
        logger.info("Reranker routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register reranker blueprint: {e}")
    
    # 9. Tools routes
    try:
        from routes import tools as tools_routes
        from routes.tools import tools_bp
        tools_routes.configure(
            manage_only_mcp_tools=config.tool_limits.manage_only_mcp_tools,
            default_min_score=config.search.default_min_score,
            agent_service=services.agent_service,
            tool_manager=services.tool_manager,
            search_tools_func=search_tools,
            read_tool_cache_func=read_tool_cache,
            read_mcp_servers_cache_func=read_mcp_servers_cache,
            process_matching_tool_func=process_matching_tool,
            init_weaviate_client_func=init_weaviate_client,
            get_weaviate_client_func=lambda: services.weaviate_client,
            is_letta_core_tool_func=models_is_letta_core_tool,
            emit_batch_event_func=emit_batch_event,
            emit_pruning_event_func=emit_pruning_event,
            audit_action_class=AuditAction,
            audit_source_class=AuditSource
        )
        app.register_blueprint(tools_bp)
        logger.info("Tools routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register tools blueprint: {e}")
    
    # 10. Evaluation routes
    try:
        from routes import evaluation as evaluation_routes
        from routes.evaluation import evaluation_bp
        evaluation_routes.configure(
            search_tools_func=search_tools,
            bm25_vector_override_service=bm25_vector_override_service,
            cache_dir=cache_dir
        )
        app.register_blueprint(evaluation_bp)
        logger.info("Evaluation routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register evaluation blueprint: {e}")
    
    # 11. Safety routes
    try:
        from routes import safety as safety_routes
        from routes.safety import safety_bp
        safety_routes.configure()
        app.register_blueprint(safety_bp)
        logger.info("Safety routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register safety blueprint: {e}")
    
    # 12. Models routes
    try:
        from routes import models as models_routes
        from routes.models import models_bp
        models_routes.configure(
            search_tools_func=search_tools,
            bm25_vector_override_service=bm25_vector_override_service
        )
        app.register_blueprint(models_bp)
        logger.info("Models routes blueprint registered")
    except Exception as e:
        logger.error(f"Failed to register models blueprint: {e}")
    
    # Load initial cache
    await read_tool_cache(force_reload=True)
    await read_mcp_servers_cache()
    logger.info("Initial cache loaded")


def get_app_services() -> ServiceContainer:
    """
    Get the service container.
    
    This is a convenience function that can be used in route handlers
    to access services.
    
    Returns:
        ServiceContainer with configured services
    """
    return get_services()
