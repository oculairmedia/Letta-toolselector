"""
Routes Package

This package contains Quart Blueprints for the API routes.
Each blueprint groups related routes together for better organization.

Blueprints:
- tools: Tool search, attach, detach, prune operations
- health: Health check endpoints
- search: Search parameter management
- webhook: Letta webhook event handlers

Usage in api_server.py:
    from routes import tools_bp, health_bp, search_bp, webhook_bp
    from routes import tools as tools_routes, health as health_routes, search as search_routes, webhook as webhook_routes
    
    # Configure blueprints with dependencies
    tools_routes.configure(search_func=search, ...)
    health_routes.configure(get_health_status_func=health_check)
    search_routes.configure(bm25_vector_override_service=service)
    webhook_routes.configure(webhook_secret=secret, weaviate_client=client)
    
    # Register blueprints
    app.register_blueprint(tools_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(webhook_bp)
"""

from .tools import tools_bp, configure as configure_tools
from .health import health_bp, configure as configure_health
from .search import search_bp, configure as configure_search
from .config import config_bp, configure as configure_config
from .ollama import ollama_bp, configure as configure_ollama
from .backup import backup_bp, configure as configure_backup
from .cost_control import cost_control_bp, configure as configure_cost_control
from .operations import maintenance_bp, logs_bp, environment_bp, configure as configure_operations
from .benchmark import benchmark_bp, configure as configure_benchmark
from .reranker import reranker_bp, configure as configure_reranker
from .enrichment import enrichment_bp, configure as configure_enrichment
from .pruning import pruning_bp
from .metrics import metrics_bp
from .webhook import webhook_bp, configure as configure_webhook

# Also expose the modules for configure() calls
from . import tools
from . import health
from . import search
from . import config
from . import ollama
from . import backup
from . import cost_control
from . import operations
from . import benchmark
from . import reranker
from . import enrichment
from . import pruning
from . import webhook

__all__ = [
    'tools_bp',
    'health_bp',
    'search_bp',
    'config_bp',
    'ollama_bp',
    'backup_bp',
    'cost_control_bp',
    'maintenance_bp',
    'logs_bp',
    'environment_bp',
    'benchmark_bp',
    'reranker_bp',
    'webhook_bp',
    'configure_tools',
    'configure_health',
    'configure_search',
    'configure_config',
    'configure_ollama',
    'configure_backup',
    'configure_cost_control',
    'configure_operations',
    'configure_benchmark',
    'configure_reranker',
    'configure_webhook',
    'tools',
    'health',
    'search',
    'config',
    'ollama',
    'backup',
    'cost_control',
    'operations',
    'benchmark',
    'reranker',
    'enrichment_bp',
    'configure_enrichment',
    'enrichment',
    'pruning_bp',
    'pruning',
    'metrics_bp',
    'webhook',
]
