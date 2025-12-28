"""
Routes Package

This package contains Quart Blueprints for the API routes.
Each blueprint groups related routes together for better organization.

Blueprints:
- tools: Tool search, attach, detach, prune operations
- health: Health check endpoints
- search: Search parameter management

Usage in api_server.py:
    from routes import tools_bp, health_bp, search_bp
    from routes import tools as tools_routes, health as health_routes, search as search_routes
    
    # Configure blueprints with dependencies
    tools_routes.configure(search_func=search, ...)
    health_routes.configure(get_health_status_func=health_check)
    search_routes.configure(bm25_vector_override_service=service)
    
    # Register blueprints
    app.register_blueprint(tools_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(search_bp)
"""

from .tools import tools_bp, configure as configure_tools
from .health import health_bp, configure as configure_health
from .search import search_bp, configure as configure_search
from .config import config_bp, configure as configure_config
from .ollama import ollama_bp, configure as configure_ollama
from .backup import backup_bp, configure as configure_backup
from .cost_control import cost_control_bp, configure as configure_cost_control
from .operations import maintenance_bp, logs_bp, environment_bp, configure as configure_operations

# Also expose the modules for configure() calls
from . import tools
from . import health
from . import search
from . import config
from . import ollama
from . import backup
from . import cost_control
from . import operations

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
    'configure_tools',
    'configure_health',
    'configure_search',
    'configure_config',
    'configure_ollama',
    'configure_backup',
    'configure_cost_control',
    'configure_operations',
    'tools',
    'health',
    'search',
    'config',
    'ollama',
    'backup',
    'cost_control',
    'operations',
]
