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

# Also expose the modules for configure() calls
from . import tools
from . import health
from . import search

__all__ = [
    'tools_bp',
    'health_bp',
    'search_bp',
    'configure_tools',
    'configure_health',
    'configure_search',
    'tools',
    'health',
    'search',
]
