"""
Routes Package

This package contains Quart Blueprints for the API routes.
Each blueprint groups related routes together for better organization.

Blueprints:
- tools: Tool search, attach, detach, prune operations
- health: Health check endpoints

Usage in api_server.py:
    from routes import tools_bp, health_bp
    from routes import tools as tools_routes, health as health_routes
    
    # Configure blueprints with dependencies
    tools_routes.configure(search_func=search, ...)
    health_routes.configure(get_health_status_func=health_check)
    
    # Register blueprints
    app.register_blueprint(tools_bp)
    app.register_blueprint(health_bp)
"""

from .tools import tools_bp, configure as configure_tools
from .health import health_bp, configure as configure_health

# Also expose the modules for configure() calls
from . import tools
from . import health

__all__ = [
    'tools_bp',
    'health_bp',
    'configure_tools',
    'configure_health',
    'tools',
    'health',
]
