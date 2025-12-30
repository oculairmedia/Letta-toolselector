#!/usr/bin/env python3
"""
Main Entry Point

This is the new entry point for the Tool Selector API server.
It uses the app_factory pattern for better testability and modularity.

Usage:
    # Via Hypercorn (production - used by Dockerfile):
    hypercorn main:app --bind 0.0.0.0:3001
    
    # Direct execution (development):
    python main.py
    
    # With blueprints enabled:
    USE_BLUEPRINTS=true python main.py
"""

import os
import asyncio
import logging

from hypercorn.config import Config
from hypercorn.asyncio import serve

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import and create app at module level for Hypercorn compatibility
# Hypercorn needs `main:app` to reference a module-level ASGI app
from app_factory import create_app

logger.info("Creating application via app_factory...")
app = create_app()


def main():
    """Main entry point for direct execution."""
    # Configure Hypercorn
    port = int(os.getenv('PORT', 3001))
    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    config.accesslog = "-"  # Log to stdout
    
    logger.info(f"Starting Hypercorn server on port {port}...")
    
    # Run the server
    asyncio.run(serve(app, config))


if __name__ == '__main__':
    main()
