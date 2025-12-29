#!/usr/bin/env python3
"""
Main Entry Point

This is the new entry point for the Tool Selector API server.
It uses the app_factory pattern for better testability and modularity.

Usage:
    python main.py              # Run with default config from environment
    USE_BLUEPRINTS=true python main.py  # Use factory-based blueprint registration
    
For backward compatibility, api_server.py can still be run directly.
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


def main():
    """Main entry point."""
    # Import here to ensure logging is configured first
    from app_factory import create_app
    
    # Create the application
    logger.info("Creating application via app_factory...")
    app = create_app()
    
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
