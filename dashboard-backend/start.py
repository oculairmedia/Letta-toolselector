#!/usr/bin/env python3
"""
LDTS Reranker Testing Dashboard Backend Startup Script
"""

import sys
import os
import uvicorn
import argparse
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description='LDTS Dashboard Backend Server')
    parser.add_argument('--host', default=settings.HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=settings.PORT, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=settings.RELOAD, 
                       help='Enable auto-reload on code changes')
    parser.add_argument('--workers', type=int, default=settings.WORKERS,
                       help='Number of worker processes')
    parser.add_argument('--log-level', default=settings.LOG_LEVEL.lower(),
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       help='Log level')
    parser.add_argument('--read-only', action='store_true',
                       help='Force read-only mode (overrides config)')
    parser.add_argument('--enable-dangerous', action='store_true',
                       help='Enable dangerous operations (overrides config)')
    
    args = parser.parse_args()
    
    # Override settings with command line arguments
    if args.read_only:
        settings.READ_ONLY_MODE = True
        print("⚠️  READ-ONLY MODE ENABLED - No write operations allowed")
    
    if args.enable_dangerous:
        settings.ENABLE_DANGEROUS_OPERATIONS = True
        print("⚠️  DANGEROUS OPERATIONS ENABLED - Use with caution")
    
    # Print startup information
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"📡 Server: {args.host}:{args.port}")
    print(f"🔒 Read-only mode: {settings.READ_ONLY_MODE}")
    print(f"⚡ Rate limiting: {settings.ENABLE_RATE_LIMITING}")
    print(f"🤖 Reranking: {settings.ENABLE_RERANKING}")
    print(f"🌐 Environment: {settings.ENVIRONMENT}")
    print(f"📚 API docs: http://{args.host}:{args.port}{settings.API_V1_STR}/docs")
    print()
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=settings.ENABLE_ACCESS_LOGS
    )

if __name__ == "__main__":
    main()