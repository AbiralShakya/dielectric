#!/usr/bin/env python3
"""
Dedalus Labs Entrypoint for Dielectric MCP Server

This is the entrypoint file that Dedalus will execute.
"""

import sys
import os
import asyncio

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run MCP server
try:
    from src.backend.mcp_servers.ngp_server import server
    
    if __name__ == "__main__":
        # Run the MCP server (openmcp uses serve())
        print("🚀 Starting Dielectric MCP Server")
        asyncio.run(server.serve())
except Exception as e:
    print(f"Error starting MCP server: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

