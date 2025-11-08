#!/usr/bin/env python3
"""
Main entry point for Neuro-Geometric Placer MCP Server

This is the main entry point that Dedalus Labs will use to run the MCP server.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our openmcp server main function
from backend.mcp_servers.ngp_server import main as server_main


if __name__ == "__main__":
    # Run the openmcp server
    asyncio.run(server_main())
