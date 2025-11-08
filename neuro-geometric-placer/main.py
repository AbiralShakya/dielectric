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

# Import our openmcp server
from backend.mcp_servers.ngp_server import server


if __name__ == "__main__":
    print("🚀 Starting Neuro-Geometric Placer MCP Server")
    print("Available tools:")
    print("  - score_delta: Compute placement score changes")
    print("  - generate_heatmap: Create thermal heatmaps")
    print("  - export_kicad: Export to KiCad format")

    # Run the openmcp server (defaults to HTTP on :8000)
    asyncio.run(server.serve())
