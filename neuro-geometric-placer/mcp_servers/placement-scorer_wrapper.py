#!/usr/bin/env python3
"""
MCP Server Wrapper for placement-scorer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mcp_servers.placement_scorer import PlacementScorerMCP

if __name__ == "__main__":
    # This would be the entry point for Dedalus deployment
    server = PlacementScorerMCP()
    print(f"🚀 Starting placement-scorer MCP server")
    # Dedalus would handle the MCP protocol here
