#!/usr/bin/env python3
"""
MCP Server Wrapper for kicad-exporter
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mcp_servers.kicad_exporter import KiCadExporterMCP

if __name__ == "__main__":
    # This would be the entry point for Dedalus deployment
    server = KiCadExporterMCP()
    print(f"🚀 Starting kicad-exporter MCP server")
    # Dedalus would handle the MCP protocol here
