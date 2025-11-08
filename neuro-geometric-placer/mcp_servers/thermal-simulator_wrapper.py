#!/usr/bin/env python3
"""
MCP Server Wrapper for thermal-simulator
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.mcp_servers.thermal_simulator import ThermalSimulatorMCP

if __name__ == "__main__":
    # This would be the entry point for Dedalus deployment
    server = ThermalSimulatorMCP()
    print(f"🚀 Starting thermal-simulator MCP server")
    # Dedalus would handle the MCP protocol here
