"""
MCP Servers for Tool Access

Single unified MCP server with multiple tools for Neuro-Geometric Placer.
"""

# Import the unified MCP server
from .ngp_server import server as mcp_server

# Import individual tools for direct access (when not using Dedalus)
from .ngp_server import score_delta, generate_heatmap, export_kicad

__all__ = ["mcp_server", "score_delta", "generate_heatmap", "export_kicad"]

