"""
MCP Servers for Tool Access

Single unified MCP server with multiple tools for Neuro-Geometric Placer.
"""

# Import the unified MCP server
from .ngp_server import app as mcp_app

# Import individual tools for direct access (when not using Dedalus)
from .ngp_server import score_delta, generate_heatmap, export_kicad

__all__ = ["mcp_app", "score_delta", "generate_heatmap", "export_kicad"]

