"""
KiCad Exporter MCP Server

Exports placements to KiCad format.
"""

from typing import Dict, Any
from backend.geometry.placement import Placement
from backend.agents.exporter_agent import ExporterAgent


class KiCadExporterMCP:
    """MCP server for KiCad export."""
    
    def __init__(self):
        """Initialize KiCad exporter MCP server."""
        self.name = "kicad_exporter"
        self.exporter = ExporterAgent()
    
    async def export(self, placement_data: Dict) -> Dict[str, Any]:
        """
        Export placement to KiCad format.
        
        Args:
            placement_data: Placement dictionary
        
        Returns:
            {"kicad_content": str, "format": "kicad"}
        """
        placement = Placement.from_dict(placement_data)
        result = await self.exporter.process(placement, format="kicad")
        
        if result["success"]:
            return {
                "kicad_content": result["output"],
                "format": "kicad"
            }
        else:
            return {
                "error": result.get("error", "Export failed"),
                "format": "kicad"
            }
    
    def get_tool_definition(self) -> Dict:
        """Get MCP tool definition."""
        return {
            "name": "export_kicad",
            "description": "Export placement to KiCad format",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement_data": {"type": "object"}
                },
                "required": ["placement_data"]
            }
        }

