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
        Export placement to KiCad format via MCP.

        Args:
            placement_data: Placement dictionary

        Returns:
            {"kicad_content": str, "format": "kicad", "metadata": dict}
        """
        placement = Placement.from_dict(placement_data)
        result = await self.exporter.process(placement, format="kicad")

        if result["success"]:
            return {
                "kicad_content": result["output"],
                "format": "kicad",
                "metadata": {
                    "component_count": len(placement.components),
                    "board_size": f"{placement.board.width}mm x {placement.board.height}mm",
                    "export_method": "dedalus_mcp"
                }
            }
        else:
            return {
                "error": result.get("error", "Export failed"),
                "format": "kicad",
                "metadata": {"status": "failed"}
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

