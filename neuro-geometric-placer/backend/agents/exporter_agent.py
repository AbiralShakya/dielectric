"""
Exporter Agent

Converts placement to KiCad/Altium formats.
"""

from typing import Dict
from backend.geometry.placement import Placement


class ExporterAgent:
    """Agent for exporting placements to CAD formats."""
    
    def __init__(self):
        """Initialize exporter agent."""
        self.name = "ExporterAgent"
    
    async def process(self, placement: Placement, format: str = "kicad") -> Dict:
        """
        Export placement to specified format.
        
        Args:
            placement: Placement to export
            format: Export format ("kicad" or "json")
        
        Returns:
            {
                "success": bool,
                "output": str,
                "format": str
            }
        """
        try:
            if format == "kicad":
                output = self._export_kicad(placement)
            elif format == "json":
                output = placement.to_dict()
            else:
                raise ValueError(f"Unknown format: {format}")
            
            return {
                "success": True,
                "output": output,
                "format": format,
                "agent": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _export_kicad(self, placement: Placement) -> str:
        """Export to KiCad format."""
        lines = [
            "(kicad_pcb (version 20211014) (generator ngp)",
            "  (general",
            f"    (thickness {placement.board.width})",
            "  )"
        ]
        
        # Add components (footprints)
        for comp in placement.components.values():
            lines.append(f"  (footprint \"{comp.package}\"")
            lines.append(f"    (at {comp.x:.3f} {comp.y:.3f} {comp.angle})")
            lines.append(f"    (layer F.Cu)")
            lines.append(f"    (des \"{comp.name}\")")
            lines.append("  )")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "export_placement",
            "description": "Export placement to CAD format",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"},
                    "format": {
                        "type": "string",
                        "enum": ["kicad", "json"],
                        "default": "kicad"
                    }
                },
                "required": ["placement"]
            }
        }

