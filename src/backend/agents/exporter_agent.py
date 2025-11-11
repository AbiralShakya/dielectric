"""
Exporter Agent

Converts placement to KiCad/Altium formats using KiCad MCP Server.
"""

from typing import Dict, Optional
try:
    from backend.geometry.placement import Placement
    try:
        from backend.export.kicad_mcp_exporter import KiCadMCPExporter, KICAD_AVAILABLE
    except ImportError:
        KiCadMCPExporter = None
        KICAD_AVAILABLE = False
except ImportError:
    from src.backend.geometry.placement import Placement
    try:
        from src.backend.export.kicad_mcp_exporter import KiCadMCPExporter, KICAD_AVAILABLE
    except ImportError:
        KiCadMCPExporter = None
        KICAD_AVAILABLE = False


class ExporterAgent:
    """Agent for exporting placements to CAD formats using KiCad MCP Server."""
    
    def __init__(self):
        """Initialize exporter agent."""
        self.name = "ExporterAgent"
        self.kicad_exporter = None
        
        # Try to initialize KiCad MCP exporter
        if KICAD_AVAILABLE and KiCadMCPExporter:
            try:
                self.kicad_exporter = KiCadMCPExporter()
                print("✅ ExporterAgent: KiCad MCP exporter initialized")
            except Exception as e:
                print(f"⚠️  ExporterAgent: KiCad MCP initialization failed, using fallback: {e}")
                self.kicad_exporter = None
        else:
            self.kicad_exporter = None
            print("⚠️  ExporterAgent: KiCad not available, using fallback")
    
    async def process(self, placement: Placement, format: str = "kicad") -> Dict:
        """
        Export placement to specified format using KiCad MCP Server.
        
        Args:
            placement: Placement to export
            format: Export format ("kicad" or "json")
        
        Returns:
            {
                "success": bool,
                "output": str,
                "format": str,
                "method": str  # "kicad_mcp" or "fallback"
            }
        """
        try:
            print(f"📤 ExporterAgent: Exporting to {format} format...")
            
            if format == "kicad":
                if self.kicad_exporter:
                    print("   Using KiCad MCP exporter...")
                    placement_data = placement.to_dict()
                    output_path = self.kicad_exporter.export(placement_data)
                    output = self.kicad_exporter.get_file_content(output_path)
                    self.kicad_exporter.cleanup()
                    print("   ✅ KiCad MCP export successful")
                    return {
                        "success": True,
                        "output": output,
                        "format": format,
                        "method": "kicad_mcp",
                        "agent": self.name
                    }
                else:
                    print("   Using fallback exporter...")
                    output = self._export_kicad_fallback(placement)
                    return {
                        "success": True,
                        "output": output,
                        "format": format,
                        "method": "fallback",
                        "agent": self.name
                    }
            elif format == "json":
                output = placement.to_dict()
                return {
                    "success": True,
                    "output": output,
                    "format": format,
                    "method": "native",
                    "agent": self.name
                }
            else:
                raise ValueError(f"Unknown format: {format}")
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"   ❌ Export failed: {error_msg}")
            print(f"   Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "agent": self.name
            }
    
    def _export_kicad_fallback(self, placement: Placement) -> str:
        """Fallback KiCad export when MCP server is not available."""
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

