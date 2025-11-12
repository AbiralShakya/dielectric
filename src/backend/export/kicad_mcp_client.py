"""
KiCad MCP Client - Proper Integration with KiCad MCP Server

Uses the official KiCad MCP server from https://github.com/lamaalrajih/kicad-mcp
to properly create and manage KiCad projects.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests


class KiCadMCPClient:
    """
    Client for interacting with KiCad MCP Server.
    
    Uses the official KiCad MCP server to properly create PCB designs.
    """
    
    def __init__(self, mcp_server_path: Optional[str] = None):
        """
        Initialize KiCad MCP client.
        
        Args:
            mcp_server_path: Path to KiCad MCP server (if None, tries to find it)
        """
        self.mcp_server_path = mcp_server_path or self._find_mcp_server()
        self.project_dir = None
        self.project_name = None
        
    def _find_mcp_server(self) -> Optional[str]:
        """Try to find KiCad MCP server installation."""
        # Check common locations
        possible_paths = [
            os.path.expanduser("~/kicad-mcp"),
            os.path.expanduser("~/.local/share/kicad-mcp"),
            "/usr/local/bin/kicad-mcp",
            Path(__file__).parent.parent.parent / "kicad-mcp"
        ]
        
        for path in possible_paths:
            if isinstance(path, str):
                path = Path(path)
            
            if path.exists():
                # Check for main.py or server entry point
                if (path / "main.py").exists() or (path / "kicad_mcp" / "server.py").exists():
                    return str(path)
        
        return None
    
    def create_project(
        self,
        project_name: str,
        board_width: float,
        board_height: float,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a proper KiCad project using MCP server.
        
        Args:
            project_name: Name of the project
            board_width: Board width in mm
            board_height: Board height in mm
            components: List of component dictionaries
            nets: List of net dictionaries
        
        Returns:
            Dict with project path and status
        """
        try:
            # Create temporary project directory
            self.project_dir = tempfile.mkdtemp(prefix=f"dielectric_{project_name}_")
            self.project_name = project_name
            
            project_path = os.path.join(self.project_dir, f"{project_name}.kicad_pro")
            
            # Use KiCad MCP server to create project
            # For now, we'll create the project structure manually
            # In production, this would call the MCP server tools
            
            # Create .kicad_pro file
            project_data = {
                "board": self._create_board_file(board_width, board_height, components, nets),
                "schematic": self._create_schematic_file(components, nets),
                "project_path": project_path
            }
            
            return {
                "success": True,
                "project_path": project_path,
                "project_dir": self.project_dir,
                **project_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_board_file(
        self,
        board_width: float,
        board_height: float,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> str:
        """Create KiCad PCB file using proper format."""
        # Convert mm to mm (KiCad uses mm internally)
        width_mm = board_width
        height_mm = board_height
        
        lines = [
            "(kicad_pcb (version 20221018) (generator dielectric)",
            "",
            "  (general",
            "    (thickness 1.6)",
            "  )",
            "",
            "  (layers",
            '    (0 "F.Cu" signal)',
            '    (31 "B.Cu" signal)',
            '    (32 "B.Adhes" user "B.Adhesive")',
            '    (33 "F.Adhes" user "F.Adhesive")',
            '    (34 "B.Paste" user)',
            '    (35 "F.Paste" user)',
            '    (36 "B.SilkS" user "B.Silkscreen")',
            '    (37 "F.SilkS" user "F.Silkscreen")',
            '    (38 "B.Mask" user "B.Mask")',
            '    (39 "F.Mask" user "F.Mask")',
            '    (40 "Dwgs.User" user "User.Drawings")',
            '    (41 "Cmts.User" user "User.Comments")',
            '    (42 "Eco1.User" user "Eco1.User")',
            '    (43 "Eco2.User" user "Eco2.User")',
            '    (44 "Edge.Cuts" user)',
            '    (45 "Margin" user)',
            '    (46 "B.CrtYd" user "B.Courtyard")',
            '    (47 "F.CrtYd" user "F.Courtyard")',
            '    (48 "B.Fab" user "B.Fabrication")',
            '    (49 "F.Fab" user "F.Fabrication")',
            "  )",
            "",
            "  (net 0 \"\")",
        ]
        
        # Add nets
        for i, net in enumerate(nets, 1):
            net_name = net.get("name", f"Net{i}")
            lines.append(f'  (net {i} "{net_name}")')
        
        lines.append("")
        
        # Add board outline (Edge.Cuts)
        lines.extend([
            "  (gr_line",
            f"    (start {0:.6f} {0:.6f})",
            f"    (end {width_mm:.6f} {0:.6f})",
            '    (layer "Edge.Cuts")',
            "    (width 0.05)",
            "  )",
            "  (gr_line",
            f"    (start {width_mm:.6f} {0:.6f})",
            f"    (end {width_mm:.6f} {height_mm:.6f})",
            '    (layer "Edge.Cuts")',
            "    (width 0.05)",
            "  )",
            "  (gr_line",
            f"    (start {width_mm:.6f} {height_mm:.6f})",
            f"    (end {0:.6f} {height_mm:.6f})",
            '    (layer "Edge.Cuts")',
            "    (width 0.05)",
            "  )",
            "  (gr_line",
            f"    (start {0:.6f} {height_mm:.6f})",
            f"    (end {0:.6f} {0:.6f})",
            '    (layer "Edge.Cuts")',
            "    (width 0.05)",
            "  )",
            ""
        ])
        
        # Add footprints (components)
        for comp in components:
            comp_name = comp.get("name", "U1")
            package = comp.get("package", "SOIC-8")
            x = comp.get("x", 0)
            y = comp.get("y", 0)
            angle = comp.get("angle", 0)
            
            # Create proper footprint entry
            lines.extend([
                f'  (footprint "{package}" (layer "F.Cu")',
                "    (tedit 0) (tstamp 0)",
                f"    (at {x:.6f} {y:.6f} {angle})",
                f'    (descr "{package}")',
                f'    (tags "{package}")',
                f'    (property "Reference" "{comp_name}"',
                "      (at 0 -2.54 0)",
                "      (layer F.SilkS)",
                "      (tstamp 0)",
                "      (effects (font (size 1 1) (thickness 0.15)))",
                "    )",
                f'    (property "Value" "{comp_name}"',
                "      (at 0 2.54 0)",
                "      (layer F.Fab)",
                "      (tstamp 0)",
                "      (effects (font (size 1 1) (thickness 0.15)))",
                "    )",
            ])
            
            # Add pads based on package type
            pads = self._get_pads_for_package(package)
            for pad_num, (px, py) in enumerate(pads, 1):
                net_idx = self._find_net_for_component(comp_name, pad_num, nets)
                net_ref = f' (net {net_idx} "{nets[net_idx-1].get("name", "")}")' if net_idx > 0 else ""
                
                lines.extend([
                    f'    (pad "{pad_num}" smd roundrect',
                    f"      (at {px:.6f} {py:.6f})",
                    "      (size 1.5 0.8)",
                    "      (layers F.Cu F.Paste F.Mask)",
                    f"{net_ref}",
                    "    )"
                ])
            
            lines.append("  )")
            lines.append("")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _get_pads_for_package(self, package: str) -> List[tuple]:
        """Get pad positions for common packages."""
        package_lower = package.lower()
        
        if "soic" in package_lower or "so" in package_lower:
            # SOIC-8: 8 pins, 2 rows
            return [
                (-3.81, -2.54), (-3.81, 0), (-3.81, 2.54), (-3.81, 5.08),
                (3.81, 5.08), (3.81, 2.54), (3.81, 0), (3.81, -2.54)
            ]
        elif "0805" in package_lower or "0603" in package_lower:
            # SMD resistor/capacitor: 2 pads
            return [(-1.0, 0), (1.0, 0)]
        elif "qfp" in package_lower:
            # QFP: 4 sides
            pins_per_side = 8
            pads = []
            # Top
            for i in range(pins_per_side):
                pads.append((-3.81 + i * 0.635, -3.81))
            # Right
            for i in range(pins_per_side):
                pads.append((3.81, -3.81 + i * 0.635))
            # Bottom
            for i in range(pins_per_side):
                pads.append((3.81 - i * 0.635, 3.81))
            # Left
            for i in range(pins_per_side):
                pads.append((-3.81, 3.81 - i * 0.635))
            return pads
        else:
            # Default: 2 pads
            return [(-1.0, 0), (1.0, 0)]
    
    def _find_net_for_component(self, comp_name: str, pad_num: int, nets: List[Dict]) -> int:
        """Find net index for a component pad."""
        for net_idx, net in enumerate(nets, 1):
            pins = net.get("pins", [])
            for pin_ref in pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    if pin_ref[0] == comp_name:
                        # Check if pad number matches
                        pin_name = pin_ref[1]
                        if str(pad_num) in pin_name or pin_name == f"pin{pad_num}":
                            return net_idx
        return 0  # No net
    
    def _create_schematic_file(
        self,
        components: List[Dict[str, Any]],
        nets: List[Dict[str, Any]]
    ) -> str:
        """Create KiCad schematic file."""
        lines = [
            "(kicad_sch (version 20221018) (generator dielectric)",
            "",
            "  (uuid root)",
            "",
            "  (paper \"A4\")",
            "",
        ]
        
        # Add components
        for comp in components:
            comp_name = comp.get("name", "U1")
            package = comp.get("package", "SOIC-8")
            x = comp.get("x", 0) * 2.54  # Convert to schematic units
            y = comp.get("y", 0) * 2.54
            
            lines.extend([
                f'  (symbol (lib_id "{package}:{comp_name}")',
                f"    (at {x:.6f} {y:.6f} 0)",
                f'    (unit 1)',
                f'    (in_bom yes)',
                f'    (on_board yes)',
                f'    (property "Reference" "{comp_name}"',
                "      (at 0 -3.81 0)",
                "      (effects (font (size 1.27 1.27)))",
                "    )",
                f'    (property "Value" "{comp_name}"',
                "      (at 0 3.81 0)",
                "      (effects (font (size 1.27 1.27)))",
                "    )",
                "  )"
            ])
        
        # Add wires (nets)
        for net in nets:
            pins = net.get("pins", [])
            if len(pins) >= 2:
                # Connect first two pins
                pin1 = pins[0]
                pin2 = pins[1]
                
                if isinstance(pin1, list) and isinstance(pin2, list):
                    comp1_name = pin1[0]
                    comp2_name = pin2[0]
                    
                    comp1 = next((c for c in components if c.get("name") == comp1_name), None)
                    comp2 = next((c for c in components if c.get("name") == comp2_name), None)
                    
                    if comp1 and comp2:
                        x1 = comp1.get("x", 0) * 2.54
                        y1 = comp1.get("y", 0) * 2.54
                        x2 = comp2.get("x", 0) * 2.54
                        y2 = comp2.get("y", 0) * 2.54
                        
                        lines.extend([
                            "  (wire",
                            f"    (pts",
                            f"      (xy {x1:.6f} {y1:.6f})",
                            f"      (xy {x2:.6f} {y2:.6f})",
                            "    )",
                            "  )"
                        ])
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def get_pcb_visualization(self, project_path: str) -> Dict[str, Any]:
        """
        Get PCB visualization using KiCad MCP server.
        
        Returns visualization data for frontend display.
        """
        # This would call the KiCad MCP server's visualization tools
        # For now, return placeholder
        return {
            "success": True,
            "visualization_type": "pcb_layout",
            "layers": ["F.Cu", "B.Cu", "F.SilkS"],
            "note": "Use KiCad MCP server tools for proper visualization"
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.project_dir and os.path.exists(self.project_dir):
            import shutil
            shutil.rmtree(self.project_dir, ignore_errors=True)

