"""
Professional KiCad PCB Exporter

Generates complete, industry-standard KiCad .kicad_pcb files with:
- Proper footprints with pads
- Net definitions and routing hints
- Board outline (Edge.Cuts)
- Multi-layer support
- Design rules
"""

from typing import Dict, List, Any, Optional
import json


class KiCadExporter:
    """Exports PCB placements to KiCad format."""
    
    def __init__(self):
        """Initialize KiCad exporter."""
        self.footprint_templates = {
            "SOIC-8": self._soic8_footprint,
            "0805": self._r0805_footprint,
            "LED-5MM": self._led5mm_footprint,
            "BGA": self._bga_footprint,
            "QFN-16": self._qfn16_footprint,
            "INDUCTOR-10MM": self._inductor_footprint,
            "CAP-10MM": self._cap_footprint,
        }
    
    def export(self, placement_data: Dict[str, Any], include_nets: bool = True) -> str:
        """
        Export placement to KiCad PCB format.
        
        Args:
            placement_data: Placement dictionary with board, components, nets
            include_nets: Whether to include net definitions
        
        Returns:
            KiCad PCB file content
        """
        board = placement_data.get("board", {})
        components = placement_data.get("components", [])
        nets = placement_data.get("nets", [])
        
        board_width = board.get("width", 100)
        board_height = board.get("height", 100)
        clearance = board.get("clearance", 0.5)
        
        lines = [
            "(kicad_pcb (version 20221018) (generator \"dielectric\")",
            "",
            "  (general",
            "    (thickness 1.6)",
            "    (legacy_teardrops no)",
            "  )",
            "",
            "  (layers",
            '    (0 "F.Cu" signal)',
            '    (1 "In1.Cu" signal)',
            '    (2 "In2.Cu" signal)',
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
            "  (net_class \"Default\" \"\"",
            f"    (clearance {clearance})",
            "    (trace_width 0.25)",
            "    (via_dia 0.8)",
            "    (via_drill 0.4)",
            "    (uvia_dia 0.3)",
            "    (uvia_drill 0.1)",
            "  )",
            ""
        ]
        
        # Add board outline (Edge.Cuts)
        lines.extend([
            "  (gr_line",
            f"    (start 0 0) (end {board_width} 0)",
            '    (layer "Edge.Cuts")',
            "    (width 0.1) (tstamp 0)",
            "  )",
            "  (gr_line",
            f"    (start {board_width} 0) (end {board_width} {board_height})",
            '    (layer "Edge.Cuts")',
            "    (width 0.1) (tstamp 0)",
            "  )",
            "  (gr_line",
            f"    (start {board_width} {board_height}) (end 0 {board_height})",
            '    (layer "Edge.Cuts")',
            "    (width 0.1) (tstamp 0)",
            "  )",
            "  (gr_line",
            f"    (start 0 {board_height}) (end 0 0)",
            '    (layer "Edge.Cuts")',
            "    (width 0.1) (tstamp 0)",
            "  )",
            ""
        ])
        
        # Add nets (nets are optional)
        net_map = {}
        if include_nets and nets:
            net_id = 1
            for net in nets:
                if isinstance(net, dict):
                    net_name = net.get("name", f"Net{net_id}")
                    net_map[net_name] = net_id
                    lines.append(f'  (net {net_id} "{net_name}")')
                    net_id += 1
            if net_map:
                lines.append("")
        
        # Add components as footprints
        footprint_id = 1
        for comp in components:
            name = comp.get("name", f"U{footprint_id}")
            package = comp.get("package", "Unknown")
            x = comp.get("x", 0)
            y = comp.get("y", 0)
            angle = comp.get("angle", 0)
            pins = comp.get("pins", [])
            
            # Get footprint generator
            footprint_gen = self.footprint_templates.get(package, self._generic_footprint)
            footprint_lines = footprint_gen(name, x, y, angle, pins, comp, net_map if include_nets else {})
            lines.extend(footprint_lines)
            footprint_id += 1
        
        lines.append(")")
        return "\n".join(lines)
    
    def _soic8_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate SOIC-8 footprint."""
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "SOIC-8 - Dielectric optimized")',
            '    (tags "dielectric soic-8")',
            "    (property \"Reference\" \"U\" (at 0 -3.81 0)",
            '      (layer "F.SilkS")',
            "      (effects (font (size 1 1) (thickness 0.15)))",
            "    )",
            f'    (property "Value" "{name}" (at 0 3.81 0)',
            '      (layer "F.Fab")',
            "      (effects (font (size 1 1) (thickness 0.15)))",
            "    )",
            ""
        ]
        
        # Add pads for SOIC-8
        pad_positions = [
            (-3.81, -1.27), (-3.81, 1.27), (-1.27, 1.27), (-1.27, -1.27),
            (1.27, -1.27), (1.27, 1.27), (3.81, 1.27), (3.81, -1.27)
        ]
        
        for i, (px, py) in enumerate(pad_positions):
            pin_name = f"pin{i+1}" if i < len(pins) else f"pad{i+1}"
            net_name = None
            if i < len(pins):
                net_name = pins[i].get("net", "")
            
            net_num = net_map.get(net_name, 0) if net_name else 0
            
            lines.extend([
                f'    (pad "{pin_name}" smd roundrect (at {px:.3f} {py:.3f} {angle}) (size 0.6 1.55) (layers "F.Cu" "F.Paste" "F.Mask")',
                f"      (roundrect_rratio 0.25) (net {net_num} \"{net_name or ''}\")",
                "    )"
            ])
        
        lines.append("  )")
        lines.append("")
        return lines
    
    def _r0805_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate 0805 resistor/capacitor footprint."""
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "0805 - Dielectric optimized")',
            '    (tags "ai-optimized 0805")',
            "    (property \"Reference\" \"R\" (at 0 -1.27 0)",
            '      (layer "F.SilkS")',
            "      (effects (font (size 1 1) (thickness 0.15)))",
            "    )",
            f'    (property "Value" "{name}" (at 0 1.27 0)',
            '      (layer "F.Fab")',
            "      (effects (font (size 1 1) (thickness 0.15)))",
            "    )",
            ""
        ]
        
        # Two pads for 0805
        pad_positions = [(-0.95, 0), (0.95, 0)]
        for i, (px, py) in enumerate(pad_positions):
            pin_name = f"pin{i+1}" if i < len(pins) else f"pad{i+1}"
            net_name = None
            if i < len(pins) and isinstance(pins[i], dict):
                net_name = pins[i].get("net", "")
            
            net_num = net_map.get(net_name, 0) if net_name else 0
            
            lines.extend([
                f'    (pad "{pin_name}" smd roundrect (at {px:.3f} {py:.3f} {angle}) (size 0.8 1.3) (layers "F.Cu" "F.Paste" "F.Mask")',
                f"      (roundrect_rratio 0.25) (net {net_num} \"{net_name or ''}\")",
                "    )"
            ])
        
        lines.append("  )")
        lines.append("")
        return lines
    
    def _led5mm_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate LED 5mm footprint."""
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "LED-5MM - Dielectric optimized")',
            '    (tags "ai-optimized led")',
            ""
        ]
        
        # Anode and cathode pads
        pad_positions = [("anode", -2.5, 0), ("cathode", 2.5, 0)]
        for pin_name, px, py in pad_positions:
            net_name = None
            for pin in pins:
                if pin.get("name") == pin_name:
                    net_name = pin.get("net", "")
                    break
            
            net_num = net_map.get(net_name, 0) if net_name else 0
            
            lines.extend([
                f'    (pad "{pin_name}" thru_hole circle (at {px:.3f} {py:.3f} {angle}) (size 1.5 1.5) (drill 0.8)',
                f'      (layers "*.Cu" "*.Mask") (net {net_num} "{net_name or ""}")',
                "    )"
            ])
        
        lines.append("  )")
        lines.append("")
        return lines
    
    def _bga_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate BGA footprint."""
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "BGA - Dielectric optimized")',
            '    (tags "ai-optimized bga")',
            ""
        ]
        
        # BGA grid (4x4 for simplicity)
        grid_size = 4
        pitch = 1.0
        start = -(grid_size - 1) * pitch / 2
        
        ball_num = 1
        for i in range(grid_size):
            for j in range(grid_size):
                px = start + i * pitch
                py = start + j * pitch
                pin_name = f"A{ball_num}"
                
                net_name = None
                if ball_num <= len(pins):
                    net_name = pins[ball_num - 1].get("net", "")
                
                net_num = net_map.get(net_name, 0) if net_name else 0
                
                lines.extend([
                    f'    (pad "{pin_name}" smd round (at {px:.3f} {py:.3f} {angle}) (size 0.5 0.5) (layers "F.Cu" "F.Paste" "F.Mask")',
                    f"      (net {net_num} \"{net_name or ''}\")",
                    "    )"
                ])
                ball_num += 1
        
        lines.append("  )")
        lines.append("")
        return lines
    
    def _qfn16_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate QFN-16 footprint."""
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "QFN-16 - Dielectric optimized")',
            '    (tags "ai-optimized qfn")',
            ""
        ]
        
        # QFN-16: 4 pads per side
        pad_width = 0.3
        pad_height = 0.8
        body_size = 3.0
        pad_positions = []
        
        # Top side
        for i in range(4):
            pad_positions.append((body_size/2 - 0.5 - i*0.5, -body_size/2))
        # Right side
        for i in range(4):
            pad_positions.append((body_size/2, -body_size/2 + 0.5 + i*0.5))
        # Bottom side
        for i in range(4):
            pad_positions.append((body_size/2 - 0.5 - i*0.5, body_size/2))
        # Left side
        for i in range(4):
            pad_positions.append((-body_size/2, body_size/2 - 0.5 - i*0.5))
        
        for i, (px, py) in enumerate(pad_positions):
            pin_name = f"pin{i+1}" if i < len(pins) else f"pad{i+1}"
            net_name = None
            if i < len(pins) and isinstance(pins[i], dict):
                net_name = pins[i].get("net", "")
            
            net_num = net_map.get(net_name, 0) if net_name else 0
            
            lines.extend([
                f'    (pad "{pin_name}" smd roundrect (at {px:.3f} {py:.3f} {angle}) (size {pad_width} {pad_height}) (layers "F.Cu" "F.Paste" "F.Mask")',
                f"      (roundrect_rratio 0.25) (net {net_num} \"{net_name or ''}\")",
                "    )"
            ])
        
        lines.append("  )")
        lines.append("")
        return lines
    
    def _inductor_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate inductor footprint."""
        return self._generic_footprint(name, x, y, angle, pins, comp, net_map, "INDUCTOR")
    
    def _cap_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict) -> List[str]:
        """Generate capacitor footprint."""
        return self._generic_footprint(name, x, y, angle, pins, comp, net_map, "CAP")
    
    def _generic_footprint(self, name: str, x: float, y: float, angle: float, pins: List, comp: Dict, net_map: Dict, package_type: str = "GENERIC") -> List[str]:
        """Generate generic footprint."""
        width = comp.get("width", 5)
        height = comp.get("height", 5)
        
        lines = [
            f'  (footprint "{name}" (version 20221018) (generator "dielectric")',
            '    (layer "F.Cu")',
            f"    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "{package_type} - Dielectric optimized")',
            '    (tags "dielectric")',
            ""
        ]
        
        # Add pads based on pins
        if pins:
            for i, pin in enumerate(pins):
                pin_name = pin.get("name", f"pin{i+1}")
                x_offset = pin.get("x_offset", 0)
                y_offset = pin.get("y_offset", 0)
                net_name = pin.get("net", "")
                net_num = net_map.get(net_name, 0) if net_name else 0
                
                lines.extend([
                    f'    (pad "{pin_name}" smd roundrect (at {x_offset:.3f} {y_offset:.3f} {angle}) (size 0.8 0.8) (layers "F.Cu" "F.Paste" "F.Mask")',
                    f"      (roundrect_rratio 0.25) (net {net_num} \"{net_name or ''}\")",
                    "    )"
                ])
        else:
            # Default: two pads
            pad_size = min(width, height) * 0.3
            lines.extend([
                f'    (pad "1" smd roundrect (at {-width/4:.3f} 0 {angle}) (size {pad_size:.2f} {pad_size:.2f}) (layers "F.Cu" "F.Paste" "F.Mask")',
                '      (roundrect_rratio 0.25) (net 0 "")',
                "    )",
                f'    (pad "2" smd roundrect (at {width/4:.3f} 0 {angle}) (size {pad_size:.2f} {pad_size:.2f}) (layers "F.Cu" "F.Paste" "F.Mask")',
                '      (roundrect_rratio 0.25) (net 0 "")',
                "    )"
            ])
        
        lines.append("  )")
        lines.append("")
        return lines

