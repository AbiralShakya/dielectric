"""
KiCAD MCP Exporter - Uses KiCAD Python API via KiCAD-MCP-Server

This exporter uses the KiCAD Python API (pcbnew) directly to create
professional PCB designs instead of manually generating .kicad_pcb files.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Add KiCAD-MCP-Server Python path
KICAD_MCP_PATH = Path(__file__).parent.parent.parent.parent / "kicad-mcp-server" / "python"
if KICAD_MCP_PATH.exists():
    sys.path.insert(0, str(KICAD_MCP_PATH))

try:
    # Try to import pcbnew - requires KiCAD installation
    import pcbnew
    KICAD_AVAILABLE = True
except ImportError:
    KICAD_AVAILABLE = False
    # Don't print warning here - let the caller handle it gracefully


class KiCadMCPExporter:
    """Exports PCB placements using KiCAD Python API."""
    
    def __init__(self):
        """Initialize KiCAD exporter."""
        if not KICAD_AVAILABLE:
            raise ImportError(
                "KiCAD Python API (pcbnew) not available. "
                "Please install KiCAD with Python support:\n"
                "  - Linux: sudo apt-get install kicad kicad-python3\n"
                "  - macOS: Install KiCAD.app from kicad.org\n"
                "  - Windows: Install KiCAD with Python support"
            )
        
        self.board = None
        self.project_path = None
        self.temp_dir = None
    
    def export(self, placement_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export placement to KiCad PCB format using KiCAD API.
        
        Args:
            placement_data: Placement dictionary with board, components, nets
            output_path: Optional path to save .kicad_pcb file
        
        Returns:
            Path to generated .kicad_pcb file
        """
        try:
            # Create temporary project directory
            self.temp_dir = tempfile.mkdtemp(prefix="dielectric_kicad_")
            project_name = "dielectric_design"
            self.project_path = os.path.join(self.temp_dir, project_name)
            
            # Create new board
            self.board = pcbnew.CreateEmptyBoard()
            
            # Set board properties
            board_info = placement_data.get("board", {})
            self._setup_board(board_info)
            
            # Add components
            components = placement_data.get("components", [])
            self._place_components(components)
            
            # Add nets and routing hints
            nets = placement_data.get("nets", [])
            self._create_nets(nets)
            
            # Save board
            if output_path is None:
                output_path = os.path.join(self.temp_dir, f"{project_name}.kicad_pcb")
            
            self.board.Save(output_path)
            
            return output_path
            
        except Exception as e:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise Exception(f"KiCAD export failed: {str(e)}")
    
    def _setup_board(self, board_info: Dict[str, Any]):
        """Set up board size and outline."""
        board_width = board_info.get("width", 100) * 1e6  # Convert mm to nanometers
        board_height = board_info.get("height", 100) * 1e6
        
        # Set board size (in nanometers)
        board_bbox = pcbnew.BOX2I(
            pcbnew.VECTOR2I(0, 0),
            pcbnew.VECTOR2I(int(board_width), int(board_height))
        )
        
        # Add board outline (Edge.Cuts layer)
        edge_layer = pcbnew.Edge_Cuts
        
        # Create rectangular outline
        outline_points = [
            pcbnew.VECTOR2I(0, 0),
            pcbnew.VECTOR2I(int(board_width), 0),
            pcbnew.VECTOR2I(int(board_width), int(board_height)),
            pcbnew.VECTOR2I(0, int(board_height)),
            pcbnew.VECTOR2I(0, 0)  # Close polygon
        ]
        
        # Add outline segments
        for i in range(len(outline_points) - 1):
            segment = pcbnew.PCB_SHAPE(self.board)
            segment.SetShape(pcbnew.SHAPE_T_SEGMENT)
            segment.SetStart(outline_points[i])
            segment.SetEnd(outline_points[i + 1])
            segment.SetLayer(edge_layer)
            segment.SetWidth(int(0.1 * 1e6))  # 0.1mm line width
            self.board.Add(segment)
        
        # Set design rules
        clearance = board_info.get("clearance", 0.5) * 1e6  # Convert mm to nanometers
        design_settings = self.board.GetDesignSettings()
        design_settings.m_MinClearance = int(clearance)
        design_settings.m_TrackMinWidth = int(0.25 * 1e6)  # 0.25mm minimum trace width
    
    def _place_components(self, components: List[Dict[str, Any]]):
        """Place components on the board."""
        for comp_data in components:
            try:
                comp_name = comp_data.get("name", "U1")
                package = comp_data.get("package", "Unknown")
                x = comp_data.get("x", 0) * 1e6  # Convert mm to nanometers
                y = comp_data.get("y", 0) * 1e6
                angle = comp_data.get("angle", 0)  # Degrees
                
                # Create footprint
                footprint = self._create_footprint(comp_name, package, comp_data)
                
                if footprint:
                    # Set position
                    footprint.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
                    footprint.SetOrientationDegrees(angle)
                    
                    # Add to board
                    self.board.Add(footprint)
                    
            except Exception as e:
                print(f"Warning: Failed to place component {comp_data.get('name', 'unknown')}: {e}")
                continue
    
    def _create_footprint(self, name: str, package: str, comp_data: Dict[str, Any]) -> Optional[pcbnew.FOOTPRINT]:
        """Create a footprint for a component."""
        try:
            # Try to load from library first
            footprint = self._load_footprint_from_library(package)
            
            if footprint is None:
                # Create generic footprint
                footprint = self._create_generic_footprint(name, package, comp_data)
            
            if footprint:
                footprint.SetReference(name)
                footprint.SetValue(name)
            
            return footprint
            
        except Exception as e:
            print(f"Warning: Failed to create footprint for {name}: {e}")
            return None
    
    def _load_footprint_from_library(self, package: str) -> Optional[pcbnew.FOOTPRINT]:
        """Try to load footprint from KiCAD library."""
        try:
            # Common footprint library paths
            lib_paths = [
                f"{package}:{package}",
                f"Resistor_SMD:R_{package}",
                f"Capacitor_SMD:C_{package}",
                f"LED_SMD:LED_{package}",
            ]
            
            for lib_path in lib_paths:
                try:
                    footprint = pcbnew.FootprintLoad(pcbnew.GetKicadDataPath(), lib_path)
                    if footprint:
                        return footprint
                except:
                    continue
            
            return None
            
        except Exception as e:
            return None
    
    def _create_generic_footprint(self, name: str, package: str, comp_data: Dict[str, Any]) -> pcbnew.FOOTPRINT:
        """Create a generic footprint when library footprint is not available."""
        width = comp_data.get("width", 5) * 1e6  # Convert mm to nanometers
        height = comp_data.get("height", 5) * 1e6
        pins = comp_data.get("pins", [])
        
        # Create new footprint
        footprint = pcbnew.FOOTPRINT(self.board)
        footprint.SetFPID(pcbnew.LIB_ID(name))
        
        # Add reference and value text
        ref_text = pcbnew.FP_TEXT(footprint)
        ref_text.SetText(name)
        ref_text.SetLayer(pcbnew.F_SilkS)
        ref_text.SetPosition(pcbnew.VECTOR2I(0, int(-height/2 - 1e6)))
        ref_text.SetTextSize(pcbnew.VECTOR2I(int(1e6), int(1e6)))
        footprint.Add(ref_text)
        
        # Add component outline
        outline_layer = pcbnew.F_Fab
        outline_width = int(0.1 * 1e6)
        
        # Rectangle outline
        corners = [
            pcbnew.VECTOR2I(int(-width/2), int(-height/2)),
            pcbnew.VECTOR2I(int(width/2), int(-height/2)),
            pcbnew.VECTOR2I(int(width/2), int(height/2)),
            pcbnew.VECTOR2I(int(-width/2), int(height/2)),
        ]
        
        for i in range(len(corners)):
            segment = pcbnew.FP_SHAPE(footprint)
            segment.SetShape(pcbnew.SHAPE_T_SEGMENT)
            segment.SetStart(corners[i])
            segment.SetEnd(corners[(i + 1) % len(corners)])
            segment.SetLayer(outline_layer)
            segment.SetWidth(outline_width)
            footprint.Add(segment)
        
        # Add pads
        if pins:
            pad_size = min(width, height) * 0.2
            pad_size = max(0.5e6, min(pad_size, 1e6))  # Clamp between 0.5mm and 1mm
            
            for i, pin in enumerate(pins):
                pin_name = pin.get("name", f"pin{i+1}") if isinstance(pin, dict) else f"pin{i+1}"
                
                # Calculate pad position
                if isinstance(pin, dict) and pin.get("x_offset") is not None:
                    x_offset = pin.get("x_offset", 0) * 1e6
                    y_offset = pin.get("y_offset", 0) * 1e6
                else:
                    # Default: distribute pads
                    if len(pins) == 2:
                        x_offset = (-width/4 if i == 0 else width/4)
                        y_offset = 0
                    else:
                        x_offset = 0
                        y_offset = 0
                
                # Create pad
                pad = pcbnew.PAD(footprint)
                pad.SetNumber(pin_name)
                pad.SetPosition(pcbnew.VECTOR2I(int(x_offset), int(y_offset)))
                pad.SetSize(pcbnew.VECTOR2I(int(pad_size), int(pad_size)))
                pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)
                pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
                pad.SetLayerSet(pcbnew.LSET(pcbnew.F_Cu))
                footprint.Add(pad)
        else:
            # Default: two pads
            pad_size = min(width, height) * 0.3
            pad_size = max(0.5e6, min(pad_size, 1e6))
            
            for i in [1, 2]:
                pad = pcbnew.PAD(footprint)
                pad.SetNumber(str(i))
                x_offset = (-width/4 if i == 1 else width/4)
                pad.SetPosition(pcbnew.VECTOR2I(int(x_offset), 0))
                pad.SetSize(pcbnew.VECTOR2I(int(pad_size), int(pad_size)))
                pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)
                pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
                pad.SetLayerSet(pcbnew.LSET(pcbnew.F_Cu))
                footprint.Add(pad)
        
        return footprint
    
    def _create_nets(self, nets: List[Dict[str, Any]]):
        """Create nets and connect component pads."""
        if not nets:
            return
        
        # Create net mapping
        net_map = {}
        for i, net_data in enumerate(nets):
            net_name = net_data.get("name", f"Net{i+1}")
            net = pcbnew.NETINFO_ITEM(self.board, net_name)
            self.board.Add(net)
            net_map[net_name] = net
        
        # Connect pads to nets
        for net_data in nets:
            net_name = net_data.get("name", "")
            if not net_name:
                continue
            
            net = net_map.get(net_name)
            if not net:
                continue
            
            net_pins = net_data.get("pins", [])
            for pin_ref in net_pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    comp_name = pin_ref[0]
                    pin_name = pin_ref[1]
                    
                    # Find component and pad
                    footprint = self._find_footprint(comp_name)
                    if footprint:
                        pad = footprint.FindPadByNumber(pin_name)
                        if pad:
                            pad.SetNet(net)
    
    def _find_footprint(self, name: str) -> Optional[pcbnew.FOOTPRINT]:
        """Find footprint by reference name."""
        for footprint in self.board.Footprints():
            if footprint.GetReference() == name:
                return footprint
        return None
    
    def get_file_content(self, file_path: str) -> str:
        """Read the generated KiCad file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

