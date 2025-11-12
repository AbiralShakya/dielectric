"""
KiCAD MCP Exporter - Uses Proper KiCad MCP Server with Routing

This exporter uses KiCad Direct Client to properly create KiCad projects with
components, nets, and ACTUALLY ROUTED TRACES.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import tempfile
import shutil
import logging

# Try to use KiCad Direct Client (has routing capabilities)
try:
    try:
        from src.backend.mcp.kicad_direct_client import KiCadDirectClient
    except ImportError:
        from backend.mcp.kicad_direct_client import KiCadDirectClient
    KICAD_DIRECT_CLIENT_AVAILABLE = True
except ImportError:
    KICAD_DIRECT_CLIENT_AVAILABLE = False
    KiCadDirectClient = None

# Try to use RoutingAgent for intelligent routing
try:
    try:
        from src.backend.agents.routing_agent import RoutingAgent
    except ImportError:
        from backend.agents.routing_agent import RoutingAgent
    ROUTING_AGENT_AVAILABLE = True
except ImportError:
    ROUTING_AGENT_AVAILABLE = False
    RoutingAgent = None

# Fallback: Try to import pcbnew directly
try:
    import pcbnew
    KICAD_AVAILABLE = True
except ImportError:
    KICAD_AVAILABLE = False
    pcbnew = None

# For type hints only
if TYPE_CHECKING:
    if KICAD_AVAILABLE:
        import pcbnew

logger = logging.getLogger(__name__)


class KiCadMCPExporter:
    """Exports PCB placements using proper KiCad MCP Server with actual routing."""
    
    def __init__(self):
        """Initialize KiCAD exporter."""
        self.kicad_client = None
        self.board = None
        self.project_path = None
        self.temp_dir = None
        self.routing_agent = None
        
        # Try to use KiCad Direct Client first (has routing capabilities)
        if KICAD_DIRECT_CLIENT_AVAILABLE and KiCadDirectClient:
            try:
                self.kicad_client = KiCadDirectClient()
                if self.kicad_client.is_available():
                    logger.info("✅ Using KiCad Direct Client with routing support")
                else:
                    logger.warning("⚠️  KiCad Direct Client not available")
                    self.kicad_client = None
            except Exception as e:
                logger.warning(f"⚠️  KiCad Direct Client initialization failed: {e}")
                self.kicad_client = None
        
        # Initialize routing agent if available
        if ROUTING_AGENT_AVAILABLE and RoutingAgent and self.kicad_client:
            try:
                self.routing_agent = RoutingAgent(kicad_client=self.kicad_client)
                logger.info("✅ RoutingAgent initialized")
            except Exception as e:
                logger.warning(f"⚠️  RoutingAgent initialization failed: {e}")
                self.routing_agent = None
        
        # Fallback to direct pcbnew API
        if not self.kicad_client and not KICAD_AVAILABLE:
            raise ImportError(
                "KiCad MCP Server not available. "
                "Please install KiCad MCP server from https://github.com/lamaalrajih/kicad-mcp\n"
                "Or install KiCAD with Python support:\n"
                "  - Linux: sudo apt-get install kicad kicad-python3\n"
                "  - macOS: Install KiCAD.app from kicad.org\n"
                "  - Windows: Install KiCAD with Python support"
            )
    
    def export(self, placement_data: Dict[str, Any], output_path: Optional[str] = None, route_traces: bool = True) -> str:
        """
        Export placement to KiCad PCB format using proper KiCad MCP Server WITH ROUTING.
        
        Args:
            placement_data: Placement dictionary with board, components, nets
            output_path: Optional path to save .kicad_pcb file
            route_traces: Whether to actually route traces (default: True)
        
        Returns:
            Path to generated .kicad_pcb file
        """
        try:
            # Use KiCad Direct Client if available (has routing support)
            if self.kicad_client and self.kicad_client.is_available():
                return self._export_with_routing(placement_data, output_path, route_traces)
            
            # Fallback to direct pcbnew API
            elif KICAD_AVAILABLE:
                return self._export_with_pcbnew(placement_data, output_path, route_traces)
            else:
                raise ImportError("No KiCad export method available")
            
        except Exception as e:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise Exception(f"KiCAD export failed: {str(e)}")
    
    def _export_with_routing(self, placement_data: Dict[str, Any], output_path: Optional[str] = None, route_traces: bool = True) -> str:
        """
        Export using KiCad Direct Client with proper routing.
        
        This creates a real board with components, nets, and ROUTED TRACES.
        """
        import pcbnew
        
        board_info = placement_data.get("board", {})
        components = placement_data.get("components", [])
        nets = placement_data.get("nets", [])
        
        # Get the board from the client
        self.board = self.kicad_client.board
        
        if not self.board:
            raise RuntimeError("KiCad board not initialized")
        
        # Set up board
        self._setup_board_direct(board_info)
        
        # Place components
        self._place_components_direct(components)
        
        # Create nets and connect pads
        self._create_nets_direct(nets)
        
        # ROUTE TRACES - This is the key missing piece!
        if route_traces and nets:
            logger.info(f"🔌 Routing {len(nets)} nets...")
            self._route_all_nets(components, nets, board_info)
        
        # Save board
        if output_path is None:
            self.temp_dir = tempfile.mkdtemp(prefix="dielectric_kicad_")
            output_path = os.path.join(self.temp_dir, "dielectric_design.kicad_pcb")
        
        self.kicad_client.save_board(output_path)
        self.project_path = output_path
        
        logger.info(f"✅ KiCad board exported with routing to {output_path}")
        return output_path
    
    def _route_all_nets(self, components: List[Dict], nets: List[Dict], board_info: Dict):
        """
        Route all nets using KiCad routing commands.
        
        This actually places tracks on the board!
        """
        import pcbnew
        
        if not self.kicad_client or not self.kicad_client.is_available():
            logger.warning("⚠️  KiCad client not available, skipping routing")
            return
        
        # Get pad positions for each net
        routed_count = 0
        
        for net_data in nets:
            net_name = net_data.get("name", "")
            if not net_name:
                continue
            
            net_pins = net_data.get("pins", [])
            if len(net_pins) < 2:
                continue
            
            # Get pad positions for this net
            pad_positions = []
            for pin_ref in net_pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    comp_name = pin_ref[0]
                    pin_name = pin_ref[1]
                    
                    # Find component
                    comp = next((c for c in components if c.get("name") == comp_name), None)
                    if not comp:
                        continue
                    
                    # Get pad position
                    pad_pos = self._get_pad_position_for_component(comp, pin_name)
                    if pad_pos:
                        pad_positions.append((pad_pos, net_name))
            
            # Route between pads using MST-like approach
            if len(pad_positions) >= 2:
                # Calculate trace width based on net type
                trace_width = self._calculate_trace_width_for_net(net_name)
                
                # Route pairs of pads
                for i in range(len(pad_positions) - 1):
                    start_pos, net = pad_positions[i]
                    end_pos, _ = pad_positions[i + 1]
                    
                    try:
                        # Route trace using KiCad client
                        result = self.kicad_client.routing_cmds.route_trace({
                            "start": {"x": start_pos[0], "y": start_pos[1], "unit": "mm"},
                            "end": {"x": end_pos[0], "y": end_pos[1], "unit": "mm"},
                            "layer": "F.Cu",
                            "width": trace_width,
                            "net": net_name
                        })
                        
                        if result.get("success"):
                            routed_count += 1
                        else:
                            logger.warning(f"Failed to route {net_name}: {result.get('errorDetails', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"Exception routing {net_name}: {e}")
        
        logger.info(f"✅ Routed {routed_count} trace segments")
    
    def _get_pad_position_for_component(self, comp: Dict, pin_name: str) -> Optional[tuple]:
        """Get pad position (x, y) in mm for a component pin."""
        comp_x = comp.get("x", 0)
        comp_y = comp.get("y", 0)
        
        # Try to get actual pad offset from component data
        pins = comp.get("pins", [])
        for pin in pins:
            pin_data = pin if isinstance(pin, dict) else {"name": str(pin)}
            if pin_data.get("name") == pin_name or str(pin_data.get("name")) == str(pin_name):
                # Get pad offset
                pad_x = comp_x + pin_data.get("x_offset", 0)
                pad_y = comp_y + pin_data.get("y_offset", 0)
                return (pad_x, pad_y)
        
        # Fallback: use component center with small offset
        # This is approximate but better than nothing
        return (comp_x, comp_y)
    
    def _calculate_trace_width_for_net(self, net_name: str) -> float:
        """Calculate appropriate trace width based on net name."""
        net_lower = net_name.lower()
        
        # Power nets: wider traces
        if any(kw in net_lower for kw in ["vcc", "vdd", "power", "supply"]):
            return 0.5  # 20 mil
        # Ground nets: medium width
        elif any(kw in net_lower for kw in ["gnd", "ground", "vss"]):
            return 0.3  # 12 mil
        # Clock signals: slightly wider
        elif any(kw in net_lower for kw in ["clk", "clock"]):
            return 0.2  # 8 mil
        # Default: standard signal width
        else:
            return 0.15  # 6 mil - manufacturable minimum
    
    def _setup_board_direct(self, board_info: Dict[str, Any]):
        """Set up board using pcbnew directly."""
        import pcbnew
        
        board_width = board_info.get("width", 100) * 1e6  # Convert mm to nanometers
        board_height = board_info.get("height", 100) * 1e6
        
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
        clearance = board_info.get("clearance", 0.15) * 1e6  # Convert mm to nanometers
        design_settings = self.board.GetDesignSettings()
        design_settings.m_MinClearance = int(clearance)
        design_settings.m_TrackMinWidth = int(0.15 * 1e6)  # 0.15mm minimum trace width
        design_settings.m_ViaMinSize = int(0.6 * 1e6)  # 0.6mm via diameter
        design_settings.m_ViaMinDrill = int(0.25 * 1e6)  # 0.25mm via drill
    
    def _place_components_direct(self, components: List[Dict[str, Any]]):
        """Place components on the board using pcbnew."""
        import pcbnew
        
        for comp_data in components:
            try:
                comp_name = comp_data.get("name", "U1")
                package = comp_data.get("package", "Unknown")
                x = comp_data.get("x", 0) * 1e6  # Convert mm to nanometers
                y = comp_data.get("y", 0) * 1e6
                angle = comp_data.get("angle", 0)  # Degrees
                
                # Create footprint
                footprint = self._create_footprint_direct(comp_name, package, comp_data)
                
                if footprint:
                    # Set position
                    footprint.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
                    footprint.SetOrientationDegrees(angle)
                    
                    # Add to board
                    self.board.Add(footprint)
                    
            except Exception as e:
                logger.warning(f"Failed to place component {comp_data.get('name', 'unknown')}: {e}")
                continue
    
    def _create_footprint_direct(self, name: str, package: str, comp_data: Dict[str, Any]) -> Optional[Any]:
        """Create a footprint using pcbnew."""
        import pcbnew
        
        try:
            # Try to load from library first
            footprint = self._load_footprint_from_library(package)
            
            if footprint is None:
                # Create generic footprint
                footprint = self._create_generic_footprint_direct(name, package, comp_data)
            
            if footprint:
                footprint.SetReference(name)
                footprint.SetValue(name)
            
            return footprint
            
        except Exception as e:
            logger.warning(f"Failed to create footprint for {name}: {e}")
            return None
    
    def _create_generic_footprint_direct(self, name: str, package: str, comp_data: Dict[str, Any]) -> Any:
        """Create a generic footprint when library footprint is not available."""
        import pcbnew
        
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
    
    def _create_nets_direct(self, nets: List[Dict[str, Any]]):
        """Create nets and connect component pads using pcbnew."""
        import pcbnew
        
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
                    footprint = self._find_footprint_direct(comp_name)
                    if footprint:
                        pad = footprint.FindPadByNumber(pin_name)
                        if pad:
                            pad.SetNet(net)
    
    def _find_footprint_direct(self, name: str) -> Optional[Any]:
        """Find footprint by reference name."""
        for footprint in self.board.Footprints():
            if footprint.GetReference() == name:
                return footprint
        return None
    
    def _export_with_pcbnew(self, placement_data: Dict[str, Any], output_path: Optional[str] = None, route_traces: bool = True) -> str:
        """Fallback export using direct pcbnew API."""
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
        
        # ROUTE TRACES if requested
        if route_traces and nets:
            logger.info(f"🔌 Routing {len(nets)} nets using pcbnew...")
            self._route_nets_pcbnew(components, nets)
        
        # Save board
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"{project_name}.kicad_pcb")
        
        self.board.Save(output_path)
        
        return output_path
    
    def _route_nets_pcbnew(self, components: List[Dict], nets: List[Dict]):
        """Route nets using pcbnew directly."""
        import pcbnew
        
        routed_count = 0
        
        for net_data in nets:
            net_name = net_data.get("name", "")
            if not net_name:
                continue
            
            # Find net object
            netinfo = self.board.GetNetInfo()
            nets_map = netinfo.NetsByName()
            if not nets_map.has_key(net_name):
                continue
            
            net_obj = nets_map[net_name]
            net_pins = net_data.get("pins", [])
            
            if len(net_pins) < 2:
                continue
            
            # Get pad positions
            pad_positions = []
            for pin_ref in net_pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    comp_name = pin_ref[0]
                    pin_name = pin_ref[1]
                    
                    footprint = self._find_footprint(comp_name)
                    if footprint:
                        pad = footprint.FindPadByNumber(pin_name)
                        if pad:
                            pad_positions.append(pad.GetPosition())
            
            # Route between pads
            if len(pad_positions) >= 2:
                trace_width = self._calculate_trace_width_for_net(net_name)
                trace_width_nm = int(trace_width * 1e6)
                
                # Route pairs
                for i in range(len(pad_positions) - 1):
                    start_pos = pad_positions[i]
                    end_pos = pad_positions[i + 1]
                    
                    try:
                        track = pcbnew.PCB_TRACK(self.board)
                        track.SetStart(start_pos)
                        track.SetEnd(end_pos)
                        track.SetLayer(pcbnew.F_Cu)
                        track.SetWidth(trace_width_nm)
                        track.SetNet(net_obj)
                        self.board.Add(track)
                        routed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to route trace for {net_name}: {e}")
        
        logger.info(f"✅ Routed {routed_count} trace segments using pcbnew")
    
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
    
    def _create_footprint(self, name: str, package: str, comp_data: Dict[str, Any]) -> Optional[Any]:
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
    
    def _load_footprint_from_library(self, package: str) -> Optional[Any]:
        """Try to load footprint from KiCAD library."""
        import pcbnew
        
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
    
    def _create_generic_footprint(self, name: str, package: str, comp_data: Dict[str, Any]) -> Any:
        """Create generic footprint - delegates to _create_generic_footprint_direct."""
        return self._create_generic_footprint_direct(name, package, comp_data)
    
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
    
    def _find_footprint(self, name: str) -> Optional[Any]:
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

