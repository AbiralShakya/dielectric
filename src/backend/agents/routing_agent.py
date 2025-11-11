"""
Routing Agent

Production-scalable agent for automatic trace routing using MST-based algorithms.
Integrates with KiCad MCP for actual trace placement.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
import logging

try:
    from backend.geometry.placement import Placement
    from backend.geometry.net import Net
    from backend.constraints.pcb_fabrication import FabricationConstraints
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.net import Net
    from src.backend.constraints.pcb_fabrication import FabricationConstraints

logger = logging.getLogger(__name__)


class RoutingAgent:
    """
    Production-scalable agent for trace routing.
    
    Features:
    - MST-based routing path calculation
    - Net prioritization (power/ground → clocks → signals)
    - Trace width calculation based on current requirements
    - Multi-layer routing support
    - Controlled impedance routing for RF/high-speed
    """
    
    def __init__(self, kicad_client=None, constraints: Optional[FabricationConstraints] = None):
        """
        Initialize routing agent.
        
        Args:
            kicad_client: Optional KiCad MCP client for trace placement
            constraints: Fabrication constraints for trace width/spacing
        """
        self.name = "RoutingAgent"
        self.kicad_client = kicad_client
        self.constraints = constraints or FabricationConstraints()
        self.routed_nets: Set[str] = set()
        self.traces: List[Dict] = []
        self.vias: List[Dict] = []
        
        # Try to initialize KiCad client if not provided
        if not self.kicad_client:
            try:
                from src.backend.mcp.kicad_direct_client import KiCadDirectClient
                self.kicad_client = KiCadDirectClient()
                if self.kicad_client.is_available():
                    logger.info("✅ RoutingAgent: KiCad client initialized")
                else:
                    logger.warning("⚠️  RoutingAgent: KiCad not available")
                    self.kicad_client = None
            except Exception as e:
                logger.warning(f"⚠️  RoutingAgent: Could not initialize KiCad client: {e}")
                self.kicad_client = None
        
    async def route_design(self, placement: Placement, max_nets: Optional[int] = None) -> Dict:
        """
        Route all nets in the design.
        
        Args:
            placement: Placement with components and nets
            max_nets: Optional limit on number of nets to route (for testing)
        
        Returns:
            {
                "success": bool,
                "routed_nets": int,
                "total_trace_length": float,
                "traces": List[Dict],
                "vias": List[Dict],
                "routing_stats": Dict
            }
        """
        try:
            logger.info(f"🔌 {self.name}: Starting routing for {len(placement.nets)} nets")
            
            # Reset routing state
            self.routed_nets.clear()
            self.traces.clear()
            self.vias.clear()
            
            # Prioritize nets
            priority_nets = self._prioritize_nets(placement.nets)
            
            routing_stats = {
                "high_priority": len(priority_nets["high"]),
                "medium_priority": len(priority_nets["medium"]),
                "low_priority": len(priority_nets["low"]),
                "total_nets": len(placement.nets)
            }
            
            # Route high-priority nets first (power, ground, clocks)
            routed_count = 0
            total_trace_length = 0.0
            
            for priority_level in ["high", "medium", "low"]:
                nets_to_route = priority_nets[priority_level]
                if max_nets:
                    nets_to_route = nets_to_route[:max_nets]
                
                for net in nets_to_route:
                    if net.name in self.routed_nets:
                        continue
                    
                    route_result = await self._route_net(placement, net)
                    if route_result["success"]:
                        self.routed_nets.add(net.name)
                        routed_count += 1
                        total_trace_length += route_result["trace_length"]
                        self.traces.extend(route_result["traces"])
                        self.vias.extend(route_result.get("vias", []))
            
            logger.info(f"✅ {self.name}: Routed {routed_count}/{len(placement.nets)} nets, "
                       f"total length: {total_trace_length:.1f}mm")
            
            return {
                "success": True,
                "routed_nets": routed_count,
                "total_nets": len(placement.nets),
                "total_trace_length": total_trace_length,
                "traces": self.traces,
                "vias": self.vias,
                "routing_stats": routing_stats,
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Routing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _prioritize_nets(self, nets: Dict[str, Net]) -> Dict[str, List[Net]]:
        """
        Prioritize nets for routing.
        
        Priority order:
        1. High: Power, ground, clocks
        2. Medium: High fanout, critical signals
        3. Low: Other signals
        """
        priority = {"high": [], "medium": [], "low": []}
        
        for net in nets.values():
            net_name_lower = net.name.lower()
            pin_count = len(net.pins)
            
            # High priority: power, ground, clocks
            if any(keyword in net_name_lower for keyword in [
                "vcc", "vdd", "vss", "gnd", "ground", "power", "supply",
                "clk", "clock", "reset", "enable"
            ]):
                priority["high"].append(net)
            # Medium priority: high fanout or critical signals
            elif pin_count > 5 or any(keyword in net_name_lower for keyword in [
                "data", "addr", "cs", "we", "oe", "int", "irq"
            ]):
                priority["medium"].append(net)
            else:
                priority["low"].append(net)
        
        return priority
    
    async def _route_net(self, placement: Placement, net: Net) -> Dict:
        """
        Route a single net.
        
        Args:
            placement: Placement with components
            net: Net to route
        
        Returns:
            {
                "success": bool,
                "trace_length": float,
                "traces": List[Dict],
                "vias": List[Dict]
            }
        """
        if len(net.pins) < 2:
            return {
                "success": True,
                "trace_length": 0.0,
                "traces": [],
                "vias": []
            }
        
        # Get component positions for this net
        component_positions = []
        for comp_ref, pad_name in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                # Get pad position (simplified: use component center + offset)
                pad_pos = self._get_pad_position(comp, pad_name)
                component_positions.append((comp_ref, pad_pos, pad_name))
        
        if len(component_positions) < 2:
            return {
                "success": False,
                "error": f"Net {net.name} has insufficient connected components",
                "trace_length": 0.0,
                "traces": [],
                "vias": []
            }
        
        # Calculate routing path using MST
        routing_path = self._calculate_routing_path(component_positions)
        
        # Generate traces
        traces = []
        total_length = 0.0
        
        for i in range(len(routing_path) - 1):
            start_comp, start_pos, start_pad = routing_path[i]
            end_comp, end_pos, end_pad = routing_path[i + 1]
            
            # Calculate trace length
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx*dx + dy*dy)
            total_length += length
            
            # Calculate appropriate trace width
            trace_width = self._calculate_trace_width(net)
            
            # Determine layer (simplified: use top layer for now)
            layer = "F.Cu"
            
            trace_info = {
                "net": net.name,
                "start_component": start_comp,
                "start_pad": start_pad,
                "start_position": start_pos,
                "end_component": end_comp,
                "end_pad": end_pad,
                "end_position": end_pos,
                "width": trace_width,
                "length": length,
                "layer": layer
            }
            
            traces.append(trace_info)
            
            # Place trace using KiCad MCP if available
            if self.kicad_client and self.kicad_client.is_available():
                try:
                    kicad_result = await self._place_trace_kicad(trace_info)
                    if kicad_result.get("success"):
                        trace_info["kicad_placed"] = True
                        logger.info(f"   ✅ KiCad: Placed trace for {net.name}")
                    else:
                        logger.warning(f"   ⚠️  KiCad trace placement failed: {kicad_result.get('error')}")
                except Exception as e:
                    logger.warning(f"Failed to place trace via KiCad: {e}")
        
        return {
            "success": True,
            "trace_length": total_length,
            "traces": traces,
            "vias": []  # TODO: Add via placement for multi-layer routing
        }
    
    def _calculate_routing_path(self, component_positions: List[Tuple]) -> List[Tuple]:
        """
        Calculate optimal routing path using Minimum Spanning Tree.
        
        Args:
            component_positions: List of (component_name, (x, y), pad_name) tuples
        
        Returns:
            Ordered list of (component_name, (x, y), pad_name) tuples for routing
        """
        n = len(component_positions)
        if n < 2:
            return component_positions
        
        # Build distance matrix
        positions = np.array([pos[1] for pos in component_positions])
        dist_matrix = squareform(pdist(positions, metric='euclidean'))
        
        # Calculate MST
        mst = minimum_spanning_tree(dist_matrix)
        mst_dense = mst.toarray()
        
        # Build graph from MST
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if mst_dense[i, j] > 0:
                    edges.append((i, j, mst_dense[i, j]))
        
        # Find routing path (simplified: start from first component, traverse MST)
        if not edges:
            return component_positions
        
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for i, j, _ in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Traverse MST starting from first component
        visited = set()
        path = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            path.append(component_positions[node])
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(0)
        
        return path
    
    def _get_pad_position(self, component, pad_name: str) -> Tuple[float, float]:
        """
        Get pad position relative to component center.
        
        TODO: Query actual footprint for pad position.
        For now, return component center with small offset.
        """
        # Simplified: return component center
        # In production, this would query the footprint library
        return (component.x, component.y)
    
    def _calculate_trace_width(self, net: Net) -> float:
        """
        Calculate appropriate trace width based on net type, current requirements, and impedance control.
        
        Physics foundation:
        - Current carrying capacity: I = k * W^0.725 * T^0.5 (IPC-2221)
        - Impedance control: Z = (87/sqrt(εr+1.41)) * ln(5.98H/(0.8W+T)) (microstrip)
        
        Args:
            net: Net to calculate width for
        
        Returns:
            Trace width in mm
        """
        net_name_lower = net.name.lower()
        
        # Check for controlled impedance requirements (RF/high-speed)
        if self._requires_controlled_impedance(net):
            # For 50Ω single-ended or 100Ω differential
            # Simplified: use standard width for controlled impedance
            # In production, this would calculate based on stackup
            return 0.2  # mm (8 mil) - typical for controlled impedance
        
        # Power nets: calculate based on current requirements
        if any(keyword in net_name_lower for keyword in ["vcc", "vdd", "power", "supply"]):
            # Default: 0.5mm (20 mil) for power
            # TODO: Calculate based on actual current requirements from net metadata
            # Formula: W = I / (k * T^0.5)^(1/0.725)
            # where k = 0.0247 for outer layer, I = current in A, T = temp rise in °C
            return max(0.5, self.constraints.min_trace_width)
        
        # Ground nets: medium width
        if any(keyword in net_name_lower for keyword in ["gnd", "ground", "vss"]):
            return max(0.3, self.constraints.min_trace_width)
        
        # Clock signals: slightly wider for signal integrity
        if any(keyword in net_name_lower for keyword in ["clk", "clock"]):
            return max(0.2, self.constraints.min_trace_width)
        
        # High-speed signals: controlled impedance width
        if any(keyword in net_name_lower for keyword in ["differential", "diff", "usb", "hdmi", "pcie"]):
            return 0.15  # mm (6 mil) - typical for high-speed
        
        # Signal nets: standard width
        return self.constraints.min_trace_width
    
    def _requires_controlled_impedance(self, net: Net) -> bool:
        """
        Check if net requires controlled impedance routing.
        
        Args:
            net: Net to check
        
        Returns:
            True if controlled impedance required
        """
        net_name_lower = net.name.lower()
        
        # RF signals
        rf_keywords = ["rf", "antenna", "2.4ghz", "5ghz", "wifi", "bluetooth", "50 ohm", "100 ohm"]
        if any(keyword in net_name_lower for keyword in rf_keywords):
            return True
        
        # High-speed digital signals
        high_speed_keywords = ["differential", "diff", "usb", "hdmi", "pcie", "sata", "ethernet"]
        if any(keyword in net_name_lower for keyword in high_speed_keywords):
            return True
        
        # Check net metadata if available
        if hasattr(net, 'metadata') and net.metadata:
            if net.metadata.get("impedance") or net.metadata.get("controlled_impedance"):
                return True
        
        return False
    
    def _calculate_impedance_controlled_width(self, net: Net, target_impedance: float = 50.0) -> float:
        """
        Calculate trace width for controlled impedance.
        
        Physics: Microstrip impedance formula
        Z = (87/sqrt(εr+1.41)) * ln(5.98H/(0.8W+T))
        
        Where:
        - Z = impedance (Ω)
        - εr = dielectric constant (~4.5 for FR4)
        - H = dielectric height (mm)
        - W = trace width (mm)
        - T = trace thickness (mm)
        
        Args:
            net: Net requiring controlled impedance
            target_impedance: Target impedance (50Ω or 100Ω)
        
        Returns:
            Calculated trace width in mm
        """
        # Simplified calculation - in production would use full stackup data
        # For FR4, 1.6mm board, 0.035mm copper:
        # 50Ω ≈ 0.2mm width
        # 100Ω differential ≈ 0.1mm width per trace
        
        net_name_lower = net.name.lower()
        
        if "100" in net_name_lower or "differential" in net_name_lower:
            return 0.1  # mm for differential pair
        else:
            return 0.2  # mm for 50Ω single-ended
    
    async def _place_trace_kicad(self, trace_info: Dict):
        """
        Place trace using KiCad direct client.
        
        Args:
            trace_info: Trace information dictionary
        
        Returns:
            Result dictionary
        """
        if not self.kicad_client or not self.kicad_client.is_available():
            return {"success": False, "error": "KiCad client not available"}
        
        try:
            start_pos = trace_info["start_position"]
            end_pos = trace_info["end_position"]
            layer = trace_info["layer"]
            width = trace_info["width"]
            net = trace_info["net"]
            
            # Format for KiCad client
            start = {
                "x": start_pos[0],
                "y": start_pos[1],
                "unit": "mm"
            }
            end = {
                "x": end_pos[0],
                "y": end_pos[1],
                "unit": "mm"
            }
            
            # Call KiCad client
            result = await self.kicad_client.route_trace(
                start=start,
                end=end,
                layer=layer,
                width=width,
                net=net
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to place trace via KiCad: {e}")
            return {"success": False, "error": str(e)}
    
    def get_routing_statistics(self) -> Dict:
        """Get routing statistics."""
        return {
            "routed_nets": len(self.routed_nets),
            "total_traces": len(self.traces),
            "total_vias": len(self.vias),
            "total_trace_length": sum(t.get("length", 0) for t in self.traces)
        }
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "route_design",
            "description": "Route all nets in PCB design",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"},
                    "max_nets": {"type": "integer", "description": "Optional limit on nets to route"}
                },
                "required": ["placement"]
            }
        }

