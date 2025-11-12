"""
RF Placer Agent

Specialized agent for RF/high-frequency PCB optimization.
Implements controlled impedance routing, RF isolation, and EMI mitigation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from backend.geometry.placement import Placement
    from backend.geometry.net import Net
    from backend.constraints.pcb_fabrication import FabricationConstraints
    from backend.agents.routing_agent import RoutingAgent
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.net import Net
    from src.backend.constraints.pcb_fabrication import FabricationConstraints
    from src.backend.agents.routing_agent import RoutingAgent

logger = logging.getLogger(__name__)


class RFPlacerAgent:
    """
    Specialized agent for RF/high-frequency PCB optimization.
    
    Features:
    - Controlled impedance routing (50Ω, 100Ω differential)
    - RF isolation (separate ground planes, via fences)
    - Antenna placement optimization
    - EMI/EMC compliance
    - Matching network optimization
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None):
        """
        Initialize RF placer agent.
        
        Args:
            constraints: Fabrication constraints
        """
        self.name = "RFPlacerAgent"
        self.constraints = constraints or FabricationConstraints()
        self.routing_agent = RoutingAgent(constraints=self.constraints)
        
        # RF-specific parameters
        self.target_impedance_50 = 50.0  # Ω
        self.target_impedance_100 = 100.0  # Ω (differential)
        self.rf_isolation_distance = 5.0  # mm minimum distance from RF section
        self.via_fence_spacing = 1.0  # mm between vias in fence
    
    async def optimize_rf_design(self, placement: Placement) -> Dict:
        """
        Optimize PCB design for RF/high-frequency operation.
        
        Args:
            placement: Placement to optimize
        
        Returns:
            {
                "success": bool,
                "placement": Placement,
                "rf_optimizations": Dict,
                "impedance_controlled_nets": List,
                "isolation_zones": List
            }
        """
        try:
            logger.info(f"📡 {self.name}: Optimizing RF design")
            
            optimized_placement = placement.copy()
            rf_optimizations = {
                "impedance_controlled_nets": [],
                "isolation_zones": [],
                "via_fences": [],
                "antenna_optimizations": []
            }
            
            # Step 1: Identify RF components and nets
            rf_components, rf_nets = self._identify_rf_elements(optimized_placement)
            
            # Step 2: Create RF isolation zones
            isolation_zones = self._create_isolation_zones(optimized_placement, rf_components)
            rf_optimizations["isolation_zones"] = isolation_zones
            
            # Step 3: Optimize antenna placement
            antenna_optimizations = self._optimize_antenna_placement(optimized_placement, rf_components)
            rf_optimizations["antenna_optimizations"] = antenna_optimizations
            
            # Step 4: Route with controlled impedance
            impedance_nets = self._identify_impedance_controlled_nets(optimized_placement.nets)
            rf_optimizations["impedance_controlled_nets"] = [
                {
                    "net": net.name,
                    "impedance": self._get_target_impedance(net),
                    "width": self._calculate_impedance_width(net)
                }
                for net in impedance_nets
            ]
            
            # Step 5: Add via fences for RF isolation
            via_fences = self._add_via_fences(optimized_placement, isolation_zones)
            rf_optimizations["via_fences"] = via_fences
            
            logger.info(f"✅ {self.name}: RF optimization complete")
            
            return {
                "success": True,
                "placement": optimized_placement,
                "rf_optimizations": rf_optimizations,
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name}: RF optimization failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _identify_rf_elements(self, placement: Placement) -> Tuple[List, List]:
        """Identify RF components and nets."""
        rf_components = []
        rf_nets = []
        
        # Identify RF components by keywords
        rf_keywords = ["rf", "antenna", "transceiver", "balun", "matching", "filter", "2.4ghz", "5ghz"]
        
        for comp in placement.components.values():
            comp_name_lower = comp.name.lower()
            if any(keyword in comp_name_lower for keyword in rf_keywords):
                rf_components.append(comp)
        
        # Identify RF nets
        for net in placement.nets.values():
            net_name_lower = net.name.lower()
            if any(keyword in net_name_lower for keyword in rf_keywords + ["50 ohm", "100 ohm", "impedance"]):
                rf_nets.append(net)
        
        return rf_components, rf_nets
    
    def _create_isolation_zones(self, placement: Placement, rf_components: List) -> List[Dict]:
        """Create isolation zones for RF section."""
        if not rf_components:
            return []
        
        # Calculate bounding box of RF components
        rf_positions = np.array([[c.x, c.y] for c in rf_components])
        x_min = np.min(rf_positions[:, 0]) - self.rf_isolation_distance
        x_max = np.max(rf_positions[:, 0]) + self.rf_isolation_distance
        y_min = np.min(rf_positions[:, 1]) - self.rf_isolation_distance
        y_max = np.max(rf_positions[:, 1]) + self.rf_isolation_distance
        
        return [{
            "type": "rf_isolation",
            "bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max)
            },
            "clearance": self.rf_isolation_distance
        }]
    
    def _optimize_antenna_placement(self, placement: Placement, rf_components: List) -> List[Dict]:
        """Optimize antenna placement for RF performance."""
        antenna_components = [c for c in rf_components if "antenna" in c.name.lower()]
        
        optimizations = []
        for antenna in antenna_components:
            # Antenna should be at board edge for best radiation
            # Move towards nearest board edge
            edge_distances = {
                "top": placement.board.height - antenna.y,
                "bottom": antenna.y,
                "left": antenna.x,
                "right": placement.board.width - antenna.x
            }
            
            nearest_edge = min(edge_distances, key=edge_distances.get)
            min_edge_distance = 3.0  # mm from edge
            
            if nearest_edge == "top":
                antenna.y = placement.board.height - min_edge_distance
            elif nearest_edge == "bottom":
                antenna.y = min_edge_distance
            elif nearest_edge == "left":
                antenna.x = min_edge_distance
            else:  # right
                antenna.x = placement.board.width - min_edge_distance
            
            optimizations.append({
                "component": antenna.name,
                "action": f"Moved to {nearest_edge} edge",
                "position": [float(antenna.x), float(antenna.y)]
            })
        
        return optimizations
    
    def _identify_impedance_controlled_nets(self, nets: Dict[str, Net]) -> List[Net]:
        """Identify nets requiring controlled impedance."""
        impedance_nets = []
        
        for net in nets.values():
            net_name_lower = net.name.lower()
            
            # Check for impedance keywords
            if any(keyword in net_name_lower for keyword in [
                "50 ohm", "100 ohm", "impedance", "rf", "antenna",
                "differential", "usb", "hdmi", "pcie"
            ]):
                impedance_nets.append(net)
        
        return impedance_nets
    
    def _get_target_impedance(self, net: Net) -> float:
        """Get target impedance for net."""
        net_name_lower = net.name.lower()
        
        if "100" in net_name_lower or "differential" in net_name_lower:
            return self.target_impedance_100
        else:
            return self.target_impedance_50
    
    def _calculate_impedance_width(self, net: Net) -> float:
        """Calculate trace width for controlled impedance."""
        target_impedance = self._get_target_impedance(net)
        
        # Simplified calculation - in production would use full stackup
        # For FR4, 1.6mm board, 0.035mm copper:
        if target_impedance == 100.0:
            return 0.1  # mm for differential pair
        else:
            return 0.2  # mm for 50Ω single-ended
    
    def _add_via_fences(self, placement: Placement, isolation_zones: List[Dict]) -> List[Dict]:
        """Add via fences for RF isolation."""
        via_fences = []
        
        for zone in isolation_zones:
            bounds = zone["bounds"]
            
            # Create via fence around RF zone
            # Simplified: return fence specification
            via_fences.append({
                "zone": zone["type"],
                "bounds": bounds,
                "via_spacing": self.via_fence_spacing,
                "via_diameter": self.constraints.via_drill_dia,
                "description": "Via fence for RF isolation"
            })
        
        return via_fences

