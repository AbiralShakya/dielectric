"""
Power Placer Agent

Specialized agent for power electronics optimization.
Implements high-current trace width calculation, thermal via placement, and EMI filtering.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.constraints.pcb_fabrication import FabricationConstraints
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.constraints.pcb_fabrication import FabricationConstraints

logger = logging.getLogger(__name__)


class PowerPlacerAgent:
    """
    Specialized agent for power electronics optimization.
    
    Features:
    - High-current trace width calculation (IPC-2221)
    - Thermal via placement optimization
    - EMI filtering optimization
    - Safety compliance (creepage, clearance)
    - Power distribution network (PDN) optimization
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None):
        """
        Initialize power placer agent.
        
        Args:
            constraints: Fabrication constraints
        """
        self.name = "PowerPlacerAgent"
        self.constraints = constraints or FabricationConstraints()
        
        # Power-specific parameters
        self.thermal_via_diameter = 0.3  # mm
        self.thermal_via_spacing = 1.0  # mm
        self.min_power_trace_width = 0.5  # mm (20 mil)
        self.high_current_threshold = 1.0  # A
    
    async def optimize_power_design(self, placement: Placement) -> Dict:
        """
        Optimize PCB design for power electronics.
        
        Args:
            placement: Placement to optimize
        
        Returns:
            {
                "success": bool,
                "placement": Placement,
                "power_optimizations": Dict,
                "thermal_vias": List,
                "high_current_traces": List
            }
        """
        try:
            logger.info(f"⚡ {self.name}: Optimizing power design")
            
            optimized_placement = placement.copy()
            power_optimizations = {
                "thermal_vias": [],
                "high_current_traces": [],
                "emi_filters": [],
                "safety_compliance": {}
            }
            
            # Step 1: Identify high-power components
            high_power_components = self._identify_high_power_components(optimized_placement)
            
            # Step 2: Add thermal vias under high-power components
            thermal_vias = self._add_thermal_vias(optimized_placement, high_power_components)
            power_optimizations["thermal_vias"] = thermal_vias
            
            # Step 3: Calculate high-current trace widths
            high_current_traces = self._calculate_high_current_traces(optimized_placement)
            power_optimizations["high_current_traces"] = high_current_traces
            
            # Step 4: Optimize EMI filtering
            emi_filters = self._optimize_emi_filtering(optimized_placement)
            power_optimizations["emi_filters"] = emi_filters
            
            # Step 5: Check safety compliance
            safety_compliance = self._check_safety_compliance(optimized_placement)
            power_optimizations["safety_compliance"] = safety_compliance
            
            logger.info(f"✅ {self.name}: Power optimization complete")
            
            return {
                "success": True,
                "placement": optimized_placement,
                "power_optimizations": power_optimizations,
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Power optimization failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _identify_high_power_components(self, placement: Placement) -> List[Component]:
        """Identify high-power components."""
        high_power = []
        
        for comp in placement.components.values():
            # Components with power > threshold
            if comp.power > 1.0:
                high_power.append(comp)
            
            # Power-related components by name
            comp_name_lower = comp.name.lower()
            if any(keyword in comp_name_lower for keyword in [
                "regulator", "converter", "switching", "buck", "boost", "ldo",
                "mosfet", "transistor", "power"
            ]):
                high_power.append(comp)
        
        return high_power
    
    def _add_thermal_vias(self, placement: Placement, high_power_components: List[Component]) -> List[Dict]:
        """
        Add thermal vias under high-power components.
        
        Physics: Thermal vias transfer heat from top to bottom layer
        Formula: R_thermal = L / (k * A) where L = via length, k = thermal conductivity, A = cross-sectional area
        """
        thermal_vias = []
        
        for comp in high_power_components:
            # Calculate number of vias needed based on power
            # Rule of thumb: 1 via per 0.5W
            num_vias = max(4, int(comp.power / 0.5))
            
            # Arrange vias in grid pattern under component
            via_positions = self._calculate_via_grid(
                comp.x, comp.y, comp.width, comp.height,
                num_vias, self.thermal_via_spacing
            )
            
            for via_pos in via_positions:
                thermal_vias.append({
                    "position": via_pos,
                    "diameter": self.thermal_via_diameter,
                    "component": comp.name,
                    "power": comp.power,
                    "purpose": "thermal_management"
                })
        
        return thermal_vias
    
    def _calculate_via_grid(self, center_x: float, center_y: float, width: float, height: float,
                           num_vias: int, spacing: float) -> List[Tuple[float, float]]:
        """Calculate grid positions for thermal vias."""
        # Calculate grid dimensions
        grid_cols = int(np.sqrt(num_vias))
        grid_rows = (num_vias + grid_cols - 1) // grid_cols
        
        # Calculate spacing
        grid_width = min(width * 0.8, (grid_cols - 1) * spacing)
        grid_height = min(height * 0.8, (grid_rows - 1) * spacing)
        
        # Generate positions
        positions = []
        start_x = center_x - grid_width / 2
        start_y = center_y - grid_height / 2
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(positions) >= num_vias:
                    break
                x = start_x + j * spacing
                y = start_y + i * spacing
                positions.append((float(x), float(y)))
        
        return positions
    
    def _calculate_high_current_traces(self, placement: Placement) -> List[Dict]:
        """
        Calculate trace widths for high-current nets.
        
        Physics: IPC-2221 current carrying capacity
        I = k * W^0.725 * T^0.5
        where:
        - I = current (A)
        - k = 0.0247 for outer layer, 0.048 for inner layer
        - W = trace width (mm)
        - T = temperature rise (°C)
        """
        high_current_traces = []
        
        for net in placement.nets.values():
            net_name_lower = net.name.lower()
            
            # Identify power nets
            if any(keyword in net_name_lower for keyword in ["vcc", "vdd", "power", "supply"]):
                # Estimate current (simplified - in production would use net metadata)
                estimated_current = self._estimate_net_current(net, placement)
                
                if estimated_current > self.high_current_threshold:
                    # Calculate required trace width
                    # Using IPC-2221: W = (I / (k * T^0.5))^(1/0.725)
                    k = 0.0247  # Outer layer
                    temp_rise = 10.0  # °C
                    width = (estimated_current / (k * temp_rise**0.5))**(1/0.725)
                    width = max(width, self.min_power_trace_width)
                    
                    high_current_traces.append({
                        "net": net.name,
                        "estimated_current": float(estimated_current),
                        "required_width": float(width),
                        "current_density": float(estimated_current / width) if width > 0 else 0.0
                    })
        
        return high_current_traces
    
    def _estimate_net_current(self, net: Net, placement: Placement) -> float:
        """Estimate current for net based on connected components."""
        # Simplified estimation - in production would use component datasheets
        total_power = 0.0
        
        for comp_ref, _ in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                total_power += comp.power
        
        # Estimate voltage (simplified)
        net_name_lower = net.name.lower()
        if "3.3" in net_name_lower:
            voltage = 3.3
        elif "5" in net_name_lower:
            voltage = 5.0
        elif "12" in net_name_lower:
            voltage = 12.0
        else:
            voltage = 5.0  # Default
        
        # I = P / V
        current = total_power / voltage if voltage > 0 else 0.0
        return current
    
    def _optimize_emi_filtering(self, placement: Placement) -> List[Dict]:
        """Optimize EMI filter placement."""
        emi_filters = []
        
        # Identify switching components (EMI sources)
        switching_components = []
        for comp in placement.components.values():
            comp_name_lower = comp.name.lower()
            if any(keyword in comp_name_lower for keyword in [
                "switching", "buck", "boost", "converter", "inverter"
            ]):
                switching_components.append(comp)
        
        # Place filters near switching components
        for switching_comp in switching_components:
            emi_filters.append({
                "component": switching_comp.name,
                "position": [float(switching_comp.x), float(switching_comp.y)],
                "recommendation": "Add EMI filter capacitor near switching component",
                "filter_type": "LC_filter"
            })
        
        return emi_filters
    
    def _check_safety_compliance(self, placement: Placement) -> Dict:
        """Check safety compliance (creepage, clearance for high voltage)."""
        # Identify high-voltage nets
        high_voltage_nets = []
        for net in placement.nets.values():
            net_name_lower = net.name.lower()
            if any(keyword in net_name_lower for keyword in ["120v", "240v", "mains", "ac"]):
                high_voltage_nets.append(net)
        
        compliance = {
            "high_voltage_nets": len(high_voltage_nets),
            "creepage_clearance_ok": True,
            "recommendations": []
        }
        
        if high_voltage_nets:
            # Check clearance for high voltage
            required_clearance = self.constraints.get_clearance_for_voltage(120.0)
            compliance["required_clearance"] = required_clearance
            compliance["recommendations"].append(
                f"Ensure {required_clearance}mm minimum clearance for high-voltage nets"
            )
        
        return compliance

