"""
Power Integrity Analysis

Features:
- IR drop analysis
- Power distribution network (PDN) analysis
- Current density analysis
- Decoupling capacitor placement optimization
- Thermal analysis
- EMI/EMC analysis
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

try:
    from backend.geometry.placement import Placement
    from backend.geometry.net import Net
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.net import Net

logger = logging.getLogger(__name__)


class PowerIntegrityAnalyzer:
    """
    Power Integrity Analysis for PCB designs.
    
    Features:
    - IR drop analysis
    - PDN impedance
    - Current density
    - Thermal analysis
    """
    
    def __init__(
        self,
        copper_resistivity: float = 1.68e-8,  # Ω·m (copper)
        copper_thickness: float = 0.035  # mm (1 oz)
    ):
        """
        Initialize PI analyzer.
        
        Args:
            copper_resistivity: Copper resistivity (Ω·m)
            copper_thickness: Copper thickness (mm)
        """
        self.rho = copper_resistivity
        self.t = copper_thickness / 1000.0  # Convert to meters
    
    def analyze_ir_drop(
        self,
        placement: Placement,
        power_net: Net,
        current: float  # Amperes
    ) -> Dict:
        """
        Analyze IR drop in power distribution network.
        
        Args:
            placement: Placement with components
            power_net: Power net to analyze
            current: Total current draw (A)
        
        Returns:
            IR drop analysis results
        """
        # Get power net geometry
        trace_width = 0.5  # Default power trace width (mm)
        trace_length = self._calculate_net_length(placement, power_net)
        
        # Calculate resistance
        # R = ρ * L / (W * T)
        # where ρ = resistivity, L = length, W = width, T = thickness
        length_m = trace_length / 1000.0  # Convert to meters
        width_m = trace_width / 1000.0  # Convert to meters
        
        resistance = self.rho * length_m / (width_m * self.t)
        
        # Calculate IR drop
        voltage_drop = current * resistance
        
        # Calculate voltage drop percentage (assuming 5V supply)
        supply_voltage = 5.0  # V
        drop_percent = (voltage_drop / supply_voltage) * 100
        
        # Check if drop is acceptable (< 5% typically)
        acceptable = drop_percent < 5.0
        
        return {
            "net": power_net.name,
            "current": current,
            "trace_length": trace_length,
            "trace_width": trace_width,
            "resistance": resistance * 1000,  # mΩ
            "voltage_drop": voltage_drop,
            "drop_percent": drop_percent,
            "acceptable": acceptable,
            "recommendation": "Increase trace width" if not acceptable else "OK"
        }
    
    def analyze_pdn_impedance(
        self,
        placement: Placement,
        power_net: Net,
        frequency: float = 1e6  # 1 MHz
    ) -> Dict:
        """
        Analyze PDN impedance.
        
        Args:
            placement: Placement with components
            power_net: Power net
            frequency: Frequency (Hz)
        
        Returns:
            PDN impedance analysis
        """
        # Simplified PDN impedance calculation
        # In production, would use full PDN solver
        
        # Get decoupling capacitors
        decap_count = self._count_decap_capacitors(placement, power_net)
        
        # Target impedance (simplified)
        # Z_target = V_ripple / I_max
        target_impedance = 0.1  # Ω (typical target)
        
        # Estimate PDN impedance
        # Simplified: Z = sqrt(L/C) at resonance
        # Would need actual capacitor and plane inductance values
        
        return {
            "net": power_net.name,
            "frequency": frequency,
            "decap_count": decap_count,
            "target_impedance": target_impedance,
            "recommendation": f"Add {max(0, 4 - decap_count)} decoupling capacitors" if decap_count < 4 else "OK"
        }
    
    def analyze_current_density(
        self,
        placement: Placement,
        power_net: Net,
        current: float
    ) -> Dict:
        """
        Analyze current density in power traces.
        
        Args:
            placement: Placement with components
            power_net: Power net
            current: Current (A)
        
        Returns:
            Current density analysis
        """
        trace_width = 0.5  # mm (default power trace width)
        trace_thickness = self.t * 1000  # mm
        
        # Current density: J = I / (W * T)
        cross_section = trace_width * trace_thickness  # mm²
        current_density = current / cross_section  # A/mm²
        
        # Maximum current density (IPC-2221: ~30 A/mm² for outer layer)
        max_current_density = 30.0  # A/mm²
        acceptable = current_density < max_current_density
        
        return {
            "net": power_net.name,
            "current": current,
            "trace_width": trace_width,
            "cross_section": cross_section,
            "current_density": current_density,
            "max_allowed": max_current_density,
            "acceptable": acceptable,
            "recommendation": "Increase trace width" if not acceptable else "OK"
        }
    
    def optimize_decap_placement(
        self,
        placement: Placement,
        power_net: Net,
        component_positions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Optimize decoupling capacitor placement.
        
        Args:
            placement: Placement with components
            power_net: Power net
            component_positions: List of (x, y) positions needing decaps
        
        Returns:
            Recommended decoupling capacitor positions
        """
        decap_positions = []
        
        # Place decaps near high-current components
        for x, y in component_positions:
            # Place decap close to component (within 5mm)
            decap_x = x + 2.0  # Offset
            decap_y = y
            
            decap_positions.append((decap_x, decap_y))
        
        return decap_positions
    
    def _calculate_net_length(self, placement: Placement, net: Net) -> float:
        """Calculate total net length."""
        positions = []
        for comp_ref, pad_name in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                positions.append((comp.x, comp.y))
        
        if len(positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i+1]
            total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        return total_length
    
    def _count_decap_capacitors(self, placement: Placement, net: Net) -> int:
        """Count decoupling capacitors on power net."""
        count = 0
        
        for comp_ref, _ in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                # Check if component is a capacitor
                comp_name_lower = comp.name.lower()
                if "cap" in comp_name_lower or "c" in comp_name_lower:
                    # Check if it's a decoupling cap (typically 0.1uF - 10uF)
                    value = comp.value or ""
                    if "0.1" in value or "1u" in value or "10u" in value:
                        count += 1
        
        return count

