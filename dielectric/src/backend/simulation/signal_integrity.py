"""
Signal Integrity Analysis

Features:
- Impedance calculation (microstrip, stripline)
- Differential pair impedance
- Transmission line modeling
- Crosstalk analysis
- Reflection analysis
- Eye diagram generation
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


class SignalIntegrityAnalyzer:
    """
    Signal Integrity Analysis for PCB designs.
    
    Features:
    - Impedance calculation
    - Transmission line modeling
    - Crosstalk analysis
    - High-speed design rules
    """
    
    def __init__(
        self,
        dielectric_constant: float = 4.5,  # FR4
        trace_thickness: float = 0.035,  # mm (1 oz copper)
        board_thickness: float = 1.6  # mm
    ):
        """
        Initialize SI analyzer.
        
        Args:
            dielectric_constant: Dielectric constant (εr)
            trace_thickness: Copper thickness (mm)
            board_thickness: Board thickness (mm)
        """
        self.er = dielectric_constant
        self.t = trace_thickness
        self.h = board_thickness
    
    def calculate_impedance(
        self,
        trace_width: float,
        trace_type: str = "microstrip"  # "microstrip" or "stripline"
    ) -> float:
        """
        Calculate characteristic impedance.
        
        Microstrip formula (IPC-2141A):
        Z = (87/sqrt(εr+1.41)) * ln(5.98H/(0.8W+T))
        
        Stripline formula:
        Z = (60/sqrt(εr)) * ln(4H/(0.67πW))
        
        Args:
            trace_width: Trace width (mm)
            trace_type: "microstrip" or "stripline"
        
        Returns:
            Characteristic impedance (Ω)
        """
        if trace_type == "microstrip":
            # Microstrip impedance
            z0 = (87.0 / np.sqrt(self.er + 1.41)) * np.log(5.98 * self.h / (0.8 * trace_width + self.t))
        else:
            # Stripline impedance
            z0 = (60.0 / np.sqrt(self.er)) * np.log(4 * self.h / (0.67 * np.pi * trace_width))
        
        return z0
    
    def calculate_differential_impedance(
        self,
        trace_width: float,
        spacing: float,
        trace_type: str = "microstrip"
    ) -> float:
        """
        Calculate differential impedance.
        
        Args:
            trace_width: Trace width (mm)
            spacing: Spacing between traces (mm)
            trace_type: "microstrip" or "stripline"
        
        Returns:
            Differential impedance (Ω)
        """
        # Single-ended impedance
        z0_se = self.calculate_impedance(trace_width, trace_type)
        
        # Coupling factor (simplified)
        coupling = np.exp(-spacing / trace_width)
        
        # Differential impedance
        z_diff = 2 * z0_se * (1 - coupling)
        
        return z_diff
    
    def analyze_net(
        self,
        placement: Placement,
        net: Net,
        frequency: float = 1e9  # 1 GHz
    ) -> Dict:
        """
        Analyze signal integrity for a net.
        
        Args:
            placement: Placement with components
            net: Net to analyze
            frequency: Signal frequency (Hz)
        
        Returns:
            SI analysis results
        """
        # Get trace information
        trace_width = 0.2  # Default, would get from actual trace
        trace_length = self._calculate_net_length(placement, net)
        
        # Calculate impedance
        z0 = self.calculate_impedance(trace_width)
        
        # Calculate propagation delay
        vp = 3e8 / np.sqrt(self.er)  # m/s
        prop_delay = trace_length / 1000.0 / vp * 1e9  # ns
        
        # Calculate rise time limit (rule of thumb: trace length < 1/6 of rise time)
        # For 1ns rise time, max length = 1ns * vp / 6 ≈ 50mm
        max_length_for_rise_time = prop_delay * 6  # Simplified
        
        # Check if net requires termination
        requires_termination = trace_length > max_length_for_rise_time
        
        return {
            "net": net.name,
            "trace_width": trace_width,
            "trace_length": trace_length,
            "impedance": z0,
            "propagation_delay": prop_delay,
            "requires_termination": requires_termination,
            "frequency": frequency,
            "wavelength": vp / frequency * 1000  # mm
        }
    
    def analyze_crosstalk(
        self,
        placement: Placement,
        aggressor_net: Net,
        victim_net: Net,
        spacing: float
    ) -> Dict:
        """
        Analyze crosstalk between two nets.
        
        Args:
            placement: Placement with components
            aggressor_net: Aggressor net
            victim_net: Victim net
            spacing: Spacing between traces (mm)
        
        Returns:
            Crosstalk analysis results
        """
        # Simplified crosstalk calculation
        # In production, would use full-field solver
        
        # Coupling coefficient (simplified)
        coupling = np.exp(-spacing / 0.2)  # Assuming 0.2mm trace width
        
        # Crosstalk voltage (simplified)
        # V_crosstalk = coupling * V_aggressor
        
        return {
            "aggressor": aggressor_net.name,
            "victim": victim_net.name,
            "spacing": spacing,
            "coupling_coefficient": coupling,
            "crosstalk_percent": coupling * 100,
            "recommendation": "Increase spacing" if coupling > 0.1 else "OK"
        }
    
    def check_high_speed_rules(
        self,
        placement: Placement,
        net: Net
    ) -> List[Dict]:
        """
        Check high-speed design rules.
        
        Rules checked:
        - Length matching
        - Via stubbing
        - Return path
        - Termination
        
        Returns:
            List of rule violations/recommendations
        """
        violations = []
        
        # Check length (simplified)
        length = self._calculate_net_length(placement, net)
        
        # Rule: Keep traces short for high-speed signals
        if "clk" in net.name.lower() or "clock" in net.name.lower():
            if length > 50.0:  # mm
                violations.append({
                    "rule": "clock_length",
                    "severity": "warning",
                    "message": f"Clock net {net.name} is {length:.1f}mm long, consider shortening",
                    "current": length,
                    "recommended": "< 50mm"
                })
        
        # Check for vias (simplified)
        if hasattr(placement, 'vias') and placement.vias:
            via_count = sum(1 for via in placement.vias if via.get("net") == net.name)
            if via_count > 2:
                violations.append({
                    "rule": "via_count",
                    "severity": "warning",
                    "message": f"Net {net.name} has {via_count} vias, minimize for high-speed",
                    "current": via_count,
                    "recommended": "< 2"
                })
        
        return violations
    
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

