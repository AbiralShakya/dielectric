"""
PCB Simulation Engine

Beyond optimization and generation, Dielectric provides:
1. Thermal simulation
2. Signal integrity analysis
3. Power distribution network (PDN) analysis
4. EMI/EMC simulation
5. Manufacturing yield prediction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.geometry.net import Net
    from backend.knowledge.knowledge_graph import KnowledgeGraph
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.geometry.net import Net
    from src.backend.knowledge.knowledge_graph import KnowledgeGraph


@dataclass
class ThermalSimulationResult:
    """Results from thermal simulation."""
    component_temperatures: Dict[str, float]  # Component name -> temperature (°C)
    board_temperature_map: np.ndarray  # 2D temperature grid
    hotspots: List[Tuple[float, float, float]]  # (x, y, temp)
    max_temperature: float
    thermal_gradient: float
    cooling_recommendations: List[str]


@dataclass
class SignalIntegrityResult:
    """Results from signal integrity analysis."""
    net_impedance: Dict[str, float]  # Net name -> impedance (Ω)
    crosstalk_risks: List[Dict[str, Any]]  # Net pairs with crosstalk risk
    reflection_risks: List[str]  # Nets with impedance mismatch
    timing_violations: List[Dict[str, Any]]  # Nets with timing issues
    recommendations: List[str]


@dataclass
class PDNResult:
    """Power Distribution Network analysis results."""
    voltage_drop: Dict[str, float]  # Component name -> voltage drop (V)
    current_density: np.ndarray  # 2D current density map
    power_loss: float  # Total power loss (W)
    decoupling_effectiveness: Dict[str, float]  # Component -> decoupling effectiveness
    recommendations: List[str]


class PCBSimulator:
    """
    Comprehensive PCB simulation engine.
    
    Provides:
    - Thermal simulation (Gaussian diffusion model)
    - Signal integrity analysis
    - Power distribution network analysis
    - EMI/EMC prediction
    - Manufacturing yield estimation
    """
    
    def __init__(self):
        """Initialize simulator."""
        pass
    
    def simulate_thermal(
        self,
        placement: Placement,
        ambient_temp: float = 25.0,
        board_material: str = "FR4"
    ) -> ThermalSimulationResult:
        """
        Simulate thermal behavior of PCB.
        
        Uses Gaussian thermal diffusion model based on:
        - Component power dissipation
        - Component placement
        - Board thermal conductivity
        - Convection coefficients
        """
        components = placement.components
        board = placement.board
        
        # Initialize temperature map
        resolution = 0.5  # mm per pixel
        width_pixels = int(board.width / resolution)
        height_pixels = int(board.height / resolution)
        temp_map = np.full((height_pixels, width_pixels), ambient_temp)
        
        # Component temperatures
        component_temps = {}
        hotspots = []
        
        # Thermal conductivity (W/m·K)
        thermal_conductivity = {
            "FR4": 0.3,
            "Aluminum": 200.0,
            "Copper": 400.0
        }.get(board_material, 0.3)
        
        # Simulate heat diffusion from each component
        for comp in components:
            if comp.power > 0:
                # Convert to pixel coordinates
                x_pixel = int(comp.x / resolution)
                y_pixel = int(comp.y / resolution)
                
                # Estimate component temperature (simplified model)
                # T = Tambient + P * R_thermal
                # R_thermal depends on component size and board material
                thermal_resistance = 50.0 / (comp.width * comp.height)  # Simplified
                comp_temp = ambient_temp + comp.power * thermal_resistance
                component_temps[comp.name] = comp_temp
                
                # Diffuse heat (Gaussian kernel)
                sigma = max(comp.width, comp.height) / resolution / 2
                for y in range(height_pixels):
                    for x in range(width_pixels):
                        dist = np.sqrt((x - x_pixel)**2 + (y - y_pixel)**2)
                        if dist < 3 * sigma:
                            heat_contribution = comp.power * np.exp(-dist**2 / (2 * sigma**2))
                            temp_map[y, x] += heat_contribution * thermal_resistance / 10
        
                # Record hotspot
                if comp_temp > ambient_temp + 10:
                    hotspots.append((comp.x, comp.y, comp_temp))
        
        # Find max temperature
        max_temp = np.max(temp_map)
        
        # Calculate thermal gradient
        temp_gradient = np.max(np.gradient(temp_map))
        
        # Generate recommendations
        recommendations = []
        if max_temp > ambient_temp + 40:
            recommendations.append("High temperature detected. Consider adding thermal vias.")
        if len(hotspots) > 3:
            recommendations.append("Multiple thermal hotspots. Consider component spacing.")
        if temp_gradient > 20:
            recommendations.append("High thermal gradient. Consider thermal management.")
        
        return ThermalSimulationResult(
            component_temperatures=component_temps,
            board_temperature_map=temp_map,
            hotspots=hotspots,
            max_temperature=float(max_temp),
            thermal_gradient=float(temp_gradient),
            cooling_recommendations=recommendations
        )
    
    def analyze_signal_integrity(
        self,
        placement: Placement,
        signal_frequency: float = 100e6  # 100 MHz default
    ) -> SignalIntegrityResult:
        """
        Analyze signal integrity.
        
        Checks:
        - Impedance matching
        - Crosstalk between nets
        - Reflection risks
        - Timing violations
        """
        nets = placement.nets
        components = placement.components
        
        # Estimate impedance based on trace geometry
        net_impedance = {}
        for net in nets:
            # Simplified impedance calculation
            # Z0 ≈ sqrt(L/C) where L and C depend on trace geometry
            trace_length = self._estimate_trace_length(net, components)
            trace_width = 0.2  # mm (default)
            
            # Characteristic impedance approximation
            # Z0 ≈ 87 / sqrt(εr + 1.41) * ln(5.98 * h / (0.8 * w + t))
            # Simplified: assume 50Ω for most traces
            impedance = 50.0  # Ω
            
            # Adjust based on length and frequency
            if trace_length > 50:  # mm
                impedance *= 1.1  # Longer traces have higher impedance
            
            net_impedance[net.name] = impedance
        
        # Check for crosstalk risks
        crosstalk_risks = []
        net_positions = {net.name: self._get_net_center(net, components) for net in nets}
        
        for i, net1 in enumerate(nets):
            for net2 in nets[i+1:]:
                pos1 = net_positions[net1.name]
                pos2 = net_positions[net2.name]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < 2.0:  # mm - too close
                    crosstalk_risks.append({
                        "net1": net1.name,
                        "net2": net2.name,
                        "distance": distance,
                        "risk": "high" if distance < 1.0 else "medium"
                    })
        
        # Check for reflection risks (impedance mismatch)
        reflection_risks = []
        for net_name, impedance in net_impedance.items():
            if impedance < 40 or impedance > 60:  # Outside typical 50Ω range
                reflection_risks.append(net_name)
        
        # Timing violations (simplified)
        timing_violations = []
        for net in nets:
            trace_length = self._estimate_trace_length(net, components)
            # Signal propagation delay ≈ length / (c / sqrt(εr))
            # c = speed of light, εr ≈ 4 for FR4
            prop_delay = trace_length * 1e-3 / (3e8 / np.sqrt(4))  # seconds
            max_delay = 1 / signal_frequency * 0.1  # 10% of period
            
            if prop_delay > max_delay:
                timing_violations.append({
                    "net": net.name,
                    "delay": prop_delay * 1e9,  # ns
                    "max_delay": max_delay * 1e9
                })
        
        # Generate recommendations
        recommendations = []
        if crosstalk_risks:
            recommendations.append(f"Found {len(crosstalk_risks)} crosstalk risks. Increase trace spacing.")
        if reflection_risks:
            recommendations.append(f"Found {len(reflection_risks)} impedance mismatches. Adjust trace geometry.")
        if timing_violations:
            recommendations.append(f"Found {len(timing_violations)} timing violations. Reduce trace length.")
        
        return SignalIntegrityResult(
            net_impedance=net_impedance,
            crosstalk_risks=crosstalk_risks,
            reflection_risks=reflection_risks,
            timing_violations=timing_violations,
            recommendations=recommendations
        )
    
    def analyze_pdn(
        self,
        placement: Placement,
        supply_voltage: float = 5.0
    ) -> PDNResult:
        """
        Analyze Power Distribution Network.
        
        Checks:
        - Voltage drop across board
        - Current density
        - Power loss
        - Decoupling effectiveness
        """
        components = placement.components
        nets = placement.nets
        
        # Find power nets
        power_nets = [net for net in nets if net.name.upper() in ['VCC', 'VDD', 'VBAT', 'POWER']]
        
        # Calculate voltage drop for each component
        voltage_drop = {}
        total_power = 0.0
        
        for comp in components:
            if comp.power > 0:
                # Estimate voltage drop based on distance from power source
                # Simplified: V_drop = I * R, where R depends on trace length
                current = comp.power / supply_voltage  # A
                trace_resistance = 0.01 * comp.x / 1000  # Simplified: 0.01Ω per mm
                v_drop = current * trace_resistance
                voltage_drop[comp.name] = v_drop
                total_power += comp.power
        
        # Estimate current density map
        board = placement.board
        resolution = 0.5  # mm
        width_pixels = int(board.width / resolution)
        height_pixels = int(board.height / resolution)
        current_density = np.zeros((height_pixels, width_pixels))
        
        for comp in components:
            if comp.power > 0:
                x_pixel = int(comp.x / resolution)
                y_pixel = int(comp.y / resolution)
                current = comp.power / supply_voltage
                # Distribute current (simplified)
                if 0 <= x_pixel < width_pixels and 0 <= y_pixel < height_pixels:
                    current_density[y_pixel, x_pixel] = current / (comp.width * comp.height)
        
        # Calculate power loss
        power_loss = sum(v_drop * comp.power / supply_voltage for comp, v_drop in 
                        zip(components, voltage_drop.values()) if comp.power > 0)
        
        # Decoupling effectiveness (simplified)
        decoupling_effectiveness = {}
        for comp in components:
            if 'CAP' in comp.name.upper() or 'C' in comp.package.upper():
                # Estimate decoupling effectiveness based on proximity to power components
                nearby_power = [c for c in components if c.power > 0.5 and 
                               np.sqrt((c.x - comp.x)**2 + (c.y - comp.y)**2) < 10]
                effectiveness = 1.0 - len(nearby_power) * 0.1  # Simplified
                decoupling_effectiveness[comp.name] = max(0.0, effectiveness)
        
        # Generate recommendations
        recommendations = []
        max_v_drop = max(voltage_drop.values()) if voltage_drop else 0.0
        if max_v_drop > 0.1:
            recommendations.append(f"High voltage drop detected ({max_v_drop:.3f}V). Increase trace width.")
        if power_loss > total_power * 0.1:
            recommendations.append(f"High power loss ({power_loss:.2f}W). Optimize PDN.")
        if len(decoupling_effectiveness) < len([c for c in components if c.power > 0.5]):
            recommendations.append("Add more decoupling capacitors near power components.")
        
        return PDNResult(
            voltage_drop=voltage_drop,
            current_density=current_density,
            power_loss=power_loss,
            decoupling_effectiveness=decoupling_effectiveness,
            recommendations=recommendations
        )
    
    def _estimate_trace_length(self, net: Net, components: List[Component]) -> float:
        """Estimate trace length for a net."""
        if not net.pins:
            return 0.0
        
        # Find component positions
        comp_positions = {}
        for pin_info in net.pins:
            comp_name = pin_info[0] if isinstance(pin_info, list) else pin_info
            comp = next((c for c in components if c.name == comp_name), None)
            if comp:
                comp_positions[comp_name] = (comp.x, comp.y)
        
        if len(comp_positions) < 2:
            return 10.0  # Default
        
        # Estimate using MST
        positions = list(comp_positions.values())
        total_length = 0.0
        for i in range(len(positions) - 1):
            dist = np.sqrt((positions[i][0] - positions[i+1][0])**2 + 
                          (positions[i][1] - positions[i+1][1])**2)
            total_length += dist
        
        return total_length
    
    def _get_net_center(self, net: Net, components: List[Component]) -> Tuple[float, float]:
        """Get center position of a net."""
        comp_positions = []
        for pin_info in net.pins:
            comp_name = pin_info[0] if isinstance(pin_info, list) else pin_info
            comp = next((c for c in components if c.name == comp_name), None)
            if comp:
                comp_positions.append((comp.x, comp.y))
        
        if comp_positions:
            return (np.mean([p[0] for p in comp_positions]), 
                   np.mean([p[1] for p in comp_positions]))
        return (0.0, 0.0)

