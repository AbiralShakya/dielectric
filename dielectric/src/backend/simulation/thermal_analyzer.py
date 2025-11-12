"""
Thermal Analysis

Features:
- Heat map generation
- Component temperature estimation
- Thermal via placement
- Airflow simulation
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component

logger = logging.getLogger(__name__)


class ThermalAnalyzer:
    """
    Thermal Analysis for PCB designs.
    
    Features:
    - Heat map generation
    - Component temperature estimation
    - Thermal via recommendations
    """
    
    def __init__(
        self,
        ambient_temp: float = 25.0,  # °C
        board_thermal_conductivity: float = 0.3  # W/(m·K) (FR4)
    ):
        """
        Initialize thermal analyzer.
        
        Args:
            ambient_temp: Ambient temperature (°C)
            board_thermal_conductivity: Board thermal conductivity
        """
        self.ambient_temp = ambient_temp
        self.k = board_thermal_conductivity
    
    def generate_heat_map(
        self,
        placement: Placement,
        resolution: int = 50
    ) -> Dict:
        """
        Generate thermal heat map.
        
        Args:
            placement: Placement with components
            resolution: Grid resolution
        
        Returns:
            Heat map data
        """
        board = placement.board
        
        # Create grid
        x_grid = np.linspace(0, board.width, resolution)
        y_grid = np.linspace(0, board.height, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize temperature grid
        temp_grid = np.zeros_like(X) + self.ambient_temp
        
        # Add heat sources (components)
        for comp_name, comp in placement.components.items():
            # Estimate power dissipation (simplified)
            power = self._estimate_power(comp)
            
            if power > 0:
                # Calculate temperature rise (simplified)
                # T = T_ambient + P * R_thermal
                # Simplified: use distance-based decay
                dx = X - comp.x
                dy = Y - comp.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Temperature rise (decays with distance)
                temp_rise = power * 10.0 * np.exp(-distance / 10.0)  # Simplified
                temp_grid += temp_rise
        
        # Find hotspots
        max_temp = np.max(temp_grid)
        hotspot_threshold = self.ambient_temp + 20.0  # 20°C above ambient
        hotspots = np.where(temp_grid > hotspot_threshold)
        
        return {
            "x_grid": x_grid.tolist(),
            "y_grid": y_grid.tolist(),
            "temperature": temp_grid.tolist(),
            "max_temperature": float(max_temp),
            "min_temperature": float(np.min(temp_grid)),
            "hotspots": [
                {"x": float(x_grid[i]), "y": float(y_grid[j]), "temp": float(temp_grid[j, i])}
                for i, j in zip(hotspots[1], hotspots[0])
            ]
        }
    
    def estimate_component_temperature(
        self,
        placement: Placement,
        component: Component
    ) -> float:
        """
        Estimate component operating temperature.
        
        Args:
            placement: Placement with components
            component: Component to analyze
        
        Returns:
            Estimated temperature (°C)
        """
        # Estimate power dissipation
        power = self._estimate_power(component)
        
        # Thermal resistance (simplified)
        # R_thermal = 1 / (h * A)
        # where h = heat transfer coefficient, A = surface area
        area = component.width * component.height / 1e6  # m²
        h = 10.0  # W/(m²·K) (natural convection)
        r_thermal = 1.0 / (h * area)
        
        # Temperature rise
        temp_rise = power * r_thermal
        
        # Add ambient
        temp = self.ambient_temp + temp_rise
        
        return temp
    
    def recommend_thermal_vias(
        self,
        placement: Placement,
        component: Component,
        max_temp: float = 85.0  # °C
    ) -> List[Tuple[float, float]]:
        """
        Recommend thermal via placement.
        
        Args:
            placement: Placement with components
            component: Component needing thermal vias
            max_temp: Maximum allowed temperature (°C)
        
        Returns:
            List of recommended via positions (x, y)
        """
        temp = self.estimate_component_temperature(placement, component)
        
        if temp < max_temp:
            return []  # No vias needed
        
        # Place thermal vias under component
        via_positions = []
        
        # Grid of vias under component
        via_spacing = 2.0  # mm
        x_start = component.x - component.width/2 + via_spacing
        x_end = component.x + component.width/2 - via_spacing
        y_start = component.y - component.height/2 + via_spacing
        y_end = component.y + component.height/2 - via_spacing
        
        x_positions = np.arange(x_start, x_end, via_spacing)
        y_positions = np.arange(y_start, y_end, via_spacing)
        
        for x in x_positions:
            for y in y_positions:
                via_positions.append((float(x), float(y)))
        
        return via_positions
    
    def _estimate_power(self, component: Component) -> float:
        """Estimate component power dissipation."""
        # Simplified power estimation based on component type
        comp_name_lower = component.name.lower()
        
        # High-power components
        if any(kw in comp_name_lower for kw in ["regulator", "ldo", "buck", "boost"]):
            return 1.0  # W
        elif any(kw in comp_name_lower for kw in ["mosfet", "transistor", "switch"]):
            return 0.5  # W
        elif any(kw in comp_name_lower for kw in ["mcu", "processor", "cpu"]):
            return 0.3  # W
        elif any(kw in comp_name_lower for kw in ["opamp", "amplifier"]):
            return 0.1  # W
        else:
            return 0.01  # W (low power)

