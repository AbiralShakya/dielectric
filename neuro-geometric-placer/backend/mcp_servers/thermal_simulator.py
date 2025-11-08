"""
Thermal Simulator MCP Server

Generates heatmaps for placements.
"""

import numpy as np
from typing import Dict, Any
from backend.geometry.placement import Placement


class ThermalSimulatorMCP:
    """MCP server for thermal simulation."""
    
    def __init__(self):
        """Initialize thermal simulator MCP server."""
        self.name = "thermal_simulator"
    
    def generate_heatmap(
        self,
        placement_data: Dict,
        grid_size: int = 64
    ) -> Dict[str, Any]:
        """
        Generate thermal heatmap using computational geometry.

        Args:
            placement_data: Placement dictionary
            grid_size: Heatmap grid size (default 64x64)

        Returns:
            {
                "heatmap": 2D array of heat values,
                "min": float,
                "max": float,
                "computation": str
            }
        """
        placement = Placement.from_dict(placement_data)

        # Create grid
        heatmap = np.zeros((grid_size, grid_size))

        # Grid to board coordinates
        x_scale = placement.board.width / grid_size
        y_scale = placement.board.height / grid_size

        # Add heat contribution from each component
        for comp in placement.components.values():
            if comp.power <= 0:
                continue

            # Component center in grid coordinates
            grid_x = int(comp.x / x_scale)
            grid_y = int(comp.y / y_scale)

            # Add heat with Gaussian falloff (computational geometry)
            for i in range(max(0, grid_x-10), min(grid_size, grid_x+10)):
                for j in range(max(0, grid_y-10), min(grid_size, grid_y+10)):
                    dist = np.sqrt((i - grid_x)**2 + (j - grid_y)**2)
                    # Gaussian with sigma = 5 grid cells
                    heat = comp.power * np.exp(-(dist**2) / (2 * 5**2))
                    heatmap[i, j] += heat

        return {
            "heatmap": heatmap.tolist(),
            "min": float(np.min(heatmap)),
            "max": float(np.max(heatmap)),
            "grid_size": grid_size,
            "computation": "gaussian_heat_convolution",
            "total_power_components": len([c for c in placement.components.values() if c.power > 0])
        }
    
    def get_tool_definition(self) -> Dict:
        """Get MCP tool definition."""
        return {
            "name": "generate_heatmap",
            "description": "Generate thermal heatmap for placement",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement_data": {"type": "object"},
                    "grid_size": {"type": "integer", "default": 64}
                },
                "required": ["placement_data"]
            }
        }

