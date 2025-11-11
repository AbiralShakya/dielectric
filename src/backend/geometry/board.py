"""
Board representation and constraints with multi-layer support
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from .component import Component


@dataclass
class LayerStackup:
    """PCB layer stackup definition."""
    layer_count: int = 2
    layers: List[Dict] = field(default_factory=lambda: [
        {"name": "F.Cu", "type": "signal", "thickness": 0.035},  # Top copper
        {"name": "B.Cu", "type": "signal", "thickness": 0.035}   # Bottom copper
    ])
    dielectric_thickness: float = 1.6  # mm (total board thickness)
    dielectric_constant: float = 4.5  # FR4
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names."""
        return [layer["name"] for layer in self.layers]
    
    def get_signal_layers(self) -> List[str]:
        """Get list of signal layer names."""
        return [layer["name"] for layer in self.layers if layer["type"] == "signal"]
    
    def get_power_layers(self) -> List[str]:
        """Get list of power plane layer names."""
        return [layer["name"] for layer in self.layers if layer["type"] == "power"]
    
    def get_ground_layers(self) -> List[str]:
        """Get list of ground plane layer names."""
        return [layer["name"] for layer in self.layers if layer["type"] == "ground"]


@dataclass
class Board:
    """
    PCB board with dimensions, constraints, and multi-layer support.
    
    Supports 2-layer, 4-layer, 6-layer, and higher layer counts with
    proper stackup definitions for impedance control and power distribution.
    """
    width: float  # mm
    height: float  # mm
    clearance: float = 0.5  # Minimum clearance between components (mm)
    layer_stackup: Optional[LayerStackup] = None
    
    def __post_init__(self):
        """Initialize default layer stackup if not provided."""
        if self.layer_stackup is None:
            self.layer_stackup = LayerStackup(layer_count=2)
    
    def contains(self, component: Component) -> bool:
        """Check if component is within board bounds."""
        x_min, y_min, x_max, y_max = component.get_bounds()
        return (x_min >= 0 and y_min >= 0 and 
                x_max <= self.width and y_max <= self.height)
    
    def get_valid_placement_area(self) -> Tuple[float, float, float, float]:
        """Get valid placement area (accounting for component sizes)."""
        # For now, return full board
        # In production, subtract keep-out zones
        return (0.0, 0.0, self.width, self.height)
    
    def get_layer_count(self) -> int:
        """Get number of layers."""
        return self.layer_stackup.layer_count if self.layer_stackup else 2
    
    def create_4_layer_stackup(self) -> LayerStackup:
        """Create standard 4-layer stackup."""
        return LayerStackup(
            layer_count=4,
            layers=[
                {"name": "F.Cu", "type": "signal", "thickness": 0.035},
                {"name": "In1.Cu", "type": "ground", "thickness": 0.035},
                {"name": "In2.Cu", "type": "power", "thickness": 0.035},
                {"name": "B.Cu", "type": "signal", "thickness": 0.035}
            ],
            dielectric_thickness=1.6
        )
    
    def create_6_layer_stackup(self) -> LayerStackup:
        """Create standard 6-layer stackup."""
        return LayerStackup(
            layer_count=6,
            layers=[
                {"name": "F.Cu", "type": "signal", "thickness": 0.035},
                {"name": "In1.Cu", "type": "ground", "thickness": 0.035},
                {"name": "In2.Cu", "type": "signal", "thickness": 0.035},
                {"name": "In3.Cu", "type": "signal", "thickness": 0.035},
                {"name": "In4.Cu", "type": "power", "thickness": 0.035},
                {"name": "B.Cu", "type": "signal", "thickness": 0.035}
            ],
            dielectric_thickness=1.6
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "width": self.width,
            "height": self.height,
            "clearance": self.clearance,
            "layer_count": self.get_layer_count()
        }
        if self.layer_stackup:
            result["layer_stackup"] = {
                "layer_count": self.layer_stackup.layer_count,
                "layers": self.layer_stackup.layers,
                "dielectric_thickness": self.layer_stackup.dielectric_thickness,
                "dielectric_constant": self.layer_stackup.dielectric_constant
            }
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Board':
        """Deserialize from dictionary."""
        board = cls(
            width=data["width"],
            height=data["height"],
            clearance=data.get("clearance", 0.5)
        )
        
        if "layer_stackup" in data:
            stackup_data = data["layer_stackup"]
            board.layer_stackup = LayerStackup(
                layer_count=stackup_data.get("layer_count", 2),
                layers=stackup_data.get("layers", []),
                dielectric_thickness=stackup_data.get("dielectric_thickness", 1.6),
                dielectric_constant=stackup_data.get("dielectric_constant", 4.5)
            )
        elif "layer_count" in data and data["layer_count"] > 2:
            # Auto-create stackup based on layer count
            layer_count = data["layer_count"]
            if layer_count == 4:
                board.layer_stackup = board.create_4_layer_stackup()
            elif layer_count >= 6:
                board.layer_stackup = board.create_6_layer_stackup()
        
        return board

