"""
Board representation and constraints
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from .component import Component


@dataclass
class Board:
    """PCB board with dimensions and constraints."""
    width: float  # mm
    height: float  # mm
    clearance: float = 0.5  # Minimum clearance between components (mm)
    
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
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "clearance": self.clearance
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Board':
        """Deserialize from dictionary."""
        return cls(
            width=data["width"],
            height=data["height"],
            clearance=data.get("clearance", 0.5)
        )

