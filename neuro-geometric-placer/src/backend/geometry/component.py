"""
Component representation for PCB placement
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Pin:
    """Represents a pin on a component."""
    name: str
    x_offset: float  # Relative to component center
    y_offset: float
    net: str = ""  # Net name this pin connects to


@dataclass
class Component:
    """PCB component with geometry and properties."""
    name: str
    package: str  # e.g., "BGA256", "0805", "SOIC-8"
    width: float  # mm
    height: float  # mm
    power: float = 0.0  # Watts
    pins: List[Pin] = field(default_factory=list)
    
    # Placement state
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0  # Rotation in degrees
    placed: bool = False
    
    def __post_init__(self):
        """Initialize default pins if none provided."""
        if not self.pins:
            # Default: 4 corner pins for rectangular components
            w2, h2 = self.width / 2, self.height / 2
            self.pins = [
                Pin("pin1", -w2, -h2),
                Pin("pin2", w2, -h2),
                Pin("pin3", w2, h2),
                Pin("pin4", -w2, h2),
            ]
    
    def get_pin_position(self, pin: Pin) -> Tuple[float, float]:
        """Get absolute position of a pin after rotation."""
        # Rotate pin offset
        angle_rad = np.radians(self.angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        x_rot = pin.x_offset * cos_a - pin.y_offset * sin_a
        y_rot = pin.x_offset * sin_a + pin.y_offset * cos_a
        
        return (self.x + x_rot, self.y + y_rot)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x_min, y_min, x_max, y_max)."""
        # Rotate corners
        w2, h2 = self.width / 2, self.height / 2
        corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
        
        angle_rad = np.radians(self.angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        rotated_corners = []
        for x, y in corners:
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            rotated_corners.append((self.x + x_rot, self.y + y_rot))
        
        xs = [c[0] for c in rotated_corners]
        ys = [c[1] for c in rotated_corners]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def overlaps(self, other: 'Component', clearance: float = 0.5) -> bool:
        """Check if this component overlaps with another."""
        x1_min, y1_min, x1_max, y1_max = self.get_bounds()
        x2_min, y2_min, x2_max, y2_max = other.get_bounds()
        
        # Add clearance
        x1_min -= clearance
        y1_min -= clearance
        x1_max += clearance
        y1_max += clearance
        
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "package": self.package,
            "width": self.width,
            "height": self.height,
            "power": self.power,
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "placed": self.placed,
            "pins": [{"name": p.name, "x_offset": p.x_offset, 
                     "y_offset": p.y_offset, "net": p.net} 
                    for p in self.pins]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Component':
        """Deserialize from dictionary."""
        pins = [Pin(**p) for p in data.get("pins", [])]
        comp = cls(
            name=data["name"],
            package=data["package"],
            width=data["width"],
            height=data["height"],
            power=data.get("power", 0.0),
            pins=pins
        )
        comp.x = data.get("x", 0.0)
        comp.y = data.get("y", 0.0)
        comp.angle = data.get("angle", 0.0)
        comp.placed = data.get("placed", False)
        return comp

