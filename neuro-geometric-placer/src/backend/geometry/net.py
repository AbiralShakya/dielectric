"""
Net representation for PCB connectivity
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Net:
    """Represents a net (connection) between component pins."""
    name: str
    pins: List[Tuple[str, str]]  # List of (component_name, pin_name) tuples
    
    def get_pin_count(self) -> int:
        """Get number of pins in this net."""
        return len(self.pins)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "pins": self.pins
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Net':
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            pins=[tuple(p) for p in data["pins"]]
        )

