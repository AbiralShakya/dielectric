"""
Placement state management
"""

import numpy as np
from typing import List, Dict, Optional
from .component import Component
from .board import Board
from .net import Net


class Placement:
    """Manages component placement state."""
    
    def __init__(self, components: List[Component], board: Board, nets: List[Net]):
        """
        Initialize placement.
        
        Args:
            components: List of components to place
            board: Board constraints
            nets: List of nets (connections)
        """
        self.components = {c.name: c for c in components}
        self.board = board
        self.nets = {n.name: n for n in nets}
        
        # Build net index for fast lookup
        self._net_index: Dict[str, List[str]] = {}
        for net_name, net in self.nets.items():
            for comp_name, pin_name in net.pins:
                if comp_name not in self._net_index:
                    self._net_index[comp_name] = []
                self._net_index[comp_name].append(net_name)
    
    def get_component(self, name: str) -> Optional[Component]:
        """Get component by name."""
        return self.components.get(name)
    
    def get_affected_nets(self, component_name: str) -> List[str]:
        """Get nets affected by a component move."""
        return self._net_index.get(component_name, [])
    
    def move_component(self, name: str, x: float, y: float, angle: Optional[float] = None):
        """Move a component."""
        comp = self.components.get(name)
        if comp:
            comp.x = x
            comp.y = y
            if angle is not None:
                comp.angle = angle
            comp.placed = True
    
    def swap_components(self, name1: str, name2: str):
        """Swap positions of two components."""
        c1 = self.components.get(name1)
        c2 = self.components.get(name2)
        if c1 and c2:
            x1, y1, a1 = c1.x, c1.y, c1.angle
            c1.x, c1.y, c1.angle = c2.x, c2.y, c2.angle
            c2.x, c2.y, c2.angle = x1, y1, a1
    
    def check_validity(self) -> List[str]:
        """Check placement validity. Returns list of violations."""
        violations = []
        
        # Check board bounds
        for comp in self.components.values():
            if not self.board.contains(comp):
                violations.append(f"{comp.name} outside board bounds")
        
        # Check overlaps
        comp_list = list(self.components.values())
        for i, c1 in enumerate(comp_list):
            for c2 in comp_list[i+1:]:
                if c1.overlaps(c2, clearance=self.board.clearance):
                    violations.append(f"{c1.name} overlaps {c2.name}")
        
        return violations
    
    def copy(self) -> 'Placement':
        """Create a deep copy of this placement."""
        new_components = [Component.from_dict(c.to_dict()) for c in self.components.values()]
        new_board = Board.from_dict(self.board.to_dict())
        new_nets = [Net.from_dict(n.to_dict()) for n in self.nets.values()]
        return Placement(new_components, new_board, new_nets)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "components": [c.to_dict() for c in self.components.values()],
            "board": self.board.to_dict(),
            "nets": [n.to_dict() for n in self.nets.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Placement':
        """Deserialize from dictionary."""
        from .component import Component
        from .board import Board
        from .net import Net
        
        components = [Component.from_dict(c) for c in data["components"]]
        board = Board.from_dict(data["board"])
        nets = [Net.from_dict(n) for n in data["nets"]]
        return cls(components, board, nets)
    
    def randomize(self, seed: Optional[int] = None):
        """Randomize component positions (for initialization)."""
        if seed is not None:
            np.random.seed(seed)
        
        for comp in self.components.values():
            # Random position within board
            margin = max(comp.width, comp.height)
            comp.x = np.random.uniform(margin, self.board.width - margin)
            comp.y = np.random.uniform(margin, self.board.height - margin)
            comp.angle = np.random.choice([0, 90, 180, 270])
            comp.placed = True

