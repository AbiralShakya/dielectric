"""
Test geometry module
"""

import pytest
import numpy as np
from backend.geometry.component import Component, Pin
from backend.geometry.board import Board
from backend.geometry.net import Net
from backend.geometry.placement import Placement


def test_component_creation():
    """Test component creation."""
    comp = Component(
        name="U1",
        package="BGA256",
        width=15.0,
        height=15.0,
        power=2.0
    )
    
    assert comp.name == "U1"
    assert comp.width == 15.0
    assert comp.power == 2.0
    assert len(comp.pins) == 4  # Default pins


def test_component_bounds():
    """Test component bounds calculation."""
    comp = Component("R1", "0805", 2.0, 1.25)
    comp.x, comp.y, comp.angle = 10.0, 20.0, 0.0
    
    x_min, y_min, x_max, y_max = comp.get_bounds()
    assert x_min == 9.0  # 10 - 2/2
    assert y_max == 20.625  # 20 + 1.25/2


def test_component_overlap():
    """Test component overlap detection."""
    c1 = Component("C1", "0805", 2.0, 1.25)
    c1.x, c1.y = 10.0, 10.0
    
    c2 = Component("C2", "0805", 2.0, 1.25)
    c2.x, c2.y = 10.5, 10.5  # Overlapping
    
    assert c1.overlaps(c2, clearance=0.5)


def test_board_contains():
    """Test board bounds checking."""
    board = Board(width=100.0, height=100.0)
    comp = Component("U1", "BGA", 10.0, 10.0)
    
    comp.x, comp.y = 50.0, 50.0
    assert board.contains(comp)
    
    comp.x, comp.y = 200.0, 50.0  # Outside
    assert not board.contains(comp)


def test_placement_creation():
    """Test placement creation."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    board = Board(100.0, 100.0)
    nets = [Net("net1", [("U1", "pin1"), ("R1", "pin1")])]
    
    placement = Placement(components, board, nets)
    
    assert len(placement.components) == 2
    assert len(placement.nets) == 1


def test_placement_validity():
    """Test placement validity checking."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    board = Board(100.0, 100.0)
    nets = []
    
    placement = Placement(components, board, nets)
    
    # Valid placement
    components[0].x, components[0].y = 50.0, 50.0
    components[1].x, components[1].y = 60.0, 50.0
    
    violations = placement.check_validity()
    assert len(violations) == 0
    
    # Invalid: outside bounds
    components[0].x, components[0].y = 200.0, 200.0
    violations = placement.check_validity()
    assert len(violations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

