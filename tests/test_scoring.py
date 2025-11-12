"""
Test scoring module
"""

import pytest
from backend.geometry.component import Component
from backend.geometry.board import Board
from backend.geometry.net import Net
from backend.geometry.placement import Placement
from backend.scoring.scorer import WorldModelScorer, ScoreWeights
from backend.scoring.incremental_scorer import IncrementalScorer


def test_score_weights():
    """Test score weights."""
    weights = ScoreWeights(alpha=0.6, beta=0.3, gamma=0.1)
    weights.normalize()
    
    assert abs(weights.alpha + weights.beta + weights.gamma - 1.0) < 1e-6


def test_trace_length():
    """Test trace length calculation."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    components[0].x, components[0].y = 10.0, 10.0
    components[1].x, components[1].y = 50.0, 50.0
    
    board = Board(100.0, 100.0)
    nets = [Net("net1", [("U1", "pin1"), ("R1", "pin1")])]
    
    placement = Placement(components, board, nets)
    
    scorer = WorldModelScorer()
    trace_length = scorer.compute_trace_length(placement)
    
    assert trace_length > 0


def test_thermal_density():
    """Test thermal density calculation."""
    components = [
        Component("U1", "BGA", 10.0, 10.0, power=2.0),
        Component("U2", "BGA", 10.0, 10.0, power=1.5)
    ]
    components[0].x, components[0].y = 10.0, 10.0
    components[1].x, components[1].y = 15.0, 15.0  # Close together
    
    board = Board(100.0, 100.0)
    nets = []
    
    placement = Placement(components, board, nets)
    
    scorer = WorldModelScorer()
    thermal = scorer.compute_thermal_density(placement)
    
    assert thermal > 0


def test_clearance_violations():
    """Test clearance violation calculation."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("U2", "BGA", 10.0, 10.0)
    ]
    components[0].x, components[0].y = 10.0, 10.0
    components[1].x, components[1].y = 12.0, 12.0  # Overlapping
    
    board = Board(100.0, 100.0, clearance=0.5)
    nets = []
    
    placement = Placement(components, board, nets)
    
    scorer = WorldModelScorer()
    violations = scorer.compute_clearance_violations(placement)
    
    assert violations > 0


def test_composite_score():
    """Test composite score calculation."""
    components = [
        Component("U1", "BGA", 10.0, 10.0, power=2.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    components[0].x, components[0].y = 10.0, 10.0
    components[1].x, components[1].y = 50.0, 50.0
    
    board = Board(100.0, 100.0)
    nets = [Net("net1", [("U1", "pin1"), ("R1", "pin1")])]
    
    placement = Placement(components, board, nets)
    
    weights = ScoreWeights(alpha=0.5, beta=0.3, gamma=0.2)
    scorer = WorldModelScorer(weights)
    
    score = scorer.score(placement)
    assert score >= 0
    
    breakdown = scorer.score_breakdown(placement)
    assert "trace_length" in breakdown
    assert "thermal_density" in breakdown
    assert "clearance_violations" in breakdown


def test_incremental_scoring():
    """Test incremental scoring."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    components[0].x, components[0].y = 10.0, 10.0
    components[1].x, components[1].y = 50.0, 50.0
    
    board = Board(100.0, 100.0)
    nets = [Net("net1", [("U1", "pin1"), ("R1", "pin1")])]
    
    placement = Placement(components, board, nets)
    
    weights = ScoreWeights()
    base_scorer = WorldModelScorer(weights)
    inc_scorer = IncrementalScorer(base_scorer)
    
    # Compute delta for a move
    delta = inc_scorer.compute_delta_score(
        placement, "R1",
        50.0, 50.0, 0.0,  # Old position
        60.0, 50.0, 0.0   # New position
    )
    
    assert isinstance(delta, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

