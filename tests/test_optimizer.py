"""
Test optimization module
"""

import pytest
from backend.geometry.component import Component
from backend.geometry.board import Board
from backend.geometry.net import Net
from backend.geometry.placement import Placement
from backend.scoring.scorer import WorldModelScorer, ScoreWeights
from backend.scoring.incremental_scorer import IncrementalScorer
from backend.optimization.simulated_annealing import SimulatedAnnealing
from backend.optimization.local_placer import LocalPlacer


def test_simulated_annealing():
    """Test simulated annealing optimizer."""
    # Create simple placement
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25),
        Component("R2", "0805", 2.0, 1.25)
    ]
    
    board = Board(100.0, 100.0)
    nets = [
        Net("net1", [("U1", "pin1"), ("R1", "pin1")]),
        Net("net2", [("R1", "pin2"), ("R2", "pin1")])
    ]
    
    placement = Placement(components, board, nets)
    placement.randomize(seed=42)
    
    # Setup scorer
    weights = ScoreWeights()
    base_scorer = WorldModelScorer(weights)
    inc_scorer = IncrementalScorer(base_scorer)
    
    # Run optimization
    optimizer = SimulatedAnnealing(
        scorer=inc_scorer,
        initial_temp=50.0,
        final_temp=0.1,
        cooling_rate=0.9,
        max_iterations=100  # Small for testing
    )
    
    best_placement, best_score, stats = optimizer.optimize(placement)
    
    assert best_placement is not None
    assert best_score >= 0
    assert stats["iterations"] > 0


def test_local_placer():
    """Test local placer (fast path)."""
    components = [
        Component("U1", "BGA", 10.0, 10.0),
        Component("R1", "0805", 2.0, 1.25)
    ]
    
    board = Board(100.0, 100.0)
    nets = [Net("net1", [("U1", "pin1"), ("R1", "pin1")])]
    
    placement = Placement(components, board, nets)
    placement.randomize(seed=42)
    
    weights = ScoreWeights()
    base_scorer = WorldModelScorer(weights)
    inc_scorer = IncrementalScorer(base_scorer)
    
    placer = LocalPlacer(inc_scorer)
    
    best_placement, best_score, stats = placer.optimize_fast(placement, max_time_ms=100.0)
    
    assert best_placement is not None
    assert best_score >= 0
    assert "time_ms" in stats
    assert stats["time_ms"] < 200.0  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

