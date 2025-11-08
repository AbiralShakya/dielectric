"""
Test full pipeline
"""

import pytest
import asyncio
from backend.geometry.component import Component
from backend.geometry.board import Board
from backend.geometry.net import Net
from backend.geometry.placement import Placement
from backend.agents.orchestrator import AgentOrchestrator


@pytest.mark.asyncio
async def test_fast_path_pipeline():
    """Test fast path optimization pipeline."""
    # Create test placement
    components = [
        Component("U1", "BGA", 10.0, 10.0, power=2.0),
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
    
    # Run optimization
    orchestrator = AgentOrchestrator()
    result = await orchestrator.optimize_fast(
        placement,
        "Optimize for minimal trace length"
    )
    
    assert result["success"]
    assert "placement" in result
    assert "score" in result
    assert "weights" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

