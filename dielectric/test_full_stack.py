#!/usr/bin/env python3
"""
Full Stack Test

Tests the complete pipeline from geometry to optimization.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.geometry.placement import Placement
from backend.agents.orchestrator import AgentOrchestrator


async def test_full_pipeline():
    """Test complete optimization pipeline."""
    print("🧪 Testing Full Stack Pipeline")
    print("=" * 50)
    
    # Load example board
    example_path = os.path.join(os.path.dirname(__file__), "examples", "simple_board.json")
    with open(example_path, "r") as f:
        placement_data = json.load(f)
    
    placement = Placement.from_dict(placement_data)
    print(f"✅ Loaded placement: {len(placement.components)} components")
    
    # Run fast path optimization
    print("\n🚀 Running fast path optimization...")
    orchestrator = AgentOrchestrator()
    
    result = await orchestrator.optimize_fast(
        placement,
        "Optimize for minimal trace length, but keep high-power components cool"
    )
    
    if result["success"]:
        print("✅ Optimization successful!")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Weights: {result['weights']}")
        print(f"   Intent: {result['intent_explanation']}")
        
        if result.get("stats"):
            stats = result["stats"]
            print(f"   Iterations: {stats.get('iterations', 0)}")
            print(f"   Time: {stats.get('time_ms', 0):.1f} ms")
        
        # Check verification
        if result.get("verification"):
            verification = result["verification"]
            print(f"   Violations: {len(verification.get('violations', []))}")
            print(f"   Warnings: {len(verification.get('warnings', []))}")
    else:
        print("❌ Optimization failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("✅ Full stack test completed!")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())

