#!/usr/bin/env python3
"""
Test MCP Servers

Tests that MCP servers work correctly.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.mcp_servers.placement_scorer import PlacementScorerMCP
from backend.mcp_servers.thermal_simulator import ThermalSimulatorMCP
from backend.mcp_servers.kicad_exporter import KiCadExporterMCP
from backend.geometry.placement import Placement


def test_placement_scorer():
    """Test Placement Scorer MCP."""
    print("🧪 Testing Placement Scorer MCP")
    print("=" * 50)

    try:
        # Create test placement
        components = [
            {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": True},
            {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 30, "angle": 0, "placed": True}
        ]
        board = {"width": 100, "height": 100, "clearance": 0.5}
        nets = [{"name": "net1", "pins": [["U1", "pin1"], ["R1", "pin1"]]}]

        placement_data = {
            "components": components,
            "board": board,
            "nets": nets
        }

        scorer = PlacementScorerMCP()
        move_data = {
            "component_name": "R1",
            "old_x": 50.0, "old_y": 30.0, "old_angle": 0.0,
            "new_x": 60.0, "new_y": 30.0, "new_angle": 0.0,
            "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
        }

        result = scorer.score_delta(placement_data, move_data)

        if "delta" in result and "new_score" in result:
            print("✅ Placement Scorer MCP: SUCCESS")
            print(f"   Delta: {result['delta']:.4f}")
            print(f"   New Score: {result['new_score']:.4f}")
            print(f"   Affected Nets: {result.get('affected_nets', 'N/A')}")
            return True
        else:
            print("❌ Placement Scorer MCP: FAILED - Missing expected fields")
            return False

    except Exception as e:
        print(f"❌ Placement Scorer MCP: EXCEPTION - {str(e)}")
        return False


def test_thermal_simulator():
    """Test Thermal Simulator MCP."""
    print("\n🧪 Testing Thermal Simulator MCP")
    print("=" * 50)

    try:
        # Create test placement
        components = [
            {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 50, "y": 50, "angle": 0, "placed": True},
            {"name": "U2", "package": "BGA", "width": 10, "height": 10, "power": 1.5, "x": 30, "y": 30, "angle": 0, "placed": True}
        ]
        board = {"width": 100, "height": 100, "clearance": 0.5}
        nets = []

        placement_data = {
            "components": components,
            "board": board,
            "nets": nets
        }

        simulator = ThermalSimulatorMCP()
        result = simulator.generate_heatmap(placement_data, grid_size=32)

        if "heatmap" in result and "min" in result and "max" in result:
            print("✅ Thermal Simulator MCP: SUCCESS")
            print(f"   Heatmap Size: {len(result['heatmap'])}x{len(result['heatmap'][0])}")
            print(f"   Min Heat: {result['min']:.4f}")
            print(f"   Max Heat: {result['max']:.4f}")
            print(f"   Computation: {result.get('computation', 'N/A')}")
            return True
        else:
            print("❌ Thermal Simulator MCP: FAILED - Missing expected fields")
            return False

    except Exception as e:
        print(f"❌ Thermal Simulator MCP: EXCEPTION - {str(e)}")
        return False


async def test_kicad_exporter():
    """Test KiCad Exporter MCP."""
    print("\n🧪 Testing KiCad Exporter MCP")
    print("=" * 50)

    try:
        # Create test placement
        components = [
            {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 50, "y": 50, "angle": 0, "placed": True},
            {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 30, "y": 30, "angle": 0, "placed": True}
        ]
        board = {"width": 100, "height": 100, "clearance": 0.5}
        nets = [{"name": "net1", "pins": [["U1", "pin1"], ["R1", "pin1"]]}]

        placement_data = {
            "components": components,
            "board": board,
            "nets": nets
        }

        exporter = KiCadExporterMCP()
        result = await exporter.export(placement_data)

        if "kicad_content" in result and result["format"] == "kicad":
            print("✅ KiCad Exporter MCP: SUCCESS")
            print(f"   Format: {result['format']}")
            print(f"   Content Length: {len(result['kicad_content'])} chars")
            print(f"   Export Method: {result.get('metadata', {}).get('export_method', 'N/A')}")
            return True
        else:
            print("❌ KiCad Exporter MCP: FAILED - Missing expected fields")
            return False

    except Exception as e:
        print(f"❌ KiCad Exporter MCP: EXCEPTION - {str(e)}")
        return False


async def test_all_mcp_servers():
    """Test all MCP servers."""
    print("🧪 Testing All MCP Servers")
    print("=" * 50)

    # Test individual servers
    scorer_ok = test_placement_scorer()
    simulator_ok = test_thermal_simulator()
    exporter_ok = await test_kicad_exporter()

    if scorer_ok and simulator_ok and exporter_ok:
        print("\n✅ All MCP Server tests PASSED!")
        print("   MCP servers are working correctly.")
        return True
    else:
        print("\n❌ Some MCP Server tests FAILED!")
        print("   Check MCP server implementations.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_all_mcp_servers())
    sys.exit(0 if success else 1)
