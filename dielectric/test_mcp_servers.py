#!/usr/bin/env python3
"""
Test Single MCP Server with Multiple Tools

Tests the unified Dielectric MCP server.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.mcp_servers.ngp_server import server as mcp_server


def test_mcp_server_setup():
    """Test that MCP server is properly configured."""
    print("🧪 Testing MCP Server Setup")
    print("=" * 40)

    try:
        # Check that the openmcp server was created
        if mcp_server is None:
            print("❌ MCP server not found")
            return False

        print("✅ MCP openmcp server created successfully")
        print(f"   Server name: {mcp_server.name}")

        # Check that tools are registered (this would be checked by openmcp internally)
        print("✅ Tools should be auto-registered by @server.tool() decorators")

        # Check that we can import the server functions
        from backend.mcp_servers.ngp_server import score_delta, generate_heatmap, export_kicad
        print("✅ All tool functions imported successfully")

        return True

    except Exception as e:
        print(f"❌ MCP Server setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_function_calls():
    """Test calling the MCP tool functions directly (for development/testing)."""
    print("\n🧪 Testing Direct Function Calls")
    print("=" * 40)

    try:
        # Test with simple mock data
        placement_data = {
            "components": [
                {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 50, "y": 50, "angle": 0, "placed": True, "pins": []}
            ],
            "board": {"width": 100, "height": 100, "clearance": 0.5},
            "nets": []
        }

        # Test generate_heatmap (simpler function)
        from backend.mcp_servers.ngp_server import generate_heatmap

        # Call synchronously since tools are sync functions
        result = generate_heatmap(placement_data, grid_size=16)
        if "heatmap" in result and len(result["heatmap"]) == 16:
            print("✅ generate_heatmap function works")
            heatmap_ok = True
        else:
            print("❌ generate_heatmap function failed")
            heatmap_ok = False

        return heatmap_ok

    except Exception as e:
        print(f"❌ Direct function calls failed: {str(e)}")
        return False


def test_kicad_export():
    """Test KiCad export function."""
    print("\n🧪 Testing KiCad Export")
    print("=" * 40)

    try:
        from backend.mcp_servers.ngp_server import export_kicad

        placement_data = {
            "components": [
                {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 50, "y": 50, "angle": 0, "placed": True, "pins": [{"name": "pin1", "x_offset": 0, "y_offset": 0, "net": "net1"}]},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 30, "y": 30, "angle": 0, "placed": True, "pins": [{"name": "pin1", "x_offset": 0, "y_offset": 0, "net": "net1"}]}
            ],
            "board": {"width": 100, "height": 100, "clearance": 0.5},
            "nets": [{"name": "net1", "pins": [["U1", "pin1"], ["R1", "pin1"]]}]
        }

        result = export_kicad(placement_data)

        if "kicad_content" in result and result["format"] == "kicad_pcb":
            print("✅ export_kicad function works")
            print(f"   Generated {len(result['kicad_content'])} chars of KiCad content")
            return True
        else:
            print("❌ export_kicad function failed")
            return False

    except Exception as e:
        print(f"❌ KiCad export failed: {str(e)}")
        return False


async def test_mcp_server():
    """Test the MCP server setup and basic functionality."""
    print("🧪 Testing MCP Server Implementation")
    print("=" * 50)

    # Test server setup
    setup_ok = test_mcp_server_setup()

    # Test direct function calls
    functions_ok = test_direct_function_calls()

    # Test KiCad export
    kicad_ok = test_kicad_export()

    if setup_ok and functions_ok and kicad_ok:
        print("\n✅ MCP Server tests PASSED!")
        print("   Server is ready for Dedalus Labs deployment.")
        print("   Tools: score_delta, generate_heatmap, export_kicad")
        return True
    else:
        print("\n❌ MCP Server tests FAILED!")
        print("   Check implementation before deploying to Dedalus.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
