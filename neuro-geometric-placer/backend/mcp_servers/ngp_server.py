"""
Neuro-Geometric Placer MCP Server

A single MCP server with multiple tools for PCB placement optimization.
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from backend.scoring.scorer import WorldModelScorer, ScoreWeights
from backend.scoring.incremental_scorer import IncrementalScorer
from backend.geometry.placement import Placement


# Create FastMCP server
app = FastMCP("neuro-geometric-placer")


@app.tool()
async def score_delta(
    placement_data: Dict[str, Any],
    move_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute score delta for a component move using incremental scoring.

    Args:
        placement_data: Placement dictionary with components, board, nets
        move_data: Move data with component_name, old_x/y/angle, new_x/y/angle, weights

    Returns:
        Dict with delta score, new_score, affected_nets, computation_method
    """
    try:
        # Parse placement data
        placement = Placement.from_dict(placement_data)

        # Get scorer with weights
        weights = move_data.get("weights", {})
        score_weights = ScoreWeights(
            alpha=weights.get("alpha", 0.5),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.2)
        )

        base_scorer = WorldModelScorer(score_weights)
        scorer = IncrementalScorer(base_scorer)

        # Compute delta score
        delta = scorer.compute_delta_score(
            placement,
            move_data["component_name"],
            move_data["old_x"],
            move_data["old_y"],
            move_data["old_angle"],
            move_data["new_x"],
            move_data["new_y"],
            move_data["new_angle"]
        )

        # Apply move temporarily to get new score
        comp = placement.get_component(move_data["component_name"])
        if comp:
            old_pos = (comp.x, comp.y, comp.angle)
            comp.x = move_data["new_x"]
            comp.y = move_data["new_y"]
            comp.angle = move_data["new_angle"]

            new_score = scorer.score(placement)

            # Restore position
            comp.x, comp.y, comp.angle = old_pos
        else:
            new_score = scorer.score(placement)

        return {
            "delta": delta,
            "new_score": new_score,
            "computation_method": "incremental_scorer",
            "affected_nets": len(placement.get_affected_nets(move_data["component_name"]))
        }

    except Exception as e:
        return {
            "error": f"Failed to compute score delta: {str(e)}",
            "delta": 0.0,
            "new_score": 0.0
        }


@app.tool()
async def generate_heatmap(
    placement_data: Dict[str, Any],
    grid_size: int = 64
) -> Dict[str, Any]:
    """
    Generate thermal heatmap using computational geometry.

    Args:
        placement_data: Placement dictionary
        grid_size: Heatmap grid size (default 64x64)

    Returns:
        Dict with heatmap data, min/max values, computation method
    """
    try:
        placement = Placement.from_dict(placement_data)

        # Create grid
        heatmap = np.zeros((grid_size, grid_size))

        # Grid to board coordinates
        x_scale = placement.board.width / grid_size
        y_scale = placement.board.height / grid_size

        # Add heat contribution from each component
        for comp in placement.components.values():
            if comp.power <= 0:
                continue

            # Component center in grid coordinates
            grid_x = int(comp.x / x_scale)
            grid_y = int(comp.y / y_scale)

            # Add heat with Gaussian falloff
            for i in range(max(0, grid_x-10), min(grid_size, grid_x+10)):
                for j in range(max(0, grid_y-10), min(grid_size, grid_y+10)):
                    dist = np.sqrt((i - grid_x)**2 + (j - grid_y)**2)
                    # Gaussian with sigma = 5 grid cells
                    heat = comp.power * np.exp(-(dist**2) / (2 * 5**2))
                    heatmap[i, j] += heat

        return {
            "heatmap": heatmap.tolist(),
            "min": float(np.min(heatmap)),
            "max": float(np.max(heatmap)),
            "grid_size": grid_size,
            "computation": "gaussian_heat_convolution",
            "total_power_components": len([c for c in placement.components.values() if c.power > 0])
        }

    except Exception as e:
        return {
            "error": f"Failed to generate heatmap: {str(e)}",
            "heatmap": [],
            "min": 0.0,
            "max": 0.0
        }


@app.tool()
async def export_kicad(placement_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export placement to KiCad .kicad_pcb format.

    Args:
        placement_data: Placement dictionary

    Returns:
        Dict with KiCad content, format info, and metadata
    """
    try:
        placement = Placement.from_dict(placement_data)

        # Generate KiCad PCB content
        kicad_content = generate_kicad_pcb(placement)

        return {
            "kicad_content": kicad_content,
            "format": "kicad_pcb",
            "metadata": {
                "export_method": "dedalus_mcp",
                "component_count": len(placement.components),
                "board_dimensions": f"{placement.board.width}x{placement.board.height}mm"
            }
        }

    except Exception as e:
        return {
            "error": f"Failed to export KiCad: {str(e)}",
            "kicad_content": "",
            "format": "error"
        }


def generate_kicad_pcb(placement: Placement) -> str:
    """Generate KiCad PCB file content."""
    lines = []

    # PCB header
    lines.append("(kicad_pcb (version 20221018) (generator \"ngp-mcp\")")
    lines.append("")

    # General settings
    lines.append("  (general")
    lines.append("    (thickness 1.6)")
    lines.append("  )")
    lines.append("")

    # Board outline
    lines.append("  (setup")
    lines.append("    (pad_to_mask_clearance 0.05)")
    lines.append("    (allow_soldermask_bridges_in_footprints false)")
    lines.append("  )")
    lines.append("")

    # Layers
    lines.append("  (layers")
    lines.append("    (0 \"F.Cu\" signal)")
    lines.append("    (31 \"B.Cu\" signal)")
    lines.append("    (32 \"B.Adhes\" user \"B.Adhesive\")")
    lines.append("    (33 \"F.Adhes\" user \"F.Adhesive\")")
    lines.append("    (34 \"B.Paste\" user)")
    lines.append("    (35 \"F.Paste\" user)")
    lines.append("    (36 \"B.SilkS\" user \"B.Silkscreen\")")
    lines.append("    (37 \"F.SilkS\" user \"F.Silkscreen\")")
    lines.append("    (38 \"B.Mask\" user)")
    lines.append("    (39 \"F.Mask\" user)")
    lines.append("    (40 \"Dwgs.User\" user \"User.Drawings\")")
    lines.append("    (41 \"Cmts.User\" user \"User.Comments\")")
    lines.append("    (42 \"Eco1.User\" user \"User.Eco1\")")
    lines.append("    (43 \"Eco2.User\" user \"User.Eco2\")")
    lines.append("    (44 \"Edge.Cuts\" user)")
    lines.append("  )")
    lines.append("")

    # Net classes
    lines.append("  (net_class \"Default\" \"\"")
    lines.append("    (clearance 0.2)")
    lines.append("    (trace_width 0.25)")
    lines.append("    (via_dia 0.8)")
    lines.append("    (via_drill 0.4)")
    lines.append("    (uvia_dia 0.3)")
    lines.append("    (uvia_drill 0.1)")
    lines.append("  )")
    lines.append("")

    # Nets
    for net_name, net_data in placement.nets.items():
        lines.append(f"  (net {len(lines)} \"{net_name}\")")

    lines.append("")

    # Footprints (components)
    for comp_name, comp in placement.components.items():
        lines.append(f"  (footprint \"{comp.package}\" (layer \"F.Cu\")")
        lines.append("    (tedit 0) (tstamp 0)")
        lines.append(f"    (at {comp.x:.3f} {comp.y:.3f} {comp.angle})")
        lines.append(f"    (descr \"{comp.package} footprint\")")
        lines.append("    (tags \"\")")

        # Add pads based on pins
        for pin in comp.pins:
            lines.append("    (pad \"\" thru_hole circle")
            lines.append("      (at 0 0) (size 1.0 1.0) (drill 0.5)")
            lines.append("      (layers *.Cu *.Mask)")
            lines.append("    )")

        lines.append("  )")
        lines.append("")

    # Close PCB
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Run the MCP server
    print("🚀 Starting Neuro-Geometric Placer MCP Server")
    print("Available tools:")
    print("  - score_delta: Compute placement score changes")
    print("  - generate_heatmap: Create thermal heatmaps")
    print("  - export_kicad: Export to KiCad format")

    import mcp.server.stdio
    mcp.server.stdio.stdio_server(app.to_server())
