#!/usr/bin/env python3
"""
Neuro-Geometric Placer MCP Server

A single MCP server with multiple tools for PCB placement optimization.
Uses Dedalus Labs openmcp framework.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openmcp import MCPServer, tool
from backend.scoring.scorer import WorldModelScorer, ScoreWeights
from backend.scoring.incremental_scorer import IncrementalScorer
from backend.geometry.placement import Placement


# Create openmcp server
server = MCPServer("neuro-geometric-placer")

# Register tools within binding context
with server.binding():
    @tool(description="Compute score delta for a component move using incremental scoring")
    def score_delta(
        placement_data: dict,
        move_data: dict
    ) -> dict:
        """Compute score delta for component moves."""
        try:
            placement = Placement.from_dict(placement_data)
            weights = move_data.get("weights", {})
            score_weights = ScoreWeights(
                alpha=weights.get("alpha", 0.5),
                beta=weights.get("beta", 0.3),
                gamma=weights.get("gamma", 0.2)
            )

            base_scorer = WorldModelScorer(score_weights)
            scorer = IncrementalScorer(base_scorer)

            delta = scorer.compute_delta_score(
                placement,
                move_data["component_name"],
                move_data["old_x"], move_data["old_y"], move_data["old_angle"],
                move_data["new_x"], move_data["new_y"], move_data["new_angle"]
            )

            # Calculate new score
            comp = placement.get_component(move_data["component_name"])
            if comp:
                old_pos = (comp.x, comp.y, comp.angle)
                comp.x, comp.y, comp.angle = move_data["new_x"], move_data["new_y"], move_data["new_angle"]
                new_score = scorer.score(placement)
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
            return {"error": str(e), "delta": 0.0, "new_score": 0.0}

    @tool(description="Generate thermal heatmap using computational geometry")
    def generate_heatmap(
        placement_data: dict,
        grid_size: int = 64
    ) -> dict:
        """Generate thermal heatmaps for placement analysis."""
        try:
            import numpy as np
            placement = Placement.from_dict(placement_data)
            heatmap = np.zeros((grid_size, grid_size))

            x_scale = placement.board.width / grid_size
            y_scale = placement.board.height / grid_size

            for comp in placement.components.values():
                if comp.power <= 0:
                    continue
                grid_x = int(comp.x / x_scale)
                grid_y = int(comp.y / y_scale)
                for i in range(max(0, grid_x-10), min(grid_size, grid_x+10)):
                    for j in range(max(0, grid_y-10), min(grid_size, grid_y+10)):
                        dist = ((i - grid_x)**2 + (j - grid_y)**2)**0.5
                        heat = comp.power * np.exp(-(dist**2) / (2 * 25))
                        heatmap[i, j] += heat

            return {
                "heatmap": heatmap.tolist(),
                "min": float(np.min(heatmap)),
                "max": float(np.max(heatmap)),
                "grid_size": grid_size,
                "total_power_components": len([c for c in placement.components.values() if c.power > 0])
            }
        except Exception as e:
            return {"error": str(e), "heatmap": []}

    @tool(description="Export placement to KiCad .kicad_pcb format")
    def export_kicad(placement_data: dict) -> dict:
        """Export optimized placement to KiCad format."""
        try:
            placement = Placement.from_dict(placement_data)
            lines = [
                "(kicad_pcb (version 20221018) (generator \"ngp-mcp\")",
                "",
                "  (general",
                "    (thickness 1.6)",
                "  )",
                "",
                "  (layers",
                '    (0 "F.Cu" signal)',
                '    (31 "B.Cu" signal)',
                "  )",
                "",
                "  (net_class \"Default\" \"\"",
                "    (clearance 0.2)",
                "    (trace_width 0.25)",
                "  )",
                ""
            ]

            # Add components
            for comp_name, comp in placement.components.items():
                lines.extend([
                    f'  (footprint "{comp.package}" (layer "F.Cu")',
                    "    (tedit 0) (tstamp 0)",
                    f"    (at {comp.x:.3f} {comp.y:.3f} {comp.angle})",
                    f'    (descr "{comp.package} footprint")',
                    "    (tags \"\")",
                    "    (pad \"\" thru_hole circle",
                    "      (at 0 0) (size 1.0 1.0) (drill 0.5)",
                    '      (layers *.Cu *.Mask)',
                    "    )",
                    "  )",
                    ""
                ])

            lines.append(")")

            return {
                "kicad_content": "\n".join(lines),
                "format": "kicad_pcb",
                "metadata": {
                    "component_count": len(placement.components),
                    "board_dimensions": f"{placement.board.width}x{placement.board.height}mm"
                }
            }
        except Exception as e:
            return {"error": str(e), "kicad_content": ""}


def main():
    """Entry point for running the MCP server."""
    print("🚀 Starting Neuro-Geometric Placer MCP Server")
    print("Available tools:")
    print("  - score_delta: Compute placement score changes")
    print("  - generate_heatmap: Create thermal heatmaps")
    print("  - export_kicad: Export to KiCad format")

    # Run the server
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
