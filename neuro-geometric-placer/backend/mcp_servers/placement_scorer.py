"""
Placement Scorer MCP Server

Exposes fast scoring as MCP tool.
"""

from typing import Dict, Any
from backend.scoring.scorer import WorldModelScorer, ScoreWeights
from backend.scoring.incremental_scorer import IncrementalScorer
from backend.geometry.placement import Placement


class PlacementScorerMCP:
    """MCP server for placement scoring."""
    
    def __init__(self):
        """Initialize scorer MCP server."""
        self.name = "placement_scorer"
        self.scorer = None
    
    def _get_scorer(self, weights: Dict) -> IncrementalScorer:
        """Get or create scorer with weights."""
        score_weights = ScoreWeights(
            alpha=weights.get("alpha", 0.5),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.2)
        )
        base_scorer = WorldModelScorer(score_weights)
        return IncrementalScorer(base_scorer)
    
    def score_delta(self, placement_data: Dict, move_data: Dict) -> Dict[str, Any]:
        """
        Compute score delta for a move using Dedalus MCP.

        Args:
            placement_data: Placement dictionary
            move_data: {
                "component_name": str,
                "old_x": float, "old_y": float, "old_angle": float,
                "new_x": float, "new_y": float, "new_angle": float,
                "weights": {"alpha": float, "beta": float, "gamma": float}
            }

        Returns:
            {"delta": float, "new_score": float}
        """
        placement = Placement.from_dict(placement_data)
        scorer = self._get_scorer(move_data.get("weights", {}))

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

        # Compute new score
        comp = placement.get_component(move_data["component_name"])
        if comp:
            comp.x = move_data["new_x"]
            comp.y = move_data["new_y"]
            comp.angle = move_data["new_angle"]

        new_score = scorer.score(placement)

        return {
            "delta": delta,
            "new_score": new_score,
            "computation_method": "incremental_scorer",
            "affected_nets": len(placement.get_affected_nets(move_data["component_name"]))
        }
    
    def get_tool_definition(self) -> Dict:
        """Get MCP tool definition."""
        return {
            "name": "score_delta",
            "description": "Compute score delta for a component move",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement_data": {"type": "object"},
                    "move_data": {
                        "type": "object",
                        "properties": {
                            "component_name": {"type": "string"},
                            "old_x": {"type": "number"},
                            "old_y": {"type": "number"},
                            "old_angle": {"type": "number"},
                            "new_x": {"type": "number"},
                            "new_y": {"type": "number"},
                            "new_angle": {"type": "number"},
                            "weights": {"type": "object"}
                        },
                        "required": ["component_name", "old_x", "old_y", "old_angle", "new_x", "new_y", "new_angle"]
                    }
                },
                "required": ["placement_data", "move_data"]
            }
        }

