"""
Local Placer Agent

Fast path optimizer for interactive UI (<200ms).
"""

import asyncio
from typing import Dict, Optional, Callable
try:
    from backend.geometry.placement import Placement
    from backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from backend.scoring.incremental_scorer import IncrementalScorer
    from backend.optimization.local_placer import LocalPlacer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from src.backend.scoring.incremental_scorer import IncrementalScorer
    from src.backend.optimization.local_placer import LocalPlacer


class LocalPlacerAgent:
    """Agent for fast local placement optimization."""
    
    def __init__(self):
        """Initialize local placer agent."""
        self.name = "LocalPlacerAgent"
        self.scorer = None
        self.placer = None
    
    def _setup_scorer(self, weights: Dict):
        """Setup scorer with weights."""
        score_weights = ScoreWeights(
            alpha=weights.get("alpha", 0.5),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.2)
        )
        base_scorer = WorldModelScorer(score_weights)
        self.scorer = IncrementalScorer(base_scorer)
        self.placer = LocalPlacer(self.scorer)
    
    async def process(
        self,
        placement: Placement,
        weights: Dict,
        max_time_ms: float = 200.0,
        callback: Optional[Callable] = None,
        random_seed: Optional[int] = None
    ) -> Dict:
        """
        Run fast local optimization.
        
        Args:
            placement: Placement to optimize
            weights: Optimization weights
            max_time_ms: Maximum time in milliseconds
            callback: Optional callback for progress
        
        Returns:
            {
                "success": bool,
                "placement": Placement,
                "score": float,
                "stats": Dict
            }
        """
        try:
            self._setup_scorer(weights)
            # Recreate placer with seed if provided
            if random_seed is not None:
                self.placer = LocalPlacer(self.scorer, random_seed=random_seed)
            
            best_placement, best_score, stats = self.placer.optimize_fast(
                placement,
                max_time_ms=max_time_ms,
                callback=callback
            )
            
            return {
                "success": True,
                "placement": best_placement,
                "score": best_score,
                "stats": stats,
                "agent": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "placement": placement,
                "score": float('inf'),
                "agent": self.name
            }
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "local_optimize",
            "description": "Fast local placement optimization (<200ms)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"},
                    "weights": {"type": "object"},
                    "max_time_ms": {"type": "number", "default": 200.0}
                },
                "required": ["placement", "weights"]
            }
        }

