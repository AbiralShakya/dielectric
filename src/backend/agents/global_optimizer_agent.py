"""
Global Optimizer Agent

Heavy batch optimization for quality results (background).
"""

from typing import Dict, Optional, Callable
try:
    from backend.geometry.placement import Placement
    from backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from backend.scoring.incremental_scorer import IncrementalScorer
    from backend.optimization.simulated_annealing import SimulatedAnnealing
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from src.backend.scoring.incremental_scorer import IncrementalScorer
    from src.backend.optimization.simulated_annealing import SimulatedAnnealing


class GlobalOptimizerAgent:
    """Agent for global placement optimization (slow path)."""
    
    def __init__(self):
        """Initialize global optimizer agent."""
        self.name = "GlobalOptimizerAgent"
        self.scorer = None
        self.optimizer = None
    
    def _setup_optimizer(self, weights: Dict, plan: Dict):
        """Setup optimizer with weights and plan."""
        score_weights = ScoreWeights(
            alpha=weights.get("alpha", 0.5),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.2)
        )
        base_scorer = WorldModelScorer(score_weights)
        self.scorer = IncrementalScorer(base_scorer)
        
        self.optimizer = SimulatedAnnealing(
            scorer=self.scorer,
            initial_temp=plan.get("initial_temp", 100.0),
            final_temp=plan.get("final_temp", 0.01),
            cooling_rate=plan.get("cooling_rate", 0.95),
            max_iterations=plan.get("max_iterations", 5000)
        )
    
    async def process(
        self,
        placement: Placement,
        weights: Dict,
        plan: Dict,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> Dict:
        """
        Run global optimization.
        
        Args:
            placement: Placement to optimize
            weights: Optimization weights
            plan: Optimization plan from PlannerAgent
            callback: Optional callback for progress
            timeout: Optional timeout in seconds
        
        Returns:
            {
                "success": bool,
                "placement": Placement,
                "score": float,
                "stats": Dict
            }
        """
        try:
            self._setup_optimizer(weights, plan)
            
            best_placement, best_score, stats = self.optimizer.optimize(
                placement,
                callback=callback,
                timeout=timeout
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
            "name": "global_optimize",
            "description": "Global placement optimization (background, quality)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"},
                    "weights": {"type": "object"},
                    "plan": {"type": "object"},
                    "timeout": {"type": "number"}
                },
                "required": ["placement", "weights", "plan"]
            }
        }

