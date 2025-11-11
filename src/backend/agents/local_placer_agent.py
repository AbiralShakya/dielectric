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
    from backend.optimization.enhanced_simulated_annealing import EnhancedSimulatedAnnealing
    from backend.ai.enhanced_xai_client import EnhancedXAIClient
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from src.backend.scoring.incremental_scorer import IncrementalScorer
    from src.backend.optimization.local_placer import LocalPlacer
    try:
        from src.backend.optimization.enhanced_simulated_annealing import EnhancedSimulatedAnnealing
        from src.backend.ai.enhanced_xai_client import EnhancedXAIClient
        ENHANCED_AVAILABLE = True
    except ImportError:
        ENHANCED_AVAILABLE = False


class LocalPlacerAgent:
    """Agent for fast local placement optimization."""
    
    def __init__(self):
        """Initialize local placer agent."""
        self.name = "LocalPlacerAgent"
        self.scorer = None
        self.placer = None
        self.xai_client = None
        
        # Try to initialize enhanced xAI client
        if ENHANCED_AVAILABLE:
            try:
                self.xai_client = EnhancedXAIClient()
                print("✅ LocalPlacerAgent: Enhanced xAI client initialized")
            except Exception as e:
                print(f"⚠️  LocalPlacerAgent: Enhanced xAI not available: {e}")
                self.xai_client = None
    
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
        random_seed: Optional[int] = None,
        user_intent: str = "Optimize placement"
    ) -> Dict:
        """
        Run fast local optimization with enhanced xAI reasoning.
        
        Args:
            placement: Placement to optimize
            weights: Optimization weights
            max_time_ms: Maximum time in milliseconds
            callback: Optional callback for progress
            random_seed: Random seed for deterministic optimization
            user_intent: User's optimization intent for xAI reasoning
        
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
            
            # Use enhanced simulated annealing if available
            if ENHANCED_AVAILABLE and self.xai_client:
                print("   🚀 Using Enhanced Simulated Annealing with xAI reasoning")
                enhanced_sa = EnhancedSimulatedAnnealing(
                    scorer=self.scorer,
                    xai_client=self.xai_client,
                    initial_temp=50.0,
                    final_temp=0.1,
                    cooling_rate=0.9,
                    max_iterations=200,
                    reasoning_interval=25,  # Call xAI every 25 iterations
                    random_seed=random_seed
                )
                
                best_placement, best_score, stats = enhanced_sa.optimize(
                    placement,
                    user_intent=user_intent,
                    callback=callback,
                    timeout=max_time_ms / 1000.0
                )
                
                stats["xai_calls"] = enhanced_sa.reasoning_calls
                stats["method"] = "enhanced_simulated_annealing"
            else:
                # Fallback to standard simulated annealing
                if random_seed is not None:
                    self.placer = LocalPlacer(self.scorer, random_seed=random_seed)
                else:
                    self.placer = LocalPlacer(self.scorer)
                
                best_placement, best_score, stats = self.placer.optimize_fast(
                    placement,
                    max_time_ms=max_time_ms,
                    callback=callback
                )
                stats["method"] = "standard_simulated_annealing"
            
            return {
                "success": True,
                "placement": best_placement,
                "score": best_score,
                "stats": stats,
                "agent": self.name
            }
        except Exception as e:
            import traceback
            print(f"   ❌ LocalPlacerAgent error: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
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

