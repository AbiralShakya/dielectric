"""
Hierarchical Reasoning Agent for Large-Scale PCB Optimization

Integrates HRM (Hierarchical Reasoning Model) with multi-agent system
for industry-scale PCB optimization (200+ components).

Based on: "Hierarchical Reasoning Model" (arXiv:2506.21734)
"""

import logging
from typing import Dict, Optional, Callable, Any
import asyncio

try:
    from backend.geometry.placement import Placement
    from backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from backend.ai.hierarchical_reasoning import HierarchicalReasoningModel
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights
    from src.backend.ai.hierarchical_reasoning import HierarchicalReasoningModel
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer

logger = logging.getLogger(__name__)


class HierarchicalReasoningAgent:
    """
    Agent that uses Hierarchical Reasoning Model for large-scale optimization.
    
    Features:
    - High-level abstract planning (modules, strategy)
    - Low-level detailed execution (components, fine-tuning)
    - Single forward pass (no explicit supervision needed)
    - Scales to 200+ component PCBs
    - Integrates with multi-agent system
    """
    
    def __init__(self):
        """Initialize hierarchical reasoning agent."""
        self.name = "HierarchicalReasoningAgent"
        self.scorer = None
        self.hrm = None
    
    async def optimize(
        self,
        placement: Placement,
        weights: Dict[str, float],
        user_intent: str,
        max_time_ms: float = 60000.0,  # 60 seconds default for large PCBs
        callback: Optional[Callable] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize using hierarchical reasoning.
        
        Args:
            placement: Initial placement
            weights: Optimization weights
            user_intent: User's optimization intent
            max_time_ms: Maximum time in milliseconds
            callback: Optional callback function
            random_seed: Random seed
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"{self.name}: Starting hierarchical reasoning optimization")
            logger.info(f"   Components: {len(placement.components)}")
            logger.info(f"   User intent: {user_intent}")
            
            # Setup scorer
            score_weights = ScoreWeights(
                alpha=weights.get("alpha", 0.3),
                beta=weights.get("beta", 0.3),
                gamma=weights.get("gamma", 0.2),
                delta=weights.get("delta", 0.2)
            )
            score_weights.normalize()
            
            self.scorer = WorldModelScorer(score_weights)
            
            # Initialize HRM
            # Adjust timescales based on problem size
            num_components = len(placement.components)
            if num_components > 150:
                high_level_timescale = 20  # Slower high-level updates for large designs
            elif num_components > 100:
                high_level_timescale = 15
            else:
                high_level_timescale = 10
            
            self.hrm = HierarchicalReasoningModel(
                scorer=self.scorer,
                high_level_timescale=high_level_timescale,
                low_level_timescale=1,
                max_reasoning_depth=5
            )
            
            # Estimate iterations from time budget
            # Assume ~50ms per iteration for large designs
            estimated_iterations = int(max_time_ms / 50)
            max_iterations = min(estimated_iterations, 2000)  # Cap at 2000
            
            logger.info(f"   Max iterations: {max_iterations}")
            logger.info(f"   High-level timescale: {high_level_timescale}")
            
            # Run hierarchical reasoning
            start_time = asyncio.get_event_loop().time()
            
            def progress_callback(pl, iter, score):
                if callback:
                    callback(pl, iter, score)
            
            optimized_placement, final_score, stats = self.hrm.reason_and_optimize(
                placement,
                user_intent,
                max_iterations=max_iterations,
                callback=progress_callback
            )
            
            elapsed_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logger.info(f"✅ {self.name}: Optimization complete")
            logger.info(f"   Final score: {final_score:.4f}")
            logger.info(f"   Reasoning depth: {stats['reasoning_depth']}")
            logger.info(f"   Time: {elapsed_time:.0f}ms")
            
            return {
                "success": True,
                "placement": optimized_placement,
                "score": final_score,
                "stats": {
                    **stats,
                    "elapsed_time_ms": elapsed_time,
                    "method": "hierarchical_reasoning",
                    "high_level_updates": stats.get("high_level_updates", 0),
                    "low_level_updates": stats.get("low_level_updates", 0),
                    "confidence": stats.get("confidence", 0.0)
                },
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name} error: {str(e)}", exc_info=True)
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
            "name": "hierarchical_optimize",
            "description": "Hierarchical reasoning optimization for large PCBs (200+ components)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"},
                    "weights": {"type": "object"},
                    "user_intent": {"type": "string"},
                    "max_time_ms": {"type": "number", "default": 60000.0}
                },
                "required": ["placement", "weights", "user_intent"]
            }
        }

