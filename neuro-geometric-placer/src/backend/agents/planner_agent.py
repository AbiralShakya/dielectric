"""
Planner Agent

Generates high-level annealing schedule and move heuristics.
"""

from typing import Dict, Optional
from backend.ai.xai_client import XAIClient


class PlannerAgent:
    """Agent for planning optimization strategy."""
    
    def __init__(self):
        """Initialize planner agent."""
        self.name = "PlannerAgent"
        self.client = XAIClient()
    
    async def process(
        self,
        placement_info: Dict,
        weights: Dict,
        optimization_type: str = "fast"
    ) -> Dict:
        """
        Generate optimization plan.
        
        Args:
            placement_info: Placement metadata (component count, board size, etc.)
            weights: Optimization weights
            optimization_type: "fast" or "quality"
        
        Returns:
            {
                "success": bool,
                "plan": {
                    "initial_temp": float,
                    "final_temp": float,
                    "cooling_rate": float,
                    "max_iterations": int,
                    "strategy": str
                }
            }
        """
        try:
            num_components = placement_info.get("num_components", 10)
            board_area = placement_info.get("board_area", 10000)
            
            if optimization_type == "fast":
                # Fast path: fewer iterations, higher cooling rate
                plan = {
                    "initial_temp": 50.0,
                    "final_temp": 0.1,
                    "cooling_rate": 0.9,
                    "max_iterations": 200,
                    "strategy": "local_optimization"
                }
            else:
                # Quality path: more iterations, slower cooling
                plan = {
                    "initial_temp": 100.0,
                    "final_temp": 0.01,
                    "cooling_rate": 0.95,
                    "max_iterations": 5000,
                    "strategy": "global_optimization"
                }
            
            # Adjust based on problem size
            if num_components > 50:
                plan["max_iterations"] = int(plan["max_iterations"] * 1.5)
            
            return {
                "success": True,
                "plan": plan,
                "agent": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "plan": {
                    "initial_temp": 50.0,
                    "final_temp": 0.1,
                    "cooling_rate": 0.9,
                    "max_iterations": 200,
                    "strategy": "default"
                },
                "agent": self.name
            }
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "plan_optimization",
            "description": "Generate optimization plan and annealing schedule",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement_info": {"type": "object"},
                    "weights": {"type": "object"},
                    "optimization_type": {
                        "type": "string",
                        "enum": ["fast", "quality"]
                    }
                },
                "required": ["placement_info", "weights"]
            }
        }

