"""
Intent Agent

Converts natural language → weight vector (α, β, γ) using computational geometry + xAI reasoning.
"""

from typing import Dict, Tuple, Optional
try:
    from backend.ai.xai_client import XAIClient
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.ai.xai_client import XAIClient
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.geometry.placement import Placement


class IntentAgent:
    """Agent for converting user intent to optimization weights using computational geometry + xAI."""
    
    def __init__(self):
        """Initialize intent agent."""
        self.name = "IntentAgent"
        self.client = XAIClient()
    
    async def process_intent(
        self,
        user_intent: str,
        context: Optional[Dict] = None,
        placement: Optional[Placement] = None
    ) -> Dict:
        """
        Process user intent using computational geometry analysis + xAI reasoning.
        
        Args:
            user_intent: Natural language description
            context: Optional context (board info, component count, etc.)
            placement: Optional placement to analyze geometrically
        
        Returns:
            {
                "success": bool,
                "weights": (alpha, beta, gamma),
                "explanation": str,
                "geometry_data": Dict
            }
        """
        try:
            geometry_data = None
            
            # Perform computational geometry analysis if placement provided
            if placement:
                analyzer = GeometryAnalyzer(placement)
                geometry_data = analyzer.analyze()
            
            # Pass computational geometry data to xAI for reasoning
            alpha, beta, gamma = self.client.intent_to_weights(
                user_intent,
                context,
                geometry_data
            )
            
            return {
                "success": True,
                "weights": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma
                },
                "explanation": f"Optimizing with priorities: trace length ({alpha:.1%}), thermal ({beta:.1%}), clearance ({gamma:.1%})",
                "geometry_data": geometry_data,
                "agent": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},  # Default
                "agent": self.name
            }
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "intent_to_weights",
            "description": "Convert natural language intent to optimization weights",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_intent": {
                        "type": "string",
                        "description": "Natural language optimization intent"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context (board size, component count)"
                    }
                },
                "required": ["user_intent"]
            }
        }

