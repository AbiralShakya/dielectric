"""
Intent Agent

Converts natural language → weight vector (α, β, γ)
"""

from typing import Dict, Tuple, Optional
from backend.ai.xai_client import XAIClient


class IntentAgent:
    """Agent for converting user intent to optimization weights."""
    
    def __init__(self):
        """Initialize intent agent."""
        self.name = "IntentAgent"
        self.client = XAIClient()
    
    async def process(self, user_intent: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user intent and return weights.
        
        Args:
            user_intent: Natural language description
            context: Optional context (board info, component count, etc.)
        
        Returns:
            {
                "success": bool,
                "weights": (alpha, beta, gamma),
                "explanation": str
            }
        """
        try:
            alpha, beta, gamma = self.client.intent_to_weights(user_intent, context)
            
            return {
                "success": True,
                "weights": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma
                },
                "explanation": f"Optimizing with priorities: trace length ({alpha:.1%}), thermal ({beta:.1%}), clearance ({gamma:.1%})",
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

