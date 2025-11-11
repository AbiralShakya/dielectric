"""
Intent Agent

Converts natural language → weight vector (α, β, γ) using computational geometry + xAI reasoning.
"""

from typing import Dict, Tuple, Optional
try:
    from backend.ai.enhanced_xai_client import EnhancedXAIClient
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.geometry.placement import Placement
except ImportError:
    try:
        from src.backend.ai.enhanced_xai_client import EnhancedXAIClient
    except ImportError:
        from src.backend.ai.xai_client import XAIClient as EnhancedXAIClient
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.geometry.placement import Placement


class IntentAgent:
    """Agent for converting user intent to optimization weights using computational geometry + xAI."""
    
    def __init__(self):
        """Initialize intent agent."""
        self.name = "IntentAgent"
        try:
            self.client = EnhancedXAIClient()
        except Exception:
            # Fallback to basic client if enhanced not available
            from src.backend.ai.xai_client import XAIClient
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
            print(f"🧠 IntentAgent: Processing intent '{user_intent}'")
            
            geometry_data = None
            
            # Perform computational geometry analysis if placement provided
            if placement:
                print("   📐 Computing computational geometry analysis...")
                analyzer = GeometryAnalyzer(placement)
                geometry_data = analyzer.analyze()
                print(f"   ✅ Geometry analysis complete: {len(geometry_data)} metrics")
            
            # Pass computational geometry data to xAI for reasoning
            print("   🤖 Calling xAI API for weight reasoning...")
            if hasattr(self.client, 'intent_to_weights_with_geometry'):
                alpha, beta, gamma = self.client.intent_to_weights_with_geometry(
                    user_intent,
                    geometry_data,
                    context
                )
            else:
                # Fallback for basic client
                alpha, beta, gamma = self.client.intent_to_weights(
                    user_intent,
                    context,
                    geometry_data
                )
            
            print(f"   ✅ xAI returned weights: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
            
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
            import traceback
            error_msg = str(e)
            print(f"   ❌ IntentAgent error: {error_msg}")
            print(f"   Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
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

