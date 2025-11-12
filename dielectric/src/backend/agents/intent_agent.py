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
        except Exception as e:
            # Fallback to basic client if enhanced not available
            try:
                from src.backend.ai.xai_client import XAIClient
                self.client = XAIClient()
            except Exception:
                # If both fail, set to None - will use fallback weights
                print(f"⚠️  IntentAgent: Could not initialize xAI client: {e}")
                print("   Will use default weights based on intent keywords")
                self.client = None
    
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
            if self.client and (hasattr(self.client, 'enabled') and self.client.enabled or not hasattr(self.client, 'enabled')):
                print("   🤖 Calling xAI API with enriched mathematical context...")
                try:
                    if hasattr(self.client, 'intent_to_weights_with_geometry'):
                        # Pass placement for full context enrichment
                        alpha, beta, gamma = self.client.intent_to_weights_with_geometry(
                            user_intent,
                            geometry_data,
                            context,
                            placement=placement  # Pass placement for full enrichment
                        )
                    elif hasattr(self.client, 'intent_to_weights'):
                        # Fallback for basic client
                        alpha, beta, gamma = self.client.intent_to_weights(
                            user_intent,
                            context,
                            geometry_data
                        )
                    else:
                        raise AttributeError("Client missing intent_to_weights method")
                    
                    print(f"   ✅ xAI returned weights: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
                except Exception as e:
                    print(f"   ⚠️  xAI call failed: {e}, using keyword-based fallback")
                    alpha, beta, gamma = self._intent_to_weights_fallback(user_intent, geometry_data)
            else:
                print("   📝 Using keyword-based weight inference (xAI not available)")
                alpha, beta, gamma = self._intent_to_weights_fallback(user_intent, geometry_data)
            
            # Include DFM weight (delta) if calculated
            weights_dict = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            }
            
            # Add delta (DFM) if available from fallback
            if hasattr(self, '_last_delta'):
                weights_dict["delta"] = self._last_delta
            
            explanation = f"Optimizing with priorities: trace length ({alpha:.1%}), thermal ({beta:.1%}), clearance ({gamma:.1%})"
            if "delta" in weights_dict:
                explanation += f", DFM ({weights_dict['delta']:.1%})"
            
            return {
                "success": True,
                "weights": weights_dict,
                "explanation": explanation,
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
    
    def _intent_to_weights_fallback(self, user_intent: str, geometry_data: Optional[Dict] = None) -> Tuple[float, float, float]:
        """
        Enhanced fallback method with DFM weights and vertical-specific terminology.
        
        Args:
            user_intent: Natural language description
            geometry_data: Optional geometry data
        
        Returns:
            (alpha, beta, gamma) weight tuple
            Note: delta (DFM weight) is stored in self._last_delta
        """
        intent_lower = user_intent.lower()
        
        # Parse explicit percentage-based weights if present in intent
        # e.g., "Signal integrity (40%), Thermal management (30%), Manufacturability (20%), Cost (10%)"
        import re
        perc = {"signal": None, "thermal": None, "manu": None, "cost": None}
        try:
            def extract(patterns):
                for p in patterns:
                    m = re.search(p, intent_lower)
                    if m:
                        return float(m.group(1))
                return None
            perc["signal"] = extract([r"signal\\s+integrity\\s*\\((\\d+)\\s*%\\)", r"signal\\s*\\((\\d+)\\s*%\\)"])
            perc["thermal"] = extract([r"thermal.*?\\((\\d+)\\s*%\\)"])
            perc["manu"] = extract([r"manufacturability\\s*\\((\\d+)\\s*%\\)", r"dfm\\s*\\((\\d+)\\s*%\\)"])
            perc["cost"] = extract([r"cost\\s*\\((\\d+)\\s*%\\)"])
            if all(v is not None for v in perc.values()):
                total = sum(perc.values())
                if total > 0:
                    alpha = perc["signal"] / total
                    beta = perc["thermal"] / total
                    manu = perc["manu"] / total
                    cost = perc["cost"] / total
                    gamma = max(0.05, manu * 0.5)
                    delta = max(0.05, manu * 0.5 + cost)
                    s = alpha + beta + gamma + delta
                    alpha, beta, gamma, delta = alpha/s, beta/s, gamma/s, delta/s
                    self._last_delta = delta
                    return (alpha, beta, gamma)
        except Exception:
            # If parsing fails, continue with keyword/geometry logic below
            pass
        
        # Initialize weights including DFM (delta)
        alpha = 0.25  # Trace length
        beta = 0.25   # Thermal
        gamma = 0.25  # Clearance
        delta = 0.25  # DFM (NEW)
        
        # Vertical-specific terminology detection
        vertical = self._detect_vertical(intent_lower)
        
        # Adjust based on vertical
        if vertical == "rf":
            # RF: Signal integrity and impedance control critical
            alpha = 0.4  # Trace length/impedance
            beta = 0.2   # Thermal
            gamma = 0.2  # Clearance
            delta = 0.2  # DFM
        elif vertical == "power":
            # Power: Thermal and current handling critical
            alpha = 0.2  # Trace length
            beta = 0.5   # Thermal (high priority)
            gamma = 0.15 # Clearance
            delta = 0.15 # DFM
        elif vertical == "medical":
            # Medical: Safety and reliability critical
            alpha = 0.2  # Trace length
            beta = 0.2   # Thermal
            gamma = 0.3  # Clearance (safety)
            delta = 0.3  # DFM (reliability)
        elif vertical == "automotive":
            # Automotive: Reliability and thermal critical
            alpha = 0.2  # Trace length
            beta = 0.4   # Thermal (wide temp range)
            gamma = 0.2  # Clearance
            delta = 0.2  # DFM
        
        # Adjust based on keywords
        if any(word in intent_lower for word in ["trace", "wire", "length", "routing", "signal", "impedance"]):
            alpha = 0.5
            beta = 0.2
            gamma = 0.15
            delta = 0.15
        elif any(word in intent_lower for word in ["thermal", "cool", "heat", "temperature", "hotspot", "power"]):
            alpha = 0.15
            beta = 0.6
            gamma = 0.15
            delta = 0.1
        elif any(word in intent_lower for word in ["violation", "clearance", "spacing", "collision"]):
            alpha = 0.2
            beta = 0.2
            gamma = 0.4
            delta = 0.2
        elif any(word in intent_lower for word in ["manufacturing", "dfm", "production", "fabrication", "assembly"]):
            alpha = 0.2
            beta = 0.2
            gamma = 0.2
            delta = 0.4  # DFM priority
        elif any(word in intent_lower for word in ["balance", "optimize", "improve"]):
            alpha = 0.3
            beta = 0.3
            gamma = 0.2
            delta = 0.2
        
        # Adjust based on geometry if available
        if geometry_data:
            voronoi_var = geometry_data.get('voronoi_variance', 0)
            mst_length = geometry_data.get('mst_length', 0)
            hotspots = geometry_data.get('thermal_hotspots', 0)
            routing_complexity = geometry_data.get('routing_complexity', 0)
            overlap_risk = geometry_data.get('overlap_risk', 0)
            
            # High routing complexity → prioritize trace length
            if routing_complexity > 10:
                alpha = min(0.5, alpha + 0.15)
                delta = max(0.1, delta - 0.05)
            
            # High Voronoi variance → prioritize thermal
            if voronoi_var > 0.5:
                beta = min(0.6, beta + 0.15)
                alpha = max(0.1, alpha - 0.1)
            
            # Long MST → prioritize trace length
            if mst_length > 100:
                alpha = min(0.5, alpha + 0.15)
                beta = max(0.1, beta - 0.1)
            
            # Many hotspots → prioritize thermal
            if hotspots > 3:
                beta = min(0.6, beta + 0.15)
                alpha = max(0.1, alpha - 0.1)
            
            # High overlap risk → prioritize clearance and DFM
            if overlap_risk > 0.5:
                gamma = min(0.4, gamma + 0.15)
                delta = min(0.3, delta + 0.1)
        
        # Normalize
        total = alpha + beta + gamma + delta
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
            delta /= total
        
        # Store delta for inclusion in weights dict
        self._last_delta = delta
        
        # Return (alpha, beta, gamma) for compatibility
        return (alpha, beta, gamma)
    
    def _detect_vertical(self, intent_lower: str) -> Optional[str]:
        """
        Detect vertical domain from intent keywords.
        
        Returns:
            "rf", "power", "medical", "automotive", or None
        """
        # RF/High-frequency keywords
        rf_keywords = [
            "rf", "radio frequency", "2.4ghz", "5ghz", "wifi", "bluetooth",
            "antenna", "impedance", "50 ohm", "100 ohm", "differential pair",
            "matching network", "balun", "transceiver", "rfid"
        ]
        if any(keyword in intent_lower for keyword in rf_keywords):
            return "rf"
        
        # Power electronics keywords
        power_keywords = [
            "power supply", "switching", "buck", "boost", "ldo", "regulator",
            "converter", "inverter", "high current", "thermal via", "heat sink",
            "emi", "emc", "filtering"
        ]
        if any(keyword in intent_lower for keyword in power_keywords):
            return "power"
        
        # Medical device keywords
        medical_keywords = [
            "medical", "fda", "isolation", "patient", "safety", "iec 60601",
            "creepage", "clearance", "barrier", "reliability", "mtbf"
        ]
        if any(keyword in intent_lower for keyword in medical_keywords):
            return "medical"
        
        # Automotive keywords
        automotive_keywords = [
            "automotive", "iso 26262", "can bus", "ecu", "wide temperature",
            "vibration", "emc", "automotive grade", "aec-q100"
        ]
        if any(keyword in intent_lower for keyword in automotive_keywords):
            return "automotive"
        
        return None
    
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

