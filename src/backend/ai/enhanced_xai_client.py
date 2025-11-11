"""
Enhanced xAI Client with Extensive Reasoning Integration

Uses xAI (Grok) for reasoning at multiple stages:
1. Design generation (extensive)
2. Intent analysis with computational geometry
3. Optimization strategy reasoning during simulated annealing
4. Post-optimization analysis and refinement suggestions
5. Hierarchical reasoning for large designs
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import time


class EnhancedXAIClient:
    """Enhanced xAI client with extensive reasoning capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize enhanced xAI client."""
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        self.call_count = 0
        
        if not self.api_key:
            print("⚠️  XAI_API_KEY not found in environment. LLM features will use fallback behavior.")
            print("   Set it with: export XAI_API_KEY=your_key")
            self.headers = None
            self.enabled = False
        else:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.enabled = True
            print(f"✅ Enhanced xAI Client initialized (endpoint: {self.endpoint})")
    
    def _call_api(
        self,
        messages: List[Dict],
        model: str = "grok-4-latest",  # Updated to use latest model
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict:
        """Make API call to xAI with extensive logging."""
        if not self.enabled or not self.headers:
            error_msg = "xAI API key not configured. Using fallback behavior."
            print(f"   ⚠️  {error_msg}")
            return {"error": error_msg, "choices": []}
        
        self.call_count += 1
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print(f"🔌 xAI API Call #{self.call_count}: {self.endpoint}")
            print(f"   Model: {model}, Temperature: {temperature}, Max tokens: {max_tokens}")
            print(f"   Messages: {len(messages)} message(s)")
            
            response = requests.post(
                self.endpoint,
                json=data,
                headers=self.headers,
                timeout=180  # Increased timeout for extensive reasoning (3 minutes)
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"   Error response: {error_text}")
                raise requests.exceptions.HTTPError(f"xAI API returned {response.status_code}: {error_text}")
            
            result = response.json()
            print(f"   ✅ xAI API call #{self.call_count} successful")
            
            if result.get("choices") and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"   Response length: {len(content)} chars")
                print(f"   Tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}")
            
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"xAI API request failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg, "choices": []}
        except Exception as e:
            error_msg = f"xAI API call error: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg, "choices": []}
    
    def generate_design_with_reasoning(
        self,
        description: str,
        board_size: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Generate PCB design using extensive xAI reasoning.
        
        Uses xAI to:
        1. Parse natural language description
        2. Identify components and their relationships
        3. Reason about thermal requirements
        4. Suggest optimal initial placement
        5. Generate complete design with modules
        """
        board_info = board_size or {"width": 100, "height": 100, "clearance": 0.5}
        
        prompt = f"""
You are an expert PCB design engineer with deep knowledge of:
- Component placement optimization
- Thermal management (Voronoi-based analysis, Gaussian thermal diffusion)
- Signal integrity (trace length minimization, MST optimization)
- Design rules and manufacturability

Generate a complete PCB design from this description: "{description}"

Board constraints:
- Width: {board_info.get('width', 100)}mm
- Height: {board_info.get('height', 100)}mm
- Minimum clearance: {board_info.get('clearance', 0.5)}mm

Reasoning process:
1. **Component Identification**: Parse the description and identify all required components (ICs, resistors, capacitors, inductors, connectors, etc.)
2. **Power Analysis**: Estimate power dissipation for each component based on typical values
3. **Thermal Considerations**: Identify high-power components that need thermal management
4. **Signal Flow**: Identify signal paths and critical nets (high-speed, differential pairs, power)
5. **Module Identification**: Group related components into functional modules (power supply, signal processing, I/O, etc.)
6. **Initial Placement Strategy**: Reason about optimal placement considering:
   - Thermal spreading (high-power components should be distributed)
   - Signal integrity (related components should be close)
   - Design rules (clearances, board boundaries)

Return a complete JSON design with this structure:
{{
    "board": {{"width": {board_info.get('width', 100)}, "height": {board_info.get('height', 100)}, "clearance": {board_info.get('clearance', 0.5)}}},
    "components": [
        {{
            "name": "U1",
            "package": "SOIC-8",
            "width": 5.0,
            "height": 4.0,
            "power": 0.5,
            "x": 50.0,
            "y": 50.0,
            "angle": 0,
            "placed": true
        }}
    ],
    "nets": [
        {{
            "name": "VCC",
            "pins": [["U1", "pin8"], ["U2", "pin1"]]
        }}
    ],
    "modules": [
        {{
            "name": "Power Supply",
            "components": ["PWR_IC", "L1", "C_PWR"],
            "thermal_priority": "high"
        }}
    ],
    "reasoning": "Brief explanation of design decisions and thermal considerations"
}}

Make it realistic, complete, and optimized for thermal management.
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert PCB design engineer. Generate complete, realistic PCB designs with proper thermal management."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, temperature=0.8, max_tokens=4000)
        
        if "error" in response:
            return {"error": response["error"], "success": False}
        
        if response.get("choices") and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            return self._parse_design_json(content, board_info)
        
        return {"error": "No response from xAI", "success": False}
    
    def reason_about_geometry_and_optimize(
        self,
        geometry_data: Dict,
        user_intent: str,
        current_score: float,
        iteration: int,
        max_iterations: int
    ) -> Dict:
        """
        Use xAI to reason about computational geometry data and suggest optimization strategy.
        
        Called periodically during simulated annealing to guide optimization.
        """
        progress = (iteration / max_iterations) * 100 if max_iterations > 0 else 0
        
        prompt = f"""
You are optimizing a PCB layout using simulated annealing. Analyze the current computational geometry state and suggest optimization strategy.

**Current State (Iteration {iteration}/{max_iterations}, {progress:.1f}% complete):**

Computational Geometry Metrics:
- Voronoi Variance: {geometry_data.get('voronoi_variance', 0):.3f} (lower = better distribution)
- MST Length: {geometry_data.get('mst_length', 0):.2f} mm (shorter = better routing)
- Convex Hull Area: {geometry_data.get('convex_hull_area', 0):.2f} mm²
- Thermal Hotspots: {geometry_data.get('thermal_hotspots', 0)} regions
- Thermal Risk Score: {geometry_data.get('thermal_risk_score', 0):.3f}
- Net Crossings: {geometry_data.get('net_crossings', 0)} potential conflicts
- Component Density: {geometry_data.get('density', 0):.3f} components/mm²
- Overlap Risk: {geometry_data.get('overlap_risk', 0):.3f}

**Current Score**: {current_score:.3f} (lower is better)

**User Intent**: "{user_intent}"

**Your Task**: Analyze these metrics and suggest:
1. What should be prioritized next? (thermal spreading, trace length, clearance)
2. Which components should be moved? (identify problematic components)
3. What optimization strategy should be used? (more aggressive moves, focus on specific regions)

Return JSON:
{{
    "priority": "thermal" | "trace_length" | "clearance" | "balanced",
    "suggested_moves": [
        {{"component": "U1", "reason": "High power, causing thermal hotspot"}},
        {{"component": "R1", "reason": "Far from connected components, increasing MST length"}}
    ],
    "strategy": "Focus on thermal spreading by moving high-power components apart",
    "temperature_adjustment": 0.95 | 1.0 | 1.05 (suggest cooling rate adjustment)
}}

Be specific and actionable.
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a PCB optimization expert analyzing computational geometry metrics to guide simulated annealing."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, temperature=0.7, max_tokens=1000)
        
        if "error" in response or not response.get("choices"):
            return {"priority": "balanced", "strategy": "Continue optimization"}
        
        content = response["choices"][0]["message"]["content"]
        return self._parse_strategy_json(content)
    
    def analyze_post_optimization(
        self,
        initial_geometry: Dict,
        final_geometry: Dict,
        initial_score: float,
        final_score: float,
        user_intent: str
    ) -> Dict:
        """
        Use xAI to analyze optimization results and suggest refinements.
        """
        improvement = ((initial_score - final_score) / initial_score * 100) if initial_score > 0 else 0
        
        prompt = f"""
Analyze PCB optimization results and provide insights.

**Initial State:**
- Voronoi Variance: {initial_geometry.get('voronoi_variance', 0):.3f}
- MST Length: {initial_geometry.get('mst_length', 0):.2f} mm
- Thermal Hotspots: {initial_geometry.get('thermal_hotspots', 0)}
- Score: {initial_score:.3f}

**Final State:**
- Voronoi Variance: {final_geometry.get('voronoi_variance', 0):.3f}
- MST Length: {final_geometry.get('mst_length', 0):.2f} mm
- Thermal Hotspots: {final_geometry.get('thermal_hotspots', 0)}
- Score: {final_score:.3f}

**Improvement**: {improvement:.1f}% score reduction

**User Intent**: "{user_intent}"

Provide:
1. Key improvements made
2. Remaining issues
3. Suggestions for further optimization
4. Thermal management assessment

Return JSON:
{{
    "improvements": ["List of key improvements"],
    "remaining_issues": ["List of remaining problems"],
    "suggestions": ["Actionable suggestions"],
    "thermal_assessment": "Assessment of thermal management",
    "overall_quality": "excellent" | "good" | "fair" | "needs_work"
}}
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a PCB design expert analyzing optimization results."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, temperature=0.7, max_tokens=2000)
        
        if "error" in response or not response.get("choices"):
            return {"overall_quality": "good", "improvements": [], "suggestions": []}
        
        content = response["choices"][0]["message"]["content"]
        return self._parse_analysis_json(content)
    
    def intent_to_weights_with_geometry(
        self,
        user_intent: str,
        geometry_data: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Enhanced intent-to-weights conversion with extensive geometry reasoning.
        """
        geometry_context = ""
        if geometry_data:
            geometry_context = f"""
**Computational Geometry Analysis:**
- Component density: {geometry_data.get('density', 0):.3f} components/mm²
- Voronoi cell variance: {geometry_data.get('voronoi_variance', 0):.3f} (lower = uniform distribution)
- Minimum spanning tree length: {geometry_data.get('mst_length', 0):.2f} mm
- Convex hull area: {geometry_data.get('convex_hull_area', 0):.2f} mm²
- Thermal hotspots: {geometry_data.get('thermal_hotspots', 0)} regions
- Thermal risk score: {geometry_data.get('thermal_risk_score', 0):.3f}
- Net crossing count: {geometry_data.get('net_crossings', 0)}
- Component overlap risk: {geometry_data.get('overlap_risk', 0):.3f}

**Geometry Interpretation:**
- High Voronoi variance → components clustered → thermal risk
- Long MST → components far apart → signal integrity issues
- Many net crossings → routing conflicts → manufacturability problems
"""
        
        prompt = f"""
You are a PCB optimization expert. Analyze user intent and computational geometry data to determine optimization weights.

**User Intent**: "{user_intent}"

**Board Context**: {json.dumps(context or {}, indent=2)}

{geometry_context}

**Your Task**: Return three weights (alpha, beta, gamma) that sum to 1.0:
- **alpha**: Weight for trace length minimization (MST-based routing optimization)
- **beta**: Weight for thermal density minimization (Voronoi-based thermal spreading)
- **gamma**: Weight for clearance violation penalties (geometric collision detection)

**Reasoning Process**:
1. Analyze geometry metrics to identify primary issues
2. Map user intent to optimization priorities
3. Balance competing objectives based on geometry data

**Examples**:
- High Voronoi variance + "optimize thermal" → beta=0.7, alpha=0.2, gamma=0.1
- Long MST + "minimize trace length" → alpha=0.8, beta=0.1, gamma=0.1
- Many net crossings + "fix violations" → gamma=0.6, alpha=0.2, beta=0.2
- Balanced optimization → alpha=0.4, beta=0.4, gamma=0.2

Return ONLY valid JSON:
{{
    "alpha": 0.4,
    "beta": 0.4,
    "gamma": 0.2,
    "reasoning": "Brief explanation of weight choices"
}}
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a PCB optimization expert. Return only valid JSON with alpha, beta, gamma weights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, temperature=0.7, max_tokens=1000)
        
        if "error" in response:
            return (0.5, 0.3, 0.2)
        
        if response.get("choices") and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            try:
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                
                weights = json.loads(content)
                alpha = float(weights.get("alpha", 0.5))
                beta = float(weights.get("beta", 0.3))
                gamma = float(weights.get("gamma", 0.2))
                
                # Normalize
                total = alpha + beta + gamma
                if total > 0:
                    alpha /= total
                    beta /= total
                    gamma /= total
                
                return (alpha, beta, gamma)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        return (0.5, 0.3, 0.2)
    
    def _parse_design_json(self, content: str, board_info: Dict) -> Dict:
        """Parse design JSON from xAI response."""
        import re
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                design_data = json.loads(json_match.group())
                if "board" not in design_data:
                    design_data["board"] = board_info
                else:
                    design_data["board"].update(board_info)
                return {"success": True, "design": design_data}
        except:
            pass
        return {"success": False, "error": "Failed to parse design JSON"}
    
    def _parse_strategy_json(self, content: str) -> Dict:
        """Parse strategy JSON from xAI response."""
        import re
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"priority": "balanced", "strategy": "Continue optimization"}
    
    def _parse_analysis_json(self, content: str) -> Dict:
        """Parse analysis JSON from xAI response."""
        import re
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"overall_quality": "good", "improvements": [], "suggestions": []}
    
    def get_call_count(self) -> int:
        """Get total number of xAI API calls made."""
        return self.call_count

