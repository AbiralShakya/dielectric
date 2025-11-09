"""
xAI (Grok) API Client

For natural language intent → weight vector mapping.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple


class XAIClient:
    """Client for xAI (Grok) API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize xAI client.
        
        Args:
            api_key: xAI API key (if None, reads from env)
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment")
        
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_api(
        self,
        messages: List[Dict],
        model: str = "grok-2-1212",
        temperature: float = 0.7
    ) -> Dict:
        """Make API call to xAI."""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=data,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "choices": []}
    
    def intent_to_weights(
        self,
        user_intent: str,
        context: Optional[Dict] = None,
        geometry_data: Optional[Dict] = None
    ) -> Tuple[float, float, float]:
        """
        Convert natural language intent to weight vector (α, β, γ) using computational geometry reasoning.
        
        Args:
            user_intent: Natural language description
            context: Optional context (board size, component count, etc.)
            geometry_data: Computational geometry data structures (Voronoi diagrams, convex hulls, etc.)
        
        Returns:
            (alpha, beta, gamma) weight tuple
        """
        # Build computational geometry context for xAI reasoning
        geometry_context = ""
        if geometry_data:
            geometry_context = f"""
Computational Geometry Analysis:
- Component density: {geometry_data.get('density', 0):.2f} components/mm²
- Convex hull area: {geometry_data.get('convex_hull_area', 0):.2f} mm²
- Voronoi cell variance: {geometry_data.get('voronoi_variance', 0):.2f}
- Minimum spanning tree length: {geometry_data.get('mst_length', 0):.2f} mm
- Thermal hotspots: {geometry_data.get('thermal_hotspots', 0)} regions
- Net crossing count: {geometry_data.get('net_crossings', 0)}
- Component overlap risk: {geometry_data.get('overlap_risk', 0):.2f}
"""
        
        prompt = f"""
You are a PCB design optimization expert with deep knowledge of computational geometry algorithms.

User intent: "{user_intent}"

Board Context: {json.dumps(context or {}, indent=2)}
{geometry_context}

You need to reason over this computational geometry data and return three weights (alpha, beta, gamma) that sum to 1.0:
- alpha: Weight for trace length minimization (Manhattan/Euclidean distance optimization)
- beta: Weight for thermal density minimization (Voronoi-based thermal spreading)  
- gamma: Weight for clearance violation penalties (geometric collision detection)

Reasoning process:
1. Analyze the computational geometry metrics (density, Voronoi cells, MST length)
2. Consider thermal hotspots and net crossings
3. Map user intent to geometric optimization priorities
4. Return weights that balance these factors

Examples:
- "Optimize for minimal trace length" → alpha=0.8, beta=0.1, gamma=0.1 (prioritize MST reduction)
- "Keep components cool" → alpha=0.2, beta=0.7, gamma=0.1 (prioritize Voronoi-based thermal spreading)
- "Prioritize cooling but keep wires short" → alpha=0.4, beta=0.5, gamma=0.1 (balance MST and thermal)
- "Minimize violations" → alpha=0.3, beta=0.2, gamma=0.5 (prioritize collision detection)

Return ONLY a JSON object with "alpha", "beta", "gamma" fields. No other text.
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a PCB design optimization expert. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, model="grok-2-1212")
        
        if "error" in response:
            # Fallback to default weights
            return (0.5, 0.3, 0.2)
        
        if response.get("choices") and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            
            # Try to parse JSON
            try:
                # Extract JSON from response
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
        
        # Fallback
        return (0.5, 0.3, 0.2)
    
    def explain_placement(
        self,
        placement: Dict,
        score_breakdown: Dict,
        weights: Tuple[float, float, float]
    ) -> str:
        """
        Generate explainable AI summary of placement.
        
        Args:
            placement: Placement dictionary
            score_breakdown: Score breakdown from scorer
            weights: Weight vector used
        
        Returns:
            Natural language explanation
        """
        prompt = f"""
You are a PCB design expert. Explain this component placement optimization result.

Placement summary:
- Components: {len(placement.get('components', []))}
- Board size: {placement.get('board', {}).get('width', 0)}mm x {placement.get('board', {}).get('height', 0)}mm

Score breakdown:
- Trace length: {score_breakdown.get('trace_length', 0):.2f}
- Thermal density: {score_breakdown.get('thermal_density', 0):.2f}
- Clearance violations: {score_breakdown.get('clearance_violations', 0):.2f}
- Total score: {score_breakdown.get('total_score', 0):.2f}

Optimization weights:
- Trace length priority: {weights[0]:.1%}
- Thermal priority: {weights[1]:.1%}
- Clearance priority: {weights[2]:.1%}

Provide a 2-3 sentence explanation of:
1. What the optimizer prioritized
2. Key improvements made
3. Any remaining issues

Be concise and technical but accessible.
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a PCB design expert explaining optimization results."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self._call_api(messages, model="grok-2-1212")
        
        if "error" in response:
            return "Optimization completed. See score breakdown for details."
        
        if response.get("choices") and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        
        return "Optimization completed."

