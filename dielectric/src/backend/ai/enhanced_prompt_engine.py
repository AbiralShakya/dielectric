"""
Enhanced Prompt Engineering for xAI/Grok

Combines:
1. Context enrichment (mathematical, physics, engineering)
2. ML feature extraction (small models for insights)
3. Structured prompt generation

Creates rich, context-aware prompts that leverage all available data.
"""

import json
from typing import Dict, List, Optional, Tuple
try:
    from backend.geometry.placement import Placement
    from backend.ai.context_enricher import ContextEnricher
    from backend.ai.ml_feature_extractor import ContextSummarizer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.ai.context_enricher import ContextEnricher
    from src.backend.ai.ml_feature_extractor import ContextSummarizer


class EnhancedPromptEngine:
    """
    Enhanced prompt engineering system.
    
    Creates rich prompts with:
    - Mathematical context (geometry, physics, optimization)
    - ML-extracted insights
    - Structured reasoning framework
    """
    
    def __init__(self):
        """Initialize enhanced prompt engine."""
        self.context_enricher = ContextEnricher()
        self.context_summarizer = ContextSummarizer()
    
    def create_intent_prompt(
        self,
        user_intent: str,
        placement: Optional[Placement] = None,
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Create enriched prompt for intent analysis.
        
        Args:
            user_intent: User's natural language input
            placement: Optional placement for analysis
            context: Optional additional context
            
        Returns:
            List of message dictionaries for xAI API
        """
        # Enrich with mathematical context
        enriched_prompt = self.context_enricher.create_enriched_prompt(
            user_intent=user_intent,
            placement=placement,
            context=context
        )
        
        # If placement available, add ML insights
        ml_summary = ""
        if placement:
            geometry_data = self.context_enricher.enrich_with_geometry(placement)
            physics_data = self.context_enricher.enrich_with_physics(placement)
            
            ml_summary = self.context_summarizer.summarize_for_xai(
                geometry_data,
                physics_data,
                user_intent
            )
        
        # Build system message with reasoning framework
        system_message = """You are an expert PCB design optimization system with deep knowledge of:

**COMPUTATIONAL GEOMETRY:**
- Voronoi Diagrams: Measure component distribution uniformity
  → Low variance = uniform distribution = better thermal spreading
  → High variance = clustering = thermal hotspots
  
- Minimum Spanning Tree (MST): Optimal trace length estimation
  → Shorter MST = shorter traces = better signal integrity
  → Longer MST = longer traces = signal integrity issues
  
- Convex Hull: Board space utilization
  → High utilization = efficient space use
  → Low utilization = wasted space = higher cost

**THERMAL PHYSICS:**
- Heat Equation: ∂T/∂t = α∇²T + Q/(ρcp)
  → Steady-state: ∇²T = -Q/(α·ρcp)
  → High thermal gradient = thermal stress = reliability issues
  
- Heat Flux: q = -k∇T
  → High heat flux = need better cooling
  → Distribute heat sources = lower gradient

**SIGNAL INTEGRITY:**
- Characteristic Impedance: Z₀ = √(L/C)
  → 50Ω for single-ended, 100Ω for differential
  → Impedance mismatch → reflections → signal degradation
  
- Reflection Coefficient: Γ = (Z_L - Z₀)/(Z_L + Z₀)
  → |Γ| < 0.1 for good signal integrity
  → Minimize trace length to reduce reflections

**POWER INTEGRITY:**
- IR Drop: V = I·R
  → Keep IR drop < 5% of supply voltage
  → Add decoupling capacitors near high-current components
  
- Current Density: J = I/A
  → High current density = heating = reliability issues
  → Use wider traces for high-current nets

**OPTIMIZATION THEORY:**
- Multi-Objective Optimization: min Σ(w_i · f_i(x))
  → Balance competing objectives (trace length, thermal, clearance)
  → Use weights (α, β, γ) to prioritize objectives
  
- Pareto Optimality: Solutions where no objective can improve without worsening another
  → Find Pareto-optimal solutions
  → Let user choose trade-offs

**YOUR ROLE:**
Analyze the mathematical and engineering context provided, reason about optimization priorities, and provide actionable insights with quantitative justification."""
        
        # Build user message
        user_message = f"""{enriched_prompt}

{ml_summary}

**TASK:**
Based on the mathematical context above, determine optimal optimization weights (α, β, γ) that sum to 1.0:
- α: Trace length minimization weight
- β: Thermal density minimization weight  
- γ: Clearance violation penalty weight

Provide:
1. Recommended weights with mathematical justification
2. Key optimization priorities based on physics constraints
3. Expected improvements from optimization

Return JSON:
{{
    "alpha": 0.4,
    "beta": 0.4,
    "gamma": 0.2,
    "reasoning": "Detailed explanation referencing mathematical metrics",
    "priorities": ["List of optimization priorities"],
    "expected_improvements": ["List of expected improvements"]
}}"""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def create_optimization_strategy_prompt(
        self,
        geometry_data: Dict,
        physics_data: Dict,
        user_intent: str,
        current_score: float,
        iteration: int,
        max_iterations: int
    ) -> List[Dict]:
        """
        Create enriched prompt for optimization strategy reasoning.
        
        Args:
            geometry_data: Current geometry metrics
            physics_data: Current physics metrics
            user_intent: User intent
            current_score: Current optimization score
            iteration: Current iteration
            max_iterations: Maximum iterations
            
        Returns:
            List of message dictionaries for xAI API
        """
        # Get ML summary
        ml_summary = self.context_summarizer.summarize_for_xai(
            geometry_data,
            physics_data,
            user_intent
        )
        
        progress = (iteration / max_iterations * 100) if max_iterations > 0 else 0
        
        system_message = """You are guiding PCB optimization using simulated annealing. Analyze computational geometry and physics metrics to suggest optimization strategy adjustments."""
        
        user_message = f"""
**OPTIMIZATION STATE (Iteration {iteration}/{max_iterations}, {progress:.1f}% complete):**

{ml_summary}

**CURRENT METRICS:**
- Score: {current_score:.4f} (lower = better)
- Voronoi Variance: {geometry_data.get('voronoi_variance', 0):.4f}
- MST Length: {geometry_data.get('mst_length', 0):.2f} mm
- Thermal Hotspots: {geometry_data.get('thermal_hotspots', 0)}
- Max Temperature: {physics_data.get('max_temperature', 25):.1f}°C

**TASK:**
Analyze current state and suggest:
1. Which components should be moved? (reference geometry/physics data)
2. What optimization strategy adjustments? (temperature, move size, focus areas)
3. What are the key bottlenecks? (thermal, routing, clearance)

Return JSON:
{{
    "priority": "thermal" | "trace_length" | "clearance" | "balanced",
    "suggested_moves": [
        {{"component": "U1", "reason": "Mathematical justification", "direction": "spread_out"}}
    ],
    "strategy": "Detailed strategy with mathematical reasoning",
    "temperature_adjustment": 0.95 | 1.0 | 1.05,
    "bottlenecks": ["List of key bottlenecks"]
}}"""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def create_post_optimization_prompt(
        self,
        initial_geometry: Dict,
        final_geometry: Dict,
        initial_physics: Dict,
        final_physics: Dict,
        initial_score: float,
        final_score: float,
        user_intent: str
    ) -> List[Dict]:
        """
        Create enriched prompt for post-optimization analysis.
        
        Args:
            initial_geometry: Initial geometry metrics
            final_geometry: Final geometry metrics
            initial_physics: Initial physics metrics
            final_physics: Final physics metrics
            initial_score: Initial score
            final_score: Final score
            user_intent: User intent
            
        Returns:
            List of message dictionaries for xAI API
        """
        improvement = ((initial_score - final_score) / initial_score * 100) if initial_score > 0 else 0
        
        system_message = """You are analyzing PCB optimization results. Provide insights on improvements, remaining issues, and recommendations based on mathematical and physics analysis."""
        
        user_message = f"""
**OPTIMIZATION RESULTS:**

**BEFORE:**
- Score: {initial_score:.4f}
- Voronoi Variance: {initial_geometry.get('voronoi_variance', 0):.4f}
- MST Length: {initial_geometry.get('mst_length', 0):.2f} mm
- Thermal Hotspots: {initial_geometry.get('thermal_hotspots', 0)}
- Max Temperature: {initial_physics.get('max_temperature', 25):.1f}°C

**AFTER:**
- Score: {final_score:.4f}
- Voronoi Variance: {final_geometry.get('voronoi_variance', 0):.4f}
- MST Length: {final_geometry.get('mst_length', 0):.2f} mm
- Thermal Hotspots: {final_geometry.get('thermal_hotspots', 0)}
- Max Temperature: {final_physics.get('max_temperature', 25):.1f}°C

**IMPROVEMENT:** {improvement:.1f}% score reduction

**User Intent:** "{user_intent}"

**TASK:**
Analyze the mathematical improvements and provide:
1. Key improvements made (quantify using metrics)
2. Remaining issues (identify bottlenecks)
3. Recommendations for further optimization (actionable, with mathematical justification)

Return JSON:
{{
    "improvements": ["Quantified improvements with metrics"],
    "remaining_issues": ["Remaining problems with metrics"],
    "recommendations": ["Actionable recommendations"],
    "overall_quality": "excellent" | "good" | "fair" | "needs_work",
    "mathematical_assessment": "Detailed mathematical analysis"
}}"""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

