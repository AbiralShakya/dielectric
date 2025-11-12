"""
Context Enrichment for xAI Reasoning

Enriches user prompts with rich mathematical, geometric, and physics context
to enable better reasoning by Grok/xAI.

Leverages:
- Computational geometry mathematics
- Physics equations and constraints
- Electrical engineering principles
- Optimization theory
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import math

try:
    from backend.geometry.placement import Placement
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.scoring.scorer import WorldModelScorer
    from backend.simulation.pcb_simulator import PCBSimulator
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.scoring.scorer import WorldModelScorer
    from src.backend.simulation.pcb_simulator import PCBSimulator


@dataclass
class MathematicalContext:
    """Structured mathematical context for AI reasoning."""
    # Geometry metrics
    voronoi_variance: float
    mst_length: float
    convex_hull_area: float
    convex_hull_utilization: float
    
    # Thermal physics
    thermal_hotspots: int
    max_temperature: float
    thermal_gradient: float
    heat_flux_estimate: float
    
    # Signal integrity
    estimated_trace_length: float
    net_crossings: int
    routing_complexity: float
    impedance_mismatch_risk: float
    
    # Power integrity
    power_density: float
    current_density_estimate: float
    ir_drop_estimate: float
    
    # Optimization metrics
    current_score: float
    score_components: Dict[str, float]
    optimization_progress: float
    
    # Component analysis
    component_density: float
    power_distribution: Dict[str, float]
    critical_nets: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_structured_prompt(self) -> str:
        """Convert to structured prompt format for xAI."""
        return f"""
**COMPUTATIONAL GEOMETRY ANALYSIS:**
- Voronoi Cell Variance: {self.voronoi_variance:.4f} (lower = uniform distribution, ideal < 0.1)
  → Mathematical interpretation: Measures spatial distribution uniformity using Voronoi diagram
  → Engineering insight: High variance indicates clustering → thermal hotspots
  
- Minimum Spanning Tree Length: {self.mst_length:.2f} mm
  → Mathematical interpretation: Optimal trace length estimate using graph theory
  → Engineering insight: Longer MST → longer traces → signal integrity issues
  
- Convex Hull Utilization: {self.convex_hull_utilization:.2%}
  → Mathematical interpretation: Board space efficiency using convex hull geometry
  → Engineering insight: Low utilization → wasted space → larger board cost

**THERMAL PHYSICS ANALYSIS:**
- Thermal Hotspots: {self.thermal_hotspots} regions
- Maximum Temperature: {self.max_temperature:.1f}°C
- Thermal Gradient: {self.thermal_gradient:.2f}°C/mm
- Heat Flux Estimate: {self.heat_flux_estimate:.2f} W/m²
  → Physics: Heat equation ∂T/∂t = α∇²T + Q/(ρcp)
  → Engineering constraint: Max temp < 85°C for most components
  → Optimization goal: Minimize thermal gradient, distribute heat sources

**SIGNAL INTEGRITY ANALYSIS:**
- Estimated Trace Length: {self.estimated_trace_length:.2f} mm
- Net Crossings: {self.net_crossings} potential conflicts
- Routing Complexity: {self.routing_complexity:.2f}
- Impedance Mismatch Risk: {self.impedance_mismatch_risk:.2f}
  → Physics: Characteristic impedance Z₀ = √(L/C), reflection coefficient Γ = (Z_L - Z₀)/(Z_L + Z₀)
  → Engineering constraint: Z₀ = 50Ω for RF, 100Ω for differential pairs
  → Optimization goal: Minimize trace length, avoid crossings, maintain impedance

**POWER INTEGRITY ANALYSIS:**
- Power Density: {self.power_density:.2f} W/mm²
- Current Density Estimate: {self.current_density_estimate:.2f} A/mm²
- IR Drop Estimate: {self.ir_drop_estimate:.3f} V
  → Physics: IR drop V = I·R, power loss P = I²R
  → Engineering constraint: IR drop < 5% of supply voltage
  → Optimization goal: Minimize trace resistance, add decoupling caps

**OPTIMIZATION STATE:**
- Current Score: {self.current_score:.4f} (lower = better)
- Score Components: {json.dumps(self.score_components, indent=2)}
- Progress: {self.optimization_progress:.1%}
  → Optimization theory: Multi-objective optimization with weighted sum
  → Current state: Balancing trace length, thermal, clearance, DFM
"""


class ContextEnricher:
    """
    Enriches prompts with mathematical and engineering context.
    
    Extracts insights from:
    - Computational geometry
    - Physics simulation
    - Electrical engineering principles
    - Optimization mathematics
    """
    
    def __init__(self):
        """Initialize context enricher."""
        self.geometry_analyzer = None
        self.scorer = None
        self.simulator = None
    
    def enrich_with_geometry(self, placement: Placement) -> Dict:
        """
        Extract rich geometric context from placement.
        
        Returns:
            Dictionary with geometric insights
        """
        if self.geometry_analyzer is None:
            self.geometry_analyzer = GeometryAnalyzer(placement)
        
        geometry_data = self.geometry_analyzer.analyze()
        
        # Extract mathematical insights
        voronoi_var = geometry_data.get('voronoi_variance', 0.0)
        mst_length = geometry_data.get('mst_length', 0.0)
        convex_hull_area = geometry_data.get('convex_hull_area', 0.0)
        board_area = placement.board.width * placement.board.height
        utilization = convex_hull_area / board_area if board_area > 0 else 0.0
        
        return {
            "voronoi_variance": voronoi_var,
            "mst_length": mst_length,
            "convex_hull_area": convex_hull_area,
            "convex_hull_utilization": utilization,
            "component_density": len(placement.components) / board_area if board_area > 0 else 0.0,
            "net_crossings": geometry_data.get('net_crossings', 0),
            "routing_complexity": geometry_data.get('routing_complexity', 0.0),
            "thermal_hotspots": geometry_data.get('thermal_hotspots', 0),
            "overlap_risk": geometry_data.get('overlap_risk', 0.0)
        }
    
    def enrich_with_physics(self, placement: Placement) -> Dict:
        """
        Extract physics-based context from placement.
        
        Returns:
            Dictionary with physics insights
        """
        if self.simulator is None:
            self.simulator = PCBSimulator()
        
        # Run thermal simulation
        thermal_results = self.simulator.simulate_thermal(placement)
        
        # Extract thermal insights
        temp_map = thermal_results.get('board_temperature_map', np.zeros((10, 10)))
        max_temp = float(np.max(temp_map)) if isinstance(temp_map, np.ndarray) else thermal_results.get('max_temperature', 25.0)
        min_temp = float(np.min(temp_map)) if isinstance(temp_map, np.ndarray) else thermal_results.get('min_temperature', 25.0)
        thermal_gradient = max_temp - min_temp
        
        # Estimate heat flux (simplified)
        total_power = sum(getattr(comp, 'power', 0.0) for comp in placement.components.values())
        board_area = placement.board.width * placement.board.height / 1e6  # Convert to m²
        heat_flux = total_power / board_area if board_area > 0 else 0.0
        
        # Power density
        power_density = total_power / (placement.board.width * placement.board.height) if (placement.board.width * placement.board.height) > 0 else 0.0
        
        # Estimate current density (simplified)
        # Assume average trace width 0.2mm, 1A current
        avg_trace_width = 0.2  # mm
        avg_current = 1.0  # A
        current_density = avg_current / (avg_trace_width * 0.001)  # A/mm²
        
        # Estimate IR drop (simplified)
        # V = I·R, R = ρ·L/A
        trace_resistivity = 1.7e-8  # Ω·m for copper
        avg_trace_length = self._estimate_avg_trace_length(placement)
        trace_cross_section = avg_trace_width * 0.035 * 1e-6  # m² (0.035mm thickness)
        trace_resistance = trace_resistivity * (avg_trace_length / 1000) / trace_cross_section
        ir_drop = avg_current * trace_resistance
        
        return {
            "max_temperature": max_temp,
            "min_temperature": min_temp,
            "thermal_gradient": thermal_gradient,
            "heat_flux_estimate": heat_flux,
            "power_density": power_density,
            "current_density_estimate": current_density,
            "ir_drop_estimate": ir_drop,
            "total_power": total_power
        }
    
    def enrich_with_optimization(self, placement: Placement, weights: Dict, current_score: float, iteration: int = 0, max_iterations: int = 1000) -> Dict:
        """
        Extract optimization context.
        
        Returns:
            Dictionary with optimization insights
        """
        if self.scorer is None:
            self.scorer = WorldModelScorer()
        
        # Compute score components
        trace_length = self.scorer.compute_trace_length(placement)
        thermal_density = self.scorer.compute_thermal_density(placement)
        clearance_violations = self.scorer.compute_clearance_violations(placement)
        
        score_components = {
            "trace_length": float(trace_length),
            "thermal_density": float(thermal_density),
            "clearance_violations": float(clearance_violations)
        }
        
        progress = iteration / max_iterations if max_iterations > 0 else 0.0
        
        return {
            "current_score": current_score,
            "score_components": score_components,
            "optimization_progress": progress,
            "weights": weights
        }
    
    def enrich_with_electrical_engineering(self, placement: Placement) -> Dict:
        """
        Extract electrical engineering context.
        
        Returns:
            Dictionary with EE insights
        """
        # Identify critical nets
        critical_nets = []
        
        for net_name, net in placement.nets.items():
            net_lower = net_name.lower()
            
            # Power nets
            if any(keyword in net_lower for keyword in ['vcc', 'vdd', 'power', 'supply']):
                critical_nets.append(f"{net_name} (Power)")
            
            # Clock nets
            if any(keyword in net_lower for keyword in ['clock', 'clk', 'osc']):
                critical_nets.append(f"{net_name} (Clock)")
            
            # Differential pairs
            if any(keyword in net_lower for keyword in ['diff', 'pair', 'd+', 'd-']):
                critical_nets.append(f"{net_name} (Differential)")
        
        # Power distribution analysis
        power_distribution = {}
        for comp in placement.components.values():
            comp_power = getattr(comp, 'power', 0.0)
            if comp_power > 0:
                power_distribution[comp.name] = comp_power
        
        # Estimate impedance mismatch risk
        # Simplified: based on trace length and frequency
        avg_trace_length = self._estimate_avg_trace_length(placement)
        # Assume 100MHz signal, trace length > λ/10 causes issues
        wavelength = 3e8 / (100e6 * np.sqrt(4.5))  # Approximate for FR4
        critical_length = wavelength / 10
        impedance_risk = min(1.0, avg_trace_length / critical_length) if critical_length > 0 else 0.0
        
        return {
            "critical_nets": critical_nets,
            "power_distribution": power_distribution,
            "impedance_mismatch_risk": impedance_risk,
            "estimated_trace_length": avg_trace_length
        }
    
    def create_enriched_prompt(
        self,
        user_intent: str,
        placement: Optional[Placement] = None,
        weights: Optional[Dict] = None,
        current_score: Optional[float] = None,
        iteration: int = 0,
        max_iterations: int = 1000
    ) -> str:
        """
        Create enriched prompt with all mathematical and engineering context.
        
        Args:
            user_intent: User's natural language input
            placement: Optional placement for analysis
            weights: Optional optimization weights
            current_score: Optional current score
            iteration: Current iteration
            max_iterations: Maximum iterations
            
        Returns:
            Enriched prompt string
        """
        # Build mathematical context
        math_context = MathematicalContext(
            voronoi_variance=0.0,
            mst_length=0.0,
            convex_hull_area=0.0,
            convex_hull_utilization=0.0,
            thermal_hotspots=0,
            max_temperature=25.0,
            thermal_gradient=0.0,
            heat_flux_estimate=0.0,
            estimated_trace_length=0.0,
            net_crossings=0,
            routing_complexity=0.0,
            impedance_mismatch_risk=0.0,
            power_density=0.0,
            current_density_estimate=0.0,
            ir_drop_estimate=0.0,
            current_score=current_score or 0.0,
            score_components={},
            optimization_progress=0.0,
            component_density=0.0,
            power_distribution={},
            critical_nets=[]
        )
        
        if placement:
            # Enrich with geometry
            geometry_data = self.enrich_with_geometry(placement)
            math_context.voronoi_variance = geometry_data.get('voronoi_variance', 0.0)
            math_context.mst_length = geometry_data.get('mst_length', 0.0)
            math_context.convex_hull_area = geometry_data.get('convex_hull_area', 0.0)
            math_context.convex_hull_utilization = geometry_data.get('convex_hull_utilization', 0.0)
            math_context.component_density = geometry_data.get('component_density', 0.0)
            math_context.net_crossings = geometry_data.get('net_crossings', 0)
            math_context.routing_complexity = geometry_data.get('routing_complexity', 0.0)
            math_context.thermal_hotspots = geometry_data.get('thermal_hotspots', 0)
            
            # Enrich with physics
            physics_data = self.enrich_with_physics(placement)
            math_context.max_temperature = physics_data.get('max_temperature', 25.0)
            math_context.thermal_gradient = physics_data.get('thermal_gradient', 0.0)
            math_context.heat_flux_estimate = physics_data.get('heat_flux_estimate', 0.0)
            math_context.power_density = physics_data.get('power_density', 0.0)
            math_context.current_density_estimate = physics_data.get('current_density_estimate', 0.0)
            math_context.ir_drop_estimate = physics_data.get('ir_drop_estimate', 0.0)
            
            # Enrich with electrical engineering
            ee_data = self.enrich_with_electrical_engineering(placement)
            math_context.estimated_trace_length = ee_data.get('estimated_trace_length', 0.0)
            math_context.impedance_mismatch_risk = ee_data.get('impedance_mismatch_risk', 0.0)
            math_context.critical_nets = ee_data.get('critical_nets', [])
            math_context.power_distribution = ee_data.get('power_distribution', {})
        
        if weights and current_score is not None:
            # Enrich with optimization
            opt_data = self.enrich_with_optimization(placement or Placement(), weights, current_score, iteration, max_iterations)
            math_context.current_score = opt_data.get('current_score', 0.0)
            math_context.score_components = opt_data.get('score_components', {})
            math_context.optimization_progress = opt_data.get('optimization_progress', 0.0)
        
        # Build enriched prompt
        enriched_prompt = f"""
**USER INTENT:** "{user_intent}"

{math_context.to_structured_prompt()}

**MATHEMATICAL REASONING FRAMEWORK:**

You are reasoning about PCB optimization using:
1. **Computational Geometry**: Voronoi diagrams, MST, convex hulls
2. **Thermal Physics**: Heat equation, thermal gradients, heat flux
3. **Signal Integrity**: Impedance matching, reflection coefficients, trace length
4. **Power Integrity**: IR drop, current density, decoupling
5. **Optimization Theory**: Multi-objective optimization, weighted sum, Pareto optimality

**YOUR TASK:**
Analyze the mathematical and engineering context above and provide actionable insights:
1. What are the primary optimization opportunities?
2. What physics constraints are most critical?
3. What geometric improvements would have the most impact?
4. What electrical engineering principles should guide optimization?

Be specific, quantitative, and reference the mathematical metrics provided.
"""
        
        return enriched_prompt
    
    def _estimate_avg_trace_length(self, placement: Placement) -> float:
        """Estimate average trace length."""
        if not placement.nets:
            return 0.0
        
        total_length = 0.0
        net_count = 0
        
        for net in placement.nets.values():
            if len(net.pins) < 2:
                continue
            
            # Get component positions
            positions = []
            for comp_name, pin_name in net.pins:
                comp = placement.get_component(comp_name)
                if comp:
                    positions.append((comp.x, comp.y))
            
            if len(positions) >= 2:
                # Estimate MST length
                positions_array = np.array(positions)
                distances = []
                for i in range(len(positions_array)):
                    for j in range(i + 1, len(positions_array)):
                        dist = np.linalg.norm(positions_array[i] - positions_array[j])
                        distances.append(dist)
                
                if distances:
                    total_length += np.mean(distances)
                    net_count += 1
        
        return total_length / net_count if net_count > 0 else 0.0

