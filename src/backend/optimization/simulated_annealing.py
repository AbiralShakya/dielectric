"""
Simulated Annealing Optimizer with Advanced Physics-Based Cooling Schedules
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, List, Literal
from enum import Enum
try:
    from backend.geometry.placement import Placement
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.scoring.incremental_scorer import IncrementalScorer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.scoring.incremental_scorer import IncrementalScorer


class CoolingSchedule(Enum):
    """Physics-based cooling schedule types."""
    EXPONENTIAL = "exponential"  # Standard exponential decay
    LOGARITHMIC = "logarithmic"  # Slower initial cooling
    LINEAR = "linear"  # Linear temperature reduction
    ADAPTIVE = "adaptive"  # Adaptive based on acceptance rate
    BOLTZMANN = "boltzmann"  # Boltzmann annealing schedule


class SimulatedAnnealing:
    """
    Advanced simulated annealing optimizer with physics-based cooling schedules.
    
    Features:
    - Multiple cooling schedule options (exponential, logarithmic, adaptive, Boltzmann)
    - Geometry-aware perturbation strategies
    - Adaptive temperature adjustment based on acceptance rate
    - Physics-based move generation using computational geometry
    """
    
    def __init__(
        self,
        scorer: IncrementalScorer,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL,
        random_seed: Optional[int] = None,
        use_geometry_guidance: bool = True
    ):
        """
        Initialize simulated annealing.
        
        Args:
            scorer: Incremental scorer
            initial_temp: Initial temperature
            final_temp: Final temperature
            cooling_rate: Temperature decay rate (for exponential schedule)
            max_iterations: Maximum iterations
            cooling_schedule: Type of cooling schedule to use
            random_seed: Random seed for deterministic optimization (None = non-deterministic)
            use_geometry_guidance: Whether to use computational geometry for move guidance
        """
        self.scorer = scorer
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.random_seed = random_seed
        self.use_geometry_guidance = use_geometry_guidance
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
        
        # Adaptive cooling parameters
        self.acceptance_history: List[float] = []
        self.target_acceptance_rate = 0.44  # Optimal acceptance rate from research
    
    def _temperature(self, iteration: int) -> float:
        """
        Compute temperature at iteration using selected cooling schedule.
        
        Physics-based cooling schedules:
        - Exponential: T(k) = T0 * α^k (standard)
        - Logarithmic: T(k) = T0 / (1 + log(1 + k)) (slower initial cooling)
        - Linear: T(k) = T0 * (1 - k/K) (linear reduction)
        - Adaptive: Adjusts based on acceptance rate
        - Boltzmann: T(k) = T0 / log(1 + k) (Boltzmann annealing)
        """
        k = iteration
        K = self.max_iterations
        
        if self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            temp = self.initial_temp * (self.cooling_rate ** k)
        
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            temp = self.initial_temp / (1.0 + np.log(1.0 + k))
        
        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            temp = self.initial_temp * (1.0 - k / K)
        
        elif self.cooling_schedule == CoolingSchedule.BOLTZMANN:
            temp = self.initial_temp / np.log(1.0 + k + 1)
        
        elif self.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # Adaptive cooling: adjust based on acceptance rate
            base_temp = self.initial_temp * (self.cooling_rate ** k)
            
            if len(self.acceptance_history) > 10:
                recent_acceptance = np.mean(self.acceptance_history[-10:])
                # If acceptance rate too high, cool faster; if too low, cool slower
                if recent_acceptance > self.target_acceptance_rate * 1.2:
                    # Too many acceptances, cool faster
                    temp = base_temp * 0.9
                elif recent_acceptance < self.target_acceptance_rate * 0.8:
                    # Too few acceptances, cool slower
                    temp = base_temp * 1.1
                else:
                    temp = base_temp
            else:
                temp = base_temp
        
        else:
            temp = self.initial_temp * (self.cooling_rate ** k)
        
        return max(temp, self.final_temp)
    
    def _perturb(self, placement: Placement, geometry_analyzer: Optional[GeometryAnalyzer] = None) -> tuple:
        """
        Generate a physics-aware perturbation (move or swap).
        
        Uses computational geometry to guide moves:
        - Prefer moving components with high thermal risk
        - Prefer moving components with poor Voronoi cell distribution
        - Use MST edges to guide component placement
        - Prefer swaps between components on same net
        
        Returns:
            (move_type, component_name, new_x, new_y, new_angle) or
            (move_type, comp1_name, comp2_name) for swap
        """
        comp_names = list(placement.components.keys())
        
        if self.use_geometry_guidance and geometry_analyzer:
            # Use geometry to guide perturbation
            geometry_data = geometry_analyzer.analyze()
            
            # Identify components with high thermal risk
            hotspots = geometry_data.get("hotspot_locations", [])
            hotspot_comps = [h["component"] for h in hotspots if h.get("component") in comp_names]
            
            # Identify components with poor Voronoi distribution
            voronoi_data = geometry_data.get("voronoi_data", {})
            cell_areas = voronoi_data.get("cell_areas", [])
            if cell_areas:
                avg_area = np.mean(cell_areas)
                std_area = np.std(cell_areas)
                # Components with unusually small/large Voronoi cells
                problematic_comps = []
                for i, comp_name in enumerate(comp_names):
                    if i < len(cell_areas):
                        if abs(cell_areas[i] - avg_area) > 2 * std_area:
                            problematic_comps.append(comp_name)
            
            # Prefer moving problematic components
            move_candidates = []
            if hotspot_comps:
                move_candidates.extend(hotspot_comps)
            if problematic_comps:
                move_candidates.extend(problematic_comps)
            
            if move_candidates and self.rng.rand() < 0.6:  # 60% chance to use guidance
                comp_name = self.rng.choice(move_candidates)
            else:
                comp_name = self.rng.choice(comp_names)
        else:
            comp_name = self.rng.choice(comp_names)
        
        comp = placement.get_component(comp_name)
        if not comp:
            comp_name = comp_names[0]
            comp = placement.get_component(comp_name)
        
        if self.rng.rand() < 0.7:  # 70% move, 30% swap
            # Move a component with geometry-aware positioning
            margin = max(comp.width, comp.height)
            
            if self.use_geometry_guidance and geometry_analyzer:
                # Try to place near MST edges or improve Voronoi distribution
                geometry_data = geometry_analyzer.analyze()
                mst_edges = geometry_data.get("mst_edges", [])
                
                if mst_edges and self.rng.rand() < 0.3:  # 30% chance to use MST guidance
                    # Place component near MST edge midpoint
                    edge = self.rng.choice(mst_edges)
                    if len(edge) >= 3:
                        i, j, weight = edge[0], edge[1], edge[2]
                        comps_list = list(placement.components.values())
                        if i < len(comps_list) and j < len(comps_list):
                            c1, c2 = comps_list[i], comps_list[j]
                            mid_x = (c1.x + c2.x) / 2
                            mid_y = (c1.y + c2.y) / 2
                            # Add some noise
                            new_x = mid_x + self.rng.normal(0, margin)
                            new_y = mid_y + self.rng.normal(0, margin)
                            new_x = np.clip(new_x, margin, placement.board.width - margin)
                            new_y = np.clip(new_y, margin, placement.board.height - margin)
                        else:
                            new_x = self.rng.uniform(margin, placement.board.width - margin)
                            new_y = self.rng.uniform(margin, placement.board.height - margin)
                    else:
                        new_x = self.rng.uniform(margin, placement.board.width - margin)
                        new_y = self.rng.uniform(margin, placement.board.height - margin)
                else:
                    # Standard random move with adaptive step size
                    # Step size decreases as optimization progresses
                    step_size = min(placement.board.width, placement.board.height) * 0.3
                    new_x = comp.x + self.rng.normal(0, step_size)
                    new_y = comp.y + self.rng.normal(0, step_size)
                    new_x = np.clip(new_x, margin, placement.board.width - margin)
                    new_y = np.clip(new_y, margin, placement.board.height - margin)
            else:
                # Standard random position
                new_x = self.rng.uniform(margin, placement.board.width - margin)
                new_y = self.rng.uniform(margin, placement.board.height - margin)
            
            new_angle = self.rng.choice([0, 90, 180, 270])
            
            return ("move", comp_name, new_x, new_y, new_angle)
        else:
            # Swap two components
            if len(comp_names) >= 2:
                # Prefer swapping components on same net
                if self.use_geometry_guidance:
                    comp1_name = comp_name
                    comp1_nets = placement.get_affected_nets(comp1_name)
                    if comp1_nets:
                        # Find another component on same net
                        candidates = []
                        for net_name in comp1_nets:
                            net = placement.nets.get(net_name)
                            if net:
                                for other_comp_name, _ in net.pins:
                                    if other_comp_name != comp1_name and other_comp_name in comp_names:
                                        candidates.append(other_comp_name)
                        
                        if candidates and self.rng.rand() < 0.5:
                            comp2_name = self.rng.choice(candidates)
                            return ("swap", comp1_name, comp2_name)
                    
                    # Fallback to random swap
                    comp1, comp2 = self.rng.choice(comp_names, size=2, replace=False)
                    return ("swap", comp1, comp2)
                else:
                    comp1, comp2 = self.rng.choice(comp_names, size=2, replace=False)
                    return ("swap", comp1, comp2)
            else:
                # Fallback to move
                margin = max(comp.width, comp.height)
                new_x = self.rng.uniform(margin, placement.board.width - margin)
                new_y = self.rng.uniform(margin, placement.board.height - margin)
                return ("move", comp_name, new_x, new_y, 0)
    
    def optimize(
        self,
        placement: Placement,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> tuple[Placement, float, Dict]:
        """
        Optimize placement using advanced simulated annealing with physics-based cooling.
        
        Args:
            placement: Initial placement
            callback: Optional callback(iteration, score, placement)
            timeout: Optional timeout in seconds
        
        Returns:
            (best_placement, best_score, stats)
        """
        start_time = time.time()
        best_placement = placement.copy()
        current_placement = placement.copy()
        
        best_score = self.scorer.score(best_placement)
        current_score = best_score
        
        # Initialize geometry analyzer if using geometry guidance
        geometry_analyzer = None
        if self.use_geometry_guidance:
            try:
                geometry_analyzer = GeometryAnalyzer(current_placement)
            except Exception:
                geometry_analyzer = None
        
        stats = {
            "iterations": 0,
            "acceptances": 0,
            "rejections": 0,
            "improvements": 0,
            "scores": [best_score],
            "cooling_schedule": self.cooling_schedule.value,
            "acceptance_rates": []
        }
        
        for iteration in range(self.max_iterations):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            temp = self._temperature(iteration)
            
            # Generate perturbation with geometry guidance
            move = self._perturb(current_placement, geometry_analyzer)
            
            if move[0] == "move":
                _, comp_name, new_x, new_y, new_angle = move
                comp = current_placement.get_component(comp_name)
                
                if not comp:
                    continue
                
                # Save old state
                old_x, old_y, old_angle = comp.x, comp.y, comp.angle
                
                # Compute delta score
                delta = self.scorer.compute_delta_score(
                    current_placement, comp_name,
                    old_x, old_y, old_angle,
                    new_x, new_y, new_angle
                )
                
                # Accept or reject
                accept = False
                if delta < 0:  # Improvement
                    accept = True
                    stats["improvements"] += 1
                elif temp > 0:
                    prob = np.exp(-delta / temp)
                    if self.rng.rand() < prob:
                        accept = True
                
                if accept:
                    comp.x, comp.y, comp.angle = new_x, new_y, new_angle
                    current_score += delta
                    stats["acceptances"] += 1
                    
                    # Update geometry analyzer after move
                    if geometry_analyzer:
                        try:
                            geometry_analyzer = GeometryAnalyzer(current_placement)
                        except Exception:
                            pass
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_placement = current_placement.copy()
                else:
                    stats["rejections"] += 1
            
            elif move[0] == "swap":
                _, comp1_name, comp2_name = move
                current_placement.swap_components(comp1_name, comp2_name)
                # Recompute full score (swap affects many nets)
                current_score = self.scorer.score(current_placement, use_cache=False)
                
                # Update geometry analyzer after swap
                if geometry_analyzer:
                    try:
                        geometry_analyzer = GeometryAnalyzer(current_placement)
                    except Exception:
                        pass
                
                if current_score < best_score:
                    best_score = current_score
                    best_placement = current_placement.copy()
                    stats["improvements"] += 1
                stats["acceptances"] += 1
            
            # Track acceptance rate for adaptive cooling
            if self.cooling_schedule == CoolingSchedule.ADAPTIVE:
                total_moves = stats["acceptances"] + stats["rejections"]
                if total_moves > 0:
                    acceptance_rate = stats["acceptances"] / total_moves
                    self.acceptance_history.append(acceptance_rate)
                    if len(self.acceptance_history) > 100:
                        self.acceptance_history.pop(0)
                    stats["acceptance_rates"].append(acceptance_rate)
            
            stats["iterations"] += 1
            stats["scores"].append(current_score)
            
            if callback:
                callback(iteration, current_score, current_placement)
        
        stats["final_score"] = best_score
        stats["time_elapsed"] = time.time() - start_time
        stats["final_temperature"] = self._temperature(stats["iterations"] - 1)
        
        return best_placement, best_score, stats

