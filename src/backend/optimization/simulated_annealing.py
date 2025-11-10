"""
Simulated Annealing Optimizer
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, List
try:
    from backend.geometry.placement import Placement
    from backend.scoring.incremental_scorer import IncrementalScorer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.incremental_scorer import IncrementalScorer


class SimulatedAnnealing:
    """Simulated annealing optimizer for fast path."""
    
    def __init__(
        self,
        scorer: IncrementalScorer,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize simulated annealing.
        
        Args:
            scorer: Incremental scorer
            initial_temp: Initial temperature
            final_temp: Final temperature
            cooling_rate: Temperature decay rate
            max_iterations: Maximum iterations
            random_seed: Random seed for deterministic optimization (None = non-deterministic)
        """
        self.scorer = scorer
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    
    def _temperature(self, iteration: int) -> float:
        """Compute temperature at iteration."""
        temp = self.initial_temp * (self.cooling_rate ** iteration)
        return max(temp, self.final_temp)
    
    def _perturb(self, placement: Placement) -> tuple:
        """
        Generate a perturbation (move or swap).
        
        Returns:
            (move_type, component_name, new_x, new_y, new_angle) or
            (move_type, comp1_name, comp2_name) for swap
        """
        comp_names = list(placement.components.keys())
        
        if self.rng.rand() < 0.7:  # 70% move, 30% swap
            # Move a component
            comp_name = self.rng.choice(comp_names)
            comp = placement.get_component(comp_name)
            
            # Random new position
            margin = max(comp.width, comp.height)
            new_x = self.rng.uniform(margin, placement.board.width - margin)
            new_y = self.rng.uniform(margin, placement.board.height - margin)
            new_angle = self.rng.choice([0, 90, 180, 270])
            
            return ("move", comp_name, new_x, new_y, new_angle)
        else:
            # Swap two components
            if len(comp_names) >= 2:
                comp1, comp2 = self.rng.choice(comp_names, size=2, replace=False)
                return ("swap", comp1, comp2)
            else:
                # Fallback to move
                comp_name = comp_names[0]
                comp = placement.get_component(comp_name)
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
        Optimize placement using simulated annealing.
        
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
        
        stats = {
            "iterations": 0,
            "acceptances": 0,
            "rejections": 0,
            "improvements": 0,
            "scores": [best_score]
        }
        
        for iteration in range(self.max_iterations):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            temp = self._temperature(iteration)
            
            # Generate perturbation
            move = self._perturb(current_placement)
            
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
                
                if current_score < best_score:
                    best_score = current_score
                    best_placement = current_placement.copy()
                    stats["improvements"] += 1
                stats["acceptances"] += 1
            
            stats["iterations"] += 1
            stats["scores"].append(current_score)
            
            if callback:
                callback(iteration, current_score, current_placement)
        
        stats["final_score"] = best_score
        stats["time_elapsed"] = time.time() - start_time
        
        return best_placement, best_score, stats

