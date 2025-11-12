"""
Enhanced Simulated Annealing with xAI Reasoning Integration

Periodically uses xAI to reason about optimization strategy based on computational geometry.
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, List, Tuple
try:
    from backend.geometry.placement import Placement
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.scoring.incremental_scorer import IncrementalScorer
    from backend.ai.enhanced_xai_client import EnhancedXAIClient
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.scoring.incremental_scorer import IncrementalScorer
    from src.backend.ai.enhanced_xai_client import EnhancedXAIClient


class EnhancedSimulatedAnnealing:
    """
    Enhanced simulated annealing with xAI reasoning integration.
    
    Uses xAI to:
    1. Analyze computational geometry metrics periodically
    2. Suggest optimization strategy adjustments
    3. Guide component moves based on geometry analysis
    """
    
    def __init__(
        self,
        scorer: IncrementalScorer,
        xai_client: Optional[EnhancedXAIClient] = None,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        reasoning_interval: int = 50,  # Call xAI every N iterations
        random_seed: Optional[int] = None
    ):
        """
        Initialize enhanced simulated annealing.
        
        Args:
            scorer: Incremental scorer
            xai_client: Enhanced xAI client for reasoning
            initial_temp: Initial temperature
            final_temp: Final temperature
            cooling_rate: Temperature decay rate
            max_iterations: Maximum iterations
            reasoning_interval: How often to call xAI for reasoning
            random_seed: Random seed
        """
        self.scorer = scorer
        self.xai_client = xai_client
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.reasoning_interval = reasoning_interval
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
        
        self.geometry_analyzer = None
        self.current_strategy = {"priority": "balanced", "temperature_adjustment": 1.0}
        self.reasoning_calls = 0
    
    def _temperature(self, iteration: int) -> float:
        """Compute temperature at iteration with strategy adjustment."""
        base_temp = self.initial_temp * (self.cooling_rate ** iteration)
        
        # Adjust based on xAI strategy
        adjusted_temp = base_temp * self.current_strategy.get("temperature_adjustment", 1.0)
        
        return max(adjusted_temp, self.final_temp)
    
    def _perturb_with_strategy(self, placement: Placement) -> tuple:
        """
        Generate perturbation guided by xAI strategy.
        
        Uses strategy to prioritize certain types of moves.
        """
        comp_names = list(placement.components.keys())
        
        # Check if strategy suggests specific moves
        suggested_moves = self.current_strategy.get("suggested_moves", [])
        
        if suggested_moves and self.rng.rand() < 0.3:  # 30% chance to follow suggestion
            # Try to follow xAI suggestion
            suggested_comp = suggested_moves[0].get("component")
            if suggested_comp in comp_names:
                comp = placement.get_component(suggested_comp)
                if comp:
                    margin = max(comp.width, comp.height)
                    new_x = self.rng.uniform(margin, placement.board.width - margin)
                    new_y = self.rng.uniform(margin, placement.board.height - margin)
                    new_angle = self.rng.choice([0, 90, 180, 270])
                    return ("move", suggested_comp, new_x, new_y, new_angle)
        
        # Default perturbation logic
        if self.rng.rand() < 0.7:  # 70% move, 30% swap
            comp_name = self.rng.choice(comp_names)
            comp = placement.get_component(comp_name)
            
            margin = max(comp.width, comp.height)
            new_x = self.rng.uniform(margin, placement.board.width - margin)
            new_y = self.rng.uniform(margin, placement.board.height - margin)
            new_angle = self.rng.choice([0, 90, 180, 270])
            
            return ("move", comp_name, new_x, new_y, new_angle)
        else:
            if len(comp_names) >= 2:
                comp1, comp2 = self.rng.choice(comp_names, size=2, replace=False)
                return ("swap", comp1, comp2)
            else:
                comp_name = comp_names[0]
                comp = placement.get_component(comp_name)
                margin = max(comp.width, comp.height)
                new_x = self.rng.uniform(margin, placement.board.width - margin)
                new_y = self.rng.uniform(margin, placement.board.height - margin)
                return ("move", comp_name, new_x, new_y, 0)
    
    def optimize(
        self,
        placement: Placement,
        user_intent: str = "Optimize placement",
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> tuple[Placement, float, Dict]:
        """
        Optimize placement using enhanced simulated annealing with xAI reasoning.
        
        Args:
            placement: Initial placement
            user_intent: User's optimization intent
            callback: Optional callback(iteration, score, placement)
            timeout: Optional timeout in seconds
        
        Returns:
            (best_placement, best_score, stats)
        """
        start_time = time.time()
        best_placement = placement.copy()
        current_placement = placement.copy()
        
        # Initialize geometry analyzer
        self.geometry_analyzer = GeometryAnalyzer(current_placement)
        
        best_score = self.scorer.score(best_placement)
        current_score = best_score
        
        initial_geometry = self.geometry_analyzer.analyze()
        
        stats = {
            "iterations": 0,
            "acceptances": 0,
            "rejections": 0,
            "improvements": 0,
            "scores": [best_score],
            "reasoning_calls": 0,
            "geometry_updates": []
        }
        
        for iteration in range(self.max_iterations):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            temp = self._temperature(iteration)
            
            # Periodic xAI reasoning
            if (self.xai_client and 
                iteration > 0 and 
                iteration % self.reasoning_interval == 0 and
                self.geometry_analyzer):
                
                try:
                    # Update geometry analysis
                    geometry_data = self.geometry_analyzer.analyze()
                    
                    # Call xAI for reasoning
                    strategy = self.xai_client.reason_about_geometry_and_optimize(
                        geometry_data=geometry_data,
                        user_intent=user_intent,
                        current_score=current_score,
                        iteration=iteration,
                        max_iterations=self.max_iterations
                    )
                    
                    self.current_strategy = strategy
                    stats["reasoning_calls"] += 1
                    stats["geometry_updates"].append({
                        "iteration": iteration,
                        "geometry": geometry_data,
                        "strategy": strategy
                    })
                    
                    print(f"   🤖 xAI Reasoning (iter {iteration}): {strategy.get('priority', 'balanced')} priority")
                    
                except Exception as e:
                    print(f"   ⚠️  xAI reasoning failed: {e}")
                    # Continue with default strategy
            
            # Generate perturbation
            move = self._perturb_with_strategy(current_placement)
            
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
                    
                    # Update geometry analyzer
                    if self.geometry_analyzer:
                        self.geometry_analyzer = GeometryAnalyzer(current_placement)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_placement = current_placement.copy()
                else:
                    stats["rejections"] += 1
            
            elif move[0] == "swap":
                _, comp1_name, comp2_name = move
                current_placement.swap_components(comp1_name, comp2_name)
                # Recompute full score
                current_score = self.scorer.score(current_placement, use_cache=False)
                
                # Update geometry analyzer
                if self.geometry_analyzer:
                    self.geometry_analyzer = GeometryAnalyzer(current_placement)
                
                if current_score < best_score:
                    best_score = current_score
                    best_placement = current_placement.copy()
                    stats["improvements"] += 1
                stats["acceptances"] += 1
            
            stats["iterations"] += 1
            stats["scores"].append(current_score)
            
            if callback:
                callback(iteration, current_score, current_placement)
        
        # Final geometry analysis
        final_geometry = None
        if self.geometry_analyzer:
            final_geometry = self.geometry_analyzer.analyze()
        
        stats["final_score"] = best_score
        stats["time_elapsed"] = time.time() - start_time
        stats["initial_geometry"] = initial_geometry
        stats["final_geometry"] = final_geometry
        
        return best_placement, best_score, stats

