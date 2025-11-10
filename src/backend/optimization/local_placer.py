"""
Local Placer - Fast Path Optimizer

Runs quick local optimization for instant UI feedback.
"""

import numpy as np
import time
from typing import Optional, Callable, Dict, List
try:
    from backend.geometry.placement import Placement
    from backend.scoring.incremental_scorer import IncrementalScorer
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.scoring.incremental_scorer import IncrementalScorer
from .simulated_annealing import SimulatedAnnealing


class LocalPlacer:
    """
    Fast local optimizer for interactive placement.
    
    Designed for <200ms response time.
    """
    
    def __init__(self, scorer: IncrementalScorer, random_seed: Optional[int] = None):
        """
        Initialize local placer.
        
        Args:
            scorer: Incremental scorer
            random_seed: Random seed for deterministic optimization
        """
        self.scorer = scorer
        self.sa = SimulatedAnnealing(
            scorer=scorer,
            initial_temp=50.0,
            final_temp=0.1,
            cooling_rate=0.9,
            max_iterations=200,  # Fast path: fewer iterations
            random_seed=random_seed
        )
    
    def optimize_fast(
        self,
        placement: Placement,
        max_time_ms: float = 200.0,
        callback: Optional[Callable] = None
    ) -> tuple[Placement, float, Dict]:
        """
        Fast optimization for interactive UI.
        
        Args:
            placement: Initial placement
            max_time_ms: Maximum time in milliseconds
            callback: Optional callback(iteration, score, placement)
        
        Returns:
            (best_placement, best_score, stats)
        """
        start_time = time.time()
        timeout = max_time_ms / 1000.0  # Convert to seconds
        
        best_placement, best_score, stats = self.sa.optimize(
            placement,
            callback=callback,
            timeout=timeout
        )
        
        stats["time_ms"] = (time.time() - start_time) * 1000.0
        
        return best_placement, best_score, stats
    
    def propose_move(
        self,
        placement: Placement,
        component_name: str,
        num_candidates: int = 10
    ) -> List[tuple]:
        """
        Propose multiple candidate moves for a component.
        
        Returns:
            List of (x, y, angle, score_delta) tuples
        """
        comp = placement.get_component(component_name)
        if not comp:
            return []
        
        candidates = []
        margin = max(comp.width, comp.height)
        old_x, old_y, old_angle = comp.x, comp.y, comp.angle
        
        for _ in range(num_candidates):
            new_x = np.random.uniform(margin, placement.board.width - margin)
            new_y = np.random.uniform(margin, placement.board.height - margin)
            new_angle = np.random.choice([0, 90, 180, 270])
            
            delta = self.scorer.compute_delta_score(
                placement, component_name,
                old_x, old_y, old_angle,
                new_x, new_y, new_angle
            )
            
            candidates.append((new_x, new_y, new_angle, delta))
        
        # Sort by score delta (lower is better)
        candidates.sort(key=lambda x: x[3])
        
        return candidates

