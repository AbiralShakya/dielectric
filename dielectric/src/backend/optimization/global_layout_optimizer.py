"""
Global Optimization Algorithms for Chip Layout and Compaction

Based on research paper: "Global Optimization Algorithms for Chip Layout and Compaction"
Implements multiple global optimization strategies:
1. Force-Directed Placement (FDP)
2. Quadratic Placement
3. Analytical Placement with Partitioning
4. Compaction Algorithms
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.scoring.scorer import WorldModelScorer, ScoreWeights
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights


class GlobalOptimizationMethod(Enum):
    """Global optimization methods from research literature."""
    FORCE_DIRECTED = "force_directed"
    QUADRATIC = "quadratic"
    ANALYTICAL = "analytical"
    COMPACTION = "compaction"
    HYBRID = "hybrid"  # Combine multiple methods


class GlobalLayoutOptimizer:
    """
    Global optimization algorithms for chip layout based on research literature.
    
    Implements algorithms from:
    - Dorneich et al. "Global Optimization Algorithms for Chip Layout and Compaction"
    - Force-directed placement
    - Quadratic placement
    - Analytical placement with partitioning
    - Compaction algorithms
    """
    
    def __init__(
        self,
        scorer: WorldModelScorer,
        method: GlobalOptimizationMethod = GlobalOptimizationMethod.HYBRID,
        max_iterations: int = 1000,
        convergence_threshold: float = 0.001
    ):
        """
        Initialize global layout optimizer.
        
        Args:
            scorer: Scoring function for evaluating placements
            method: Optimization method to use
            max_iterations: Maximum iterations
            convergence_threshold: Convergence threshold for early stopping
        """
        self.scorer = scorer
        self.method = method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.geometry_analyzer = None
    
    def optimize(
        self,
        placement: Placement,
        weights: Dict[str, float],
        timeout: Optional[float] = None,
        callback: Optional[callable] = None
    ) -> Tuple[Placement, float, Dict]:
        """
        Run global optimization on placement.
        
        Args:
            placement: Initial placement
            weights: Optimization weights (alpha, beta, gamma)
            timeout: Optional timeout in seconds
            callback: Optional progress callback
        
        Returns:
            (optimized_placement, best_score, stats)
        """
        start_time = time.time()
        self.geometry_analyzer = GeometryAnalyzer(placement)
        
        # Update scorer weights
        score_weights = ScoreWeights(
            alpha=weights.get("alpha", 0.3),
            beta=weights.get("beta", 0.3),
            gamma=weights.get("gamma", 0.2),
            delta=weights.get("delta", 0.2)
        )
        self.scorer.weights = score_weights
        
        if self.method == GlobalOptimizationMethod.FORCE_DIRECTED:
            return self._force_directed_placement(placement, timeout, callback)
        elif self.method == GlobalOptimizationMethod.QUADRATIC:
            return self._quadratic_placement(placement, timeout, callback)
        elif self.method == GlobalOptimizationMethod.ANALYTICAL:
            return self._analytical_placement(placement, timeout, callback)
        elif self.method == GlobalOptimizationMethod.COMPACTION:
            return self._compaction_optimization(placement, timeout, callback)
        else:  # HYBRID
            return self._hybrid_optimization(placement, timeout, callback)
    
    def _force_directed_placement(
        self,
        placement: Placement,
        timeout: Optional[float],
        callback: Optional[callable]
    ) -> Tuple[Placement, float, Dict]:
        """
        Force-Directed Placement (FDP) algorithm.
        
        Based on spring-mass system where:
        - Components connected by nets attract each other (spring forces)
        - All components repel each other (electrostatic forces)
        - System converges to equilibrium
        """
        optimized_placement = placement.copy()
        best_score = self.scorer.score(optimized_placement)
        best_placement = optimized_placement.copy()
        
        stats = {
            "method": "force_directed",
            "iterations": 0,
            "converged": False,
            "final_force": 0.0
        }
        
        start_time = time.time()
        damping = 0.9  # Damping factor for stability
        dt = 0.1  # Time step
        
        for iteration in range(self.max_iterations):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Calculate forces for all components
            forces = self._calculate_force_directed_forces(optimized_placement)
            
            # Update positions based on forces
            max_force = 0.0
            for comp_name, comp in optimized_placement.components.items():
                if comp_name in forces:
                    fx, fy = forces[comp_name]
                    max_force = max(max_force, abs(fx), abs(fy))
                    
                    # Update position with damping
                    comp.x += fx * dt * damping
                    comp.y += fy * dt * damping
                    
                    # Enforce board boundaries
                    comp.x = max(comp.width/2, min(optimized_placement.board.width - comp.width/2, comp.x))
                    comp.y = max(comp.height/2, min(optimized_placement.board.height - comp.height/2, comp.y))
            
            # Evaluate new score
            current_score = self.scorer.score(optimized_placement)
            if current_score < best_score:
                best_score = current_score
                best_placement = optimized_placement.copy()
            
            stats["iterations"] = iteration + 1
            stats["final_force"] = max_force
            
            # Check convergence
            if max_force < self.convergence_threshold:
                stats["converged"] = True
                break
            
            if callback:
                callback(iteration, current_score, optimized_placement)
        
        return best_placement, best_score, stats
    
    def _calculate_force_directed_forces(self, placement: Placement) -> Dict[str, Tuple[float, float]]:
        """
        Calculate force-directed forces for all components.
        
        Forces:
        - Attractive: Components on same net attract (spring force)
        - Repulsive: All components repel (electrostatic force)
        """
        forces = {name: (0.0, 0.0) for name in placement.components.keys()}
        components = list(placement.components.values())
        
        # Attractive forces from nets (spring forces)
        spring_constant = 0.1
        for net in placement.nets.values():
            net_components = [pin[0] for pin in net.pins]
            for i, comp1_name in enumerate(net_components):
                comp1 = placement.get_component(comp1_name)
                if not comp1:
                    continue
                
                for comp2_name in net_components[i+1:]:
                    comp2 = placement.get_component(comp2_name)
                    if not comp2:
                        continue
                    
                    dx = comp2.x - comp1.x
                    dy = comp2.y - comp1.y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist > 0:
                        # Spring force (attractive)
                        force_magnitude = spring_constant * dist
                        fx = force_magnitude * (dx / dist)
                        fy = force_magnitude * (dy / dist)
                        
                        forces[comp1_name] = (forces[comp1_name][0] + fx, forces[comp1_name][1] + fy)
                        forces[comp2_name] = (forces[comp2_name][0] - fx, forces[comp2_name][1] - fy)
        
        # Repulsive forces (electrostatic)
        repulsion_constant = 100.0
        comp_list = list(placement.components.values())
        for i, comp1 in enumerate(comp_list):
            for comp2 in comp_list[i+1:]:
                dx = comp2.x - comp1.x
                dy = comp2.y - comp1.y
                dist_sq = dx**2 + dy**2
                
                if dist_sq > 0:
                    # Electrostatic repulsion
                    force_magnitude = repulsion_constant / dist_sq
                    fx = force_magnitude * (dx / np.sqrt(dist_sq))
                    fy = force_magnitude * (dy / np.sqrt(dist_sq))
                    
                    forces[comp1.name] = (forces[comp1.name][0] - fx, forces[comp1.name][1] - fy)
                    forces[comp2.name] = (forces[comp2.name][0] + fx, forces[comp2.name][1] + fy)
        
        return forces
    
    def _quadratic_placement(
        self,
        placement: Placement,
        timeout: Optional[float],
        callback: Optional[callable]
    ) -> Tuple[Placement, float, Dict]:
        """
        Quadratic Placement algorithm.
        
        Minimizes quadratic wire length:
        minimize: Σ (x_i - x_j)² + (y_i - y_j)² for all connected components
        
        Solves using conjugate gradient or direct solver.
        """
        optimized_placement = placement.copy()
        
        # Build connectivity matrix
        n = len(optimized_placement.components)
        comp_names = list(optimized_placement.components.keys())
        comp_to_idx = {name: i for i, name in enumerate(comp_names)}
        
        # Build Laplacian matrix for x and y coordinates
        L_x = np.zeros((n, n))
        L_y = np.zeros((n, n))
        
        for net in optimized_placement.nets.values():
            net_comps = [pin[0] for pin in net.pins]
            for i, comp1_name in enumerate(net_comps):
                idx1 = comp_to_idx.get(comp1_name)
                if idx1 is None:
                    continue
                
                for comp2_name in net_comps[i+1:]:
                    idx2 = comp_to_idx.get(comp2_name)
                    if idx2 is None:
                        continue
                    
                    # Add connectivity
                    weight = 1.0 / len(net_comps)  # Normalize by net size
                    L_x[idx1, idx2] -= weight
                    L_x[idx2, idx1] -= weight
                    L_x[idx1, idx1] += weight
                    L_x[idx2, idx2] += weight
                    
                    L_y[idx1, idx2] -= weight
                    L_y[idx2, idx1] -= weight
                    L_y[idx1, idx1] += weight
                    L_y[idx2, idx2] += weight
        
        # Solve quadratic system (simplified - use current positions as constraints)
        # In practice, would use more sophisticated solver
        x_coords = np.array([optimized_placement.get_component(name).x for name in comp_names])
        y_coords = np.array([optimized_placement.get_component(name).y for name in comp_names])
        
        # Iterative refinement
        stats = {"method": "quadratic", "iterations": 0, "converged": False}
        start_time = time.time()
        
        for iteration in range(min(100, self.max_iterations)):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Solve for new positions (simplified iterative update)
            x_new = np.linalg.solve(L_x + np.eye(n) * 0.1, x_coords)
            y_new = np.linalg.solve(L_y + np.eye(n) * 0.1, y_coords)
            
            # Update positions
            for i, name in enumerate(comp_names):
                comp = optimized_placement.get_component(name)
                comp.x = max(comp.width/2, min(optimized_placement.board.width - comp.width/2, x_new[i]))
                comp.y = max(comp.height/2, min(optimized_placement.board.height - comp.height/2, y_new[i]))
            
            x_coords = x_new
            y_coords = y_new
            
            stats["iterations"] = iteration + 1
            
            if callback:
                score = self.scorer.score(optimized_placement)
                callback(iteration, score, optimized_placement)
        
        best_score = self.scorer.score(optimized_placement)
        return optimized_placement, best_score, stats
    
    def _analytical_placement(
        self,
        placement: Placement,
        timeout: Optional[float],
        callback: Optional[callable]
    ) -> Tuple[Placement, float, Dict]:
        """
        Analytical Placement with Partitioning.
        
        Uses recursive bisection:
        1. Partition components into two groups
        2. Place groups optimally
        3. Refine placement
        """
        optimized_placement = placement.copy()
        stats = {"method": "analytical", "iterations": 0, "partitions": 0}
        
        # Recursive partitioning and placement
        def partition_and_place(comps: List[str], bounds: Tuple[float, float, float, float], depth: int):
            if len(comps) <= 1 or depth > 5:
                return
            
            # Simple bisection (in practice, use better partitioning)
            mid = len(comps) // 2
            comps1 = comps[:mid]
            comps2 = comps[mid:]
            
            x_min, y_min, x_max, y_max = bounds
            
            # Place first half
            x_mid = (x_min + x_max) / 2
            for comp_name in comps1:
                comp = optimized_placement.get_component(comp_name)
                comp.x = (x_min + x_mid) / 2
                comp.y = (y_min + y_max) / 2
            
            # Place second half
            for comp_name in comps2:
                comp = optimized_placement.get_component(comp_name)
                comp.x = (x_mid + x_max) / 2
                comp.y = (y_min + y_max) / 2
            
            stats["partitions"] += 1
            
            # Recursive refinement
            partition_and_place(comps1, (x_min, y_min, x_mid, y_max), depth + 1)
            partition_and_place(comps2, (x_mid, y_min, x_max, y_max), depth + 1)
        
        comp_names = list(optimized_placement.components.keys())
        bounds = (0, 0, optimized_placement.board.width, optimized_placement.board.height)
        partition_and_place(comp_names, bounds, 0)
        
        # Refinement iterations
        start_time = time.time()
        best_score = self.scorer.score(optimized_placement)
        best_placement = optimized_placement.copy()
        
        for iteration in range(min(50, self.max_iterations)):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Local refinement using force-directed
            forces = self._calculate_force_directed_forces(optimized_placement)
            for comp_name, comp in optimized_placement.components.items():
                if comp_name in forces:
                    fx, fy = forces[comp_name]
                    comp.x += fx * 0.1
                    comp.y += fy * 0.1
                    comp.x = max(comp.width/2, min(optimized_placement.board.width - comp.width/2, comp.x))
                    comp.y = max(comp.height/2, min(optimized_placement.board.height - comp.height/2, comp.y))
            
            current_score = self.scorer.score(optimized_placement)
            if current_score < best_score:
                best_score = current_score
                best_placement = optimized_placement.copy()
            
            stats["iterations"] = iteration + 1
            
            if callback:
                callback(iteration, current_score, optimized_placement)
        
        return best_placement, best_score, stats
    
    def _compaction_optimization(
        self,
        placement: Placement,
        timeout: Optional[float],
        callback: Optional[callable]
    ) -> Tuple[Placement, float, Dict]:
        """
        Compaction algorithm to minimize board area while maintaining constraints.
        
        Iteratively compacts placement by:
        1. Finding minimum bounding box
        2. Shifting components to reduce area
        3. Maintaining design rules
        """
        optimized_placement = placement.copy()
        stats = {"method": "compaction", "iterations": 0, "area_reduction": 0.0}
        
        initial_area = optimized_placement.board.width * optimized_placement.board.height
        
        start_time = time.time()
        best_score = self.scorer.score(optimized_placement)
        best_placement = optimized_placement.copy()
        
        for iteration in range(min(100, self.max_iterations)):
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Find bounding box of current placement
            if optimized_placement.components:
                x_coords = [c.x for c in optimized_placement.components.values()]
                y_coords = [c.y for c in optimized_placement.components.values()]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Calculate required margins
                max_width = max(c.width for c in optimized_placement.components.values())
                max_height = max(c.height for c in optimized_placement.components.values())
                
                # Compact by shifting towards origin
                x_shift = x_min - max_width/2 - 1.0
                y_shift = y_min - max_height/2 - 1.0
                
                # Apply compaction shift
                for comp in optimized_placement.components.values():
                    comp.x -= x_shift
                    comp.y -= y_shift
                    
                    # Ensure within bounds
                    comp.x = max(comp.width/2, min(optimized_placement.board.width - comp.width/2, comp.x))
                    comp.y = max(comp.height/2, min(optimized_placement.board.height - comp.height/2, comp.y))
            
            current_score = self.scorer.score(optimized_placement)
            if current_score < best_score:
                best_score = current_score
                best_placement = optimized_placement.copy()
            
            stats["iterations"] = iteration + 1
            
            if callback:
                callback(iteration, current_score, optimized_placement)
        
        final_area = optimized_placement.board.width * optimized_placement.board.height
        stats["area_reduction"] = (initial_area - final_area) / initial_area
        
        return best_placement, best_score, stats
    
    def _hybrid_optimization(
        self,
        placement: Placement,
        timeout: Optional[float],
        callback: Optional[callable]
    ) -> Tuple[Placement, float, Dict]:
        """
        Hybrid optimization combining multiple methods.
        
        Strategy:
        1. Start with analytical placement (coarse)
        2. Refine with force-directed (medium)
        3. Final compaction (fine)
        """
        optimized_placement = placement.copy()
        total_time = timeout or 60.0
        phase_time = total_time / 3.0
        
        stats = {"method": "hybrid", "phases": []}
        
        # Phase 1: Analytical placement
        optimized_placement, score1, stats1 = self._analytical_placement(
            optimized_placement, phase_time, callback
        )
        stats["phases"].append({"name": "analytical", "score": score1, "iterations": stats1.get("iterations", 0)})
        
        # Phase 2: Force-directed refinement
        optimized_placement, score2, stats2 = self._force_directed_placement(
            optimized_placement, phase_time, callback
        )
        stats["phases"].append({"name": "force_directed", "score": score2, "iterations": stats2.get("iterations", 0)})
        
        # Phase 3: Compaction
        optimized_placement, final_score, stats3 = self._compaction_optimization(
            optimized_placement, phase_time, callback
        )
        stats["phases"].append({"name": "compaction", "score": final_score, "iterations": stats3.get("iterations", 0)})
        
        return optimized_placement, final_score, stats

