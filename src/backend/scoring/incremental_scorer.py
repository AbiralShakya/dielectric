"""
Incremental Scoring for Fast Path

Only recomputes affected nets/components on move.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
from .scorer import WorldModelScorer, ScoreWeights


def _fast_manhattan(x1: float, y1: float, x2: float, y2: float) -> float:
    """Manhattan distance."""
    return abs(x1 - x2) + abs(y1 - y2)


class IncrementalScorer:
    """
    Incremental scorer that only recomputes affected regions.
    
    Maintains cached scores and updates only on component moves.
    """
    
    def __init__(self, base_scorer: WorldModelScorer):
        """
        Initialize incremental scorer.
        
        Args:
            base_scorer: Base world model scorer
        """
        self.base_scorer = base_scorer
        self._cached_scores: Dict[str, float] = {}
        self._cached_net_lengths: Dict[str, float] = {}
        self._last_placement_hash: Optional[str] = None
    
    def _get_placement_hash(self, placement: Placement) -> str:
        """Get hash of placement state for cache invalidation."""
        # Simple hash based on component positions
        positions = []
        for comp in sorted(placement.components.values(), key=lambda c: c.name):
            positions.append(f"{comp.name}:{comp.x:.2f},{comp.y:.2f},{comp.angle:.1f}")
        return "|".join(positions)
    
    def compute_delta_score(
        self,
        placement: Placement,
        component_name: str,
        old_x: float,
        old_y: float,
        old_angle: float,
        new_x: float,
        new_y: float,
        new_angle: float
    ) -> float:
        """
        Compute score delta for a component move.
        
        Only recomputes affected nets and local thermal/clearance.
        """
        comp = placement.get_component(component_name)
        if not comp:
            return 0.0
        
        # Get affected nets
        affected_nets = placement.get_affected_nets(component_name)
        
        # Compute delta for trace length (only affected nets)
        delta_L = 0.0
        
        # Temporarily move component to old position
        comp.x, comp.y, comp.angle = old_x, old_y, old_angle
        old_net_length = self._compute_nets_length(placement, affected_nets)
        
        # Move to new position
        comp.x, comp.y, comp.angle = new_x, new_y, new_angle
        new_net_length = self._compute_nets_length(placement, affected_nets)
        
        delta_L = new_net_length - old_net_length
        
        # Compute delta for thermal (local region only)
        delta_D = self._compute_local_thermal_delta(
            placement, component_name, old_x, old_y, new_x, new_y
        )
        
        # Compute delta for clearance (local region only)
        delta_C = self._compute_local_clearance_delta(
            placement, component_name, old_x, old_y, old_angle, new_x, new_y, new_angle
        )
        
        # Compute delta for DFM (local region only)
        delta_DFM = self._compute_local_dfm_delta(
            placement, component_name, old_x, old_y, new_x, new_y
        )
        
        # Compute total delta with DFM
        delta_score = (self.base_scorer.weights.alpha * delta_L +
                      self.base_scorer.weights.beta * delta_D +
                      self.base_scorer.weights.gamma * delta_C +
                      self.base_scorer.weights.delta * delta_DFM)
        
        return delta_score
    
    def _compute_nets_length(self, placement: Placement, net_names: List[str]) -> float:
        """Compute total length for specific nets."""
        total = 0.0
        
        for net_name in net_names:
            net = placement.nets.get(net_name)
            if not net or len(net.pins) < 2:
                continue
            
            # Get pin positions
            pin_positions = []
            for comp_name, pin_name in net.pins:
                comp = placement.get_component(comp_name)
                if comp:
                    pin = next((p for p in comp.pins if p.name == pin_name), None)
                    if pin:
                        pos = comp.get_pin_position(pin)
                        pin_positions.append(pos)
            
            if len(pin_positions) >= 2:
                # Center-based approximation
                center_x = sum(p[0] for p in pin_positions) / len(pin_positions)
                center_y = sum(p[1] for p in pin_positions) / len(pin_positions)
                
                for x, y in pin_positions:
                    total += _fast_manhattan(x, y, center_x, center_y)
        
        return total
    
    def _compute_local_thermal_delta(
        self,
        placement: Placement,
        comp_name: str,
        old_x: float,
        old_y: float,
        new_x: float,
        new_y: float
    ) -> float:
        """Compute thermal density delta for local region."""
        comp = placement.get_component(comp_name)
        if not comp or comp.power <= 0:
            return 0.0
        
        delta = 0.0
        comp_list = list(placement.components.values())
        
        for other in comp_list:
            if other.name == comp_name or other.power <= 0:
                continue
            
            # Old distance
            old_dist = np.sqrt((old_x - other.x)**2 + (old_y - other.y)**2)
            # New distance
            new_dist = np.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
            
            if old_dist < 10.0 or new_dist < 10.0:
                old_contrib = other.power / (1.0 + old_dist / 5.0) if old_dist < 10.0 else 0.0
                new_contrib = other.power / (1.0 + new_dist / 5.0) if new_dist < 10.0 else 0.0
                delta += new_contrib - old_contrib
        
        return delta
    
    def _compute_local_clearance_delta(
        self,
        placement: Placement,
        comp_name: str,
        old_x: float,
        old_y: float,
        old_angle: float,
        new_x: float,
        new_y: float,
        new_angle: float
    ) -> float:
        """Compute clearance violation delta for local region."""
        comp = placement.get_component(comp_name)
        if not comp:
            return 0.0
        
        delta = 0.0
        
        # Check board bounds
        comp.x, comp.y, comp.angle = old_x, old_y, old_angle
        old_in_bounds = placement.board.contains(comp)
        
        comp.x, comp.y, comp.angle = new_x, new_y, new_angle
        new_in_bounds = placement.board.contains(comp)
        
        if not old_in_bounds and new_in_bounds:
            delta -= 10.0  # Fixed violation
        elif old_in_bounds and not new_in_bounds:
            delta += 10.0  # Created violation
        
        # Check overlaps with nearby components
        comp_list = list(placement.components.values())
        for other in comp_list:
            if other.name == comp_name:
                continue
            
            comp.x, comp.y, comp.angle = old_x, old_y, old_angle
            old_overlap = comp.overlaps(other, clearance=placement.board.clearance)
            
            comp.x, comp.y, comp.angle = new_x, new_y, new_angle
            new_overlap = comp.overlaps(other, clearance=placement.board.clearance)
            
            if old_overlap and not new_overlap:
                delta -= 5.0  # Fixed overlap
            elif not old_overlap and new_overlap:
                delta += 5.0  # Created overlap
        
        return delta
    
    def _compute_local_dfm_delta(
        self,
        placement: Placement,
        comp_name: str,
        old_x: float,
        old_y: float,
        new_x: float,
        new_y: float
    ) -> float:
        """Compute DFM penalty delta for local region."""
        try:
            from src.backend.constraints.pcb_fabrication import FabricationConstraints
        except ImportError:
            try:
                from backend.constraints.pcb_fabrication import FabricationConstraints
            except ImportError:
                return 0.0
        
        constraints = FabricationConstraints()
        comp = placement.get_component(comp_name)
        if not comp:
            return 0.0
        
        delta = 0.0
        
        # Check edge clearance
        edge_margin = constraints.min_pad_to_pad_clearance
        old_edge_violation = (old_x < edge_margin or old_x > placement.board.width - edge_margin or
                             old_y < edge_margin or old_y > placement.board.height - edge_margin)
        new_edge_violation = (new_x < edge_margin or new_x > placement.board.width - edge_margin or
                             new_y < edge_margin or new_y > placement.board.height - edge_margin)
        
        if old_edge_violation and not new_edge_violation:
            delta -= 2.0  # Fixed edge violation
        elif not old_edge_violation and new_edge_violation:
            delta += 2.0  # Created edge violation
        
        # Check spacing with nearby components
        comp_list = list(placement.components.values())
        for other in comp_list:
            if other.name == comp_name:
                continue
            
            old_dist = np.sqrt((old_x - other.x)**2 + (old_y - other.y)**2)
            new_dist = np.sqrt((new_x - other.x)**2 + (new_y - other.y)**2)
            
            min_clearance = constraints.min_pad_to_pad_clearance
            min_distance = (comp.width + other.width) / 2 + min_clearance
            
            old_violation = old_dist < min_distance
            new_violation = new_dist < min_distance
            
            if old_violation and not new_violation:
                delta -= 5.0  # Fixed spacing violation
            elif not old_violation and new_violation:
                violation = min_distance - new_dist
                delta += violation * 5.0  # Created spacing violation
        
        return delta
    
    def score(self, placement: Placement, use_cache: bool = True) -> float:
        """
        Compute full score (with caching).
        
        Args:
            placement: Placement to score
            use_cache: Whether to use cached scores
        """
        if use_cache:
            placement_hash = self._get_placement_hash(placement)
            if placement_hash == self._last_placement_hash and placement_hash in self._cached_scores:
                return self._cached_scores[placement_hash]
        
        # Compute full score
        score = self.base_scorer.score(placement)
        
        if use_cache:
            placement_hash = self._get_placement_hash(placement)
            self._cached_scores[placement_hash] = score
            self._last_placement_hash = placement_hash
        
        return score

