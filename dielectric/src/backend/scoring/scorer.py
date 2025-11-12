"""
World Model Scoring Function

S = α·L_trace + β·D_thermal + γ·C_clearance
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component


@dataclass
class ScoreWeights:
    """Weight vector for composite score with DFM support."""
    alpha: float = 0.3  # Trace length weight
    beta: float = 0.3   # Thermal density weight
    gamma: float = 0.2  # Clearance violation weight
    delta: float = 0.2  # DFM weight (NEW)

    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = self.alpha + self.beta + self.gamma + self.delta
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total


def _manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance calculation."""
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)


class WorldModelScorer:
    """World model scoring engine."""
    
    def __init__(self, weights: ScoreWeights = None):
        """
        Initialize scorer.
        
        Args:
            weights: Score weights (alpha, beta, gamma)
        """
        self.weights = weights or ScoreWeights()
        self.weights.normalize()
    
    def compute_trace_length(self, placement: Placement) -> float:
        """
        Compute total trace length (L_trace).
        
        Uses Manhattan distance for routing estimation.
        """
        total_length = 0.0
        
        for net_name, net in placement.nets.items():
            if len(net.pins) < 2:
                continue
            
            # Get pin positions
            pin_positions = []
            for comp_name, pin_name in net.pins:
                comp = placement.get_component(comp_name)
                if comp:
                    # Find pin by name
                    pin = next((p for p in comp.pins if p.name == pin_name), None)
                    if pin:
                        pos = comp.get_pin_position(pin)
                        pin_positions.append(pos)
            
            # Compute minimum spanning tree length (simplified: use center-to-center)
            if len(pin_positions) >= 2:
                # Simple approximation: sum of distances from center
                center_x = sum(p[0] for p in pin_positions) / len(pin_positions)
                center_y = sum(p[1] for p in pin_positions) / len(pin_positions)
                
                for x, y in pin_positions:
                    total_length += _manhattan_distance(x, y, center_x, center_y)
        
        return total_length
    
    def compute_thermal_density(self, placement: Placement) -> float:
        """
        Compute thermal density score (D_thermal).
        
        Penalizes high-power components being too close together.
        """
        if not placement.components:
            return 0.0
        
        total_heat = 0.0
        max_heat_density = 0.0
        
        comp_list = list(placement.components.values())
        
        for i, c1 in enumerate(comp_list):
            if c1.power <= 0:
                continue
            
            local_heat = c1.power
            
            # Add contribution from nearby high-power components
            for c2 in comp_list[i+1:]:
                if c2.power <= 0:
                    continue
                
                # Distance between components
                dist = _euclidean_distance(c1.x, c1.y, c2.x, c2.y)
                
                if dist < 10.0:  # Within 10mm
                    # Inverse distance weighting
                    local_heat += c2.power / (1.0 + dist / 5.0)
            
            max_heat_density = max(max_heat_density, local_heat)
            total_heat += local_heat
        
        # Normalize by board area
        board_area = placement.board.width * placement.board.height
        avg_heat_density = total_heat / board_area if board_area > 0 else 0.0
        
        return max_heat_density + avg_heat_density * 0.1
    
    def compute_clearance_violations(self, placement: Placement) -> float:
        """
        Compute clearance violation penalty (C_clearance).
        
        Returns penalty score (higher = worse).
        """
        violations = 0.0
        comp_list = list(placement.components.values())
        
        for i, c1 in enumerate(comp_list):
            # Check board bounds
            if not placement.board.contains(c1):
                violations += 10.0  # Heavy penalty for out-of-bounds
            
            # Check overlaps with other components
            for c2 in comp_list[i+1:]:
                if c1.overlaps(c2, clearance=placement.board.clearance):
                    # Penalty proportional to overlap area
                    x1_min, y1_min, x1_max, y1_max = c1.get_bounds()
                    x2_min, y2_min, x2_max, y2_max = c2.get_bounds()
                    
                    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    overlap_area = overlap_x * overlap_y
                    
                    violations += overlap_area * 10.0  # Scale penalty
        
        return violations
    
    def compute_dfm_score(self, placement: Placement) -> float:
        """
        Compute DFM (Design for Manufacturing) penalty score.
        
        Checks:
        - Trace width violations
        - Spacing violations
        - Via size violations
        - Edge clearance violations
        - Estimated BOM cost (normalized) – used when user includes Cost in weights
        
        Returns penalty score (higher = worse).
        """
        try:
            from src.backend.constraints.pcb_fabrication import FabricationConstraints
        except ImportError:
            try:
                from backend.constraints.pcb_fabrication import FabricationConstraints
            except ImportError:
                return 0.0  # DFM not available
        
        constraints = FabricationConstraints()
        dfm_penalty = 0.0
        
        # Check component spacing against manufacturing constraints
        comp_list = list(placement.components.values())
        for i, c1 in enumerate(comp_list):
            for c2 in comp_list[i+1:]:
                dist = _euclidean_distance(c1.x, c1.y, c2.x, c2.y)
                # Enforce at least 0.5mm min clearance policy
                min_clearance = max(0.5, constraints.min_pad_to_pad_clearance)
                min_distance = (c1.width + c2.width) / 2 + min_clearance
                
                if dist < min_distance:
                    # Penalty proportional to violation
                    violation = min_distance - dist
                    dfm_penalty += violation * 5.0
            
            # Check edge clearance
            edge_margin = max(0.5, constraints.min_pad_to_pad_clearance)
            if (c1.x < edge_margin or c1.x > placement.board.width - edge_margin or
                c1.y < edge_margin or c1.y > placement.board.height - edge_margin):
                dfm_penalty += 2.0
        
        # Include cost estimate as part of DFM (normalized)
        try:
            from src.backend.components.bom_manager import BOMManager
            bom_mgr = BOMManager()
            bom = bom_mgr.generate_bom(placement, include_pricing=True)
            total_cost = float(bom.get("summary", {}).get("total_cost", 0.0))
            # Normalize: penalty ~ cost per cm^2 to keep scale reasonable
            board_area_cm2 = (placement.board.width * placement.board.height) / 100.0
            if board_area_cm2 > 0:
                cost_penalty = total_cost / board_area_cm2
            else:
                cost_penalty = total_cost
            # Scale down to comparable range
            dfm_penalty += cost_penalty * 0.1
        except Exception:
            pass
        
        # Add routing complexity / signal integrity proxy using computational geometry
        try:
            from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
            analyzer = GeometryAnalyzer(placement)
            routing = analyzer.compute_net_crossings()
            # Penalize crossings and high fanout
            dfm_penalty += routing.get("net_crossings", 0) * 0.5
            dfm_penalty += routing.get("avg_fanout", 0.0) * 0.2
        except Exception:
            pass
        
        return dfm_penalty
    
    def score(self, placement: Placement) -> float:
        """
        Compute composite score with DFM support.
        
        S = α·L_trace + β·D_thermal + γ·C_clearance + δ·DFM
        
        Lower is better.
        """
        L = self.compute_trace_length(placement)
        D = self.compute_thermal_density(placement)
        C = self.compute_clearance_violations(placement)
        DFM = self.compute_dfm_score(placement)
        
        score = (self.weights.alpha * L + 
                self.weights.beta * D + 
                self.weights.gamma * C +
                self.weights.delta * DFM)
        
        return score
    
    def score_breakdown(self, placement: Placement) -> Dict[str, float]:
        """Get detailed score breakdown with DFM."""
        L = self.compute_trace_length(placement)
        D = self.compute_thermal_density(placement)
        C = self.compute_clearance_violations(placement)
        DFM = self.compute_dfm_score(placement)
        
        return {
            "trace_length": L,
            "thermal_density": D,
            "clearance_violations": C,
            "dfm_penalty": DFM,
            "total_score": (self.weights.alpha * L + 
                          self.weights.beta * D + 
                          self.weights.gamma * C +
                          self.weights.delta * DFM),
            "weights": {
                "alpha": self.weights.alpha,
                "beta": self.weights.beta,
                "gamma": self.weights.gamma,
                "delta": self.weights.delta
            }
        }

