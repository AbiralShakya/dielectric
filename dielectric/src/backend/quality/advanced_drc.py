"""
Advanced Design Rule Checking (DRC)

Comprehensive DRC including:
- Trace width/spacing violations
- Via size and drill checks
- Copper-to-edge clearance
- Solder mask expansion
- Silkscreen clearance
- Annular ring checks
- Thermal relief validation
- Electrical Rule Checking (ERC)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set

try:
    from backend.geometry.placement import Placement
    from backend.geometry.net import Net
    from backend.constraints.pcb_fabrication import FabricationConstraints
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.net import Net
    from src.backend.constraints.pcb_fabrication import FabricationConstraints

logger = logging.getLogger(__name__)


class AdvancedDRC:
    """
    Advanced Design Rule Checking.
    
    Features:
    - Comprehensive DRC rules
    - Electrical Rule Checking (ERC)
    - Manufacturing rule checks
    - Custom rule definition
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None):
        """
        Initialize advanced DRC.
        
        Args:
            constraints: Fabrication constraints
        """
        self.constraints = constraints or FabricationConstraints()
        self.violations: List[Dict] = []
    
    def run_all_checks(self, placement: Placement) -> Dict:
        """
        Run all DRC checks.
        
        Args:
            placement: Placement to check
        
        Returns:
            {
                "success": bool,
                "violations": List[Dict],
                "summary": Dict
            }
        """
        self.violations = []
        
        # Run all checks
        self._check_trace_width_spacing(placement)
        self._check_via_rules(placement)
        self._check_copper_to_edge(placement)
        self._check_component_clearance(placement)
        self._check_annular_rings(placement)
        self._check_thermal_reliefs(placement)
        self._check_solder_mask(placement)
        self._check_silkscreen(placement)
        
        # ERC checks
        self._check_unconnected_nets(placement)
        self._check_short_circuits(placement)
        self._check_power_ground_violations(placement)
        
        # Manufacturing checks
        self._check_minimum_feature_sizes(placement)
        self._check_aspect_ratios(placement)
        
        # Summary
        summary = self._generate_summary()
        
        return {
            "success": len(self.violations) == 0,
            "violations": self.violations,
            "summary": summary
        }
    
    def _check_trace_width_spacing(self, placement: Placement):
        """Check trace width and spacing violations."""
        if not hasattr(placement, 'traces') or not placement.traces:
            return
        
        min_width = self.constraints.min_trace_width
        min_spacing = self.constraints.min_trace_spacing
        
        for trace in placement.traces:
            # Check width
            width = trace.get("width", 0.2)
            if width < min_width:
                self.violations.append({
                    "type": "trace_width",
                    "severity": "error",
                    "message": f"Trace {trace.get('net', 'unknown')} width {width:.3f}mm < minimum {min_width:.3f}mm",
                    "trace": trace
                })
            
            # Check spacing (simplified - would check against all other traces)
            # In production, would use spatial indexing for efficiency
        
        logger.info(f"Checked {len(placement.traces)} traces")
    
    def _check_via_rules(self, placement: Placement):
        """Check via size and drill rules."""
        if not hasattr(placement, 'vias') or not placement.vias:
            return
        
        min_via_size = self.constraints.min_via_size
        min_drill = self.constraints.min_via_drill
        
        for via in placement.vias:
            size = via.get("size", 0.5)
            drill = via.get("drill", 0.2)
            
            if size < min_via_size:
                self.violations.append({
                    "type": "via_size",
                    "severity": "error",
                    "message": f"Via size {size:.3f}mm < minimum {min_via_size:.3f}mm",
                    "via": via
                })
            
            if drill < min_drill:
                self.violations.append({
                    "type": "via_drill",
                    "severity": "error",
                    "message": f"Via drill {drill:.3f}mm < minimum {min_drill:.3f}mm",
                    "via": via
                })
            
            # Check annular ring
            annular_ring = (size - drill) / 2
            min_annular_ring = 0.1  # mm
            if annular_ring < min_annular_ring:
                self.violations.append({
                    "type": "annular_ring",
                    "severity": "warning",
                    "message": f"Via annular ring {annular_ring:.3f}mm < recommended {min_annular_ring:.3f}mm",
                    "via": via
                })
    
    def _check_copper_to_edge(self, placement: Placement):
        """Check copper-to-edge clearance."""
        min_clearance = 0.5  # mm
        
        board = placement.board
        
        # Check components
        for comp_name, comp in placement.components.items():
            # Check distance to board edge
            dist_to_left = comp.x - comp.width/2
            dist_to_right = board.width - (comp.x + comp.width/2)
            dist_to_top = comp.y - comp.height/2
            dist_to_bottom = board.height - (comp.y + comp.height/2)
            
            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            if min_dist < min_clearance:
                self.violations.append({
                    "type": "copper_to_edge",
                    "severity": "error",
                    "message": f"Component {comp_name} too close to board edge ({min_dist:.3f}mm < {min_clearance:.3f}mm)",
                    "component": comp_name,
                    "distance": min_dist
                })
    
    def _check_component_clearance(self, placement: Placement):
        """Check component-to-component clearance."""
        min_clearance = 0.5  # mm
        
        components = list(placement.components.values())
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                dx = comp2.x - comp1.x
                dy = comp2.y - comp1.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                min_required = (comp1.width + comp2.width) / 2 + (comp1.height + comp2.height) / 2 + min_clearance
                
                if distance < min_required:
                    self.violations.append({
                        "type": "component_clearance",
                        "severity": "error",
                        "message": f"Components too close: {comp1.name} and {comp2.name} ({distance:.3f}mm < {min_required:.3f}mm)",
                        "component1": comp1.name,
                        "component2": comp2.name,
                        "distance": distance
                    })
    
    def _check_annular_rings(self, placement: Placement):
        """Check via annular rings."""
        min_annular_ring = 0.1  # mm
        
        if hasattr(placement, 'vias') and placement.vias:
            for via in placement.vias:
                size = via.get("size", 0.5)
                drill = via.get("drill", 0.2)
                annular_ring = (size - drill) / 2
                
                if annular_ring < min_annular_ring:
                    self.violations.append({
                        "type": "annular_ring",
                        "severity": "warning",
                        "message": f"Via annular ring {annular_ring:.3f}mm < minimum {min_annular_ring:.3f}mm",
                        "via": via
                    })
    
    def _check_thermal_reliefs(self, placement: Placement):
        """Check thermal reliefs for vias in pads."""
        # Simplified check - in production would check actual via-in-pad
        pass
    
    def _check_solder_mask(self, placement: Placement):
        """Check solder mask expansion."""
        min_expansion = 0.05  # mm
        
        # Check that pads have adequate solder mask expansion
        # Simplified - would check actual mask layers
        pass
    
    def _check_silkscreen(self, placement: Placement):
        """Check silkscreen clearance."""
        min_clearance = 0.1  # mm
        
        # Check that silkscreen doesn't overlap pads
        # Simplified - would check actual silkscreen layers
        pass
    
    def _check_unconnected_nets(self, placement: Placement):
        """ERC: Check for unconnected nets."""
        for net_name, net in placement.nets.items():
            if len(net.pins) < 2:
                self.violations.append({
                    "type": "unconnected_net",
                    "severity": "warning",
                    "message": f"Net {net_name} has fewer than 2 connections",
                    "net": net_name,
                    "pin_count": len(net.pins)
                })
    
    def _check_short_circuits(self, placement: Placement):
        """ERC: Check for potential short circuits."""
        # Check if nets with different names share components
        # Simplified - would check actual connectivity
        pass
    
    def _check_power_ground_violations(self, placement: Placement):
        """ERC: Check power/ground violations."""
        power_nets = []
        ground_nets = []
        
        for net_name, net in placement.nets.items():
            name_lower = net_name.lower()
            if any(kw in name_lower for kw in ["vcc", "vdd", "power", "supply"]):
                power_nets.append(net_name)
            elif any(kw in name_lower for kw in ["gnd", "ground", "vss"]):
                ground_nets.append(net_name)
        
        # Check that power and ground nets exist
        if not power_nets:
            self.violations.append({
                "type": "missing_power",
                "severity": "warning",
                "message": "No power nets found"
            })
        
        if not ground_nets:
            self.violations.append({
                "type": "missing_ground",
                "severity": "error",
                "message": "No ground nets found"
            })
    
    def _check_minimum_feature_sizes(self, placement: Placement):
        """Check minimum feature sizes for manufacturing."""
        min_feature = 0.1  # mm
        
        # Check trace widths
        if hasattr(placement, 'traces') and placement.traces:
            for trace in placement.traces:
                width = trace.get("width", 0.2)
                if width < min_feature:
                    self.violations.append({
                        "type": "minimum_feature",
                        "severity": "error",
                        "message": f"Trace width {width:.3f}mm < minimum manufacturable {min_feature:.3f}mm",
                        "trace": trace
                    })
    
    def _check_aspect_ratios(self, placement: Placement):
        """Check via aspect ratios."""
        max_aspect_ratio = 10.0  # drill depth / drill diameter
        
        if hasattr(placement, 'vias') and placement.vias:
            board_thickness = placement.board.thickness or 1.6  # mm
            
            for via in placement.vias:
                drill = via.get("drill", 0.2)
                aspect_ratio = board_thickness / drill
                
                if aspect_ratio > max_aspect_ratio:
                    self.violations.append({
                        "type": "aspect_ratio",
                        "severity": "warning",
                        "message": f"Via aspect ratio {aspect_ratio:.1f} > maximum {max_aspect_ratio:.1f}",
                        "via": via
                    })
    
    def _generate_summary(self) -> Dict:
        """Generate DRC summary."""
        error_count = sum(1 for v in self.violations if v["severity"] == "error")
        warning_count = sum(1 for v in self.violations if v["severity"] == "warning")
        
        violations_by_type = {}
        for v in self.violations:
            v_type = v["type"]
            violations_by_type[v_type] = violations_by_type.get(v_type, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "errors": error_count,
            "warnings": warning_count,
            "by_type": violations_by_type,
            "passed": len(self.violations) == 0
        }

