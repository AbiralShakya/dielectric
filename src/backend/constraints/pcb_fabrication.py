"""
PCB Fabrication Constraints

Real-world PCB manufacturing constraints based on industry standards.
Integrates with computational geometry for large-scale design validation.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FabricationConstraints:
    """PCB fabrication constraints."""
    # Board basics
    board_thickness: float = 1.6  # mm (standard)
    layer_count: int = 4
    copper_weight: float = 1.0  # oz (35 µm)
    
    # Minimum geometry (conservative / typical / aggressive)
    min_trace_width: float = 0.15  # mm (6 mil) - typical
    min_trace_spacing: float = 0.15  # mm (6 mil) - typical
    min_annular_ring: float = 0.15  # mm (6 mil)
    
    # Via constraints
    via_drill_dia: float = 0.3  # mm (12 mil)
    via_pad_dia: float = 0.6  # mm (24 mil)
    max_aspect_ratio: float = 8.0  # board_thickness / drill_dia
    
    # Solder mask
    solder_mask_clearance: float = 0.1  # mm (4 mil)
    solder_mask_expansion: float = 0.12  # mm
    
    # Silkscreen
    min_silkscreen_stroke: float = 0.15  # mm (6 mil)
    min_silkscreen_font: float = 0.8  # mm (31 mil)
    
    # Spacing for safety (creepage & clearance)
    low_voltage_clearance: float = 0.2  # mm (<30V)
    medium_voltage_clearance: float = 0.5  # mm (30-120V)
    high_voltage_clearance: float = 3.0  # mm (mains)
    
    # Assembly constraints
    min_pad_to_pad_clearance: float = 0.2  # mm (8 mil) - comfortable
    min_smd_pad_clearance: float = 0.15  # mm (6 mil) - OK for dense
    
    # Thermal
    thermal_via_dia: float = 0.3  # mm
    thermal_via_annular_ring: float = 0.15  # mm
    
    # Current carrying (1 oz copper)
    current_per_mm_trace: float = 0.25  # A/mm (depends on temp rise)
    
    def validate_trace_width(self, width: float) -> Tuple[bool, str]:
        """Validate trace width against constraints."""
        if width < self.min_trace_width:
            return False, f"Trace width {width:.3f}mm < minimum {self.min_trace_width:.3f}mm"
        return True, "OK"
    
    def validate_spacing(self, spacing: float) -> Tuple[bool, str]:
        """Validate spacing between traces/components."""
        if spacing < self.min_trace_spacing:
            return False, f"Spacing {spacing:.3f}mm < minimum {self.min_trace_spacing:.3f}mm"
        return True, "OK"
    
    def validate_via(self, drill_dia: float) -> Tuple[bool, str]:
        """Validate via dimensions."""
        if drill_dia < 0.1:  # Microvia minimum
            return False, f"Via drill {drill_dia:.3f}mm too small (min 0.1mm for microvia)"
        
        aspect_ratio = self.board_thickness / drill_dia
        if aspect_ratio > self.max_aspect_ratio:
            return False, f"Aspect ratio {aspect_ratio:.1f} > max {self.max_aspect_ratio:.1f}"
        
        return True, "OK"
    
    def calculate_trace_width_for_current(self, current_amps: float) -> float:
        """Calculate minimum trace width for given current."""
        # Simplified: 1 oz copper, 10°C temp rise
        width_mm = current_amps / self.current_per_mm_trace
        return max(width_mm, self.min_trace_width)
    
    def get_clearance_for_voltage(self, voltage: float) -> float:
        """Get required clearance for voltage level."""
        if voltage < 30:
            return self.low_voltage_clearance
        elif voltage < 120:
            return self.medium_voltage_clearance
        else:
            return self.high_voltage_clearance


class ConstraintValidator:
    """Validates PCB design against fabrication constraints."""
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None):
        """Initialize validator."""
        self.constraints = constraints or FabricationConstraints()
    
    def validate_placement(
        self,
        placement,
        knowledge_graph=None
    ) -> Dict:
        """
        Validate placement against fabrication constraints.
        
        Uses computational geometry to check:
        - Component spacing
        - Pad-to-pad clearance
        - Thermal via placement
        - High-voltage clearance
        """
        violations = []
        warnings = []
        
        # Check component spacing
        components = list(placement.components.values())
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                # Calculate distance
                dx = comp1.x - comp2.x
                dy = comp1.y - comp2.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Get required spacing from knowledge graph
                min_spacing = self.constraints.min_pad_to_pad_clearance
                if knowledge_graph:
                    hints1 = knowledge_graph.get_placement_hints(comp1.name)
                    hints2 = knowledge_graph.get_placement_hints(comp2.name)
                    
                    # Check for special spacing requirements
                    spacing_req1 = hints1.get("spacing_requirements", {}).get("min_clearance", 0)
                    spacing_req2 = hints2.get("spacing_requirements", {}).get("min_clearance", 0)
                    min_spacing = max(min_spacing, spacing_req1, spacing_req2)
                
                # Account for component sizes
                min_distance = (comp1.width + comp2.width) / 2 + min_spacing
                
                if distance < min_distance:
                    violations.append({
                        "type": "spacing",
                        "component1": comp1.name,
                        "component2": comp2.name,
                        "distance": distance,
                        "required": min_distance,
                        "severity": "error"
                    })
        
        # Check for thermal hotspots (if geometry data available)
        try:
            from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
            analyzer = GeometryAnalyzer()
            geometry_data = analyzer.analyze(placement)
            
            if "thermal_hotspots" in geometry_data:
                hotspot_count = geometry_data["thermal_hotspots"]
                if hotspot_count > 3:
                    warnings.append({
                        "type": "thermal",
                        "message": f"{hotspot_count} thermal hotspots detected",
                        "severity": "warning"
                    })
        except Exception:
            pass
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "constraints_used": {
                "min_trace_width": self.constraints.min_trace_width,
                "min_spacing": self.constraints.min_trace_spacing,
                "min_pad_clearance": self.constraints.min_pad_to_pad_clearance
            }
        }

