"""
PCB Design Quality Validator

Validates design quality using multiple metrics:
- Design rules (clearance, spacing)
- Thermal performance
- Signal integrity
- Manufacturing feasibility
- Performance benchmarks
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component


class DesignQualityValidator:
    """Validates and scores PCB design quality."""
    
    def __init__(self):
        """Initialize validator."""
        self.quality_metrics = {}
    
    def validate_design(self, placement: Placement) -> Dict:
        """
        Comprehensive design validation.
        
        Returns:
            Dictionary with validation results, scores, and recommendations
        """
        results = {
            "overall_score": 0.0,
            "pass": False,
            "categories": {},
            "issues": [],
            "recommendations": []
        }
        
        # Design rule checks
        drc_results = self._check_design_rules(placement)
        results["categories"]["design_rules"] = drc_results
        
        # Thermal analysis
        thermal_results = self._analyze_thermal(placement)
        results["categories"]["thermal"] = thermal_results
        
        # Signal integrity
        si_results = self._analyze_signal_integrity(placement)
        results["categories"]["signal_integrity"] = si_results
        
        # Manufacturing feasibility
        dfm_results = self._check_manufacturability(placement)
        results["categories"]["manufacturability"] = dfm_results
        
        # Component distribution
        distribution_results = self._analyze_distribution(placement)
        results["categories"]["distribution"] = distribution_results
        
        # Calculate overall score (weighted average)
        weights = {
            "design_rules": 0.3,
            "thermal": 0.25,
            "signal_integrity": 0.2,
            "manufacturability": 0.15,
            "distribution": 0.1
        }
        
        overall_score = sum(
            results["categories"][cat].get("score", 0) * weights[cat]
            for cat in weights.keys()
            if cat in results["categories"]
        )
        
        results["overall_score"] = overall_score
        results["pass"] = overall_score >= 0.7 and len(results["issues"]) == 0
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def _check_design_rules(self, placement: Placement) -> Dict:
        """Check design rule compliance."""
        violations = []
        score = 1.0
        
        components = list(placement.components.values())
        
        # Check clearance violations
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                min_clearance = (c1.width + c2.width) / 2 + placement.board.clearance
                
                if dist < min_clearance:
                    violations.append({
                        "type": "clearance_violation",
                        "components": [c1.name, c2.name],
                        "distance": float(dist),
                        "required": float(min_clearance),
                        "severity": "error"
                    })
                    score -= 0.1
        
        # Check board bounds
        for comp in components:
            if not placement.board.contains(comp):
                violations.append({
                    "type": "out_of_bounds",
                    "component": comp.name,
                    "severity": "error"
                })
                score -= 0.2
        
        score = max(0.0, score)
        
        return {
            "score": score,
            "violations": violations,
            "violation_count": len(violations),
            "pass": len(violations) == 0
        }
    
    def _analyze_thermal(self, placement: Placement) -> Dict:
        """Analyze thermal performance."""
        components = list(placement.components.values())
        high_power = [c for c in components if c.power > 1.0]
        
        if not high_power:
            return {"score": 1.0, "hotspots": 0, "pass": True}
        
        # Calculate thermal hotspots
        positions = np.array([[c.x, c.y] for c in high_power])
        powers = np.array([c.power for c in high_power])
        
        # Check spacing between high-power components
        min_spacing = 15.0  # mm
        hotspots = 0
        
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions[i+1:], i+1):
                dist = np.linalg.norm(p1 - p2)
                if dist < min_spacing:
                    hotspots += 1
        
        # Score based on hotspot count
        score = max(0.0, 1.0 - (hotspots * 0.2))
        
        return {
            "score": score,
            "hotspots": hotspots,
            "high_power_count": len(high_power),
            "pass": hotspots == 0
        }
    
    def _analyze_signal_integrity(self, placement: Placement) -> Dict:
        """Analyze signal integrity."""
        issues = []
        score = 1.0
        
        # Check for long traces
        for net in placement.nets.values():
            net_pins = net.pins
            if len(net_pins) < 2:
                continue
            
            positions = []
            for pin_ref in net_pins:
                comp_name = pin_ref[0]
                comp = placement.get_component(comp_name)
                if comp:
                    positions.append([comp.x, comp.y])
            
            if len(positions) >= 2:
                total_length = 0
                for i in range(len(positions) - 1):
                    x0, y0 = positions[i]
                    x1, y1 = positions[i+1]
                    total_length += abs(x1 - x0) + abs(y1 - y0)  # Manhattan
                
                # Long trace threshold
                if total_length > 100:  # mm
                    issues.append({
                        "net": net.name,
                        "length": float(total_length),
                        "severity": "warning" if total_length < 150 else "error"
                    })
                    score -= 0.1 if total_length < 150 else 0.2
        
        score = max(0.0, score)
        
        return {
            "score": score,
            "issues": issues,
            "issue_count": len(issues),
            "pass": len(issues) == 0
        }
    
    def _check_manufacturability(self, placement: Placement) -> Dict:
        """Check manufacturing feasibility."""
        issues = []
        score = 1.0
        
        components = list(placement.components.values())
        board = placement.board
        
        # Check edge clearance
        edge_clearance = 2.0  # mm
        for comp in components:
            if (comp.x - comp.width/2 < edge_clearance or
                comp.x + comp.width/2 > board.width - edge_clearance or
                comp.y - comp.height/2 < edge_clearance or
                comp.y + comp.height/2 > board.height - edge_clearance):
                issues.append({
                    "type": "edge_clearance",
                    "component": comp.name,
                    "severity": "warning"
                })
                score -= 0.05
        
        # Check component density
        board_area = board.width * board.height
        component_area = sum(c.width * c.height for c in components)
        density = component_area / board_area
        
        if density > 0.7:
            issues.append({
                "type": "high_density",
                "density": float(density),
                "severity": "warning"
            })
            score -= 0.1
        
        score = max(0.0, score)
        
        return {
            "score": score,
            "issues": issues,
            "density": float(density),
            "pass": len([i for i in issues if i.get("severity") == "error"]) == 0
        }
    
    def _analyze_distribution(self, placement: Placement) -> Dict:
        """Analyze component distribution uniformity."""
        components = list(placement.components.values())
        if len(components) < 2:
            return {"score": 1.0, "uniformity": 1.0}
        
        positions = np.array([[c.x, c.y] for c in components])
        
        # Calculate spread
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        board_area = placement.board.width * placement.board.height
        
        # Ideal spread would use most of board
        used_area = x_range * y_range
        utilization = used_area / board_area
        
        # Score based on utilization (0.3-0.7 is good)
        if 0.3 <= utilization <= 0.7:
            score = 1.0
        elif utilization < 0.3:
            score = utilization / 0.3  # Underutilized
        else:
            score = max(0.0, 1.0 - (utilization - 0.7) / 0.3)  # Overcrowded
        
        return {
            "score": score,
            "utilization": float(utilization),
            "spread": {"x": float(x_range), "y": float(y_range)}
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Design rules
        drc = results["categories"].get("design_rules", {})
        if drc.get("violation_count", 0) > 0:
            recommendations.append(f"Fix {drc['violation_count']} design rule violations (clearance, bounds)")
        
        # Thermal
        thermal = results["categories"].get("thermal", {})
        if thermal.get("hotspots", 0) > 0:
            recommendations.append(f"Improve thermal spacing: {thermal['hotspots']} hotspots detected")
        
        # Signal integrity
        si = results["categories"].get("signal_integrity", {})
        if si.get("issue_count", 0) > 0:
            recommendations.append(f"Optimize {si['issue_count']} long traces for signal integrity")
        
        # Manufacturing
        dfm = results["categories"].get("manufacturability", {})
        if dfm.get("density", 0) > 0.7:
            recommendations.append("Reduce component density for better manufacturability")
        
        if not recommendations:
            recommendations.append("Design quality is excellent - ready for manufacturing")
        
        return recommendations

