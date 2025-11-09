"""
Error Fixer Agent

Automatically fixes design errors using computational geometry and reasoning.
Agentic: Actually fixes issues, not just reports them.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.quality.design_validator import DesignQualityValidator
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.quality.design_validator import DesignQualityValidator


class ErrorFixerAgent:
    """Agentic error fixer - automatically resolves design issues."""
    
    def __init__(self):
        """Initialize error fixer."""
        self.validator = DesignQualityValidator()
        self.name = "ErrorFixerAgent"
    
    async def fix_design(self, placement: Placement, max_iterations: int = 10) -> Dict:
        """
        Automatically fix all design errors.
        
        Args:
            placement: Placement to fix
            max_iterations: Maximum fix iterations
        
        Returns:
            Fixed placement and fix report
        """
        fixed_placement = placement.copy()
        fixes_applied = []
        iteration = 0
        
        while iteration < max_iterations:
            # Validate current design
            quality = self.validator.validate_design(fixed_placement)
            
            # Check if design passes
            if quality.get("pass", False) and quality.get("overall_score", 0) >= 0.7:
                break
            
            # Get issues
            issues = self._extract_issues(quality)
            
            if not issues:
                break
            
            # Fix issues
            for issue in issues:
                fix_result = self._fix_issue(fixed_placement, issue)
                if fix_result["fixed"]:
                    fixes_applied.append(fix_result)
            
            iteration += 1
        
        # Final validation
        final_quality = self.validator.validate_design(fixed_placement)
        
        return {
            "success": True,
            "placement": fixed_placement,
            "fixes_applied": fixes_applied,
            "iterations": iteration,
            "initial_quality": quality.get("overall_score", 0),
            "final_quality": final_quality.get("overall_score", 0),
            "pass": final_quality.get("pass", False),
            "agent": self.name
        }
    
    def _extract_issues(self, quality: Dict) -> List[Dict]:
        """Extract actionable issues from quality report."""
        issues = []
        
        # Design rule violations
        drc = quality.get("categories", {}).get("design_rules", {})
        for violation in drc.get("violations", []):
            issues.append({
                "type": "design_rule",
                "violation": violation
            })
        
        # Thermal hotspots
        thermal = quality.get("categories", {}).get("thermal", {})
        if thermal.get("hotspots", 0) > 0:
            issues.append({
                "type": "thermal",
                "hotspots": thermal.get("hotspots", 0)
            })
        
        # Signal integrity
        si = quality.get("categories", {}).get("signal_integrity", {})
        for issue in si.get("issues", []):
            issues.append({
                "type": "signal_integrity",
                "issue": issue
            })
        
        # Manufacturing
        dfm = quality.get("categories", {}).get("manufacturability", {})
        for issue in dfm.get("issues", []):
            if issue.get("severity") == "error":
                issues.append({
                    "type": "manufacturability",
                    "issue": issue
                })
        
        return issues
    
    def _fix_issue(self, placement: Placement, issue: Dict) -> Dict:
        """Fix a specific issue."""
        issue_type = issue.get("type")
        
        if issue_type == "design_rule":
            return self._fix_design_rule(placement, issue["violation"])
        elif issue_type == "thermal":
            return self._fix_thermal(placement, issue)
        elif issue_type == "signal_integrity":
            return self._fix_signal_integrity(placement, issue)
        elif issue_type == "manufacturability":
            return self._fix_manufacturability(placement, issue)
        
        return {"fixed": False, "reason": "Unknown issue type"}
    
    def _fix_design_rule(self, placement: Placement, violation: Dict) -> Dict:
        """Fix design rule violation."""
        violation_type = violation.get("type")
        
        if violation_type == "clearance_violation":
            # Move components apart
            comp_names = violation.get("components", [])
            if len(comp_names) >= 2:
                comp1 = placement.get_component(comp_names[0])
                comp2 = placement.get_component(comp_names[1])
                
                if comp1 and comp2:
                    # Calculate required separation
                    required_dist = violation.get("required", 10.0)
                    current_dist = np.sqrt((comp1.x - comp2.x)**2 + (comp1.y - comp2.y)**2)
                    
                    if current_dist < required_dist:
                        # Move components apart
                        direction = np.array([comp2.x - comp1.x, comp2.y - comp1.y])
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            move_distance = (required_dist - current_dist) / 2 + 1.0
                            
                            # Move comp1 away
                            comp1.x -= direction[0] * move_distance
                            comp1.y -= direction[1] * move_distance
                            
                            # Move comp2 away
                            comp2.x += direction[0] * move_distance
                            comp2.y += direction[1] * move_distance
                            
                            # Ensure within bounds
                            comp1.x = max(comp1.width/2, min(placement.board.width - comp1.width/2, comp1.x))
                            comp1.y = max(comp1.height/2, min(placement.board.height - comp1.height/2, comp1.y))
                            comp2.x = max(comp2.width/2, min(placement.board.width - comp2.width/2, comp2.x))
                            comp2.y = max(comp2.height/2, min(placement.board.height - comp2.height/2, comp2.y))
                            
                            return {
                                "fixed": True,
                                "type": "clearance_violation",
                                "components": comp_names,
                                "action": "Moved components apart"
                            }
        
        elif violation_type == "out_of_bounds":
            # Move component back into bounds
            comp_name = violation.get("component")
            comp = placement.get_component(comp_name)
            
            if comp:
                # Move to center if out of bounds
                comp.x = max(comp.width/2, min(placement.board.width - comp.width/2, comp.x))
                comp.y = max(comp.height/2, min(placement.board.height - comp.height/2, comp.y))
                
                return {
                    "fixed": True,
                    "type": "out_of_bounds",
                    "component": comp_name,
                    "action": "Moved component within board bounds"
                }
        
        return {"fixed": False, "reason": "Could not fix violation"}
    
    def _fix_thermal(self, placement: Placement, issue: Dict) -> Dict:
        """Fix thermal hotspots by spacing high-power components."""
        components = list(placement.components.values())
        high_power = [c for c in components if c.power > 1.0]
        
        if len(high_power) < 2:
            return {"fixed": False, "reason": "Not enough high-power components"}
        
        # Calculate optimal spacing (15mm minimum)
        min_spacing = 15.0
        positions = np.array([[c.x, c.y] for c in high_power])
        
        fixes = 0
        for i, c1 in enumerate(high_power):
            for j, c2 in enumerate(high_power[i+1:], i+1):
                dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                
                if dist < min_spacing:
                    # Move apart
                    direction = np.array([c2.x - c1.x, c2.y - c1.y])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        move_distance = (min_spacing - dist) / 2 + 2.0
                        
                        c1.x -= direction[0] * move_distance
                        c1.y -= direction[1] * move_distance
                        c2.x += direction[0] * move_distance
                        c2.y += direction[1] * move_distance
                        
                        # Ensure bounds
                        c1.x = max(c1.width/2, min(placement.board.width - c1.width/2, c1.x))
                        c1.y = max(c1.height/2, min(placement.board.height - c1.height/2, c1.y))
                        c2.x = max(c2.width/2, min(placement.board.width - c2.width/2, c2.x))
                        c2.y = max(c2.height/2, min(placement.board.height - c2.height/2, c2.y))
                        
                        fixes += 1
        
        return {
            "fixed": fixes > 0,
            "type": "thermal",
            "hotspots_fixed": fixes,
            "action": f"Spaced {fixes} high-power component pairs"
        }
    
    def _fix_signal_integrity(self, placement: Placement, issue: Dict) -> Dict:
        """Fix signal integrity issues by optimizing component placement."""
        si_issue = issue.get("issue", {})
        net_name = si_issue.get("net")
        
        if not net_name:
            return {"fixed": False, "reason": "No net specified"}
        
        # Find net
        net = placement.nets.get(net_name)
        if not net:
            return {"fixed": False, "reason": "Net not found"}
        
        # Get components in this net
        net_pins = net.pins
        comp_names = [pin[0] for pin in net_pins if isinstance(pin, list)]
        components = [placement.get_component(name) for name in comp_names if placement.get_component(name)]
        
        if len(components) < 2:
            return {"fixed": False, "reason": "Not enough components in net"}
        
        # Optimize placement to minimize trace length
        # Use centroid approach
        positions = np.array([[c.x, c.y] for c in components])
        centroid = np.mean(positions, axis=0)
        
        # Move components closer to centroid (but not too close)
        target_dist = 20.0  # mm
        fixes = 0
        
        for comp in components:
            dist_to_centroid = np.sqrt((comp.x - centroid[0])**2 + (comp.y - centroid[1])**2)
            
            if dist_to_centroid > target_dist:
                # Move closer
                direction = np.array([centroid[0] - comp.x, centroid[1] - comp.y])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    move_distance = min(dist_to_centroid - target_dist, 10.0)
                    
                    comp.x += direction[0] * move_distance
                    comp.y += direction[1] * move_distance
                    
                    # Ensure bounds
                    comp.x = max(comp.width/2, min(placement.board.width - comp.width/2, comp.x))
                    comp.y = max(comp.height/2, min(placement.board.height - comp.height/2, comp.y))
                    
                    fixes += 1
        
        return {
            "fixed": fixes > 0,
            "type": "signal_integrity",
            "net": net_name,
            "components_moved": fixes,
            "action": f"Optimized placement for net {net_name}"
        }
    
    def _fix_manufacturability(self, placement: Placement, issue: Dict) -> Dict:
        """Fix manufacturability issues."""
        dfm_issue = issue.get("issue", {})
        issue_type = dfm_issue.get("type")
        
        if issue_type == "edge_clearance":
            comp_name = dfm_issue.get("component")
            comp = placement.get_component(comp_name)
            
            if comp:
                # Move away from edge
                edge_clearance = 3.0  # mm
                comp.x = max(edge_clearance + comp.width/2, 
                           min(placement.board.width - edge_clearance - comp.width/2, comp.x))
                comp.y = max(edge_clearance + comp.height/2,
                           min(placement.board.height - edge_clearance - comp.height/2, comp.y))
                
                return {
                    "fixed": True,
                    "type": "edge_clearance",
                    "component": comp_name,
                    "action": "Moved component away from board edge"
                }
        
        return {"fixed": False, "reason": "Could not fix manufacturability issue"}

