"""
DFM Agent

Specialized agent for Design for Manufacturing (DFM) validation.
Comprehensive manufacturing constraint checking and optimization recommendations.
"""

from typing import Dict, List, Optional
import logging

try:
    from backend.geometry.placement import Placement
    from backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator
    from backend.agents.verifier_agent import VerifierAgent
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator
    from src.backend.agents.verifier_agent import VerifierAgent

logger = logging.getLogger(__name__)


class DFMAgent:
    """
    Specialized agent for Design for Manufacturing validation.
    
    Features:
    - Comprehensive DFM checks
    - Manufacturing constraint validation
    - Manufacturer-specific DFM rules (JLCPCB, PCBWay)
    - DFM score calculation
    - Optimization recommendations
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None, manufacturer: str = "standard"):
        """
        Initialize DFM agent.
        
        Args:
            constraints: Fabrication constraints
            manufacturer: Manufacturer name ("jlcpcb", "pcbway", "standard")
        """
        self.name = "DFMAgent"
        self.constraints = constraints or FabricationConstraints()
        self.constraint_validator = ConstraintValidator(self.constraints)
        self.verifier_agent = VerifierAgent(constraints=self.constraints)
        self.manufacturer = manufacturer
        
        # Manufacturer-specific constraints
        self.manufacturer_constraints = self._get_manufacturer_constraints(manufacturer)
    
    async def validate_dfm(self, placement: Placement) -> Dict:
        """
        Comprehensive DFM validation.
        
        Args:
            placement: Placement to validate
        
        Returns:
            {
                "success": bool,
                "dfm_score": float,
                "dfm_ready": bool,
                "violations": List[Dict],
                "warnings": List[Dict],
                "recommendations": List[Dict],
                "manufacturer_compliance": Dict
            }
        """
        try:
            logger.info(f"✅ {self.name}: Running DFM validation")
            
            violations = []
            warnings = []
            recommendations = []
            
            # Use VerifierAgent for basic DFM checks
            verification_result = await self.verifier_agent.process(placement, include_dfm=True)
            violations.extend(verification_result.get("violations", []))
            warnings.extend(verification_result.get("warnings", []))
            
            # Additional DFM checks
            dfm_checks = await self._run_dfm_checks(placement)
            violations.extend(dfm_checks.get("violations", []))
            warnings.extend(dfm_checks.get("warnings", []))
            recommendations.extend(dfm_checks.get("recommendations", []))
            
            # Manufacturer-specific compliance
            manufacturer_compliance = self._check_manufacturer_compliance(placement)
            
            # Calculate DFM score
            dfm_score = self._calculate_dfm_score(placement, violations, warnings)
            dfm_ready = dfm_score >= 0.9 and len([v for v in violations if v.get("severity") == "error"]) == 0
            
            logger.info(f"✅ {self.name}: DFM validation complete - Score: {dfm_score:.2f}, Ready: {dfm_ready}")
            
            return {
                "success": True,
                "dfm_score": dfm_score,
                "dfm_ready": dfm_ready,
                "violations": violations,
                "warnings": warnings,
                "recommendations": recommendations,
                "manufacturer_compliance": manufacturer_compliance,
                "agent": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name}: DFM validation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    async def _run_dfm_checks(self, placement: Placement) -> Dict:
        """Run additional DFM-specific checks."""
        violations = []
        warnings = []
        recommendations = []
        
        # Check component density
        board_area = placement.board.width * placement.board.height
        component_density = len(placement.components) / board_area if board_area > 0 else 0
        
        if component_density > 0.1:  # components/mm²
            warnings.append({
                "type": "component_density",
                "severity": "warning",
                "message": f"High component density: {component_density:.2f} components/mm²",
                "recommendation": "Consider larger board or multi-layer design"
            })
        
        # Check for components too close to board edge
        edge_margin = 2.0  # mm
        for comp in placement.components.values():
            if comp.x < edge_margin or comp.x > placement.board.width - edge_margin:
                warnings.append({
                    "type": "edge_clearance",
                    "severity": "warning",
                    "component": comp.name,
                    "message": f"{comp.name} too close to board edge (x={comp.x:.2f}mm)"
                })
            if comp.y < edge_margin or comp.y > placement.board.height - edge_margin:
                warnings.append({
                    "type": "edge_clearance",
                    "severity": "warning",
                    "component": comp.name,
                    "message": f"{comp.name} too close to board edge (y={comp.y:.2f}mm)"
                })
        
        # Check for proper test points
        if len(placement.components) > 10:
            recommendations.append({
                "type": "test_points",
                "priority": "medium",
                "message": "Consider adding test points for key signals",
                "benefit": "Easier testing and debugging"
            })
        
        return {
            "violations": violations,
            "warnings": warnings,
            "recommendations": recommendations
        }
    
    def _check_manufacturer_compliance(self, placement: Placement) -> Dict:
        """Check compliance with manufacturer-specific constraints."""
        compliance = {
            "manufacturer": self.manufacturer,
            "compliant": True,
            "violations": []
        }
        
        # Check against manufacturer constraints
        if self.manufacturer == "jlcpcb":
            # JLCPCB specific checks
            if self.constraints.min_trace_width < 0.1:  # 4 mil minimum
                compliance["violations"].append({
                    "type": "trace_width",
                    "message": f"Trace width {self.constraints.min_trace_width}mm below JLCPCB minimum (0.1mm)"
                })
                compliance["compliant"] = False
        
        elif self.manufacturer == "pcbway":
            # PCBWay specific checks
            if self.constraints.min_trace_width < 0.075:  # 3 mil minimum
                compliance["violations"].append({
                    "type": "trace_width",
                    "message": f"Trace width {self.constraints.min_trace_width}mm below PCBWay minimum (0.075mm)"
                })
                compliance["compliant"] = False
        
        return compliance
    
    def _calculate_dfm_score(self, placement: Placement, violations: List[Dict], warnings: List[Dict]) -> float:
        """Calculate comprehensive DFM score."""
        base_score = 1.0
        
        # Deduct for violations
        for violation in violations:
            if violation.get("severity") == "error":
                base_score -= 0.1
            else:
                base_score -= 0.05
        
        # Deduct for warnings
        for warning in warnings:
            base_score -= 0.02
        
        # Bonus for good practices
        # Check component distribution
        if len(placement.components) > 0:
            comp_positions = [(c.x, c.y) for c in placement.components.values()]
            # Check if components are well-distributed (not all clustered)
            # Simplified check
            if len(set(int(x/10) for x, y in comp_positions)) > len(comp_positions) * 0.5:
                base_score += 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _get_manufacturer_constraints(self, manufacturer: str) -> Dict:
        """Get manufacturer-specific constraints."""
        constraints = {
            "jlcpcb": {
                "min_trace_width": 0.1,  # mm (4 mil)
                "min_trace_spacing": 0.1,  # mm (4 mil)
                "min_via_drill": 0.2,  # mm (8 mil)
                "max_layers": 6,
                "board_thickness_options": [0.8, 1.0, 1.2, 1.6, 2.0]
            },
            "pcbway": {
                "min_trace_width": 0.075,  # mm (3 mil)
                "min_trace_spacing": 0.075,  # mm (3 mil)
                "min_via_drill": 0.15,  # mm (6 mil)
                "max_layers": 32,
                "board_thickness_options": [0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0]
            },
            "standard": {
                "min_trace_width": 0.15,  # mm (6 mil)
                "min_trace_spacing": 0.15,  # mm (6 mil)
                "min_via_drill": 0.3,  # mm (12 mil)
                "max_layers": 10
            }
        }
        
        return constraints.get(manufacturer, constraints["standard"])

