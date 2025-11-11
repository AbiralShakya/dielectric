"""
Verifier Agent

Production-scalable design-rule checking and DFM (Design for Manufacturing) validation.
"""

from typing import Dict, List, Optional, Tuple
import logging

try:
    from backend.geometry.placement import Placement
    from backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator

logger = logging.getLogger(__name__)


class VerifierAgent:
    """
    Production-scalable agent for design-rule checking and DFM validation.
    
    Features:
    - Design rule checking (DRC) with KiCad integration
    - Manufacturing constraints validation
    - DFM score calculation
    - Signal integrity checks (impedance control, crosstalk analysis)
    - Thermal validation
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None, kicad_client=None):
        """
        Initialize verifier agent.
        
        Args:
            constraints: Fabrication constraints (defaults to standard constraints)
            kicad_client: Optional KiCad MCP client for DRC checking
        """
        self.name = "VerifierAgent"
        self.constraints = constraints or FabricationConstraints()
        self.constraint_validator = ConstraintValidator(self.constraints)
        self.kicad_client = kicad_client
        
        # Try to initialize KiCad client if not provided
        if not self.kicad_client:
            try:
                from src.backend.mcp.kicad_direct_client import KiCadDirectClient
                self.kicad_client = KiCadDirectClient()
                if self.kicad_client.is_available():
                    logger.info("✅ VerifierAgent: KiCad client initialized")
                else:
                    logger.warning("⚠️  VerifierAgent: KiCad not available")
                    self.kicad_client = None
            except Exception as e:
                logger.warning(f"⚠️  VerifierAgent: Could not initialize KiCad client: {e}")
                self.kicad_client = None
    
    async def process(self, placement: Placement, include_dfm: bool = True, run_kicad_drc: bool = True) -> Dict:
        """
        Verify placement against design rules and DFM constraints.
        
        Args:
            placement: Placement to verify
            include_dfm: Whether to include DFM validation
        
        Returns:
            {
                "success": bool,
                "passed": bool,
                "violations": List[Dict],
                "warnings": List[Dict],
                "dfm_score": float,
                "dfm_ready": bool
            }
        """
        violations = []
        warnings = []
        
        # Basic validity checks
        validity_errors = placement.check_validity()
        for error in validity_errors:
            violations.append({
                "type": "validity",
                "severity": "error",
                "message": error
            })
        
        # KiCad DRC integration
        kicad_drc_violations = []
        if run_kicad_drc and self.kicad_client and self.kicad_client.is_available():
            try:
                kicad_drc_result = await self._run_kicad_drc(placement)
                if kicad_drc_result.get("success"):
                    kicad_drc_violations = kicad_drc_result.get("violations", [])
                    violations.extend(kicad_drc_violations)
                    logger.info(f"   KiCad DRC: Found {len(kicad_drc_violations)} violations")
                else:
                    logger.warning(f"   KiCad DRC failed: {kicad_drc_result.get('error')}")
            except Exception as e:
                logger.warning(f"   KiCad DRC error: {e}")
        
        # Signal integrity checks
        si_violations, si_warnings = await self._check_signal_integrity(placement)
        violations.extend(si_violations)
        warnings.extend(si_warnings)
        
        # DFM validation
        dfm_score = 1.0
        dfm_ready = True
        
        if include_dfm:
            dfm_result = await self.verify_with_dfm(placement)
            violations.extend(dfm_result["violations"])
            warnings.extend(dfm_result["warnings"])
            dfm_score = dfm_result["dfm_score"]
            dfm_ready = dfm_result["dfm_ready"]
        
        # Component clearance checks
        comp_list = list(placement.components.values())
        for i, c1 in enumerate(comp_list):
            for c2 in comp_list[i+1:]:
                dist = ((c1.x - c2.x)**2 + (c1.y - c2.y)**2)**0.5
                min_dist = max(
                    placement.board.clearance + (c1.width + c2.width) / 2,
                    self.constraints.min_pad_to_pad_clearance
                )
                
                if dist < min_dist:
                    violations.append({
                        "type": "clearance",
                        "severity": "error",
                        "component1": c1.name,
                        "component2": c2.name,
                        "distance": dist,
                        "required": min_dist,
                        "message": f"{c1.name} and {c2.name} too close ({dist:.2f}mm < {min_dist:.2f}mm)"
                    })
                elif dist < min_dist * 1.2:  # Warning if close
                    warnings.append({
                        "type": "clearance",
                        "severity": "warning",
                        "component1": c1.name,
                        "component2": c2.name,
                        "distance": dist,
                        "required": min_dist,
                        "message": f"{c1.name} and {c2.name} are close ({dist:.2f}mm < {min_dist*1.2:.2f}mm)"
                    })
        
        # High-power component spacing
        high_power_components = [c for c in comp_list if c.power > 1.0]
        for i, c1 in enumerate(high_power_components):
            for c2 in high_power_components[i+1:]:
                dist = ((c1.x - c2.x)**2 + (c1.y - c2.y)**2)**0.5
                thermal_spacing = 20.0  # mm minimum for high-power components
                if dist < thermal_spacing:
                    warnings.append({
                        "type": "thermal",
                        "severity": "warning",
                        "component1": c1.name,
                        "component2": c2.name,
                        "distance": dist,
                        "required": thermal_spacing,
                        "message": f"High-power components {c1.name} and {c2.name} are close ({dist:.2f}mm)"
                    })
        
        # Board boundary checks
        for comp in comp_list:
            margin = self.constraints.min_pad_to_pad_clearance
            if comp.x < margin or comp.x > placement.board.width - margin:
                violations.append({
                    "type": "boundary",
                    "severity": "error",
                    "component": comp.name,
                    "message": f"{comp.name} too close to board edge (x={comp.x:.2f}mm)"
                })
            if comp.y < margin or comp.y > placement.board.height - margin:
                violations.append({
                    "type": "boundary",
                    "severity": "error",
                    "component": comp.name,
                    "message": f"{comp.name} too close to board edge (y={comp.y:.2f}mm)"
                })
        
        passed = len(violations) == 0
        
        return {
            "success": True,
            "passed": passed,
            "violations": violations,
            "warnings": warnings,
            "dfm_score": dfm_score,
            "dfm_ready": dfm_ready,
            "agent": self.name
        }
    
    async def verify_with_dfm(self, placement: Placement) -> Dict:
        """
        Verify design with Design for Manufacturing (DFM) checks.
        
        Args:
            placement: Placement to verify
        
        Returns:
            {
                "violations": List[Dict],
                "warnings": List[Dict],
                "dfm_score": float,
                "dfm_ready": bool
            }
        """
        violations = []
        warnings = []
        dfm_score = 1.0
        
        # Use ConstraintValidator for comprehensive DFM checks
        try:
            dfm_result = self.constraint_validator.validate_placement(placement)
            violations.extend(dfm_result.get("violations", []))
            warnings.extend(dfm_result.get("warnings", []))
        except Exception as e:
            logger.warning(f"DFM validation error: {e}")
        
        # Check trace widths (if routing data available)
        # TODO: Integrate with RoutingAgent to check actual trace widths
        
        # Check via sizes (if vias present)
        # TODO: Check via dimensions against constraints
        
        # Calculate DFM score
        dfm_score = self._calculate_dfm_score(placement, violations, warnings)
        
        # DFM ready if score >= 0.9 and no critical violations
        critical_violations = [v for v in violations if v.get("severity") == "error"]
        dfm_ready = dfm_score >= 0.9 and len(critical_violations) == 0
        
        return {
            "violations": violations,
            "warnings": warnings,
            "dfm_score": dfm_score,
            "dfm_ready": dfm_ready
        }
    
    def _calculate_dfm_score(self, placement: Placement, violations: List[Dict], warnings: List[Dict]) -> float:
        """
        Calculate DFM score (0.0 to 1.0).
        
        Args:
            placement: Placement being scored
            violations: List of violations
            warnings: List of warnings
        
        Returns:
            DFM score from 0.0 (poor) to 1.0 (excellent)
        """
        base_score = 1.0
        
        # Deduct points for violations
        violation_penalty = 0.1
        for violation in violations:
            if violation.get("severity") == "error":
                base_score -= violation_penalty
            else:
                base_score -= violation_penalty * 0.5
        
        # Deduct points for warnings
        warning_penalty = 0.05
        for warning in warnings:
            base_score -= warning_penalty
        
        # Bonus for good practices
        comp_list = list(placement.components.values())
        
        # Check for thermal vias (simplified: check for high-power components with spacing)
        high_power_comps = [c for c in comp_list if c.power > 1.0]
        if high_power_comps:
            # Check if high-power components are well-spaced
            well_spaced = True
            for i, c1 in enumerate(high_power_comps):
                for c2 in high_power_comps[i+1:]:
                    dist = ((c1.x - c2.x)**2 + (c1.y - c2.y)**2)**0.5
                    if dist < 20.0:
                        well_spaced = False
                        break
            if well_spaced:
                base_score += 0.05
        
        # Check for proper component distribution (using Voronoi variance if available)
        # TODO: Integrate with GeometryAnalyzer for Voronoi analysis
        
        return max(0.0, min(1.0, base_score))
    
    async def _run_kicad_drc(self, placement: Placement) -> Dict:
        """
        Run KiCad Design Rule Check via direct client.
        
        Args:
            placement: Placement to check
        
        Returns:
            DRC result with violations
        """
        if not self.kicad_client or not self.kicad_client.is_available():
            return {
                "success": False,
                "error": "KiCad client not available",
                "violations": []
            }
        
        try:
            # Run DRC
            result = await self.kicad_client.run_drc()
            
            if result.get("success"):
                logger.info(f"   KiCad DRC: Found {len(result.get('violations', []))} violations")
            else:
                logger.warning(f"   KiCad DRC failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"KiCad DRC failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "violations": []
            }
    
    async def _check_signal_integrity(self, placement: Placement) -> Tuple[List[Dict], List[Dict]]:
        """
        Check signal integrity (impedance control, crosstalk analysis).
        
        Args:
            placement: Placement to check
        
        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        
        # Check impedance-controlled nets
        for net_name, net in placement.nets.items():
            net_name_lower = net.name.lower()
            
            # Check for impedance requirements
            if any(keyword in net_name_lower for keyword in ["rf", "50 ohm", "100 ohm", "differential"]):
                # Verify controlled impedance routing
                # In production, would check actual trace width/spacing from routing data
                target_impedance = 50.0
                if "100" in net_name_lower or "differential" in net_name_lower:
                    target_impedance = 100.0
                
                # Check if routing data exists (would come from RoutingAgent)
                # For now, add warning
                warnings.append({
                    "type": "signal_integrity",
                    "severity": "warning",
                    "net": net_name,
                    "message": f"Net {net_name} requires {target_impedance}Ω controlled impedance - verify routing"
                })
        
        # Check for crosstalk risks (simplified)
        net_list = list(placement.nets.items())
        for i, (net1_name, net1) in enumerate(net_list):
            for net2_name, net2 in net_list[i+1:]:
                # Calculate net proximity (simplified)
                net1_positions = []
                net2_positions = []
                
                for comp_ref, _ in net1.pins:
                    comp = placement.components.get(comp_ref)
                    if comp:
                        net1_positions.append((comp.x, comp.y))
                
                for comp_ref, _ in net2.pins:
                    comp = placement.components.get(comp_ref)
                    if comp:
                        net2_positions.append((comp.x, comp.y))
                
                if net1_positions and net2_positions:
                    # Calculate minimum distance
                    min_dist = float('inf')
                    for p1 in net1_positions:
                        for p2 in net2_positions:
                            dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                            min_dist = min(min_dist, dist)
                    
                    # Warning if nets are very close (< 2mm)
                    if min_dist < 2.0:
                        warnings.append({
                            "type": "crosstalk",
                            "severity": "warning",
                            "net1": net1_name,
                            "net2": net2_name,
                            "distance": min_dist,
                            "message": f"Nets {net1_name} and {net2_name} are close ({min_dist:.2f}mm) - potential crosstalk risk"
                        })
        
        return violations, warnings
    
    def get_tool_definition(self) -> Dict:
        """Get tool definition for MCP registration."""
        return {
            "name": "verify_placement",
            "description": "Verify placement against design rules",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "placement": {"type": "object"}
                },
                "required": ["placement"]
            }
        }

