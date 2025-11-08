"""
Verifier Agent

Design-rule checks and manufacturability verification.
"""

from typing import Dict, List
from backend.geometry.placement import Placement


class VerifierAgent:
    """Agent for design-rule checking."""
    
    def __init__(self):
        """Initialize verifier agent."""
        self.name = "VerifierAgent"
    
    async def process(self, placement: Placement) -> Dict:
        """
        Verify placement against design rules.
        
        Args:
            placement: Placement to verify
        
        Returns:
            {
                "success": bool,
                "violations": List[Dict],
                "warnings": List[Dict]
            }
        """
        violations = []
        warnings = []
        
        # Check validity
        validity_errors = placement.check_validity()
        for error in validity_errors:
            violations.append({
                "type": "validity",
                "severity": "error",
                "message": error
            })
        
        # Check minimum clearances
        comp_list = list(placement.components.values())
        for i, c1 in enumerate(comp_list):
            for c2 in comp_list[i+1:]:
                dist = ((c1.x - c2.x)**2 + (c1.y - c2.y)**2)**0.5
                min_dist = placement.board.clearance + (c1.width + c2.width) / 2
                
                if dist < min_dist * 0.8:  # Warning if too close
                    warnings.append({
                        "type": "clearance",
                        "severity": "warning",
                        "message": f"{c1.name} and {c2.name} are very close ({dist:.2f}mm < {min_dist:.2f}mm)"
                    })
        
        # Check high-power component spacing
        high_power_components = [c for c in comp_list if c.power > 1.0]
        for i, c1 in enumerate(high_power_components):
            for c2 in high_power_components[i+1:]:
                dist = ((c1.x - c2.x)**2 + (c1.y - c2.y)**2)**0.5
                if dist < 20.0:  # High-power components should be spaced
                    warnings.append({
                        "type": "thermal",
                        "severity": "warning",
                        "message": f"High-power components {c1.name} and {c2.name} are close ({dist:.2f}mm)"
                    })
        
        return {
            "success": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "agent": self.name
        }
    
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

