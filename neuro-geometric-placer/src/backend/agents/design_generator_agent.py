"""
Design Generator Agent

Generates PCB designs from natural language descriptions using xAI.
"""

from typing import Dict, List, Optional, Any
try:
    from backend.ai.xai_client import XAIClient
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.geometry.board import Board
    from backend.geometry.net import Net
except ImportError:
    from src.backend.ai.xai_client import XAIClient
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.geometry.board import Board
    from src.backend.geometry.net import Net


class DesignGeneratorAgent:
    """Generates PCB designs from natural language using xAI."""
    
    def __init__(self):
        """Initialize design generator."""
        self.xai_client = XAIClient()
        self.name = "DesignGeneratorAgent"
    
    async def generate_from_natural_language(
        self,
        description: str,
        board_size: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Generate PCB design from natural language description.
        
        Args:
            description: Natural language description of the PCB design
            board_size: Optional board dimensions {"width": 100, "height": 100}
        
        Returns:
            Dictionary with placement data
        """
        try:
            # Use xAI to parse natural language and generate design
            prompt = f"""
            Generate a PCB design from this description: {description}
            
            Parse the description and identify:
            1. Components needed (ICs, resistors, capacitors, etc.)
            2. Component specifications (package types, power requirements)
            3. Connections/nets between components
            4. Board size requirements
            5. Functional modules
            
            Return a structured JSON with:
            {{
                "board": {{"width": 100, "height": 100, "clearance": 0.5}},
                "components": [
                    {{
                        "name": "U1",
                        "package": "SOIC-8",
                        "width": 5,
                        "height": 4,
                        "power": 0.5,
                        "x": 50,
                        "y": 50,
                        "angle": 0,
                        "placed": true
                    }}
                ],
                "nets": [
                    {{
                        "name": "VCC",
                        "pins": [["U1", "pin1"], ["U2", "pin1"]]
                    }}
                ],
                "modules": [
                    {{
                        "name": "Power Supply",
                        "components": ["U1", "C1", "L1"]
                    }}
                ]
            }}
            
            Make it realistic and complete.
            """
            
            response = self.xai_client._call_api(prompt, max_tokens=2000)
            
            # Parse response to extract JSON
            design_data = self._parse_design_response(response, board_size)
            
            # Validate and enhance design
            design_data = self._enhance_design(design_data, description)
            
            return {
                "success": True,
                "placement": design_data,
                "agent": self.name,
                "description": description
            }
            
        except Exception as e:
            # Fallback: Generate basic design
            return {
                "success": True,
                "placement": self._generate_fallback_design(description, board_size),
                "agent": self.name,
                "description": description,
                "warning": f"Used fallback design due to: {str(e)}"
            }
    
    def _parse_design_response(self, response: str, board_size: Optional[Dict] = None) -> Dict:
        """Parse xAI response to extract design data."""
        import json
        import re
        
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                design_data = json.loads(json_match.group())
                if board_size:
                    design_data["board"].update(board_size)
                return design_data
        except:
            pass
        
        # Fallback: Generate from keywords
        return self._generate_from_keywords(response, board_size)
    
    def _generate_from_keywords(self, description: str, board_size: Optional[Dict] = None) -> Dict:
        """Generate design from keywords in description."""
        description_lower = description.lower()
        
        # Determine components from keywords
        components = []
        nets = []
        
        # Common patterns
        if "amplifier" in description_lower or "amp" in description_lower:
            components.extend([
                {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 40, "y": 40, "angle": 0, "placed": True},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 60, "y": 40, "angle": 0, "placed": True},
                {"name": "C1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 50, "angle": 0, "placed": True},
            ])
            nets.append({"name": "VCC", "pins": [["U1", "pin8"]]})
            nets.append({"name": "GND", "pins": [["U1", "pin4"], ["R1", "pin2"], ["C1", "pin2"]]})
        
        if "power" in description_lower or "supply" in description_lower:
            components.extend([
                {"name": "PWR_IC", "package": "SOIC-8", "width": 5, "height": 4, "power": 1.5, "x": 30, "y": 30, "angle": 0, "placed": True},
                {"name": "L1", "package": "INDUCTOR-10MM", "width": 10, "height": 10, "power": 0.0, "x": 50, "y": 30, "angle": 0, "placed": True},
                {"name": "C_PWR", "package": "CAP-10MM", "width": 10, "height": 10, "power": 0.0, "x": 70, "y": 30, "angle": 0, "placed": True},
            ])
            nets.append({"name": "VBAT", "pins": [["PWR_IC", "VIN"]]})
            nets.append({"name": "VCC", "pins": [["PWR_IC", "VOUT"], ["L1", "pin1"]]})
        
        if "sensor" in description_lower:
            components.extend([
                {"name": "SENSOR", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.1, "x": 50, "y": 50, "angle": 0, "placed": True},
                {"name": "R_SENSE", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 60, "y": 50, "angle": 0, "placed": True},
            ])
            nets.append({"name": "SENSOR_OUT", "pins": [["SENSOR", "OUT"], ["R_SENSE", "pin1"]]})
        
        # Default board size
        board = board_size or {"width": 100, "height": 100, "clearance": 0.5}
        
        return {
            "board": board,
            "components": components if components else [
                {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 50, "y": 50, "angle": 0, "placed": True}
            ],
            "nets": nets if nets else [],
            "modules": []
        }
    
    def _enhance_design(self, design_data: Dict, description: str) -> Dict:
        """Enhance design with additional details from description."""
        # Add missing fields, validate structure
        if "board" not in design_data:
            design_data["board"] = {"width": 100, "height": 100, "clearance": 0.5}
        
        if "components" not in design_data:
            design_data["components"] = []
        
        if "nets" not in design_data:
            design_data["nets"] = []
        
        # Ensure all components have required fields
        for comp in design_data["components"]:
            comp.setdefault("placed", True)
            comp.setdefault("angle", 0)
            comp.setdefault("power", 0.0)
        
        return design_data
    
    def _generate_fallback_design(self, description: str, board_size: Optional[Dict] = None) -> Dict:
        """Generate fallback design when xAI fails."""
        return self._generate_from_keywords(description, board_size)

