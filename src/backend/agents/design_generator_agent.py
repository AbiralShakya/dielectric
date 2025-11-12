"""
Design Generator Agent

Production-scalable agent for generating PCB designs from natural language.
Integrates with KiCad libraries, supports multi-layer boards, and considers manufacturing availability.
"""

from typing import Dict, List, Optional, Any
import logging

try:
    from backend.ai.enhanced_xai_client import EnhancedXAIClient
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.geometry.board import Board, LayerStackup
    from backend.geometry.net import Net
except ImportError:
    try:
        from src.backend.ai.enhanced_xai_client import EnhancedXAIClient
    except ImportError:
        from src.backend.ai.xai_client import XAIClient as EnhancedXAIClient
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.geometry.board import Board, LayerStackup
    from src.backend.geometry.net import Net

logger = logging.getLogger(__name__)


class ComponentLibrary:
    """Component library lookup for real footprints."""
    
    def __init__(self):
        """Initialize component library."""
        self.kicad_library_manager = None
        self.jlcpcb_parts = {}  # Cache for JLCPCB parts
        
        # Try to initialize KiCad library manager
        try:
            import sys
            sys.path.insert(0, '/Users/abiralshakya/Documents/hackprinceton2025/dielectric/kicad-mcp-server/python')
            from commands.library import LibraryManager
            self.kicad_library_manager = LibraryManager()
            logger.info("ComponentLibrary: KiCad library manager initialized")
        except Exception as e:
            logger.warning(f"ComponentLibrary: KiCad library not available: {e}")
            self.kicad_library_manager = None
    
    def lookup_footprint(self, package: str, component_type: Optional[str] = None) -> Optional[str]:
        """
        Lookup footprint from KiCad library.
        
        Args:
            package: Package name (e.g., "0805", "SOIC-8")
            component_type: Component type (e.g., "resistor", "capacitor", "ic")
        
        Returns:
            Footprint specification (e.g., "Resistor_SMD:R_0805_2012Metric") or None
        """
        if not self.kicad_library_manager:
            return None
        
        try:
            # Common footprint mappings
            footprint_mappings = {
                "0805": {
                    "resistor": "Resistor_SMD:R_0805_2012Metric",
                    "capacitor": "Capacitor_SMD:C_0805_2012Metric",
                    "inductor": "Inductor_SMD:L_0805_2012Metric"
                },
                "0603": {
                    "resistor": "Resistor_SMD:R_0603_1608Metric",
                    "capacitor": "Capacitor_SMD:C_0603_1608Metric"
                },
                "SOIC-8": {
                    "ic": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
                },
                "SOIC-14": {
                    "ic": "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm"
                }
            }
            
            # Try direct lookup
            if package in footprint_mappings and component_type:
                footprint_spec = footprint_mappings[package].get(component_type)
                if footprint_spec:
                    # Verify footprint exists
                    result = self.kicad_library_manager.find_footprint(footprint_spec)
                    if result:
                        return footprint_spec
            
            # Try search
            search_pattern = f"*{package}*"
            results = self.kicad_library_manager.search_footprints(search_pattern, limit=5)
            if results:
                return results[0].get("full_name")
            
        except Exception as e:
            logger.warning(f"Footprint lookup failed for {package}: {e}")
        
        return None
    
    def check_jlcpcb_availability(self, part_number: str) -> Optional[Dict]:
        """
        Check JLCPCB parts database for component availability.
        
        Args:
            part_number: Component part number or LCSC number
        
        Returns:
            Part information if available, None otherwise
        """
        try:
            from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager
            parts_manager = JLCPCBPartsManager()
            
            # Try as LCSC number first
            part_info = parts_manager.get_part_info(part_number)
            if part_info:
                return {
                    "available": part_info.get("stock", 0) > 0,
                    "stock": part_info.get("stock", 0),
                    "price": self._extract_price(part_info.get("prices", [])),
                    "library_type": part_info.get("library_type", "Extended"),
                    "lcsc": part_info.get("lcsc"),
                    "description": part_info.get("description", "")
                }
            
            # Try searching by description/package
            results = parts_manager.search_parts(query=part_number, limit=1)
            if results:
                part_info = results[0]
                return {
                    "available": part_info.get("stock", 0) > 0,
                    "stock": part_info.get("stock", 0),
                    "price": self._extract_price(json.loads(part_info.get("price_json", "[]"))),
                    "library_type": part_info.get("library_type", "Extended"),
                    "lcsc": part_info.get("lcsc"),
                    "description": part_info.get("description", "")
                }
        except ImportError:
            # Fallback to cached/common parts if database not available
            pass
        except Exception as e:
            logger.warning(f"JLCPCB lookup failed: {str(e)}")
        
        # Fallback: return cached/common parts
        common_jlcpcb_parts = {
            "C0805": {"available": True, "price": 0.001, "stock": 10000, "library_type": "Basic"},
            "R0805": {"available": True, "price": 0.001, "stock": 10000, "library_type": "Basic"},
            "SOIC-8": {"available": True, "price": 0.01, "stock": 5000, "library_type": "Basic"}
        }
        
        return common_jlcpcb_parts.get(part_number)
    
    def _extract_price(self, prices: List[Dict]) -> float:
        """Extract unit price from price breaks."""
        if not prices:
            return 0.0
        # Return price for quantity 1 (first break)
        return float(prices[0].get("price", 0.0)) if isinstance(prices[0], dict) else 0.0


class DesignGeneratorAgent:
    """
    Production-scalable agent for generating PCB designs.
    
    Features:
    - KiCad library integration for real footprints
    - Multi-layer board generation with proper stackup
    - JLCPCB parts database integration
    - Natural language design generation
    """
    
    def __init__(self):
        """Initialize design generator."""
        try:
            self.xai_client = EnhancedXAIClient()
        except Exception:
            # Fallback to basic client
            from src.backend.ai.xai_client import XAIClient
            self.xai_client = XAIClient()
        self.name = "DesignGeneratorAgent"
        self.component_library = ComponentLibrary()
    
    async def generate_from_natural_language(
        self,
        description: str,
        board_size: Optional[Dict[str, float]] = None,
        layer_count: int = 2,
        use_real_footprints: bool = True,
        check_jlcpcb: bool = False
    ) -> Dict:
        """
        Generate PCB design from natural language description.
        
        Args:
            description: Natural language description of the PCB design
            board_size: Optional board dimensions {"width": 100, "height": 100}
            layer_count: Number of layers (2, 4, 6, etc.)
            use_real_footprints: Whether to use real KiCad footprints
            check_jlcpcb: Whether to check JLCPCB parts availability
        
        Returns:
            Dictionary with placement data
        """
        import asyncio
        import concurrent.futures
        
        try:
            logger.info(f"{self.name}: Generating design from description")
            
            # Use enhanced xAI client for extensive reasoning
            if hasattr(self.xai_client, 'generate_design_with_reasoning'):
                logger.info("   Using Enhanced xAI Client for design generation")
                loop = asyncio.get_event_loop()
                
                # Retry logic with exponential backoff
                max_retries = 2
                retry_delay = 1.0
                
                for attempt in range(max_retries + 1):
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = loop.run_in_executor(
                                executor,
                                lambda: self.xai_client.generate_design_with_reasoning(description, board_size)
                            )
                            # Increased timeout to 180 seconds to match HTTP timeout and allow for extensive reasoning
                            result = await asyncio.wait_for(future, timeout=180.0)
                            break  # Success, exit retry loop
                    except asyncio.TimeoutError:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                    except Exception as e:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                
                if result.get("success"):
                    design_data = result.get("design", {})
                else:
                    raise Exception(result.get("error", "Design generation failed"))
            else:
                # Fallback to basic xAI generation
                logger.info("   Using Basic xAI Client for design generation")
                prompt = self._create_design_prompt(description, layer_count)
                
                loop = asyncio.get_event_loop()
                
                # Retry logic with exponential backoff
                max_retries = 2
                retry_delay = 1.0
                
                for attempt in range(max_retries + 1):
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = loop.run_in_executor(
                                executor,
                                lambda: self.xai_client._call_api([{"role": "user", "content": prompt}], max_tokens=2000)
                            )
                            # Increased timeout to 180 seconds for extensive reasoning
                            response = await asyncio.wait_for(future, timeout=180.0)
                            break  # Success, exit retry loop
                    except asyncio.TimeoutError as e:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                    except Exception as e:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                
                if response.get("choices"):
                    content = response["choices"][0]["message"]["content"]
                    design_data = self._parse_design_response(content, board_size)
                else:
                    raise Exception("No response from xAI")
            
            # Enhance design with real footprints and multi-layer support
            design_data = self._enhance_design(
                design_data, 
                description,
                layer_count=layer_count,
                use_real_footprints=use_real_footprints,
                check_jlcpcb=check_jlcpcb
            )
            
            return {
                "success": True,
                "placement": design_data,
                "agent": self.name,
                "description": description,
                "layer_count": layer_count,
                "enhanced": True
            }
            
        except asyncio.TimeoutError:
            error_msg = "xAI API call timed out after 180 seconds (with retries). The request may be too complex or the API is slow. Try simplifying the description or try again later."
            logger.error(f"{error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Design generation failed: {str(e)}"
            logger.error(f"{error_msg}")
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)
    
    def _create_design_prompt(self, description: str, layer_count: int) -> str:
        """Create enhanced design prompt with multi-layer support."""
        return f"""
        Generate a PCB design from this description: {description}
        
        Board specifications:
        - Layer count: {layer_count}
        - Standard stackup for {layer_count}-layer board
        
        Parse the description and identify:
        1. Components needed (ICs, resistors, capacitors, etc.)
        2. Component specifications (package types, power requirements)
        3. Connections/nets between components
        4. Board size requirements
        5. Functional modules
        6. Layer assignment (signal layers, power planes, ground planes)
        
        Return a structured JSON with:
        {{
            "board": {{
                "width": 100, 
                "height": 100, 
                "clearance": 0.5,
                "layer_count": {layer_count}
            }},
            "components": [
                {{
                    "name": "U1",
                    "package": "SOIC-8",
                    "component_type": "ic",
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
                    "pins": [["U1", "pin1"], ["U2", "pin1"]],
                    "layer": "In2.Cu"
                }}
            ],
            "modules": [
                {{
                    "name": "Power Supply",
                    "components": ["U1", "C1", "L1"]
                }}
            ]
        }}
        
        Make it realistic and complete for a {layer_count}-layer board.
        """
    
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
    
    def _enhance_design(
        self, 
        design_data: Dict, 
        description: str,
        layer_count: int = 2,
        use_real_footprints: bool = True,
        check_jlcpcb: bool = False
    ) -> Dict:
        """
        Enhance design with real footprints, multi-layer support, and manufacturing checks.
        
        Args:
            design_data: Initial design data
            description: Original description
            layer_count: Number of layers
            use_real_footprints: Whether to use real KiCad footprints
            check_jlcpcb: Whether to check JLCPCB availability
        """
        # Add missing fields, validate structure
        if "board" not in design_data:
            design_data["board"] = {"width": 100, "height": 100, "clearance": 0.5}
        
        # Set layer count
        design_data["board"]["layer_count"] = layer_count
        
        # Create proper stackup based on layer count
        if layer_count == 4:
            design_data["board"]["layer_stackup"] = {
                "layer_count": 4,
                "layers": [
                    {"name": "F.Cu", "type": "signal", "thickness": 0.035},
                    {"name": "In1.Cu", "type": "ground", "thickness": 0.035},
                    {"name": "In2.Cu", "type": "power", "thickness": 0.035},
                    {"name": "B.Cu", "type": "signal", "thickness": 0.035}
                ],
                "dielectric_thickness": 1.6
            }
        elif layer_count >= 6:
            design_data["board"]["layer_stackup"] = {
                "layer_count": layer_count,
                "layers": [
                    {"name": "F.Cu", "type": "signal", "thickness": 0.035},
                    {"name": "In1.Cu", "type": "ground", "thickness": 0.035},
                    {"name": "In2.Cu", "type": "signal", "thickness": 0.035},
                    {"name": "In3.Cu", "type": "signal", "thickness": 0.035},
                    {"name": "In4.Cu", "type": "power", "thickness": 0.035},
                    {"name": "B.Cu", "type": "signal", "thickness": 0.035}
                ],
                "dielectric_thickness": 1.6
            }
        
        if "components" not in design_data:
            design_data["components"] = []
        
        if "nets" not in design_data:
            design_data["nets"] = []
        
        # Enhance components with real footprints
        for comp in design_data["components"]:
            comp.setdefault("placed", True)
            comp.setdefault("angle", 0)
            comp.setdefault("power", 0.0)
            
            # Lookup real footprint if requested
            if use_real_footprints:
                package = comp.get("package", "")
                component_type = comp.get("component_type", "")
                footprint = self.component_library.lookup_footprint(package, component_type)
                if footprint:
                    comp["footprint"] = footprint
                    comp["footprint_source"] = "kicad_library"
                    logger.info(f"   Found footprint for {comp.get('name')}: {footprint}")
            
            # Check JLCPCB availability if requested
            if check_jlcpcb:
                jlcpcb_info = self.component_library.check_jlcpcb_availability(package)
                if jlcpcb_info:
                    comp["jlcpcb"] = jlcpcb_info
        
        return design_data
    
    def _generate_fallback_design(self, description: str, board_size: Optional[Dict] = None) -> Dict:
        """Generate fallback design when xAI fails."""
        return self._generate_from_keywords(description, board_size)

