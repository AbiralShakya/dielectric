"""
Design Generator Agent

Production-scalable agent for generating PCB designs from natural language.
Integrates with KiCad libraries, supports multi-layer boards, and considers manufacturing availability.
"""

from typing import Dict, List, Optional, Any
import logging
import json

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
            
            # Try as LCSC number first (if it looks like one)
            if part_number.startswith("C") and len(part_number) >= 4:
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
            logger.debug("JLCPCB database not available, using fallback")
        except Exception as e:
            logger.warning(f"JLCPCB lookup failed for '{part_number}': {str(e)}")
        
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

            # Initialize design_data
            design_data = None

            # Use enhanced xAI client for extensive reasoning
            if hasattr(self.xai_client, 'generate_design_with_reasoning'):
                logger.info("   Using Enhanced xAI Client for design generation")
                loop = asyncio.get_event_loop()
                
                # Retry logic with exponential backoff
                max_retries = 2
                retry_delay = 1.0

                enhanced_success = False
                
                for attempt in range(max_retries + 1):
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = loop.run_in_executor(
                                executor,
                                lambda: self.xai_client.generate_design_with_reasoning(description, board_size)
                            )
                            # Increased timeout to 600 seconds for very complex designs (100+ components)
                            result = await asyncio.wait_for(future, timeout=600.0)

                            if result.get("success"):
                                design_data = result.get("design", {})
                                enhanced_success = True
                                break  # Success, exit retry loop
                            else:
                                logger.warning(f"   Enhanced xAI client failed (attempt {attempt + 1}): {result.get('error', 'Unknown error')}")
                                if attempt == max_retries:
                                    break  # All retries failed, will fall back to basic client

                    except asyncio.TimeoutError:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            break
                    except Exception as e:
                        if attempt < max_retries:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"   xAI API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            break

                if not enhanced_success:
                    logger.info("   Enhanced xAI client failed, falling back to basic client")

            # Fallback to basic xAI generation (used when enhanced fails or doesn't exist)
            if design_data is None:
                # Check if xAI client is enabled for basic generation
                if hasattr(self.xai_client, 'enabled') and self.xai_client.enabled:
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
                                # Increased timeout to 600 seconds for very complex designs
                                response = await asyncio.wait_for(future, timeout=600.0)
                                break  # Success, exit retry loop
                        except asyncio.TimeoutError as e:
                            if attempt < max_retries:
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"   xAI API timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                break
                        except Exception as e:
                            if attempt < max_retries:
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"   xAI API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}, retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                break

                        if response.get("choices"):
                            content = response["choices"][0]["message"]["content"]
                            design_data = self._parse_design_response(content, board_size)
                            break
                        else:
                            logger.warning("   Basic xAI client failed, falling back to keyword generation")

                # If xAI client is not enabled or failed, use keyword generation
                if design_data is None:
                    logger.info("   Using keyword-based generation")
                    design_data = self._generate_from_keywords(description, board_size)
            
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
        """Generate design from keywords in description. Enhanced for complex designs."""
        description_lower = description.lower()
        
        # Get board dimensions
        board_width = board_size.get('width', 100) if board_size else 100
        board_height = board_size.get('height', 100) if board_size else 100

        # Initialize components and nets
        components = []
        nets = []
        
        # Parse complex structured descriptions
        self._parse_complex_description(description, components, nets, board_width, board_height)

        # If no components found from complex parsing, use basic patterns
        if not components:
            self._parse_basic_patterns(description_lower, components, nets, board_width, board_height)

        # If still no components, create a minimal design
        if not components:
            components.append({
                "name": "U1",
                "package": "SOIC-8",
                "width": 5,
                "height": 4,
                "power": 0.5,
                "x": board_width // 2,
                "y": board_height // 2,
                "angle": 0,
                "placed": True
            })

        return {
            "board": {
                "width": board_width,
                "height": board_height,
                "clearance": board_size.get('clearance', 0.15) if board_size else 0.15,
                "layer_count": 2
            },
            "components": components,
            "nets": nets,
            "modules": []
        }

    def _parse_complex_description(self, description: str, components: list, nets: list, board_width: float, board_height: float):
        """Parse complex structured descriptions with detailed component lists."""
        import re

        logger.info(f"   Parsing complex description ({len(description)} characters)")

        # Limit description length to prevent performance issues
        if len(description) > 10000:
            logger.warning(f"   Description too long ({len(description)} chars), truncating to 10000 chars")
            description = description[:10000]

        # Pre-compile regex patterns for better performance
        component_patterns = {
            'analog_input': (re.compile(r'(\d+)\s*analog\s*inputs?', re.IGNORECASE), 'Analog Input'),
            'digital_io': (re.compile(r'(\d+)\s*digital\s*i/?o\s*pins?', re.IGNORECASE), 'Digital I/O'),
            'high_side_driver': (re.compile(r'(\d+)\s*high.?side\s*driver', re.IGNORECASE), 'High-Side Driver'),
            'push_button': (re.compile(r'(\d+)\s*push\s*buttons?', re.IGNORECASE), 'Push Button'),
            'status_led': (re.compile(r'(\d+)\s*status\s*leds?', re.IGNORECASE), 'Status LED'),
            'relay': (re.compile(r'(\d+)\s*relay', re.IGNORECASE), 'Relay'),
            'mcu': (re.compile(r'mcu|microcontroller', re.IGNORECASE), 'MCU'),
            'ethernet': (re.compile(r'ethernet', re.IGNORECASE), 'Ethernet'),
            'can_bus': (re.compile(r'can\s*bus', re.IGNORECASE), 'CAN Bus'),
            'rs485': (re.compile(r'rs485', re.IGNORECASE), 'RS485'),
            'wifi': (re.compile(r'wifi', re.IGNORECASE), 'WiFi'),
            'bluetooth': (re.compile(r'ble|bluetooth', re.IGNORECASE), 'Bluetooth'),
            'display': (re.compile(r'lcd|tft', re.IGNORECASE), 'Display'),
            'encoder': (re.compile(r'rotary\s*encoder', re.IGNORECASE), 'Encoder'),
            'buzzer': (re.compile(r'buzzer', re.IGNORECASE), 'Buzzer'),
            'microsd': (re.compile(r'microsd|sd\s*card', re.IGNORECASE), 'MicroSD'),
            'eeprom': (re.compile(r'eeprom', re.IGNORECASE), 'EEPROM'),
            'optocoupler': (re.compile(r'optocoupler', re.IGNORECASE), 'Optocoupler'),
            'fuse': (re.compile(r'fuse', re.IGNORECASE), 'Fuse'),
            'tvs': (re.compile(r'tvs|transient', re.IGNORECASE), 'TVS Diode'),
            'spi_flash': (re.compile(r'spi\s*flash', re.IGNORECASE), 'SPI Flash'),
            'sdram': (re.compile(r'sdram', re.IGNORECASE), 'SDRAM'),
            'capacitor': (re.compile(r'capacitor|cap', re.IGNORECASE), 'Capacitor'),
            'resistor': (re.compile(r'resistor', re.IGNORECASE), 'Resistor'),
            'inductor': (re.compile(r'inductor', re.IGNORECASE), 'Inductor'),
            'diode': (re.compile(r'diode', re.IGNORECASE), 'Diode')
        }

        found_components = {}

        # Find all component matches with pre-compiled patterns
        total_matches = 0
        for comp_type, (pattern, display_name) in component_patterns.items():
            matches = pattern.findall(description)
            if matches:
                # For numbered components, sum them up
                count = 0
                for match in matches:
                    if isinstance(match, str) and match.isdigit():
                        count += int(match)
                    elif isinstance(match, tuple) and match and match[0].isdigit():
                        count += int(match[0])

                # For non-numbered components, count occurrences (but cap at reasonable number)
                if count == 0:
                    count = min(len(matches), 20)  # Cap at 20 for non-numbered components

                if count > 0:
                    found_components[comp_type] = min(count, 100)  # Cap total per type at 100
                    total_matches += count

                # Early exit if we have too many components (performance optimization)
                if total_matches > 200:
                    logger.warning(f"   Too many components found ({total_matches}), limiting to prevent performance issues")
                    break

        logger.info(f"   Found {len(found_components)} component types, total estimated: {total_matches}")

        # Generate components based on findings
        if found_components:
            self._generate_components_from_findings(found_components, components, nets, board_width, board_height)
            logger.info(f"   Generated {len(components)} components and {len(nets)} nets")
        else:
            logger.info("   No complex patterns found, will use basic patterns")

    def _generate_components_from_findings(self, found_components: dict, components: list, nets: list, board_width: float, board_height: float):
        """Generate actual component objects from parsed findings."""
        comp_id = 1
        net_id = 1

        # Define component templates
        templates = {
            'mcu': {
                'package': 'LQFP-100', 'width': 14, 'height': 14, 'power': 0.5,
                'description': 'Main Microcontroller'
            },
            'analog_input': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.01,
                'description': 'Analog Input'
            },
            'digital_io': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.01,
                'description': 'Digital I/O'
            },
            'high_side_driver': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.1,
                'description': 'High-Side Driver'
            },
            'push_button': {
                'package': 'BUTTON-6MM', 'width': 6, 'height': 6, 'power': 0.0,
                'description': 'Push Button'
            },
            'status_led': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.02,
                'description': 'Status LED'
            },
            'relay': {
                'package': 'RELAY-SPDT', 'width': 15, 'height': 12, 'power': 0.5,
                'description': 'Relay'
            },
            'ethernet': {
                'package': 'QFN-32', 'width': 6, 'height': 6, 'power': 0.3,
                'description': 'Ethernet PHY'
            },
            'can_bus': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.1,
                'description': 'CAN Transceiver'
            },
            'rs485': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.1,
                'description': 'RS485 Transceiver'
            },
            'wifi': {
                'package': 'QFN-32', 'width': 6, 'height': 6, 'power': 0.5,
                'description': 'WiFi Module'
            },
            'bluetooth': {
                'package': 'QFN-32', 'width': 6, 'height': 6, 'power': 0.3,
                'description': 'Bluetooth Module'
            },
            'display': {
                'package': 'DISPLAY-1.8IN', 'width': 35, 'height': 50, 'power': 0.2,
                'description': 'TFT Display'
            },
            'encoder': {
                'package': 'ENCODER-12MM', 'width': 12, 'height': 12, 'power': 0.01,
                'description': 'Rotary Encoder'
            },
            'buzzer': {
                'package': 'BUZZER-9MM', 'width': 9, 'height': 9, 'power': 0.05,
                'description': 'Buzzer'
            },
            'microsd': {
                'package': 'MICROSD', 'width': 12, 'height': 15, 'power': 0.01,
                'description': 'MicroSD Card Slot'
            },
            'eeprom': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.01,
                'description': 'EEPROM'
            },
            'optocoupler': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.1,
                'description': 'Optocoupler'
            },
            'fuse': {
                'package': 'FUSE-5MM', 'width': 5, 'height': 20, 'power': 0.0,
                'description': 'Fuse'
            },
            'tvs': {
                'package': 'DO-214AC', 'width': 3, 'height': 2, 'power': 0.0,
                'description': 'TVS Diode'
            },
            'spi_flash': {
                'package': 'SOIC-8', 'width': 5, 'height': 4, 'power': 0.05,
                'description': 'SPI Flash'
            },
            'sdram': {
                'package': 'TSOP-54', 'width': 18, 'height': 8, 'power': 0.3,
                'description': 'SDRAM'
            },
            'capacitor': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.0,
                'description': 'Capacitor'
            },
            'resistor': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.0,
                'description': 'Resistor'
            },
            'inductor': {
                'package': '0805', 'width': 2, 'height': 1.25, 'power': 0.0,
                'description': 'Inductor'
            },
            'diode': {
                'package': 'DO-214AC', 'width': 3, 'height': 2, 'power': 0.0,
                'description': 'Diode'
            }
        }

        # Simple board zoning to avoid clustering and improve readability
        # Zones: power (left), processing (center), comms (top-right), io/ui (bottom-right), storage (right)
        zones = {
            "power":  {"x0": 0.05 * board_width, "x1": 0.25 * board_width, "y0": 0.15 * board_height, "y1": 0.85 * board_height},
            "proc":   {"x0": 0.30 * board_width, "x1": 0.55 * board_width, "y0": 0.20 * board_height, "y1": 0.80 * board_height},
            "comms":  {"x0": 0.60 * board_width, "x1": 0.90 * board_width, "y0": 0.55 * board_height, "y1": 0.90 * board_height},
            "io":     {"x0": 0.60 * board_width, "x1": 0.90 * board_width, "y0": 0.10 * board_height, "y1": 0.50 * board_height},
            "store":  {"x0": 0.56 * board_width, "x1": 0.90 * board_width, "y0": 0.40 * board_height, "y1": 0.60 * board_height},
        }

        def pick_zone(comp_type: str) -> dict:
            t = comp_type.lower()
            if any(k in t for k in ["fuse", "tvs", "power", "inductor", "capacitor"]) or "high_side_driver" in t:
                return zones["power"]
            if any(k in t for k in ["mcu", "sdram", "spi_flash"]):
                return zones["proc"]
            if any(k in t for k in ["ethernet", "can_bus", "rs485", "wifi", "bluetooth"]):
                return zones["comms"]
            if any(k in t for k in ["display", "encoder", "buzzer", "push_button", "status_led", "analog_input", "digital_io", "optocoupler"]):
                return zones["io"]
            if any(k in t for k in ["microsd", "eeprom"]):
                return zones["store"]
            return zones["proc"]

        # Generate components with optimized placement and named pins for better KiCad export
        total_components = sum(found_components.values())
        logger.info(f"   Generating {total_components} components...")

        for comp_type, count in found_components.items():
            template = templates.get(comp_type)
            if template:
                # Limit per type and use batch generation for performance
                actual_count = min(count, 30)  # Reduced limit for better performance

                for i in range(actual_count):
                    # Zone-aware grid placement
                    zone = pick_zone(comp_type)
                    grid_cols = min(8, max(1, int((zone["x1"] - zone["x0"]) / max(template['width'], 5))))
                    grid_rows = max(1, (actual_count + grid_cols - 1) // grid_cols)
                    col = i % grid_cols
                    row = i // grid_cols
                    x_spacing = (zone["x1"] - zone["x0"]) / max(grid_cols, 1)
                    y_spacing = (zone["y1"] - zone["y0"]) / max(grid_rows, 1)
                    x = zone["x0"] + (col + 0.5) * x_spacing
                    y = zone["y0"] + (row + 0.5) * y_spacing

                    component = {
                        "name": f"{comp_type.upper()}{comp_id}",
                        "package": template['package'],
                        "width": template['width'],
                        "height": template['height'],
                        "power": template['power'],
                        "x": float(x),
                        "y": float(y),
                        "angle": 0,
                        "placed": True,
                        "component_type": comp_type,
                        # Provide named pins for improved KiCad readability and net attachment
                        "pins": self._default_pins_for(comp_type, template['package'])
                    }
                    components.append(component)
                    comp_id += 1

        # Generate realistic functional nets (rails + buses + IO)
        try:
            self._generate_functional_nets(components, nets)
        except Exception as e:
            logger.warning(f"   Net generation failed, falling back to VCC/GND only: {e}")
            if components:
                vcc_components = [c['name'] for c in components if c.get('power', 0) > 0]
                gnd_components = [c['name'] for c in components]
                if vcc_components:
                    nets.append({"name": "VCC", "pins": [[comp, "VCC"] for comp in vcc_components[:20]]})
                if gnd_components:
                    nets.append({"name": "GND", "pins": [[comp, "GND"] for comp in gnd_components[:50]]})

    def _parse_basic_patterns(self, description_lower: str, components: list, nets: list, board_width: float, board_height: float):
        """Parse basic patterns for simple designs."""
        # Common patterns for simple designs
        if "amplifier" in description_lower or "amp" in description_lower:
            components.extend([
                {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": board_width*0.4, "y": board_height*0.4, "angle": 0, "placed": True},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": board_width*0.6, "y": board_height*0.4, "angle": 0, "placed": True},
                {"name": "C1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": board_width*0.5, "y": board_height*0.5, "angle": 0, "placed": True},
            ])
            nets.append({"name": "VCC", "pins": [["U1", "pin8"]]})
            nets.append({"name": "GND", "pins": [["U1", "pin4"], ["R1", "pin2"], ["C1", "pin2"]]})
        
        if "power" in description_lower or "supply" in description_lower:
            components.extend([
                {"name": "PWR_IC", "package": "SOIC-8", "width": 5, "height": 4, "power": 1.5, "x": board_width*0.3, "y": board_height*0.3, "angle": 0, "placed": True},
                {"name": "L1", "package": "INDUCTOR-10MM", "width": 10, "height": 10, "power": 0.0, "x": board_width*0.5, "y": board_height*0.3, "angle": 0, "placed": True},
                {"name": "C_PWR", "package": "CAP-10MM", "width": 10, "height": 10, "power": 0.0, "x": board_width*0.7, "y": board_height*0.3, "angle": 0, "placed": True},
            ])
            nets.append({"name": "VBAT", "pins": [["PWR_IC", "VIN"]]})
            nets.append({"name": "VCC", "pins": [["PWR_IC", "VOUT"], ["L1", "pin1"]]})
        
        if "sensor" in description_lower:
            components.extend([
                {"name": "SENSOR", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.1, "x": board_width*0.5, "y": board_height*0.5, "angle": 0, "placed": True},
                {"name": "R_SENSE", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": board_width*0.6, "y": board_height*0.5, "angle": 0, "placed": True},
            ])
            nets.append({"name": "SENSOR_OUT", "pins": [["SENSOR", "OUT"], ["R_SENSE", "pin1"]]})
    
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
            comp.setdefault("component_type", "")
            if "pins" not in comp:
                comp["pins"] = self._default_pins_for(comp.get("component_type",""), comp.get("package",""))
            
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
                try:
                    # Prefer Basic library parts in the same package
                    from src.backend.integrations.jlcpcb_parts import JLCPCBPartsManager
                    pm = JLCPCBPartsManager()
                    pkg = comp.get("package","")
                    ctype = comp.get("component_type","")
                    category_hint = None
                    if "resistor" in ctype: category_hint = "Resistors"
                    elif "capacitor" in ctype: category_hint = "Capacitors"
                    elif "inductor" in ctype: category_hint = "Inductors"
                    elif "mcu" in ctype or "ic" in ctype: category_hint = "Integrated Circuits (ICs)"
                    results = pm.search_parts(category=category_hint, package=pkg, library_type="Basic", in_stock=True, limit=1)
                    if not results:
                        results = pm.search_parts(package=pkg, in_stock=True, limit=1)
                    if results:
                        part = results[0]
                        comp["jlcpcb"] = {
                            "lcsc": part.get("lcsc"),
                            "library_type": part.get("library_type"),
                            "stock": part.get("stock", 0),
                            "price": self.component_library._extract_price(part.get("prices", [])) if "prices" in part else 0.0,
                            "description": part.get("description","")
                        }
                        # Map to a concrete KiCad footprint if available
                        fps = pm.map_package_to_footprint(pkg, ctype)
                        if fps:
                            comp["footprint"] = fps[0]
                            comp["footprint_source"] = "jlcpcb_mapping"
                except Exception as e:
                    logger.debug(f"   JLCPCB lookup failed for {comp.get('name')}: {e}")
        
        return design_data

    # ----------------- Helpers for pins and nets -----------------
    def _default_pins_for(self, comp_type: str, package: str) -> List[Dict[str, str]]:
        """
        Return a reasonable set of named pins for a component.
        These names become pad numbers in the generic KiCad footprint → greatly improves readability.
        """
        t = (comp_type or "").lower()
        pkg = (package or "").upper()
        if "mcu" in t:
            # Minimal functional subset instead of 100+ pins
            return [{"name": n} for n in [
                "VDD","VSS","SPI_MOSI","SPI_MISO","SPI_SCK","SPI_CS",
                "I2C_SCL","I2C_SDA","UART_TX","UART_RX",
                "CAN1_TX","CAN1_RX","CAN2_TX","CAN2_RX",
                "ETH_TXP","ETH_TXN","ETH_RXP","ETH_RXN","ETH_REFCLK",
                "GPIO1","GPIO2","GPIO3","GPIO4"
            ]]
        if "ethernet" in t:
            return [{"name": n} for n in ["TXP","TXN","RXP","RXN","REFCLK","VDD","VSS"]]
        if "can_bus" in t:
            return [{"name": n} for n in ["TXD","RXD","VDD","VSS","CANH","CANL"]]
        if "rs485" in t:
            return [{"name": n} for n in ["RO","DI","RE","DE","VDD","VSS","A","B"]]
        if "spi_flash" in t or "eeprom" in t:
            return [{"name": n} for n in ["VDD","VSS","MOSI","MISO","SCK","CS"]]
        if "sdram" in t:
            return [{"name": n} for n in ["VDD","VSS","DQ0","DQ1","DQ2","DQ3","CK","CS","RAS","CAS","WE"]]
        if "display" in t:
            return [{"name": n} for n in ["VDD","VSS","BLK","RST","SCK","MOSI","CS","DC"]]
        if "microsd" in t:
            return [{"name": n} for n in ["VDD","VSS","MOSI","MISO","SCK","CS"]]
        if "encoder" in t:
            return [{"name": n} for n in ["A","B","SW","VDD","VSS"]]
        if "push_button" in t or "status_led" in t or "buzzer" in t or "digital_io" in t or "analog_input" in t or "optocoupler" in t:
            return [{"name": "1"}, {"name": "2"}]
        if "high_side_driver" in t or "relay" in t or "fuse" in t or "tvs" in t or "diode" in t or "inductor" in t or "resistor" in t or "capacitor" in t:
            return [{"name": "1"}, {"name": "2"}]
        # Default to two pins
        return [{"name": "1"}, {"name": "2"}]

    def _generate_functional_nets(self, components: List[Dict], nets: List[Dict]):
        """
        Build realistic nets for common embedded systems based on available components.
        Creates rails (24V, 12V, 5V, 3V3, GND) and functional buses (SPI/I2C/UART/CAN/Ethernet/GPIO).
        """
        name_to_comp = {c["name"]: c for c in components}

        # Group components by rough role using their generated names/types
        def comps_with(prefix: str) -> List[str]:
            return [c["name"] for c in components if c.get("component_type","") == prefix]

        mcu_names = [c["name"] for c in components if c.get("component_type","") == "mcu"]
        can_names = comps_with("can_bus")
        rs485_names = comps_with("rs485")
        eth_names = comps_with("ethernet")
        spi_flash = comps_with("spi_flash")
        sdram = comps_with("sdram")
        eeproms = comps_with("eeprom")
        microsd = comps_with("microsd")
        displays = comps_with("display")
        wifi_bt = comps_with("wifi") + comps_with("bluetooth")
        analog_inputs = comps_with("analog_input")
        digital_ios = comps_with("digital_io")
        buttons = comps_with("push_button")
        leds = comps_with("status_led")
        encoder = comps_with("encoder")
        drivers = comps_with("high_side_driver")
        relays = comps_with("relay")
        fuses = comps_with("fuse")
        tvs = comps_with("tvs")

        # Rails
        rails = {
            "VIN_24V": [],
            "V12": [],
            "V5": [],
            "V3V3": [],
            "GND": [c["name"] for c in components]
        }
        # Basic power assignments
        for n in mcu_names + spi_flash + eeproms + microsd + displays + wifi_bt + eth_names + can_names + rs485_names + sdram:
            rails["V3V3"].append(n)
        for n in drivers + relays:
            rails["V12"].append(n)
        for n in fuses + tvs:
            rails["VIN_24V"].append(n)

        # Convert rails to nets
        for rail, comps in rails.items():
            if comps:
                pin_name = "VSS" if rail == "GND" else ("VIN" if rail == "VIN_24V" else rail)
                nets.append({"name": rail, "pins": [[c, pin_name if pin_name in {p.get('name') for p in name_to_comp[c].get('pins',[])} else "1"] for c in comps]})

        # SPI bus (MCU ↔ flash/microsd/display)
        if mcu_names:
            mcu = mcu_names[0]
            def add_spi_net(name, mcu_pin, devs, dev_pin):
                pins = [[mcu, mcu_pin]]
                for d in devs:
                    pins.append([d, dev_pin])
                if len(pins) > 1:
                    nets.append({"name": name, "pins": pins})
            add_spi_net("SPI_MOSI", "SPI_MOSI", spi_flash + microsd + displays, "MOSI")
            add_spi_net("SPI_MISO", "SPI_MISO", spi_flash + microsd, "MISO")
            add_spi_net("SPI_SCK",  "SPI_SCK",  spi_flash + microsd + displays, "SCK")
            # Individual chip selects
            for idx, dev in enumerate(spi_flash + microsd + displays):
                nets.append({"name": f"SPI_CS_{dev}", "pins": [[mcu, "SPI_CS"], [dev, "CS"]]})

            # I2C bus (MCU ↔ EEPROM + sensors/IO if present)
            i2c_devs = eeproms + analog_inputs + digital_ios
            if i2c_devs:
                nets.append({"name":"I2C_SCL", "pins": [[mcu,"I2C_SCL"]] + [[d,"I2C_SCL"] if any(p.get("name")=="I2C_SCL" for p in name_to_comp[d].get("pins",[])) else [d,"1"] for d in i2c_devs]})
                nets.append({"name":"I2C_SDA", "pins": [[mcu,"I2C_SDA"]] + [[d,"I2C_SDA"] if any(p.get("name")=="I2C_SDA" for p in name_to_comp[d].get("pins",[])) else [d,"2"] for d in i2c_devs]})

            # UART (MCU ↔ RS485)
            if rs485_names:
                dev = rs485_names[0]
                nets.append({"name":"UART_TX", "pins": [[mcu,"UART_TX"], [dev,"DI"]]})
                nets.append({"name":"UART_RX", "pins": [[mcu,"UART_RX"], [dev,"RO"]]})

            # CAN buses
            if len(can_names) >= 1:
                nets.append({"name":"CAN1_TX", "pins": [[mcu,"CAN1_TX"], [can_names[0],"TXD"]]})
                nets.append({"name":"CAN1_RX", "pins": [[mcu,"CAN1_RX"], [can_names[0],"RXD"]]})
            if len(can_names) >= 2:
                nets.append({"name":"CAN2_TX", "pins": [[mcu,"CAN2_TX"], [can_names[1],"TXD"]]})
                nets.append({"name":"CAN2_RX", "pins": [[mcu,"CAN2_RX"], [can_names[1],"RXD"]]})

            # Ethernet (RMII/RGMII simplified)
            if eth_names:
                eth = eth_names[0]
                for pair in [("ETH_TXP","TXP"), ("ETH_TXN","TXN"), ("ETH_RXP","RXP"), ("ETH_RXN","RXN"), ("ETH_REFCLK","REFCLK")]:
                    nets.append({"name": pair[0], "pins": [[mcu, pair[0]], [eth, pair[1]]]})

            # GPIO to buttons/LEDs/encoder
            gpio_targets = buttons + leds + encoder
            for gi, tgt in enumerate(gpio_targets[:8], start=1):
                nets.append({"name": f"GPIO{gi}", "pins": [[mcu, f"GPIO{min(gi,4)}"], [tgt, "1"]]})

            # Memory interfaces
            for mem in sdram:
                for sig in ["DQ0","DQ1","DQ2","DQ3","CK","CS","RAS","CAS","WE"]:
                    nets.append({"name": f"SDRAM_{sig}", "pins": [[mcu, sig], [mem, sig]]})

        # High-side drivers to relays and flyback diodes (simplified)
        for idx, drv in enumerate(drivers):
            # Connect driver output to relay coil pin 1
            if idx < len(relays):
                relay = relays[idx]
                nets.append({"name": f"RELAY_DRV_{idx+1}", "pins": [[drv, "1"], [relay, "1"]]})
                # Coil other end to 12V
                nets.append({"name": f"RELAY_COIL_{idx+1}", "pins": [[relay, "2"], ["V12", "V12"]]})
                # Flyback diode across coil if a diode exists
                if idx < len([c for c in components if c.get("component_type","")=="diode"]):
                    # Find a diode (any)
                    diode_name = next((c["name"] for c in components if c.get("component_type","")=="diode"), None)
                    if diode_name:
                        nets.append({"name": f"RELAY_FLYBACK_{idx+1}_A", "pins": [[diode_name, "1"], [relay, "1"]]})
                        nets.append({"name": f"RELAY_FLYBACK_{idx+1}_K", "pins": [[diode_name, "2"], [relay, "2"]]})

    
    def _generate_fallback_design(self, description: str, board_size: Optional[Dict] = None) -> Dict:
        """Generate fallback design when xAI fails."""
        return self._generate_from_keywords(description, board_size)

