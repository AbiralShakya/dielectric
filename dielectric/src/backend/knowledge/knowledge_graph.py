"""
Knowledge Graph System for Hierarchical PCB Design

Based on research:
- Hierarchical abstraction for large-scale optimization
- Module-based design decomposition
- Thermal management at multiple abstraction levels
- Knowledge graph representation of component relationships
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component


class ModuleType(Enum):
    """Types of functional modules in PCB design."""
    POWER_SUPPLY = "power_supply"
    SIGNAL_PROCESSING = "signal_processing"
    ANALOG_FRONTEND = "analog_frontend"
    DIGITAL_LOGIC = "digital_logic"
    RF_CIRCUIT = "rf_circuit"
    IO_INTERFACE = "io_interface"
    SENSOR_INTERFACE = "sensor_interface"
    CLOCK_DISTRIBUTION = "clock_distribution"
    UNKNOWN = "unknown"


@dataclass
class Module:
    """Represents a functional module in the PCB design."""
    name: str
    module_type: ModuleType
    components: List[str]  # Component names
    bounds: Dict[str, float]  # Bounding box
    thermal_priority: str = "medium"  # "high", "medium", "low"
    power_dissipation: float = 0.0
    critical_nets: List[str] = field(default_factory=list)
    parent_module: Optional[str] = None  # For hierarchical structure
    child_modules: List[str] = field(default_factory=list)


@dataclass
class ComponentNode:
    """Node in knowledge graph representing a component."""
    name: str
    component_type: str  # "IC", "resistor", "capacitor", etc.
    package: str
    power: float
    thermal_zone: Optional[str] = None
    module: Optional[str] = None
    criticality: str = "normal"  # "critical", "normal", "low"


@dataclass
class NetEdge:
    """Edge in knowledge graph representing a net connection."""
    net_name: str
    source_component: str
    target_component: str
    signal_type: str = "digital"  # "digital", "analog", "power", "ground", "clock"
    criticality: str = "normal"  # "critical", "normal", "low"
    estimated_length: float = 0.0


class KnowledgeGraph:
    """
    Knowledge graph representing PCB design hierarchy and relationships.
    
    Uses hierarchical abstraction for large-scale optimization:
    - Level 0: Individual components
    - Level 1: Functional modules
    - Level 2: System-level blocks
    """
    
    def __init__(self, placement: Placement):
        """Initialize knowledge graph from placement."""
        self.placement = placement
        self.components: Dict[str, ComponentNode] = {}
        self.nets: Dict[str, NetEdge] = {}
        self.modules: Dict[str, Module] = {}
        self.thermal_zones: Dict[str, List[str]] = {}  # Zone name -> component names
        self.hierarchy_levels: Dict[int, List[str]] = {}  # Level -> module names
        
        self._build_graph()
        self._identify_modules()
        self._build_hierarchy()
        self._assign_thermal_zones()
    
    def _build_graph(self):
        """Build component and net graph from placement."""
        # Build component nodes
        for comp_name, comp in self.placement.components.items():
            comp_type = self._classify_component(comp)
            self.components[comp_name] = ComponentNode(
                name=comp_name,
                component_type=comp_type,
                package=comp.package,
                power=comp.power,
                criticality=self._assess_criticality(comp)
            )
        
        # Build net edges
        for net in self.placement.nets:
            net_pins = net.pins
            if len(net_pins) < 2:
                continue
            
            # Create edges between all pairs of connected components
            for i in range(len(net_pins)):
                for j in range(i + 1, len(net_pins)):
                    source = net_pins[i][0] if isinstance(net_pins[i], list) else net_pins[i]
                    target = net_pins[j][0] if isinstance(net_pins[j], list) else net_pins[j]
                    
                    edge_key = f"{source}->{target}"
                    signal_type = self._classify_signal(net.name)
                    
                    self.nets[edge_key] = NetEdge(
                        net_name=net.name,
                        source_component=source,
                        target_component=target,
                        signal_type=signal_type,
                        criticality=self._assess_net_criticality(net.name, signal_type)
                    )
    
    def _classify_component(self, comp: Component) -> str:
        """Classify component type."""
        name_lower = comp.name.lower()
        package_lower = comp.package.lower()
        
        if any(x in name_lower for x in ["u", "ic", "chip", "mcu", "cpu", "fpga"]):
            return "IC"
        elif "r" in name_lower or "resistor" in package_lower:
            return "resistor"
        elif "c" in name_lower or "cap" in package_lower:
            return "capacitor"
        elif "l" in name_lower or "inductor" in package_lower:
            return "inductor"
        elif "conn" in name_lower or "header" in package_lower:
            return "connector"
        elif "crystal" in name_lower or "osc" in name_lower:
            return "oscillator"
        else:
            return "other"
    
    def _assess_criticality(self, comp: Component) -> str:
        """Assess component criticality."""
        if comp.power > 1.0:
            return "critical"
        elif comp.power > 0.1:
            return "normal"
        else:
            return "low"
    
    def _classify_signal(self, net_name: str) -> str:
        """Classify signal type from net name."""
        name_lower = net_name.lower()
        
        if any(x in name_lower for x in ["vcc", "vdd", "power", "vin", "vout"]):
            return "power"
        elif any(x in name_lower for x in ["gnd", "ground", "vss"]):
            return "ground"
        elif any(x in name_lower for x in ["clk", "clock", "osc"]):
            return "clock"
        elif any(x in name_lower for x in ["diff", "differential", "+", "-"]):
            return "analog"
        else:
            return "digital"
    
    def _assess_net_criticality(self, net_name: str, signal_type: str) -> str:
        """Assess net criticality."""
        if signal_type in ["clock", "power"]:
            return "critical"
        elif signal_type == "analog":
            return "critical"
        else:
            return "normal"
    
    def _identify_modules(self):
        """
        Identify functional modules using graph clustering.
        
        Based on:
        - Component connectivity (nets)
        - Component types
        - Power relationships
        - Signal flow
        """
        # Group components by connectivity and type
        visited = set()
        module_id = 0
        
        for comp_name, comp_node in self.components.items():
            if comp_name in visited:
                continue
            
            # Start new module
            module_components = [comp_name]
            visited.add(comp_name)
            
            # Find connected components with similar characteristics
            queue = [comp_name]
            while queue:
                current = queue.pop(0)
                
                # Find neighbors
                neighbors = []
                for edge_key, edge in self.nets.items():
                    if edge.source_component == current:
                        neighbors.append(edge.target_component)
                    elif edge.target_component == current:
                        neighbors.append(edge.source_component)
                
                # Add similar neighbors to module
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    
                    neighbor_node = self.components.get(neighbor)
                    if not neighbor_node:
                        continue
                    
                    # Check if similar (same type or connected by critical net)
                    is_similar = (
                        neighbor_node.component_type == comp_node.component_type or
                        any(e.criticality == "critical" for e in self.nets.values() 
                            if (e.source_component == current and e.target_component == neighbor) or
                               (e.source_component == neighbor and e.target_component == current))
                    )
                    
                    if is_similar:
                        module_components.append(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Create module
            if len(module_components) > 1 or comp_node.power > 0.5:
                module_name = f"Module_{module_id}"
                module_type = self._classify_module_type(module_components)
                
                # Calculate module bounds
                bounds = self._calculate_module_bounds(module_components)
                
                # Calculate power dissipation
                power = sum(self.components[c].power for c in module_components if c in self.components)
                
                # Identify critical nets
                critical_nets = []
                for edge_key, edge in self.nets.items():
                    if edge.source_component in module_components and edge.target_component in module_components:
                        if edge.criticality == "critical":
                            critical_nets.append(edge.net_name)
                
                self.modules[module_name] = Module(
                    name=module_name,
                    module_type=module_type,
                    components=module_components,
                    bounds=bounds,
                    thermal_priority="high" if power > 1.0 else "medium" if power > 0.5 else "low",
                    power_dissipation=power,
                    critical_nets=list(set(critical_nets))
                )
                
                # Assign components to module
                for comp_name in module_components:
                    if comp_name in self.components:
                        self.components[comp_name].module = module_name
                
                module_id += 1
    
    def _classify_module_type(self, components: List[str]) -> ModuleType:
        """Classify module type from components."""
        comp_types = [self.components[c].component_type for c in components if c in self.components]
        comp_names = [c.lower() for c in components]
        
        # Check for power supply indicators
        if any("pwr" in n or "power" in n or "supply" in n for n in comp_names):
            return ModuleType.POWER_SUPPLY
        elif any("sensor" in n for n in comp_names):
            return ModuleType.SENSOR_INTERFACE
        elif any("rf" in n or "antenna" in n for n in comp_names):
            return ModuleType.RF_CIRCUIT
        elif any("clk" in n or "osc" in n or "crystal" in n for n in comp_names):
            return ModuleType.CLOCK_DISTRIBUTION
        elif "IC" in comp_types and len([c for c in comp_types if c == "IC"]) > 1:
            return ModuleType.DIGITAL_LOGIC
        else:
            return ModuleType.UNKNOWN
    
    def _calculate_module_bounds(self, components: List[str]) -> Dict[str, float]:
        """Calculate bounding box for module."""
        if not components:
            return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}
        
        positions = []
        for comp_name in components:
            comp = self.placement.get_component(comp_name)
            if comp:
                positions.append((comp.x, comp.y))
        
        if not positions:
            return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}
        
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        return {
            "x_min": min(x_coords),
            "y_min": min(y_coords),
            "x_max": max(x_coords),
            "y_max": max(y_coords)
        }
    
    def _build_hierarchy(self):
        """Build hierarchical structure of modules."""
        # Level 0: Individual components (already in graph)
        self.hierarchy_levels[0] = list(self.components.keys())
        
        # Level 1: Functional modules
        self.hierarchy_levels[1] = list(self.modules.keys())
        
        # Level 2: System-level blocks (group related modules)
        system_blocks = self._group_modules_into_blocks()
        self.hierarchy_levels[2] = system_blocks
    
    def _group_modules_into_blocks(self) -> List[str]:
        """Group modules into system-level blocks."""
        blocks = []
        
        # Group by module type
        type_groups = {}
        for module_name, module in self.modules.items():
            if module.module_type not in type_groups:
                type_groups[module.module_type] = []
            type_groups[module.module_type].append(module_name)
        
        # Create blocks
        for module_type, module_names in type_groups.items():
            if len(module_names) > 1:
                block_name = f"Block_{module_type.value}"
                blocks.append(block_name)
        
        return blocks
    
    def _assign_thermal_zones(self):
        """
        Assign components to thermal zones based on power and proximity.
        
        Uses Voronoi-based thermal analysis to create zones.
        """
        # Group high-power components into thermal zones
        high_power_components = [
            name for name, node in self.components.items()
            if node.power > 0.5
        ]
        
        if not high_power_components:
            return
        
        # Create zones based on proximity
        zone_id = 0
        assigned = set()
        
        for comp_name in high_power_components:
            if comp_name in assigned:
                continue
            
            zone_name = f"ThermalZone_{zone_id}"
            zone_components = [comp_name]
            assigned.add(comp_name)
            
            comp = self.placement.get_component(comp_name)
            if not comp:
                continue
            
            # Find nearby high-power components
            for other_name in high_power_components:
                if other_name in assigned:
                    continue
                
                other_comp = self.placement.get_component(other_name)
                if not other_comp:
                    continue
                
                distance = np.sqrt((comp.x - other_comp.x)**2 + (comp.y - other_comp.y)**2)
                
                # If within thermal interaction distance (20mm)
                if distance < 20.0:
                    zone_components.append(other_name)
                    assigned.add(other_name)
            
            self.thermal_zones[zone_name] = zone_components
            
            # Assign zone to components
            for c in zone_components:
                if c in self.components:
                    self.components[c].thermal_zone = zone_name
            
            zone_id += 1
    
    def get_optimization_strategy(self) -> Dict:
        """
        Get optimization strategy based on knowledge graph analysis.
        
        Returns hierarchical optimization plan.
        """
        strategy = {
            "level_0": {
                "description": "Component-level optimization",
                "focus": "Individual component placement",
                "components": len(self.components)
            },
            "level_1": {
                "description": "Module-level optimization",
                "focus": "Functional module placement",
                "modules": len(self.modules),
                "thermal_zones": len(self.thermal_zones)
            },
            "level_2": {
                "description": "System-level optimization",
                "focus": "Block-level placement",
                "blocks": len(self.hierarchy_levels.get(2, []))
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if len(self.thermal_zones) > 0:
            strategy["recommendations"].append(
                f"Optimize thermal zones: {len(self.thermal_zones)} zones identified"
            )
        
        high_power_modules = [
            m for m in self.modules.values() if m.power_dissipation > 1.0
        ]
        if high_power_modules:
            strategy["recommendations"].append(
                f"Focus on high-power modules: {len(high_power_modules)} modules"
            )
        
        critical_nets = [
            e for e in self.nets.values() if e.criticality == "critical"
        ]
        if critical_nets:
            strategy["recommendations"].append(
                f"Optimize critical nets: {len(critical_nets)} critical connections"
            )
        
        return strategy
    
    def to_dict(self) -> Dict:
        """Convert knowledge graph to dictionary."""
        return {
            "components": {
                name: {
                    "type": node.component_type,
                    "package": node.package,
                    "power": node.power,
                    "thermal_zone": node.thermal_zone,
                    "module": node.module,
                    "criticality": node.criticality
                }
                for name, node in self.components.items()
            },
            "modules": {
                name: {
                    "type": module.module_type.value,
                    "components": module.components,
                    "bounds": module.bounds,
                    "thermal_priority": module.thermal_priority,
                    "power_dissipation": module.power_dissipation,
                    "critical_nets": module.critical_nets
                }
                for name, module in self.modules.items()
            },
            "thermal_zones": self.thermal_zones,
            "hierarchy_levels": {
                level: modules for level, modules in self.hierarchy_levels.items()
            }
        }

