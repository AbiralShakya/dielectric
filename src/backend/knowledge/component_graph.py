"""
Component Knowledge Graph

Represents component relationships, dependencies, and design patterns
for large-scale PCB design with computational geometry reasoning.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class ComponentNode:
    """Node in component knowledge graph."""
    name: str
    package: str
    category: str  # "power", "analog", "digital", "rf", "passive"
    properties: Dict = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # Connected component names
    design_rules: Dict = field(default_factory=dict)  # Spacing, thermal, etc.


@dataclass
class NetEdge:
    """Edge representing a net connection."""
    net_name: str
    components: List[Tuple[str, str]]  # (component_name, pin_name)
    signal_type: str  # "power", "signal", "ground", "clock", "differential"
    constraints: Dict = field(default_factory=dict)  # Impedance, length, etc.


class ComponentKnowledgeGraph:
    """
    Knowledge graph for PCB component relationships.
    
    Uses computational geometry to understand spatial relationships
    and design patterns for large-scale designs.
    """
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.components: Dict[str, ComponentNode] = {}
        self.nets: Dict[str, NetEdge] = {}
        self.modules: Dict[str, List[str]] = {}  # Module name -> component names
        self.hierarchical_levels: Dict[str, int] = {}  # Component -> abstraction level
    
    def add_component(
        self,
        name: str,
        package: str,
        category: str,
        properties: Optional[Dict] = None,
        design_rules: Optional[Dict] = None
    ):
        """Add component to knowledge graph."""
        self.components[name] = ComponentNode(
            name=name,
            package=package,
            category=category,
            properties=properties or {},
            design_rules=design_rules or {}
        )
    
    def add_relationship(self, comp1: str, comp2: str, relationship_type: str = "connected"):
        """Add relationship between components."""
        if comp1 in self.components:
            self.components[comp1].relationships.append(comp2)
        if comp2 in self.components:
            self.components[comp2].relationships.append(comp1)
    
    def add_net(
        self,
        net_name: str,
        components: List[Tuple[str, str]],
        signal_type: str = "signal",
        constraints: Optional[Dict] = None
    ):
        """Add net to knowledge graph."""
        self.nets[net_name] = NetEdge(
            net_name=net_name,
            components=components,
            signal_type=signal_type,
            constraints=constraints or {}
        )
        
        # Update component relationships
        comp_names = [c[0] for c in components]
        for i, (comp1, _) in enumerate(components):
            for comp2, _ in components[i+1:]:
                self.add_relationship(comp1, comp2, "net_connected")
    
    def identify_modules(self, placement) -> Dict[str, List[str]]:
        """
        Identify functional modules using computational geometry.
        
        Groups components by:
        - Spatial proximity (Voronoi clustering)
        - Net connectivity (graph analysis)
        - Functional categories (knowledge graph)
        """
        modules = {}
        
        # Group by category
        category_groups = defaultdict(list)
        for name, comp in self.components.items():
            category_groups[comp.category].append(name)
        
        # Group by spatial proximity (if placement available)
        if placement:
            from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
            analyzer = GeometryAnalyzer()
            geometry_data = analyzer.analyze(placement)
            
            # Use Voronoi clusters
            if "voronoi_clusters" in geometry_data:
                for cluster_id, comp_names in geometry_data["voronoi_clusters"].items():
                    module_name = f"Module_{cluster_id}"
                    modules[module_name] = comp_names
        
        # Group by net connectivity
        net_groups = defaultdict(list)
        for net_name, net_edge in self.nets.items():
            comp_names = [c[0] for c in net_edge.components]
            if len(comp_names) > 1:
                # Find existing group or create new
                group_key = tuple(sorted(comp_names))
                net_groups[group_key].extend(comp_names)
        
        # Merge groups
        for group_key, comp_names in net_groups.items():
            module_name = f"NetModule_{len(modules)}"
            modules[module_name] = list(set(comp_names))
        
        self.modules = modules
        return modules
    
    def get_design_rules(self, component_name: str) -> Dict:
        """Get design rules for a component."""
        if component_name in self.components:
            return self.components[component_name].design_rules
        return {}
    
    def get_placement_hints(self, component_name: str) -> Dict:
        """
        Get placement hints based on knowledge graph.
        
        Returns:
            Dict with hints like:
            - preferred_location: "top_left", "center", etc.
            - spacing_requirements: min distances to other components
            - thermal_considerations: heat dissipation needs
            - routing_priority: high/low priority for routing
        """
        if component_name not in self.components:
            return {}
        
        comp = self.components[component_name]
        hints = {
            "category": comp.category,
            "related_components": comp.relationships,
            "design_rules": comp.design_rules
        }
        
        # Category-based hints
        if comp.category == "power":
            hints["preferred_location"] = "edge"
            hints["thermal_priority"] = "high"
            hints["spacing_requirements"] = {"min_clearance": 2.0}  # mm
        elif comp.category == "analog":
            hints["preferred_location"] = "isolated"
            hints["routing_priority"] = "high"
            hints["spacing_requirements"] = {"min_clearance": 1.5}  # mm
        elif comp.category == "digital":
            hints["preferred_location"] = "center"
            hints["routing_priority"] = "medium"
        elif comp.category == "rf":
            hints["preferred_location"] = "isolated"
            hints["routing_priority"] = "critical"
            hints["spacing_requirements"] = {"min_clearance": 3.0}  # mm
        
        return hints
    
    def to_dict(self) -> Dict:
        """Serialize knowledge graph to dictionary."""
        return {
            "components": {
                name: {
                    "package": comp.package,
                    "category": comp.category,
                    "properties": comp.properties,
                    "relationships": comp.relationships,
                    "design_rules": comp.design_rules
                }
                for name, comp in self.components.items()
            },
            "nets": {
                name: {
                    "components": edge.components,
                    "signal_type": edge.signal_type,
                    "constraints": edge.constraints
                }
                for name, edge in self.nets.items()
            },
            "modules": self.modules
        }
    
    @classmethod
    def from_placement(cls, placement) -> 'ComponentKnowledgeGraph':
        """Build knowledge graph from placement."""
        kg = cls()
        
        # Add components
        for name, comp in placement.components.items():
            # Infer category from package/name
            category = "digital"
            if "power" in name.lower() or "pwr" in name.lower():
                category = "power"
            elif "rf" in name.lower() or "antenna" in name.lower():
                category = "rf"
            elif comp.package in ["0805", "0603", "0402"]:
                category = "passive"
            
            kg.add_component(name, comp.package, category)
        
        # Add nets
        for net_name, net in placement.nets.items():
            components = [(pin[0], pin[1]) for pin in net.pins]
            
            # Infer signal type
            signal_type = "signal"
            if "vcc" in net_name.lower() or "vdd" in net_name.lower() or "power" in net_name.lower():
                signal_type = "power"
            elif "gnd" in net_name.lower() or "ground" in net_name.lower():
                signal_type = "ground"
            elif "clk" in net_name.lower() or "clock" in net_name.lower():
                signal_type = "clock"
            
            kg.add_net(net_name, components, signal_type)
        
        return kg

