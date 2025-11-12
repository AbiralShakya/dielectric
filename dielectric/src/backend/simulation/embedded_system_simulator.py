"""
Embedded System Simulator with Hierarchical Reasoning

Simulates embedded systems (MCUs, sensors, actuators) with:
- Multi-agent coordination
- Hierarchical reasoning for system-level optimization
- Real-time constraints
- Power management
- Communication protocols

Integrates with HRM for system-level reasoning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    from backend.geometry.placement import Placement
    from backend.ai.hierarchical_reasoning import HierarchicalReasoningModel
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.ai.hierarchical_reasoning import HierarchicalReasoningModel


class ComponentType(Enum):
    """Types of embedded system components."""
    MCU = "mcu"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "communication"
    POWER = "power"
    MEMORY = "memory"


@dataclass
class EmbeddedComponent:
    """Represents an embedded system component."""
    name: str
    component_type: ComponentType
    power_consumption: float  # Watts
    communication_protocol: str  # I2C, SPI, UART, etc.
    real_time_constraints: bool
    criticality: float  # 0-1, how critical for system operation
    position: Tuple[float, float]


class EmbeddedSystemSimulator:
    """
    Simulates embedded system behavior with hierarchical reasoning.
    
    Uses HRM for:
    - High-level: System architecture, communication topology
    - Low-level: Component placement, signal routing, power distribution
    """
    
    def __init__(self):
        """Initialize embedded system simulator."""
        self.components: Dict[str, EmbeddedComponent] = {}
        self.communication_graph: Dict[str, List[str]] = {}
        self.power_network: Dict[str, float] = {}
        
    def add_component(
        self,
        name: str,
        component_type: ComponentType,
        power_consumption: float,
        communication_protocol: str = "I2C",
        real_time_constraints: bool = False,
        criticality: float = 0.5
    ):
        """Add component to embedded system."""
        self.components[name] = EmbeddedComponent(
            name=name,
            component_type=component_type,
            power_consumption=power_consumption,
            communication_protocol=communication_protocol,
            real_time_constraints=real_time_constraints,
            criticality=criticality,
            position=(0.0, 0.0)
        )
    
    def simulate_system(
        self,
        placement: Placement,
        hrm: Optional[HierarchicalReasoningModel] = None
    ) -> Dict[str, Any]:
        """
        Simulate embedded system with hierarchical reasoning.
        
        Args:
            placement: PCB placement
            hrm: Optional hierarchical reasoning model
            
        Returns:
            Simulation results
        """
        # Map placement components to embedded components
        self._map_components(placement)
        
        # High-level reasoning: System architecture
        if hrm:
            system_architecture = self._reason_about_architecture(hrm, placement)
        else:
            system_architecture = self._analyze_architecture(placement)
        
        # Low-level reasoning: Component interactions
        component_interactions = self._analyze_interactions(placement)
        
        # Simulate power consumption
        power_analysis = self._simulate_power(placement)
        
        # Simulate communication
        communication_analysis = self._simulate_communication(placement)
        
        # Simulate real-time constraints
        real_time_analysis = self._simulate_real_time(placement)
        
        return {
            "system_architecture": system_architecture,
            "component_interactions": component_interactions,
            "power_analysis": power_analysis,
            "communication_analysis": communication_analysis,
            "real_time_analysis": real_time_analysis,
            "overall_health": self._compute_system_health(
                power_analysis,
                communication_analysis,
                real_time_analysis
            )
        }
    
    def _map_components(self, placement: Placement):
        """Map PCB components to embedded system components."""
        for comp_name, comp in placement.components.items():
            # Infer component type from name/package
            comp_lower = comp_name.lower()
            
            if any(keyword in comp_lower for keyword in ["mcu", "cpu", "processor", "stm32", "esp32"]):
                comp_type = ComponentType.MCU
            elif any(keyword in comp_lower for keyword in ["sensor", "adc", "temp", "imu"]):
                comp_type = ComponentType.SENSOR
            elif any(keyword in comp_lower for keyword in ["motor", "actuator", "servo", "relay"]):
                comp_type = ComponentType.ACTUATOR
            elif any(keyword in comp_lower for keyword in ["wifi", "bluetooth", "uart", "spi", "i2c"]):
                comp_type = ComponentType.COMMUNICATION
            elif any(keyword in comp_lower for keyword in ["power", "regulator", "ldo", "buck"]):
                comp_type = ComponentType.POWER
            else:
                comp_type = ComponentType.MCU  # Default
            
            # Estimate power consumption
            power = getattr(comp, 'power', 0.0)
            if power == 0.0:
                # Estimate based on component type
                if comp_type == ComponentType.MCU:
                    power = 0.1  # 100mW typical
                elif comp_type == ComponentType.SENSOR:
                    power = 0.01  # 10mW typical
                elif comp_type == ComponentType.ACTUATOR:
                    power = 0.5  # 500mW typical
                elif comp_type == ComponentType.COMMUNICATION:
                    power = 0.2  # 200mW typical
                elif comp_type == ComponentType.POWER:
                    power = 0.05  # 50mW typical
                else:
                    power = 0.05
            
            # Add to embedded components
            self.components[comp_name] = EmbeddedComponent(
                name=comp_name,
                component_type=comp_type,
                power_consumption=power,
                communication_protocol="I2C",  # Default
                real_time_constraints=(comp_type == ComponentType.ACTUATOR),
                criticality=1.0 if comp_type == ComponentType.MCU else 0.5,
                position=(comp.x, comp.y)
            )
    
    def _reason_about_architecture(
        self,
        hrm: HierarchicalReasoningModel,
        placement: Placement
    ) -> Dict[str, Any]:
        """Use HRM to reason about system architecture."""
        # Get high-level state from HRM
        high_level_state = hrm.high_level_state
        
        architecture = {
            "modules": high_level_state.get("modules", []),
            "strategic_focus": high_level_state.get("strategic_focus", {}),
            "module_strategy": high_level_state.get("module_strategy", {}),
            "reasoning_confidence": hrm.state.confidence
        }
        
        return architecture
    
    def _analyze_architecture(self, placement: Placement) -> Dict[str, Any]:
        """Analyze system architecture without HRM."""
        # Group components by type
        type_groups = {}
        for comp_name, comp in placement.components.items():
            comp_lower = comp_name.lower()
            
            if any(kw in comp_lower for kw in ["mcu", "cpu"]):
                comp_type = "mcu"
            elif any(kw in comp_lower for kw in ["sensor"]):
                comp_type = "sensor"
            elif any(kw in comp_lower for kw in ["power"]):
                comp_type = "power"
            else:
                comp_type = "other"
            
            if comp_type not in type_groups:
                type_groups[comp_type] = []
            type_groups[comp_type].append(comp_name)
        
        return {
            "component_groups": type_groups,
            "total_components": len(placement.components),
            "architecture_type": self._classify_architecture(type_groups)
        }
    
    def _classify_architecture(self, type_groups: Dict[str, List[str]]) -> str:
        """Classify system architecture type."""
        if "mcu" in type_groups and len(type_groups.get("mcu", [])) > 1:
            return "multi-mcu"
        elif "mcu" in type_groups:
            return "single-mcu"
        elif "sensor" in type_groups and len(type_groups.get("sensor", [])) > 5:
            return "sensor-network"
        else:
            return "mixed"
    
    def _analyze_interactions(self, placement: Placement) -> Dict[str, Any]:
        """Analyze component interactions."""
        interactions = {}
        
        for comp_name in placement.components.keys():
            # Find components connected via nets
            connected = []
            for net_name, net in placement.nets.items():
                if any(p[0] == comp_name for p in net.pins):
                    for other_comp, _ in net.pins:
                        if other_comp != comp_name:
                            connected.append(other_comp)
            
            interactions[comp_name] = {
                "connected_components": connected,
                "connection_count": len(connected),
                "criticality": self.components.get(comp_name, EmbeddedComponent(
                    comp_name, ComponentType.MCU, 0.0
                )).criticality
            }
        
        return interactions
    
    def _simulate_power(self, placement: Placement) -> Dict[str, Any]:
        """Simulate power consumption and distribution."""
        total_power = sum(
            self.components.get(name, EmbeddedComponent(name, ComponentType.MCU, 0.0)).power_consumption
            for name in placement.components.keys()
        )
        
        # Analyze power distribution
        power_density = {}
        for comp_name, comp in placement.components.items():
            embedded_comp = self.components.get(comp_name)
            if embedded_comp:
                power_density[comp_name] = embedded_comp.power_consumption
        
        return {
            "total_power_watts": total_power,
            "power_density": power_density,
            "power_efficiency": self._compute_power_efficiency(placement),
            "thermal_risk": "high" if total_power > 2.0 else "medium" if total_power > 1.0 else "low"
        }
    
    def _compute_power_efficiency(self, placement: Placement) -> float:
        """Compute power efficiency score."""
        # Simplified: based on component distribution
        # Better distribution = better efficiency
        from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
        
        analyzer = GeometryAnalyzer(placement)
        geometry_data = analyzer.analyze()
        
        voronoi_var = geometry_data.get('voronoi_variance', 0.0)
        # Lower variance = better distribution = better efficiency
        efficiency = max(0.0, 1.0 - voronoi_var)
        
        return efficiency
    
    def _simulate_communication(self, placement: Placement) -> Dict[str, Any]:
        """Simulate communication protocols and routing."""
        protocols = {}
        communication_load = {}
        
        for comp_name, embedded_comp in self.components.items():
            protocol = embedded_comp.communication_protocol
            if protocol not in protocols:
                protocols[protocol] = []
            protocols[protocol].append(comp_name)
            
            # Estimate communication load
            connections = sum(1 for net in placement.nets.values()
                           if any(p[0] == comp_name for p in net.pins))
            communication_load[comp_name] = connections
        
        return {
            "protocols": protocols,
            "communication_load": communication_load,
            "routing_complexity": sum(communication_load.values()) / len(communication_load) if communication_load else 0.0,
            "bottlenecks": self._identify_communication_bottlenecks(communication_load)
        }
    
    def _identify_communication_bottlenecks(
        self,
        communication_load: Dict[str, int]
    ) -> List[str]:
        """Identify communication bottlenecks."""
        if not communication_load:
            return []
        
        avg_load = sum(communication_load.values()) / len(communication_load)
        bottlenecks = [
            comp for comp, load in communication_load.items()
            if load > avg_load * 1.5
        ]
        
        return bottlenecks
    
    def _simulate_real_time(self, placement: Placement) -> Dict[str, Any]:
        """Simulate real-time constraints."""
        real_time_components = [
            name for name, comp in self.components.items()
            if comp.real_time_constraints
        ]
        
        # Analyze timing constraints
        timing_analysis = {}
        for comp_name in real_time_components:
            comp = placement.get_component(comp_name)
            if comp:
                # Estimate signal propagation delay
                # Based on distance to connected components
                max_distance = 0.0
                for net_name, net in placement.nets.items():
                    if any(p[0] == comp_name for p in net.pins):
                        for other_comp_name, _ in net.pins:
                            if other_comp_name != comp_name:
                                other_comp = placement.get_component(other_comp_name)
                                if other_comp:
                                    distance = np.sqrt(
                                        (comp.x - other_comp.x)**2 +
                                        (comp.y - other_comp.y)**2
                                    )
                                    max_distance = max(max_distance, distance)
                
                # Estimate delay (simplified: 1ns per mm)
                delay_ns = max_distance * 1.0
                timing_analysis[comp_name] = {
                    "max_distance_mm": max_distance,
                    "estimated_delay_ns": delay_ns,
                    "meets_constraints": delay_ns < 100.0  # 100ns typical constraint
                }
        
        return {
            "real_time_components": real_time_components,
            "timing_analysis": timing_analysis,
            "constraint_violations": [
                comp for comp, analysis in timing_analysis.items()
                if not analysis["meets_constraints"]
            ]
        }
    
    def _compute_system_health(
        self,
        power_analysis: Dict,
        communication_analysis: Dict,
        real_time_analysis: Dict
    ) -> Dict[str, Any]:
        """Compute overall system health."""
        health_score = 1.0
        
        # Power health
        if power_analysis.get("thermal_risk") == "high":
            health_score -= 0.3
        elif power_analysis.get("thermal_risk") == "medium":
            health_score -= 0.1
        
        # Communication health
        bottlenecks = communication_analysis.get("bottlenecks", [])
        if len(bottlenecks) > 0:
            health_score -= 0.2 * min(len(bottlenecks) / 5, 1.0)
        
        # Real-time health
        violations = real_time_analysis.get("constraint_violations", [])
        if len(violations) > 0:
            health_score -= 0.3 * min(len(violations) / 3, 1.0)
        
        health_score = max(0.0, health_score)
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "critical",
            "issues": {
                "power": power_analysis.get("thermal_risk"),
                "communication": len(bottlenecks),
                "real_time": len(violations)
            }
        }

