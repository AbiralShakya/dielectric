"""
Research Components for Dielectric ML

This module contains research prototypes for:
- Physics-Informed Machine Learning (PIML)
- Geometric Deep Learning
- Multi-Agent Reinforcement Learning (MARL)

Note: These are research prototypes. Production-ready code is in ../dielectric/
"""

try:
    from .neural_em import NeuralEMField, PhysicsInformedLoss, SParameterPredictor, NeuralEMSimulator
    from .signal_integrity_gnn import SignalIntegrityGNN, NetGraph
    from .thermal_neural import ThermalNeuralField, PhysicsInformedThermalLoss
    from .routing_gnn import RoutingGNN, RoutingGraph
    from .differentiable_geometry import (
        soft_voronoi,
        differentiable_mst,
        DifferentiablePlacementOptimizer
    )
    from .marl import (
        PCBDesignEnvironment,
        RLPlacerAgent,
        RLRouterAgent,
        MARLOrchestrator,
        RFDomainAgent
    )
    from .unified_co_optimizer import UnifiedCoOptimizer
    
    __all__ = [
        "NeuralEMField",
        "PhysicsInformedLoss",
        "SParameterPredictor",
        "NeuralEMSimulator",
        "SignalIntegrityGNN",
        "NetGraph",
        "ThermalNeuralField",
        "PhysicsInformedThermalLoss",
        "RoutingGNN",
        "RoutingGraph",
        "soft_voronoi",
        "differentiable_mst",
        "DifferentiablePlacementOptimizer",
        "PCBDesignEnvironment",
        "RLPlacerAgent",
        "RLRouterAgent",
        "MARLOrchestrator",
        "RFDomainAgent",
        "UnifiedCoOptimizer",
    ]
except ImportError as e:
    # Graceful degradation if dependencies are missing
    import warnings
    warnings.warn(f"Some ML components could not be imported: {e}")
    __all__ = []
