# Machine Learning Components for Dielectric

This module implements the cutting-edge research components outlined in the Technical Research Deep Dive document.

## Overview

The ML module contains three main research pillars:

1. **Physics-Informed Machine Learning (PIML)** - Neural field methods for EM simulation
2. **Geometric Deep Learning** - Graph neural networks for routing prediction
3. **Multi-Agent Reinforcement Learning (MARL)** - Collaborative, learning agents

Plus a **Unified Co-Optimization Framework** that combines all three.

## Components

### Physics-Informed ML (`neural_em.py`, `thermal_neural.py`)

- **NeuralEMField**: Neural field for EM field prediction (E/H fields)
- **PhysicsInformedLoss**: Enforces Maxwell's equations via loss terms
- **SParameterPredictor**: Predicts S-parameters from geometry
- **ThermalNeuralField**: Neural field for 3D temperature distribution
- **NeuralEMSimulator**: High-level interface for EM simulation

**Usage:**
```python
from dielectric.src.backend.ml import NeuralEMSimulator, ThermalNeuralField

# EM simulation
em_sim = NeuralEMSimulator()
results = em_sim.simulate(geometry, frequency=1e9)

# Thermal simulation
thermal_field = ThermalNeuralField()
T = thermal_field(x, y, z, component_powers, board_material)
```

### Signal Integrity GNN (`signal_integrity_gnn.py`)

- **NetGraph**: Converts PCB placement to graph representation
- **SignalIntegrityGNN**: Predicts impedance, crosstalk, timing violations

**Usage:**
```python
from dielectric.src.backend.ml import SignalIntegrityGNN

gnn = SignalIntegrityGNN()
results = gnn.predict(placement)
# Returns: {"impedance": ..., "crosstalk": ..., "timing": ...}
```

### Routing GNN (`routing_gnn.py`)

- **RoutingGraph**: Graph representation for routing prediction
- **RoutingGNN**: Predicts routing paths, via locations, layer assignments

**Usage:**
```python
from dielectric.src.backend.ml import RoutingGNN

routing_gnn = RoutingGNN()
routing = routing_gnn.predict_routing(placement)
# Returns: {"paths": ..., "vias": ..., "layer_assignments": ...}
```

### Differentiable Geometry (`differentiable_geometry.py`)

- **soft_voronoi**: Differentiable Voronoi approximation
- **differentiable_mst**: Differentiable MST computation
- **DifferentiablePlacementOptimizer**: Gradient-based placement optimization

**Usage:**
```python
from dielectric.src.backend.ml import DifferentiablePlacementOptimizer

optimizer = DifferentiablePlacementOptimizer(placement, target_metrics)
optimized = optimizer.optimize(num_iterations=1000)
```

### Multi-Agent RL (`marl.py`)

- **PCBDesignEnvironment**: RL environment for PCB design
- **RLPlacerAgent**: RL agent for component placement
- **RLRouterAgent**: RL agent for routing
- **MARLOrchestrator**: Coordinates multiple RL agents
- **RFDomainAgent**: Specialized agent for RF design

**Usage:**
```python
from dielectric.src.backend.ml import MARLOrchestrator

orchestrator = MARLOrchestrator(initial_placement)
optimized = orchestrator.optimize(placement, user_intent, max_episodes=100)
```

### Unified Co-Optimizer (`unified_co_optimizer.py`)

- **UnifiedCoOptimizer**: Combines all three pillars for simultaneous optimization

**Usage:**
```python
from dielectric.src.backend.ml import UnifiedCoOptimizer

optimizer = UnifiedCoOptimizer(device="cuda")
optimized = optimizer.optimize(initial_placement, user_intent, max_iterations=100)
```

## Research Papers

These implementations are based on:

- **Physics-Informed ML**: Raissi et al., "Physics-Informed Neural Networks" (2019)
- **Geometric Deep Learning**: Bronstein et al., "Geometric Deep Learning" (2021)
- **Multi-Agent RL**: Lowe et al., "MADDPG" (2017)

## Status

⚠️ **Research Prototypes**: These are research prototypes implementing cutting-edge ideas. They require:
- Training data (FDTD/FEM simulations, routing examples)
- Model training (neural fields, GNNs, RL agents)
- Integration with existing Dielectric components

## Next Steps

1. **Data Collection**: Gather training data (PCB designs, FDTD simulations)
2. **Model Training**: Train neural fields, GNNs, and RL agents
3. **Integration**: Integrate with existing Dielectric agents and optimizers
4. **Evaluation**: Measure improvements (speedup, accuracy, quality)

See `TECHNICAL_RESEARCH_DEEP_DIVE.md` for detailed implementation roadmap.

