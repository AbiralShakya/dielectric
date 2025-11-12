# Dielectric ML Research

**Status:** 🔬 Research & Development  
**Purpose:** Cutting-edge ML research components for Dielectric

This folder contains research prototypes for:
- Physics-Informed Machine Learning (PIML)
- Geometric Deep Learning
- Multi-Agent Reinforcement Learning (MARL)

**Note:** These are research prototypes. Production-ready enhancements are in the `dielectric/` folder.

---

## 📁 Structure

```
dielectric_ml_research/
├── research_components/     # ML research implementations
│   ├── neural_em.py         # Neural EM Field
│   ├── signal_integrity_gnn.py
│   ├── thermal_neural.py
│   ├── routing_gnn.py
│   ├── differentiable_geometry.py
│   ├── marl.py
│   └── unified_co_optimizer.py
├── data_collection/         # Data collection scripts
│   ├── collect_dataset.py   # Main collection script
│   └── ...
├── model_training/          # Training scripts (TODO)
│   └── ...
└── docs/                    # Research documentation
    ├── DATASET_AND_VALIDATION.md
    └── ...
```

---

## 🎯 Where to Get Training Data

### 1. Open Source PCB Repositories

**GitHub:**
- `adafruit/Adafruit-PCB-Library` - Adafruit designs
- `sparkfun/SparkFun-KiCad-Libraries` - SparkFun designs
- `raspberrypi/pico-examples` - Raspberry Pi designs
- Search GitHub for `.kicad_pcb` files

**KiCad Library:**
```bash
git clone https://gitlab.com/kicad/libraries/kicad-footprints.git
git clone https://gitlab.com/kicad/libraries/kicad-symbols.git
```

### 2. PCB Manufacturer Galleries

- **JLCPCB Design Gallery** - Thousands of user-submitted designs
- **PCBWay Gallery** - Real manufactured designs
- **OSHPark** - Open source hardware designs

### 3. Synthetic Data Generation

Use `data_collection/collect_dataset.py` to generate synthetic designs:
```bash
python data_collection/collect_dataset.py --synthetic 10000
```

### 4. FDTD/FEM Simulation Data

For physics training:
- Run FDTD/FEM simulations on collected designs
- Extract E/H fields, S-parameters
- Use as ground truth for neural field training

---

## 🚀 Quick Start

### Collect Data

```bash
# Collect from GitHub
python data_collection/collect_dataset.py --github --max-repos 10

# Generate synthetic designs
python data_collection/collect_dataset.py --synthetic 1000

# Validate KiCad exports
python data_collection/collect_dataset.py --validate exports/
```

### Use Research Components

```python
from research_components.neural_em import NeuralEMSimulator
from research_components.routing_gnn import RoutingGNN
from research_components.marl import MARLOrchestrator

# Neural EM simulation
em_sim = NeuralEMSimulator()
results = em_sim.simulate(geometry, frequency=1e9)

# Routing prediction
routing_gnn = RoutingGNN()
routing = routing_gnn.predict_routing(placement)

# Multi-agent RL
orchestrator = MARLOrchestrator(initial_placement)
optimized = orchestrator.optimize(placement, user_intent)
```

---

## 📊 Data Requirements

### Neural EM Simulator
- **10,000+ PCB geometries**
- **FDTD/FEM simulation results** (E/H fields, S-parameters)
- **Frequency range:** 1MHz - 10GHz

### Routing GNN
- **50,000+ routing examples**
- **Successful autorouting results**
- **Via locations, layer assignments**

### MARL Agents
- **10,000+ optimization episodes**
- **Before/after placements**
- **Reward signals** (physics + geometry + manufacturability)

---

## 🔬 Research Status

- ✅ **Components Implemented** - All research components coded
- ⏳ **Data Collection** - In progress
- ⏳ **Model Training** - Pending data
- ⏳ **Integration** - Pending training

---

## 📚 Documentation

- `docs/DATASET_AND_VALIDATION.md` - Dataset collection strategy
- `research_components/README.md` - Component documentation
- `../dielectric/docs/TECHNICAL_RESEARCH_DEEP_DIVE.md` - Technical details

---

**Note:** Production-ready enhancements are in `../dielectric/` folder.

