# Training Data Sources

**Where to Get Training Data for Dielectric ML Models**

---

## 🎯 Overview

We need training data for three main components:

1. **Neural EM Simulator** - 10,000+ PCB geometries with FDTD/FEM simulation results
2. **Routing GNN** - 50,000+ routing examples from successful designs
3. **MARL Agents** - 10,000+ optimization episodes with before/after placements

---

## 📦 Source 1: Open Source PCB Repositories

### GitHub Repositories

**High-Quality Sources:**
- `adafruit/Adafruit-PCB-Library` - Professional Adafruit designs
- `sparkfun/SparkFun-KiCad-Libraries` - SparkFun open-source designs
- `raspberrypi/pico-examples` - Raspberry Pi Pico designs
- `kicad/kicad-footprints` - Official KiCad footprint library
- `kicad/kicad-symbols` - Official KiCad symbol library

**Search GitHub:**
```bash
# Search for .kicad_pcb files
gh search code "extension:kicad_pcb" --limit 1000

# Or use GitHub API
curl "https://api.github.com/search/code?q=extension:kicad_pcb" | jq '.items[].html_url'
```

**Collection Script:**
```bash
python collect_dataset.py --github --max-repos 50
```

### KiCad Library

**Official KiCad Libraries:**
```bash
# Clone KiCad libraries
git clone https://gitlab.com/kicad/libraries/kicad-footprints.git
git clone https://gitlab.com/kicad/libraries/kicad-symbols.git

# Extract example boards
find kicad-footprints -name "*.kicad_pcb" -type f
```

---

## 🏭 Source 2: PCB Manufacturer Galleries

### JLCPCB Design Gallery

**Access:**
- Website: https://jlcpcb.com/design-gallery
- Thousands of user-submitted designs
- Real manufacturing data

**Collection Method:**
- Web scraping (with permission)
- API access (if available)
- Manual download

### PCBWay Gallery

**Access:**
- Website: https://www.pcbway.com/project/
- Real manufactured designs
- Various complexity levels

### OSHPark

**Access:**
- Website: https://oshpark.com/
- Open source hardware designs
- High-quality designs

---

## 🎨 Source 3: Synthetic Data Generation

**Generate synthetic designs:**

```bash
python collect_dataset.py --synthetic 10000
```

**Features:**
- Vary complexity (simple, medium, complex)
- Random component placement
- Random net topologies
- Configurable board sizes

**Advantages:**
- Unlimited data
- Controlled complexity
- Known ground truth

---

## 🔬 Source 4: FDTD/FEM Simulation Data

### For Neural EM Simulator

**Required:**
- PCB geometries (from sources above)
- FDTD/FEM simulation results
- E/H fields at sample points
- S-parameters vs. frequency

**Simulation Tools:**
- **ANSYS HFSS** - Full-wave EM simulation
- **CST Studio Suite** - FDTD/FEM simulation
- **OpenEMS** - Open-source FDTD
- **Meep** - MIT FDTD simulator

**Process:**
1. Load PCB geometry
2. Run FDTD/FEM simulation
3. Extract E/H fields
4. Compute S-parameters
5. Store as training data

**Script:**
```python
# TODO: Create FDTD simulation script
def run_fdtd_simulation(geometry, frequencies):
    """
    Run FDTD simulation and extract fields.
    """
    # Use OpenEMS or Meep
    pass
```

---

## 📊 Source 5: Academic Datasets

### Research Papers

**Look for:**
- Papers on PCB design automation
- Papers on routing algorithms
- Papers on EM simulation

**Datasets:**
- Contact authors for datasets
- Check supplementary materials
- Look for GitHub repositories

### Benchmark Suites

**ICCAD Benchmarks:**
- CAD contest benchmarks
- Routing benchmarks
- Placement benchmarks

---

## 🗂️ Dataset Structure

**Recommended structure:**

```
datasets/
├── kicad_designs/          # Real KiCad designs
│   ├── pcb_files/         # .kicad_pcb files
│   ├── metadata.json      # Design metadata
│   └── labels.json        # Labels (routing, physics)
├── synthetic_designs/     # Generated designs
│   ├── simple/            # 5-15 components
│   ├── medium/            # 15-50 components
│   └── complex/           # 50+ components
├── simulation_data/       # FDTD/FEM results
│   ├── em_fields/         # E/H fields
│   ├── s_parameters/      # S-parameters
│   └── thermal/           # Thermal simulation
├── routing_data/          # Routing examples
│   ├── paths/             # Routing paths
│   ├── vias/              # Via locations
│   └── layers/            # Layer assignments
└── training/              # Training splits
    ├── train/             # 80%
    ├── val/               # 10%
    └── test/              # 10%
```

---

## 🏃 Quick Start

### Step 1: Collect Real Designs

```bash
cd data_collection
python collect_dataset.py --github --max-repos 20
```

### Step 2: Generate Synthetic Designs

```bash
python collect_dataset.py --synthetic 5000
```

### Step 3: Label Designs

```python
# TODO: Create labeling script
from research_components import RoutingGNN, NeuralEMSimulator

# Label routing
routing_gnn = RoutingGNN()
routing_labels = routing_gnn.predict_routing(placement)

# Label physics
em_sim = NeuralEMSimulator()
physics_labels = em_sim.simulate(geometry, frequency)
```

### Step 4: Prepare Training Data

```python
# TODO: Create data preparation script
def prepare_training_data(dataset_dir):
    """
    Prepare data for training.
    """
    # Load designs
    # Extract features
    # Create train/val/test splits
    pass
```

---

## 📈 Data Requirements Summary

| Component | Data Type | Quantity | Source |
|-----------|-----------|----------|--------|
| Neural EM | PCB geometries + FDTD results | 10,000+ | GitHub + Synthetic + FDTD |
| Routing GNN | Routing examples | 50,000+ | GitHub + Synthetic |
| MARL Agents | Optimization episodes | 10,000+ | Synthetic + Real designs |

---

## 🔗 Useful Links

- **GitHub PCB Search:** https://github.com/search?q=extension%3Akicad_pcb
- **KiCad Library:** https://gitlab.com/kicad/libraries
- **JLCPCB Gallery:** https://jlcpcb.com/design-gallery
- **OSHPark:** https://oshpark.com/
- **OpenEMS:** https://openems.de/
- **Meep:** https://meep.readthedocs.io/

---

## ⚠️ Important Notes

1. **Respect Licenses** - Check licenses before using designs
2. **Attribution** - Credit original designers
3. **Privacy** - Don't collect proprietary designs
4. **Quality** - Validate collected designs before training

---

**Next Steps:** Run `collect_dataset.py` to start collecting data!

