# Model Training

**Status:** ⏳ TODO - Pending data collection

This folder will contain training scripts for:
- Neural EM Field training
- Routing GNN training
- MARL agent training

---

## 📋 Training Pipeline

### Phase 1: Neural EM Simulator (Months 1-2)

**Data Requirements:**
- 10,000 PCB geometries
- FDTD/FEM simulation results
- E/H fields, S-parameters

**Training Script:** `train_neural_em.py` (TODO)

**Expected Results:**
- 100x speedup over FDTD
- <5% error on test set

---

### Phase 2: Routing GNN (Months 2-3)

**Data Requirements:**
- 50,000 routing examples
- Successful autorouting results
- Via locations, layer assignments

**Training Script:** `train_routing_gnn.py` (TODO)

**Expected Results:**
- 80% routing accuracy
- 10x faster than autorouter

---

### Phase 3: MARL Agents (Months 3-4)

**Data Requirements:**
- 10,000 optimization episodes
- Before/after placements
- Reward signals

**Training Script:** `train_marl_agents.py` (TODO)

**Expected Results:**
- Agents learn to collaborate
- Cooperation score >0.8

---

## 🚀 Quick Start (When Ready)

```bash
# Train Neural EM Simulator
python train_neural_em.py --data datasets/simulation_data --epochs 1000

# Train Routing GNN
python train_routing_gnn.py --data datasets/routing_data --epochs 500

# Train MARL Agents
python train_marl_agents.py --data datasets/optimization_episodes --episodes 10000
```

---

**Status:** Waiting for data collection to complete.

