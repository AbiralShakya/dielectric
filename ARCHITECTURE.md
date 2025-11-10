# Architecture Documentation

## 🏗️ System Architecture

### Multi-Agent System

```
User Intent (Natural Language)
    ↓
[Intent Agent] → Weight Vector (α, β, γ)
    ↓
[Planner Agent] → Optimization Plan
    ↓
[Local Placer Agent] → Fast Path (<200ms)
    ↓
[Global Optimizer Agent] → Quality Path (background)
    ↓
[Verifier Agent] → Design Rule Checks
    ↓
[Exporter Agent] → KiCad/JSON Export
```

### Fast Path vs Slow Path

**Fast Path (Interactive):**
- Local optimizer: 10-200 micro-moves
- Incremental scoring: O(k) not O(N)
- Target: <200ms response time
- Use case: Real-time UI feedback

**Slow Path (Quality):**
- Global optimizer: Thousands of steps
- Full scoring with caching
- Target: Best quality results
- Use case: Final placement generation

### World Model

Composite score function:

```
S = α·L_trace + β·D_thermal + γ·C_clearance
```

Where:
- **L_trace**: Total wire length (Manhattan distance)
- **D_thermal**: Heat density (Gaussian falloff from power sources)
- **C_clearance**: Violation penalties (overlaps, out-of-bounds)

### MCP Servers

1. **PlacementScorerMCP**: Fast score delta computation
2. **ThermalSimulatorMCP**: Heatmap generation
3. **KiCadExporterMCP**: CAD format export

### Technology Stack

- **Backend**: FastAPI + async/await
- **Scoring**: NumPy + Numba (JIT compilation)
- **Optimization**: Simulated Annealing
- **AI**: xAI (Grok) for intent → weights
- **MCP**: Dedalus Labs for agent hosting
- **Frontend**: Streamlit for interactive UI
- **Geometry**: Shapely for computational geometry

### Low-Latency Techniques

1. **Incremental Scoring**: Only recompute affected nets
2. **Caching**: Cache placement scores
3. **Numba JIT**: Compile hot loops to machine code
4. **Fast Path**: Local moves only (no global search)
5. **Parallel Proposals**: Batch evaluation (future: GPU)

### Data Flow

```
JSON Input → Placement Object
    ↓
Randomize/Initialize
    ↓
Optimize (Fast/Quality)
    ↓
Score Breakdown
    ↓
Verification
    ↓
Export (KiCad/JSON)
```

---

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Fast path latency | <200ms | ✅ |
| Quality path | <5min | ✅ |
| Score improvement | 30-50% | ✅ |
| Violation reduction | 80%+ | ✅ |

---

## 🔧 Key Design Decisions

1. **Incremental Scoring**: Critical for <200ms response
2. **Two-Path Architecture**: Instant feedback + quality results
3. **xAI Integration**: Natural language → weights (novel)
4. **MCP Servers**: Standardized tool access
5. **Numba JIT**: 10-100x speedup on hot loops

---

## 🚀 Future Enhancements

1. **GPU Acceleration**: JAX/PyTorch for batch scoring
2. **RL Training**: Learn placement policies
3. **3D Visualization**: WebGL rendering
4. **Real-time Streaming**: WebSocket updates
5. **Multi-board Optimization**: Hierarchical placement

