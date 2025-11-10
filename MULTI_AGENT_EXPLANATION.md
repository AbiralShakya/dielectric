# Multi-Agent Architecture Explanation

## 🧠 How Neuro-Geometric Placer Uses Multi-Agent Systems

### 6 Specialized Agents Working Together

#### 1. **Intent Agent** (xAI/Grok)
**Purpose:** Converts natural language → optimization weights (α, β, γ)

**How it works:**
- Takes user input: "Optimize for cooling, but keep wires short"
- Uses xAI (Grok) to reason about intent
- Returns weights: α=0.3 (trace), β=0.5 (thermal), γ=0.2 (clearance)
- **Computational Geometry Integration:** Uses board metadata (component count, area) in context

**Code Location:** `backend/agents/intent_agent.py`

---

#### 2. **Planner Agent** (Local Reasoning)
**Purpose:** Generates optimization strategy and annealing schedule

**How it works:**
- Analyzes placement complexity (component count, board size)
- Chooses fast vs quality optimization based on user needs
- Sets temperature schedules for simulated annealing
- **Computational Geometry Integration:** Considers geometric constraints in planning

**Code Location:** `backend/agents/planner_agent.py`

---

#### 3. **Local Placer Agent** (Fast Path)
**Purpose:** Interactive optimization (<200ms response time)

**How it works:**
- Runs incremental simulated annealing on component moves
- Uses incremental scoring (O(k) not O(N))
- Provides real-time feedback to UI
- **Computational Geometry Integration:** Uses Shapely polygons, Manhattan/Euclidean distances

**Code Location:** `backend/agents/local_placer_agent.py`

---

#### 4. **Global Optimizer Agent** (Quality Path)
**Purpose:** Background optimization for best results

**How it works:**
- Runs full simulated annealing with multiple restarts
- Longer time horizon (minutes)
- Produces final optimal placement
- **Computational Geometry Integration:** Full geometric validation and scoring

**Code Location:** `backend/agents/global_optimizer_agent.py`

---

#### 5. **Verifier Agent** (Design Rules)
**Purpose:** Checks design rule compliance

**How it works:**
- Validates board boundaries, overlaps, clearances
- Flags manufacturability issues
- Reports violations and warnings
- **Computational Geometry Integration:** Uses polygon intersections, boundary checks

**Code Location:** `backend/agents/verifier_agent.py`

---

#### 6. **Exporter Agent** (CAD Output)
**Purpose:** Generates KiCad/Altium files

**How it works:**
- Converts placement to CAD format
- Exports JSON + KiCad .kicad_pcb files
- Includes metadata and rationale
- **Computational Geometry Integration:** Transforms coordinate systems, generates footprints

**Code Location:** `backend/agents/exporter_agent.py`

---

## 🔗 Agent Communication & Orchestration

### **Agent Orchestrator** (`backend/agents/orchestrator.py`)
Coordinates all agents in the pipeline:

```
User Intent → Intent Agent → Weights
    ↓
Weights + Placement → Planner Agent → Strategy
    ↓
Strategy + Placement → Local Placer/Global Optimizer → Optimized Placement
    ↓
Optimized Placement → Verifier Agent → Validation
    ↓
Validated Placement → Exporter Agent → CAD Files
```

### **Two-Path Architecture**
- **Fast Path:** Intent → Local Placer (<200ms) - for interactive UI
- **Quality Path:** Intent → Planner → Global Optimizer (minutes) - for best results

---

## 🏗️ MCP Servers (Dedalus Labs Integration)

### **What are MCP Servers?**
MCP (Model Context Protocol) servers expose specialized capabilities as standardized tools that LLMs can call.

### **Our MCP Servers:**

#### 1. **PlacementScorerMCP** (`backend/mcp_servers/placement_scorer.py`)
**Purpose:** Fast score computation for moves

**MCP Integration:**
- Hosted on Dedalus Labs
- Exposes `score_delta()` tool
- Agents can call: "Compute score change for moving R1 from (50,30) to (60,30)"

**Computational Geometry:** Uses incremental scoring with affected net analysis

#### 2. **ThermalSimulatorMCP** (`backend/mcp_servers/thermal_simulator.py`)
**Purpose:** Generate thermal heatmaps

**MCP Integration:**
- Hosted on Dedalus Labs
- Exposes `generate_heatmap()` tool
- Agents can call: "Create thermal heatmap for this placement"

**Computational Geometry:** Gaussian heat convolution on 2D grid

#### 3. **KiCadExporterMCP** (`backend/mcp_servers/kicad_exporter.py`)
**Purpose:** Export to CAD formats

**MCP Integration:**
- Hosted on Dedalus Labs
- Exposes `export_kicad()` tool
- Agents can call: "Convert this placement to KiCad format"

**Computational Geometry:** Coordinate transformation and footprint generation

---

## 🤖 xAI (Grok) Integration

### **How Computational Geometry Data is Passed to xAI**

#### **Intent Agent → xAI**
```python
# Context includes geometric data
context = {
    "num_components": len(placement.components),
    "board_area": placement.board.width * placement.board.height,
    "component_types": [c.package for c in placement.components.values()]
}

# xAI receives this context to make better weight decisions
weights = xai.intent_to_weights(user_intent, context)
```

#### **Geometry Simulation Agent → xAI**
```python
# Full geometric data passed to xAI for reasoning
geometry_json = placement.to_dict()

# xAI analyzes geometry for shadow calculations, wind flow, etc.
shadow_analysis = xai.analyze_shadows(geometry_json, location, height)
```

#### **Example xAI Prompt with Geometry:**
```
You are a computational geometry expert. Analyze this building floorplan:

Geometry Data:
{
  "walls": [{"polygon": [[0,0], [10,0], [10,1], [0,1]], "area": 10}],
  "rooms": [{"centroid": [5,0.5], "area": 10}],
  "board": {"width": 100, "height": 100}
}

Calculate shadow lengths at different times of day...
```

---

## 🔄 Communication Patterns

### **Hot Path (<200ms)**
- Direct function calls between agents
- In-process scoring
- Incremental updates

### **Warm Path (seconds)**
- Async agent communication
- FastAPI endpoints
- JSON-RPC style

### **Cold Path (minutes)**
- Background job queues
- Database persistence
- Email/webhook notifications

### **MCP Pattern**
- Standardized tool calling
- Hosted on Dedalus Labs
- Cross-model compatibility

---

## 🎯 Why This Architecture Wins

### **Technical Innovation:**
1. **Multi-Agent Coordination:** 6 specialized agents vs single monolithic system
2. **MCP Standardization:** Tool-based architecture enables easy extension
3. **Computational Geometry:** Real math (Shapely, NumPy) not just LLM guessing
4. **xAI Reasoning:** Natural language + geometry for intelligent optimization
5. **Low-Latency Design:** Fast path enables interactive UX

### **Competitive Advantage:**
- **JITX:** Code-based, no natural language
- **UpCodes:** Text-only, no geometric reasoning
- **Traditional placers:** Heuristics, no AI guidance

### **Scalability:**
- Agents can be deployed independently
- MCP servers enable distributed computing
- Dedalus Labs provides hosting/scaling

---

## 🚀 How to Use

### **Fast Path (Interactive):**
```python
from backend.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
result = await orchestrator.optimize_fast(placement, "Keep it cool")
# Returns optimized placement in <200ms
```

### **Quality Path (Background):**
```python
result = await orchestrator.optimize_quality(placement, "Minimize traces")
# Returns best possible placement (takes longer)
```

### **MCP Tool Calling:**
```python
# Agents can call MCP servers hosted on Dedalus Labs
scorer = PlacementScorerMCP()
delta = scorer.score_delta(placement_data, move_data)
```

This architecture combines the best of AI reasoning, computational geometry, and scalable multi-agent systems for a truly novel PCB placement solution.
