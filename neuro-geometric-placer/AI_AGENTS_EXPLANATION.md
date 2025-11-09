# AI Agents in Dielectric: Complete Explanation

## 🤖 Multi-Agent Architecture Overview

Dielectric uses **6 specialized AI agents** working together to automate PCB design from natural language to production-ready files. Each agent has a specific role and uses computational geometry, xAI reasoning, or optimization algorithms.

---

## 1. IntentAgent (xAI-Powered)

**Purpose**: Converts natural language → optimization weights (α, β, γ) using computational geometry + xAI reasoning

**Technology**: xAI Grok API + Computational Geometry Analysis

**How It Works**:
1. Takes user input: "Optimize for thermal management, but keep traces short"
2. Performs computational geometry analysis (Voronoi, MST, thermal hotspots)
3. Passes geometry data + user intent to xAI Grok
4. xAI reasons over geometric metrics to understand trade-offs
5. Returns optimization weights: α=0.3 (trace), β=0.6 (thermal), γ=0.1 (clearance)

**Key Features**:
- **Computational Geometry Integration**: Analyzes Voronoi variance, MST length, thermal hotspots
- **xAI Reasoning**: Chain-of-thought reasoning over geometric data structures
- **Context-Aware**: Considers board size, component count, design complexity

**Code Location**: `src/backend/agents/intent_agent.py`

**Example**:
```python
# User intent: "Optimize for thermal management"
# Geometry analysis shows:
#   - Voronoi variance: 0.85 (clustered → thermal risk)
#   - Thermal hotspots: 3 (high-power components grouped)
#   - MST length: 45mm (short traces, but thermal risk)

# xAI reasoning:
#   1. High Voronoi variance → components clustered
#   2. 3 thermal hotspots → thermal problems
#   3. User wants thermal optimization
#   4. Decision: Prioritize thermal spreading (β = 0.7)

# Returns: {"alpha": 0.2, "beta": 0.7, "gamma": 0.1}
```

**Why It Matters**:
- Makes PCB design accessible (natural language input)
- Understands geometric relationships (not just keywords)
- Balances conflicting objectives intelligently

---

## 2. DesignGeneratorAgent (xAI-Powered)

**Purpose**: Generates complete PCB designs from natural language descriptions

**Technology**: xAI Grok API + Component Knowledge Base

**How It Works**:
1. Takes description: "Design an audio amplifier with power supply and op-amp"
2. xAI parses description to identify:
   - Components needed (ICs, resistors, capacitors)
   - Component specifications (package types, power)
   - Connections/nets between components
   - Board size requirements
   - Functional modules
3. Generates structured JSON with complete placement data
4. Validates and enhances design with missing details

**Key Features**:
- **Natural Language → Design**: Complete PCB from description
- **Component Selection**: Automatically chooses appropriate components
- **Net Generation**: Creates connections between components
- **Module Identification**: Groups related components

**Code Location**: `src/backend/agents/design_generator_agent.py`

**Example**:
```python
# Input: "Design a multi-module audio amplifier with:
#         - Power supply: 12V to 3.3V converter
#         - Analog section: Op-amp with feedback
#         - Digital section: MCU with crystal"

# Output: Complete placement JSON with:
#   - Board: 100mm × 80mm
#   - Components: 15 components (regulator, op-amp, MCU, passives)
#   - Nets: 8 nets (VCC, GND, signal paths)
#   - Modules: 3 modules (power, analog, digital)
```

**Why It Matters**:
- Eliminates manual component selection
- Creates complete designs from scratch
- Saves hours of manual work

---

## 3. PlannerAgent (Rule-Based + xAI)

**Purpose**: Generates optimization strategy and simulated annealing schedule

**Technology**: Rule-based heuristics + xAI for complex cases

**How It Works**:
1. Analyzes placement complexity (component count, board size)
2. Chooses optimization strategy: "fast" (<200ms) or "quality" (background)
3. Sets temperature schedule for simulated annealing:
   - Initial temperature
   - Final temperature
   - Cooling rate
   - Max iterations
4. Adjusts parameters based on problem size

**Key Features**:
- **Fast Path**: 200 iterations, high cooling rate (<200ms)
- **Quality Path**: 5000 iterations, slow cooling (background)
- **Adaptive**: Adjusts based on component count and board size

**Code Location**: `src/backend/agents/planner_agent.py`

**Example**:
```python
# Fast path (interactive):
{
    "initial_temp": 50.0,
    "final_temp": 0.1,
    "cooling_rate": 0.9,
    "max_iterations": 200,
    "strategy": "local_optimization"
}

# Quality path (background):
{
    "initial_temp": 100.0,
    "final_temp": 0.01,
    "cooling_rate": 0.95,
    "max_iterations": 5000,
    "strategy": "global_optimization"
}
```

**Why It Matters**:
- Enables real-time interactive optimization
- Provides best-quality results when time allows
- Adapts to problem complexity

---

## 4. LocalPlacerAgent (Computational Geometry)

**Purpose**: Fast interactive optimization (<500ms) using simulated annealing

**Technology**: Simulated Annealing + Incremental Scoring + Computational Geometry

**How It Works**:
1. Takes initial placement + optimization weights (from IntentAgent)
2. Runs simulated annealing with incremental scoring:
   - Only recomputes affected nets (O(k) not O(N))
   - Uses Shapely polygons for collision detection
   - Manhattan/Euclidean distance for trace length
   - Gaussian thermal model for hotspots
3. Generates perturbations (component moves/swaps)
4. Accepts/rejects based on Metropolis criterion
5. Returns optimized placement in <500ms

**Key Features**:
- **Incremental Scoring**: O(k) updates (not O(N) full scoring)
- **Deterministic**: Same input = same output (seeded from user intent)
- **Real-time**: <500ms for interactive UI
- **Geometric Validation**: Uses computational geometry for constraints

**Code Location**: `src/backend/agents/local_placer_agent.py`

**Example**:
```python
# Input: Placement + weights (α=0.3, β=0.6, γ=0.1)
# Process:
#   1. Initialize simulated annealing (T=50.0)
#   2. For 200 iterations:
#      - Generate move: Move component U1 from (50,50) to (55,52)
#      - Compute delta score: Only affected nets (O(k))
#      - Accept if better or Metropolis criterion
#   3. Return optimized placement

# Output: Optimized placement with score = 0.85
```

**Why It Matters**:
- Enables real-time interactive optimization
- Fast enough for UI feedback
- Mathematically rigorous (simulated annealing)

---

## 5. GlobalOptimizerAgent (Background Quality)

**Purpose**: Background optimization for best-quality results

**Technology**: Simulated Annealing + Full Scoring + Multiple Restarts

**How It Works**:
1. Takes placement + weights + optimization plan
2. Runs comprehensive simulated annealing:
   - Higher initial temperature (100.0)
   - Lower final temperature (0.01)
   - Slower cooling rate (0.95)
   - More iterations (5000+)
3. Full scoring with caching
4. Multiple restarts for global optimum
5. Returns best-quality placement (may take minutes)

**Key Features**:
- **Quality Focus**: Best results, not speed
- **Global Search**: Explores entire solution space
- **Background Processing**: Doesn't block UI
- **Multiple Restarts**: Finds global optimum

**Code Location**: `src/backend/agents/global_optimizer_agent.py`

**Example**:
```python
# Input: Placement + weights + plan (quality path)
# Process:
#   1. Initialize simulated annealing (T=100.0, max_iter=5000)
#   2. For 5000 iterations:
#      - Full scoring (with caching)
#      - Multiple move types (translation, rotation, swap)
#      - Global search
#   3. Multiple restarts (3-5 times)
#   4. Return best placement found

# Output: Best-quality placement with score = 0.92
```

**Why It Matters**:
- Provides best-quality results when time allows
- Finds global optimum (not just local)
- Runs in background (doesn't block UI)

---

## 6. VerifierAgent (Design Rules)

**Purpose**: Validates optimized placement against design rules

**Technology**: Geometric Collision Detection + Design Rule Checking

**How It Works**:
1. Checks component validity (within board bounds)
2. Checks clearance violations (minimum spacing)
3. Checks overlap detection (collision detection)
4. Validates against fabrication constraints:
   - Trace width limits
   - Via size limits
   - Pad-to-pad clearance
5. Generates violation report

**Key Features**:
- **Geometric Validation**: Uses Shapely for collision detection
- **Design Rules**: Real fabrication constraints
- **Comprehensive**: Checks all design rule categories
- **Detailed Reports**: Lists all violations with locations

**Code Location**: `src/backend/agents/verifier_agent.py`

**Example**:
```python
# Input: Optimized placement
# Process:
#   1. Check bounds: All components within board?
#   2. Check clearance: Minimum spacing met?
#   3. Check overlaps: Any component collisions?
#   4. Check fabrication: Trace width, via size OK?

# Output:
{
    "passed": False,
    "violations": [
        {
            "type": "clearance_violation",
            "components": ["U1", "U2"],
            "distance": 0.3,  # mm (too close)
            "required": 0.5   # mm
        }
    ]
}
```

**Why It Matters**:
- Ensures manufacturability
- Catches errors before export
- Provides detailed feedback

---

## 7. ErrorFixerAgent (Agentic)

**Purpose**: **Automatically fixes design errors** (not just reports them)

**Technology**: Computational Geometry + Geometric Algorithms

**How It Works**:
1. Validates design to find issues
2. Categorizes issues:
   - Design rule violations (clearance, out-of-bounds)
   - Thermal hotspots
   - Signal integrity issues
   - Manufacturability problems
3. **Automatically fixes each issue**:
   - Clearance violations → Move components apart
   - Thermal hotspots → Space high-power components
   - Signal integrity → Optimize net placement
   - Edge clearance → Move away from board edge
4. Iterates until all issues fixed (max 10 iterations)
5. Returns fixed placement + fix report

**Key Features**:
- **Agentic**: Actually fixes issues (not just reports)
- **Automatic**: No human intervention needed
- **Iterative**: Fixes multiple issues in sequence
- **Geometric**: Uses computational geometry for fixes

**Code Location**: `src/backend/agents/error_fixer_agent.py`

**Example**:
```python
# Input: Placement with violations
# Process:
#   1. Detect: Clearance violation between U1 and U2 (0.3mm, need 0.5mm)
#   2. Fix: Calculate direction vector, move components apart
#   3. Detect: Thermal hotspot (3 high-power components clustered)
#   4. Fix: Space components to 15mm minimum
#   5. Validate: All issues fixed?

# Output:
{
    "success": True,
    "fixes_applied": [
        {"type": "clearance_violation", "components": ["U1", "U2"]},
        {"type": "thermal", "hotspots_fixed": 2}
    ],
    "quality_improvement": 0.65 → 0.85
}
```

**Why It Matters**:
- **Zero error rate**: Automatically fixes all issues
- **Saves time**: No manual fixing needed
- **Agentic**: Demonstrates true AI agency (not just analysis)

---

## 8. ExporterAgent (Format Conversion)

**Purpose**: Converts placement to production-ready CAD formats

**Technology**: Format-specific exporters (KiCad, Altium, JSON)

**How It Works**:
1. Takes optimized placement
2. Converts to target format:
   - **KiCad**: PCB file with footprints, nets, layers
   - **Altium**: (Future) Altium Designer format
   - **JSON**: Structured data format
3. Validates export format
4. Returns export file

**Key Features**:
- **Production-Ready**: Generates manufacturable files
- **Multi-Format**: Supports multiple CAD tools
- **Validated**: Ensures export correctness

**Code Location**: `src/backend/agents/exporter_agent.py`

**Example**:
```python
# Input: Optimized placement
# Process:
#   1. Convert components to KiCad footprints
#   2. Generate net connections
#   3. Add board layers
#   4. Format as KiCad PCB file

# Output: KiCad .kicad_pcb file ready for manufacturing
```

**Why It Matters**:
- Enables production use
- Integrates with existing EDA tools
- Complete end-to-end workflow

---

## 🔄 Agent Orchestration

### Workflow: Design Generation

```
User: "Design an audio amplifier"
    ↓
[DesignGeneratorAgent] → Generates complete PCB design
    ↓
[IntentAgent] → Analyzes geometry, generates weights
    ↓
[LocalPlacerAgent] → Optimizes placement
    ↓
[VerifierAgent] → Validates design rules
    ↓
[ErrorFixerAgent] → Automatically fixes issues
    ↓
[ExporterAgent] → Exports to KiCad
    ↓
Production-ready PCB file
```

### Workflow: Optimization

```
User: "Optimize for thermal management"
    ↓
[IntentAgent] → 
    ├─ GeometryAnalyzer: Computes Voronoi, MST, thermal hotspots
    └─ xAI Grok: Reasons over geometry → weights (α, β, γ)
    ↓
[PlannerAgent] → Generates optimization strategy
    ↓
[LocalPlacerAgent] → Fast optimization (<500ms)
    OR
[GlobalOptimizerAgent] → Quality optimization (background)
    ↓
[VerifierAgent] → Validates design rules
    ↓
[ErrorFixerAgent] → Automatically fixes violations
    ↓
Optimized placement
```

---

## 🎯 Why Multi-Agent Architecture?

### 1. **Specialization**
Each agent is an expert in one task:
- IntentAgent: Natural language understanding
- LocalPlacerAgent: Fast optimization
- VerifierAgent: Design rule checking
- ErrorFixerAgent: Automatic fixing

### 2. **Modularity**
- Agents can be improved independently
- Easy to add new agents
- Clear separation of concerns

### 3. **Explainability**
- Each agent's role is clear
- Easy to understand what's happening
- Not a black box

### 4. **Scalability**
- Fast path for interactive use
- Quality path for background processing
- Can parallelize agents

### 5. **Agentic Behavior**
- ErrorFixerAgent actually fixes issues (not just reports)
- Demonstrates true AI agency
- Autonomous problem-solving

---

## 📊 Agent Comparison

| Agent | Technology | Speed | Purpose | AI Type |
|-------|-----------|-------|---------|---------|
| **IntentAgent** | xAI Grok + Geometry | <2s | Natural language → weights | Neural (xAI) |
| **DesignGeneratorAgent** | xAI Grok | <3s | Generate designs from text | Neural (xAI) |
| **PlannerAgent** | Rule-based | <10ms | Optimization strategy | Symbolic |
| **LocalPlacerAgent** | Simulated Annealing | <500ms | Fast optimization | Symbolic |
| **GlobalOptimizerAgent** | Simulated Annealing | Minutes | Quality optimization | Symbolic |
| **VerifierAgent** | Geometric algorithms | <10ms | Design rule checking | Symbolic |
| **ErrorFixerAgent** | Geometric algorithms | <1s | Automatic error fixing | Symbolic |
| **ExporterAgent** | Format conversion | <100ms | Export to CAD formats | Symbolic |

---

## 🔬 Research Foundation

### Multi-Agent Systems
- **Wooldridge (2009)**: "An Introduction to MultiAgent Systems" - Agent coordination
- **Hohpe & Woolf (2003)**: "Enterprise Integration Patterns" - Orchestration patterns

### AI Reasoning
- **Wei et al. (2022)**: Chain-of-thought reasoning improves LLM performance
- **Brown et al. (2020)**: Few-shot learning for domain-specific tasks

### Optimization
- **Kirkpatrick et al. (1983)**: Simulated annealing algorithm
- **PCB Placement**: NP-hard problem, requires heuristic optimization

---

## 🎓 Key Takeaways

1. **6 Specialized Agents**: Each agent is an expert in one task
2. **Hybrid AI**: Combines neural (xAI) and symbolic (geometry) reasoning
3. **Agentic Behavior**: ErrorFixerAgent actually fixes issues (not just reports)
4. **Real-Time + Quality**: Fast path for UI, quality path for background
5. **Complete Workflow**: From natural language to production-ready files

**The Result**: A system that combines specialized AI agents to automate PCB design with explainable, agentic behavior.

