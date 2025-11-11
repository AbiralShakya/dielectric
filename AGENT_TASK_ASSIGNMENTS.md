# Dielectric: Agent Task Assignments

## Overview

This document assigns specific tasks to each agent in the Dielectric multi-agent system based on the strategic analysis and implementation priorities.

---

## 🔴 Critical Priority Tasks (This Week)

### RoutingAgent (NEW - Must Create)
**Status:** Not yet created - **BLOCKING PRODUCTION USE**

**Tasks:**
1. ✅ **CREATE RoutingAgent class** (`src/backend/agents/routing_agent.py`)
   - Implement MST-based routing path calculation
   - Add net prioritization (power/ground → clocks → signals)
   - Integrate with KiCad MCP for trace placement

2. ✅ **Implement trace routing logic**
   - Calculate optimal routing paths using Minimum Spanning Tree
   - Determine trace widths based on current requirements
   - Handle multi-layer routing

3. ✅ **Integrate into optimization pipeline**
   - Add routing step after placement optimization
   - Route high-priority nets first
   - Validate routing against design rules

**Dependencies:** KiCad MCP server, LocalPlacerAgent output

---

### VerifierAgent (Enhancement)
**Status:** Exists but needs DFM integration

**Tasks:**
1. ✅ **Integrate FabricationConstraints class**
   - Import `src.backend.constraints.pcb_fabrication.FabricationConstraints`
   - Add DFM validation methods
   - Check trace widths, spacing, via sizes

2. ✅ **Add manufacturing constraint checks**
   - Minimum clearance validation
   - Pad-to-pad spacing checks
   - Board boundary validation
   - Via size validation

3. ✅ **Calculate DFM score**
   - Score from 0.0 to 1.0
   - Deduct points for violations
   - Bonus for good practices (thermal vias, ground planes)

**Dependencies:** `pcb_fabrication.py` constraints

---

### ExporterAgent (Enhancement)
**Status:** Basic export works, needs production files

**Tasks:**
1. ✅ **Enhance KiCad export**
   - Include proper net connections
   - Export routing (traces, vias)
   - Validate exported files open correctly

2. ✅ **Add Gerber file generation**
   - Generate all layers (F.Cu, B.Cu, F.SilkS, B.SilkS, F.Mask, B.Mask)
   - Proper file naming convention
   - Validate Gerber files

3. ✅ **Add drill file generation**
   - NC drill file for via and hole drilling
   - Proper drill file format

**Dependencies:** KiCad MCP server, routing data

---

## 🟡 High Priority Tasks (This Month)

### IntentAgent (Enhancement)
**Status:** Working, needs DFM and vertical support

**Tasks:**
1. ✅ **Add DFM weight to optimization priorities**
   - Include manufacturing constraints in weight calculation
   - Balance DFM with other objectives (thermal, routing)

2. ✅ **Enhance geometry data extraction**
   - Add net crossing analysis
   - Add routing complexity metrics
   - Improve thermal hotspot detection

3. ✅ **Add vertical-specific intent understanding**
   - RF terminology (impedance, matching, isolation)
   - Power terminology (current, thermal, EMI)
   - Medical terminology (isolation, safety, compliance)

**Dependencies:** GeometryAnalyzer enhancements

---

### DesignGeneratorAgent (Enhancement)
**Status:** Working, needs library integration

**Tasks:**
1. ✅ **Integrate component library lookup**
   - Use real footprints from KiCad libraries
   - Resolve footprint library paths
   - Place components with correct footprints

2. ✅ **Add multi-layer board support**
   - Generate 4+ layer boards
   - Proper layer stackup configuration
   - Multi-layer routing support

3. ✅ **Enhance component selection**
   - Consider manufacturing availability
   - Integrate JLCPCB parts database
   - Filter by availability and cost

**Dependencies:** KiCad library path detection, JLCPCB API

---

### ErrorFixerAgent (Enhancement)
**Status:** Working, needs DFM violation fixing

**Tasks:**
1. ✅ **Add DFM violation auto-fixing**
   - Fix trace width violations (increase width)
   - Fix spacing violations (move components)
   - Fix via size violations (adjust via size)

2. ✅ **Enhance component repositioning**
   - Intelligent clearance violation fixing
   - Maintain optimization objectives while fixing
   - Avoid creating new violations

**Dependencies:** VerifierAgent DFM checks

---

### LocalPlacerAgent (Enhancement)
**Status:** Working, needs scaling improvements

**Tasks:**
1. ✅ **Optimize for 100+ component designs**
   - Improve incremental scoring performance
   - Add hierarchical optimization support
   - Optimize memory usage

2. ✅ **Integrate DFM constraints into scoring**
   - Add manufacturing constraint penalties
   - Balance DFM with other objectives
   - Weight DFM appropriately

**Dependencies:** VerifierAgent DFM validation

---

## 🟢 Medium Priority Tasks (Next 3 Months)

### PhysicsSimulationAgent (Enhancement)
**Status:** Basic simulation exists

**Tasks:**
1. ✅ **Enhance thermal simulation to 3D**
   - Move from 2D Gaussian to 3D thermal modeling
   - Add thermal via effects
   - Add heat sink modeling

2. ✅ **Add SPICE simulation integration**
   - Circuit analysis before/after optimization
   - Signal integrity simulation
   - Power analysis

3. ✅ **Add signal integrity analysis**
   - Impedance control validation
   - Crosstalk analysis
   - EMI/EMC simulation

**Dependencies:** External simulation tools (SPICE, thermal simulators)

---

### PlannerAgent (Enhancement)
**Status:** Exists but needs workflow planning

**Tasks:**
1. ✅ **Add production workflow planning**
   - Design → Optimize → Route → Validate → Export
   - Coordinate agent execution order
   - Handle workflow dependencies

2. ✅ **Add vertical-specific workflow planning**
   - RF workflow (RFPlacerAgent → ImpedanceAgent → EMIAgent)
   - Power workflow (PowerPlacerAgent → ThermalAgent → SafetyAgent)
   - Medical workflow (SafetyAgent → IsolationAgent → ComplianceAgent)

**Dependencies:** New vertical agents

---

### GlobalOptimizerAgent (Enhancement)
**Status:** Exists but needs scaling

**Tasks:**
1. ✅ **Implement quality path optimization**
   - Background optimization for 100+ components
   - Comprehensive search algorithms
   - Convergence criteria

2. ✅ **Add parallel module optimization**
   - Optimize modules independently
   - Coordinate inter-module optimization
   - Merge module results

**Dependencies:** Hierarchical abstraction

---

### AgentOrchestrator (Enhancement)
**Status:** Working, needs production workflow

**Tasks:**
1. ✅ **Integrate RoutingAgent into workflow**
   - Add routing step after placement
   - Coordinate routing with other agents
   - Handle routing failures

2. ✅ **Create production optimization endpoint**
   - `/optimize/production` API endpoint
   - Complete workflow: Design → Optimize → Route → Validate → Export
   - Production readiness scoring

3. ✅ **Implement agent communication protocol**
   - Message passing between agents
   - Shared state management
   - Error propagation

**Dependencies:** All agents, RoutingAgent

---

## 🔵 New Agents to Create (Expansion)

### ManufacturingAgent (NEW)
**Purpose:** Generate manufacturing files and handle production

**Tasks:**
1. ✅ **CREATE ManufacturingAgent class**
   - Generate Gerber files
   - Generate drill files
   - Generate BOM
   - Generate pick-and-place files

2. ✅ **Add manufacturing cost estimation**
   - Calculate cost based on board area
   - Layer count pricing
   - Quantity discounts

3. ✅ **Integrate with manufacturer APIs**
   - JLCPCB API integration
   - PCBWay API integration
   - Quote generation

**Dependencies:** ExporterAgent, manufacturer APIs

---

### DFMAgent (NEW)
**Purpose:** Specialized Design for Manufacturing validation

**Tasks:**
1. ✅ **CREATE DFMAgent class**
   - Comprehensive DFM checks
   - Manufacturing constraint validation
   - DFM score calculation

2. ✅ **Add manufacturer-specific DFM rules**
   - JLCPCB capabilities
   - PCBWay capabilities
   - Custom manufacturer rules

**Dependencies:** FabricationConstraints, VerifierAgent

---

### RFPlacerAgent (NEW - Vertical)
**Purpose:** RF/high-frequency PCB optimization

**Tasks:**
1. ✅ **CREATE RFPlacerAgent class**
   - RF component placement optimization
   - Controlled impedance routing
   - RF isolation

2. ✅ **Implement RF-specific features**
   - 50Ω impedance control
   - 100Ω differential pairs
   - RF ground plane separation
   - Via fence placement

**Dependencies:** RoutingAgent, ImpedanceAgent (to create)

---

### PowerPlacerAgent (NEW - Vertical)
**Purpose:** Power electronics optimization

**Tasks:**
1. ✅ **CREATE PowerPlacerAgent class**
   - High-current path optimization
   - Thermal management
   - EMI filtering

2. ✅ **Implement power-specific features**
   - High-current trace width calculation
   - Thermal via placement
   - EMI filter placement
   - Safety compliance (creepage, clearance)

**Dependencies:** ThermalAgent, EMCAgent (to create)

---

## Task Dependencies Graph

```
Component Library Fix
    ↓
DesignGeneratorAgent (library integration)
    ↓
LocalPlacerAgent (placement with real footprints)
    ↓
RoutingAgent (CREATE) ← VerifierAgent (DFM)
    ↓
ExporterAgent (production files)
    ↓
ManufacturingAgent (CREATE)
    ↓
Production Ready ✅
```

---

## Priority Matrix

| Agent | Task | Priority | Effort | Impact | Status |
|-------|------|----------|--------|--------|--------|
| **RoutingAgent** | Create agent | 🔴 Critical | High | High | Not Started |
| **VerifierAgent** | DFM integration | 🔴 Critical | Medium | High | Not Started |
| **ExporterAgent** | Production files | 🔴 Critical | Medium | High | Not Started |
| **DesignGeneratorAgent** | Library integration | 🟡 High | Medium | High | Not Started |
| **ErrorFixerAgent** | DFM fixing | 🟡 High | Medium | Medium | Not Started |
| **IntentAgent** | DFM weights | 🟡 High | Low | Medium | Not Started |
| **LocalPlacerAgent** | Scaling | 🟡 High | High | Medium | Not Started |
| **PhysicsSimulationAgent** | 3D thermal | 🟢 Medium | High | Medium | Not Started |
| **ManufacturingAgent** | Create agent | 🟢 Medium | High | High | Not Started |
| **RFPlacerAgent** | Create agent | 🔵 Future | High | Medium | Not Started |
| **PowerPlacerAgent** | Create agent | 🔵 Future | High | Medium | Not Started |

---

## Weekly Sprint Plan

### Week 1: Critical Fixes
- **Monday-Tuesday:** Create RoutingAgent
- **Wednesday:** Integrate DFM into VerifierAgent
- **Thursday:** Enhance ExporterAgent for production files
- **Friday:** Test production workflow

### Week 2: High Priority
- **Monday:** DesignGeneratorAgent library integration
- **Tuesday:** ErrorFixerAgent DFM fixing
- **Wednesday:** IntentAgent DFM weights
- **Thursday:** LocalPlacerAgent scaling improvements
- **Friday:** Integration testing

### Week 3: Medium Priority
- **Monday-Wednesday:** PhysicsSimulationAgent enhancements
- **Thursday-Friday:** ManufacturingAgent creation

### Week 4: Testing & Polish
- **Monday-Wednesday:** End-to-end testing
- **Thursday-Friday:** Documentation and bug fixes

---

## Success Criteria

### Production Readiness (Week 1-2)
- [ ] RoutingAgent routes all nets correctly
- [ ] VerifierAgent validates DFM constraints
- [ ] ExporterAgent generates production files
- [ ] 90%+ designs pass DFM checks

### Quality Improvements (Week 3-4)
- [ ] DesignGeneratorAgent uses real footprints
- [ ] ErrorFixerAgent fixes DFM violations automatically
- [ ] LocalPlacerAgent handles 100+ components efficiently
- [ ] Production workflow completes in <5 minutes

### Expansion (Future)
- [ ] ManufacturingAgent generates all production files
- [ ] RFPlacerAgent optimizes RF designs
- [ ] PowerPlacerAgent optimizes power designs
- [ ] Vertical workflows operational

---

## Agent Communication Protocol

### Message Format
```python
class AgentMessage:
    sender: str  # Agent name
    receiver: str  # Target agent
    message_type: str  # "request", "response", "notification"
    payload: dict  # Message data
    timestamp: float
```

### Example Workflow
```
IntentAgent → LocalPlacerAgent: "Optimize with weights α=0.3, β=0.6, γ=0.1"
LocalPlacerAgent → VerifierAgent: "Verify optimized placement"
VerifierAgent → ErrorFixerAgent: "Fix violations: [list]"
ErrorFixerAgent → RoutingAgent: "Route nets after fixing"
RoutingAgent → ExporterAgent: "Export with routing"
```

---

**Each agent has specific responsibilities. Focus on critical tasks first, then expand to verticals.**

