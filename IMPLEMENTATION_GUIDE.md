# Dielectric: Implementation Guide for Production Readiness

## Quick Reference: What Needs to Be Done

### 🔴 Critical (Blocking Production Use)
1. **Component Library Integration** - Fix footprint library access
2. **Routing Integration** - Add auto-routing to optimization pipeline
3. **Manufacturing Constraints** - Integrate DFM validation

### 🟡 High Priority (Needed for Quality)
4. **KiCad Export Enhancement** - Export with proper nets and footprints
5. **DRC Integration** - Design rule checking after optimization
6. **Signal Integrity** - Impedance control and crosstalk analysis

### 🟢 Medium Priority (Nice to Have)
7. **IPC Backend** - Real-time KiCad integration
8. **Multi-Layer Support** - 4+ layer board optimization
9. **3D Thermal Simulation** - Advanced thermal analysis

---

## Part 1: Fixing Component Library Integration

### Current Problem
- 153 libraries discovered, but component placement fails
- Error: "Could not find footprint library"
- Root cause: MCP server doesn't have access to KiCad library paths

### Solution Implementation

#### Step 1: Fix Library Path Detection
**File:** `dielectric/kicad-mcp-server/python/kicad_interface.py`

```python
def detect_kicad_library_paths():
    """Detect KiCad library paths from environment and system"""
    paths = []
    
    # Method 1: Environment variable
    if os.getenv("KICAD_LIBRARY_PATH"):
        paths.append(os.getenv("KICAD_LIBRARY_PATH"))
    
    # Method 2: KiCad config file
    kicad_config = os.path.expanduser("~/.config/kicad/kicad_common")
    if os.path.exists(kicad_config):
        # Parse config for library paths
        with open(kicad_config, 'r') as f:
            for line in f:
                if 'KICAD_SYMBOL_DIR' in line or 'KICAD_FOOTPRINT_DIR' in line:
                    paths.append(parse_path(line))
    
    # Method 3: System defaults
    system_paths = [
        "/usr/share/kicad/footprints",  # Linux
        "~/Library/Application Support/kicad/footprints",  # macOS
        "C:/Program Files/KiCad/share/kicad/footprints",  # Windows
    ]
    for path in system_paths:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            paths.append(expanded)
    
    return paths
```

#### Step 2: Update Component Placement
**File:** `dielectric/kicad-mcp-server/python/commands/place_component.py`

```python
def place_component_with_footprint(board, component_name, footprint_name, x, y, rotation=0):
    """Place component with proper footprint library resolution"""
    # Resolve footprint library path
    library_paths = detect_kicad_library_paths()
    footprint_path = None
    
    for lib_path in library_paths:
        potential_path = os.path.join(lib_path, f"{footprint_name}.kicad_mod")
        if os.path.exists(potential_path):
            footprint_path = potential_path
            break
    
    if not footprint_path:
        # Try to find in KiCad's footprint library database
        footprint_path = search_footprint_database(footprint_name)
    
    if not footprint_path:
        raise ValueError(f"Footprint {footprint_name} not found in any library")
    
    # Load footprint and place component
    footprint = load_footprint(footprint_path)
    component = create_component(component_name, footprint, x, y, rotation)
    board.Add(component)
    
    return component
```

#### Step 3: Test Component Placement
**File:** `dielectric/test_component_placement.py`

```python
import sys
sys.path.append('kicad-mcp-server/python')

from kicad_interface import KiCadInterface
from commands.place_component import place_component_with_footprint

def test_component_placement():
    """Test placing a component with real footprint"""
    kicad = KiCadInterface()
    board = kicad.open_board("test_board.kicad_pcb")
    
    # Test placing a common component
    try:
        component = place_component_with_footprint(
            board, 
            "R1", 
            "Resistor_SMD:R_0805_2012Metric", 
            x=50.0, 
            y=50.0
        )
        print(f"✅ Successfully placed {component.GetReference()}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    test_component_placement()
```

### Testing Checklist
- [ ] Library paths detected correctly
- [ ] Footprint files found and loaded
- [ ] Components placed with correct footprints
- [ ] Components visible in KiCad UI
- [ ] Footprint pads match component package

---

## Part 2: Integrating Routing into Optimization Pipeline

### Current State
- Routing operations tested and working
- Not integrated into optimization workflow
- Manual routing required after placement

### Solution Implementation

#### Step 1: Create RoutingAgent
**File:** `dielectric/src/backend/agents/routing_agent.py`

```python
from typing import Dict, List, Tuple
from src.backend.geometry.placement import Placement
from src.backend.geometry.net import Net

class RoutingAgent:
    """Agent responsible for trace routing"""
    
    def __init__(self, kicad_client=None):
        self.kicad_client = kicad_client
    
    def route_design(self, placement: Placement) -> Placement:
        """Route all nets in the design"""
        routed_placement = placement.copy()
        
        # Group nets by priority
        priority_nets = self._prioritize_nets(placement.nets)
        
        # Route high-priority nets first (power, ground, clocks)
        for net in priority_nets["high"]:
            self._route_net(routed_placement, net)
        
        # Route medium-priority nets (signal nets)
        for net in priority_nets["medium"]:
            self._route_net(routed_placement, net)
        
        # Route low-priority nets (optional connections)
        for net in priority_nets["low"]:
            self._route_net(routed_placement, net)
        
        return routed_placement
    
    def _prioritize_nets(self, nets: List[Net]) -> Dict[str, List[Net]]:
        """Prioritize nets for routing"""
        priority = {"high": [], "medium": [], "low": []}
        
        for net in nets:
            net_name = net.name.lower()
            if any(keyword in net_name for keyword in ["vcc", "vdd", "gnd", "ground", "power"]):
                priority["high"].append(net)
            elif any(keyword in net_name for keyword in ["clk", "clock", "reset"]):
                priority["high"].append(net)
            elif len(net.components) > 5:  # High fanout
                priority["medium"].append(net)
            else:
                priority["low"].append(net)
        
        return priority
    
    def _route_net(self, placement: Placement, net: Net):
        """Route a single net"""
        if len(net.components) < 2:
            return  # No routing needed
        
        # Get component positions
        component_positions = []
        for comp_ref, pad_name in net.components:
            comp = placement.components.get(comp_ref)
            if comp:
                pad_pos = self._get_pad_position(comp, pad_name)
                component_positions.append((comp_ref, pad_pos))
        
        # Calculate routing path (MST-based)
        routing_path = self._calculate_routing_path(component_positions)
        
        # Add traces using KiCad MCP
        for i in range(len(routing_path) - 1):
            start_pos = routing_path[i][1]
            end_pos = routing_path[i+1][1]
            
            if self.kicad_client:
                self.kicad_client.add_trace(
                    net.name,
                    start_pos,
                    end_pos,
                    layer="F.Cu",
                    width=self._calculate_trace_width(net)
                )
    
    def _calculate_routing_path(self, component_positions: List[Tuple]) -> List[Tuple]:
        """Calculate optimal routing path using MST"""
        from scipy.sparse.csgraph import minimum_spanning_tree
        import numpy as np
        
        # Build distance matrix
        n = len(component_positions)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                pos1 = component_positions[i][1]
                pos2 = component_positions[j][1]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        
        # Calculate MST
        mst = minimum_spanning_tree(dist_matrix)
        
        # Extract routing path from MST
        routing_path = []
        # ... MST to path conversion logic ...
        
        return routing_path
    
    def _calculate_trace_width(self, net: Net) -> float:
        """Calculate appropriate trace width based on net type"""
        net_name = net.name.lower()
        
        # Power nets: wider traces
        if any(keyword in net_name for keyword in ["vcc", "vdd", "power"]):
            return 0.5  # mm (20 mil)
        
        # Signal nets: standard width
        return 0.15  # mm (6 mil)
    
    def _get_pad_position(self, component, pad_name: str) -> Tuple[float, float]:
        """Get pad position relative to component center"""
        # This would query the footprint for pad position
        # For now, return component center
        return (component.x, component.y)
```

#### Step 2: Integrate RoutingAgent into Orchestrator
**File:** `dielectric/src/backend/agents/orchestrator.py`

```python
class AgentOrchestrator:
    def __init__(self):
        # ... existing agents ...
        self.routing_agent = RoutingAgent(kicad_client=self.kicad_client)
    
    def optimize_with_routing(self, placement: Placement, intent: str) -> Dict:
        """Complete optimization workflow including routing"""
        # Step 1: Placement optimization (existing)
        optimized_placement = self.optimize_placement(placement, intent)
        
        # Step 2: Routing
        routed_placement = self.routing_agent.route_design(optimized_placement)
        
        # Step 3: Verification
        verification_result = self.verifier_agent.verify(routed_placement)
        
        return {
            "placement": routed_placement,
            "verification": verification_result,
            "routing_stats": {
                "nets_routed": len(routed_placement.nets),
                "total_trace_length": self._calculate_total_trace_length(routed_placement)
            }
        }
```

### Testing Checklist
- [ ] RoutingAgent routes all nets
- [ ] High-priority nets routed first
- [ ] Trace widths calculated correctly
- [ ] Traces visible in KiCad
- [ ] No routing violations

---

## Part 3: Manufacturing Constraints Integration

### Current State
- `FabricationConstraints` class exists in `pcb_fabrication.py`
- Not integrated into optimization pipeline
- No DFM validation

### Solution Implementation

#### Step 1: Enhance VerifierAgent with DFM Checks
**File:** `dielectric/src/backend/agents/verifier_agent.py`

```python
from src.backend.constraints.pcb_fabrication import FabricationConstraints

class VerifierAgent:
    def __init__(self):
        self.fabrication_constraints = FabricationConstraints()
    
    def verify_with_dfm(self, placement: Placement) -> Dict:
        """Verify design with Design for Manufacturing checks"""
        violations = []
        warnings = []
        
        # Check trace widths
        for net in placement.nets:
            trace_width = self._get_net_trace_width(net)
            is_valid, error = self.fabrication_constraints.validate_trace_width(trace_width)
            if not is_valid:
                violations.append({
                    "type": "trace_width",
                    "net": net.name,
                    "width": trace_width,
                    "minimum": self.fabrication_constraints.min_trace_width,
                    "message": error
                })
        
        # Check component spacing
        for comp1, comp2 in self._get_component_pairs(placement):
            distance = self._calculate_distance(comp1, comp2)
            min_clearance = self._calculate_min_clearance(comp1, comp2)
            
            if distance < min_clearance:
                violations.append({
                    "type": "clearance",
                    "component1": comp1.name,
                    "component2": comp2.name,
                    "distance": distance,
                    "minimum": min_clearance,
                    "message": f"Components too close: {distance}mm < {min_clearance}mm"
                })
        
        # Check via sizes
        for via in placement.vias:
            is_valid, error = self.fabrication_constraints.validate_via(via)
            if not is_valid:
                violations.append({
                    "type": "via",
                    "via": via.name,
                    "message": error
                })
        
        # Check board boundaries
        for comp in placement.components.values():
            if not self._is_within_board(comp, placement.board):
                violations.append({
                    "type": "boundary",
                    "component": comp.name,
                    "message": f"Component {comp.name} outside board boundaries"
                })
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "dfm_score": self._calculate_dfm_score(placement, violations)
        }
    
    def _calculate_dfm_score(self, placement: Placement, violations: List) -> float:
        """Calculate DFM score (0.0 to 1.0)"""
        base_score = 1.0
        
        # Deduct points for violations
        violation_penalty = 0.1
        for violation in violations:
            base_score -= violation_penalty
        
        # Bonus for good practices
        if self._has_thermal_vias(placement):
            base_score += 0.05
        
        if self._has_proper_ground_plane(placement):
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))
```

#### Step 2: Add DFM to Optimization Objectives
**File:** `dielectric/src/backend/scoring/scorer.py`

```python
class Scorer:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.fabrication_constraints = FabricationConstraints()
    
    def score_with_dfm(self, placement: Placement) -> float:
        """Score placement including DFM considerations"""
        # Existing scoring (trace length, thermal, clearance)
        base_score = self.score(placement)
        
        # DFM scoring
        dfm_score = self._score_dfm(placement)
        
        # Weighted combination
        dfm_weight = self.weights.get("dfm", 0.2)
        return base_score * (1 - dfm_weight) + dfm_score * dfm_weight
    
    def _score_dfm(self, placement: Placement) -> float:
        """Score based on manufacturing constraints"""
        score = 1.0
        
        # Check trace widths
        for net in placement.nets:
            trace_width = self._get_net_trace_width(net)
            if trace_width < self.fabrication_constraints.min_trace_width:
                score -= 0.1
        
        # Check component spacing
        for comp1, comp2 in self._get_component_pairs(placement):
            distance = self._calculate_distance(comp1, comp2)
            min_clearance = self._calculate_min_clearance(comp1, comp2)
            if distance < min_clearance:
                score -= 0.05
        
        return max(0.0, score)
```

### Testing Checklist
- [ ] Trace width validation works
- [ ] Component spacing validation works
- [ ] Via size validation works
- [ ] DFM score calculated correctly
- [ ] Violations reported clearly

---

## Part 4: Production Workflow Integration

### Complete Production Workflow

#### API Endpoint: `/optimize/production`
**File:** `dielectric/src/backend/api/main.py`

```python
@app.post("/optimize/production")
async def optimize_for_production(request: ProductionOptimizationRequest):
    """Complete production optimization workflow"""
    
    # Step 1: Generate/load design
    if request.design_description:
        placement = design_generator_agent.generate(request.design_description)
    else:
        placement = Placement.from_dict(request.placement)
    
    # Step 2: Optimize placement
    optimized_placement = orchestrator.optimize_placement(
        placement, 
        request.optimization_intent
    )
    
    # Step 3: Route traces
    routed_placement = routing_agent.route_design(optimized_placement)
    
    # Step 4: Verify with DFM
    verification = verifier_agent.verify_with_dfm(routed_placement)
    
    # Step 5: Auto-fix violations
    if not verification["passed"]:
        fixed_placement = error_fixer_agent.fix_violations(
            routed_placement, 
            verification["violations"]
        )
        verification = verifier_agent.verify_with_dfm(fixed_placement)
    
    # Step 6: Export production files
    if verification["passed"]:
        export_files = exporter_agent.export_production_files(
            fixed_placement,
            formats=["kicad_pcb", "gerber", "drill", "bom"]
        )
    else:
        export_files = None
    
    return {
        "placement": fixed_placement.to_dict(),
        "verification": verification,
        "export_files": export_files,
        "production_ready": verification["passed"]
    }
```

### Testing Production Workflow

```python
# Test script: test_production_workflow.py
import requests
import json

def test_production_workflow():
    """Test complete production workflow"""
    
    # Step 1: Generate design
    design_response = requests.post("http://localhost:8000/generate", json={
        "description": "Design a production IoT sensor board",
        "board_size": {"width": 60, "height": 40}
    })
    placement = design_response.json()["placement"]
    
    # Step 2: Optimize for production
    optimization_response = requests.post("http://localhost:8000/optimize/production", json={
        "placement": placement,
        "optimization_intent": "Optimize for manufacturing: ensure all traces meet minimum width, proper component spacing",
        "fabrication_constraints": {
            "min_trace_width": 0.15,
            "min_trace_spacing": 0.15,
            "min_clearance": 0.2
        }
    })
    
    result = optimization_response.json()
    
    # Verify results
    assert result["production_ready"] == True, "Design should pass DFM checks"
    assert result["verification"]["passed"] == True, "Verification should pass"
    assert result["export_files"] is not None, "Export files should be generated"
    
    print("✅ Production workflow test passed!")
    return result

if __name__ == "__main__":
    test_production_workflow()
```

---

## Part 5: Quick Wins for Immediate Impact

### 1. Add Production Readiness Indicator
**File:** `dielectric/frontend/app_dielectric.py`

```python
def display_production_readiness(placement):
    """Display production readiness score"""
    dfm_score = calculate_dfm_score(placement)
    
    st.metric("Production Readiness", f"{dfm_score*100:.1f}%")
    
    if dfm_score >= 0.9:
        st.success("✅ Production Ready")
    elif dfm_score >= 0.7:
        st.warning("⚠️ Needs Minor Fixes")
    else:
        st.error("❌ Needs Significant Fixes")
    
    # Show violations
    violations = get_dfm_violations(placement)
    if violations:
        st.subheader("DFM Violations")
        for violation in violations:
            st.error(f"{violation['type']}: {violation['message']}")
```

### 2. Add One-Click Production Export
```python
def export_production_files_button(placement):
    """One-click export for production"""
    if st.button("📦 Export Production Files", type="primary"):
        with st.spinner("Generating production files..."):
            files = exporter_agent.export_production_files(
                placement,
                formats=["kicad_pcb", "gerber", "drill", "bom", "pick_place"]
            )
            
            # Create download buttons
            for file_type, file_content in files.items():
                st.download_button(
                    label=f"Download {file_type.upper()}",
                    data=file_content,
                    file_name=f"board.{file_type}",
                    mime="application/octet-stream"
                )
```

### 3. Add Manufacturing Cost Estimate
```python
def estimate_manufacturing_cost(placement):
    """Estimate manufacturing cost"""
    # Calculate board area
    board_area = placement.board.width * placement.board.height
    
    # Estimate cost based on area and layer count
    cost_per_cm2 = {
        2: 0.05,  # $0.05 per cm² for 2-layer
        4: 0.10,  # $0.10 per cm² for 4-layer
        6: 0.15,  # $0.15 per cm² for 6-layer
    }
    
    layer_count = placement.board.layer_count or 2
    cost_per_unit = board_area * cost_per_cm2.get(layer_count, 0.10)
    
    # Quantity pricing
    quantities = [10, 50, 100, 500, 1000]
    costs = {}
    for qty in quantities:
        if qty < 100:
            unit_cost = cost_per_unit * 1.5  # Higher cost for small quantities
        else:
            unit_cost = cost_per_unit * 0.8  # Lower cost for large quantities
        costs[qty] = unit_cost * qty
    
    st.subheader("Manufacturing Cost Estimate")
    st.table(pd.DataFrame({
        "Quantity": quantities,
        "Total Cost": [f"${c:.2f}" for c in costs.values()],
        "Unit Cost": [f"${c/q:.2f}" for c, q in zip(costs.values(), quantities)]
    }))
```

---

## Implementation Priority

### Week 1: Critical Fixes
1. ✅ Fix component library integration
2. ✅ Integrate routing agent
3. ✅ Add DFM validation

### Week 2: Production Workflow
4. ✅ Create production optimization endpoint
5. ✅ Add production readiness indicator
6. ✅ Test with real PCB designs

### Week 3: Quality & Polish
7. ✅ Enhance error messages
8. ✅ Add manufacturing cost estimate
9. ✅ Document production workflow

### Week 4: Testing & Validation
10. ✅ Test with 50+ component designs
11. ✅ Validate exported files with manufacturers
12. ✅ Get user feedback

---

## Success Metrics

**Production Readiness:**
- [ ] 90%+ designs pass DFM checks after optimization
- [ ] All exported files open correctly in KiCad
- [ ] Manufacturing cost estimates within 20% of actual
- [ ] Production workflow completes in <5 minutes

**Quality:**
- [ ] Zero routing violations
- [ ] All traces meet minimum width requirements
- [ ] Component spacing meets manufacturing constraints
- [ ] Thermal hotspots identified and resolved

**User Experience:**
- [ ] One-click production export
- [ ] Clear violation messages
- [ ] Production readiness score visible
- [ ] Manufacturing cost estimate available

---

**This implementation guide provides concrete steps to make Dielectric production-ready. Focus on the critical fixes first, then build out the production workflow.**

