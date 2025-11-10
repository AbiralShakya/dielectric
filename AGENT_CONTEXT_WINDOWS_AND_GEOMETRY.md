# Agent Context Windows & Computational Geometry Usage

## 🎯 Overview

Each of the 6 agents in Dielectric has specific context windows (data they receive) and uses computational geometry in different ways. This document explains both.

---

## 1. IntentAgent (xAI-Powered)

### **Context Window**

**Input Data:**
```python
{
    "user_intent": str,              # Natural language: "Optimize for thermal management"
    "context": {                     # Board metadata
        "board_width": float,
        "board_height": float,
        "component_count": int,
        "net_count": int
    },
    "geometry_data": {               # Computational geometry analysis
        "density": float,             # components/mm²
        "convex_hull_area": float,    # mm²
        "voronoi_variance": float,    # Distribution uniformity
        "mst_length": float,          # mm (trace length estimate)
        "thermal_hotspots": int,      # High-power regions
        "net_crossings": int,         # Routing conflicts
        "overlap_risk": float,        # Collision probability
        "voronoi_data": {...},        # Detailed Voronoi analysis
        "mst_edges": [...],           # MST edge list
        "hull_vertices": [...],       # Convex hull vertices
        "hotspot_locations": [...]    # Thermal hotspot coordinates
    }
}
```

**xAI Model**: `grok-2-1212`
- **Context Window**: ~128K tokens (xAI Grok-2)
- **Actual Usage**: ~500-1000 tokens per request
- **Temperature**: 0.7 (for reasoning)

### **How It Uses Computational Geometry**

**Step 1: Geometry Analysis**
```python
# From intent_agent.py
analyzer = GeometryAnalyzer(placement)
geometry_data = analyzer.analyze()
# Computes: Voronoi, MST, Convex Hull, Thermal hotspots, Net crossings
```

**Step 2: Format for xAI**
```python
# From xai_client.py
geometry_context = f"""
Computational Geometry Analysis:
- Component density: {geometry_data.get('density', 0):.2f} components/mm²
- Convex hull area: {geometry_data.get('convex_hull_area', 0):.2f} mm²
- Voronoi cell variance: {geometry_data.get('voronoi_variance', 0):.2f}
- Minimum spanning tree length: {geometry_data.get('mst_length', 0):.2f} mm
- Thermal hotspots: {geometry_data.get('thermal_hotspots', 0)} regions
- Net crossing count: {geometry_data.get('net_crossings', 0)}
- Component overlap risk: {geometry_data.get('overlap_risk', 0):.2f}
"""
```

**Step 3: xAI Reasoning**
- xAI receives: User intent + Board context + Geometry data
- xAI reasons: "High Voronoi variance (0.85) → clustered → thermal risk → increase β"
- xAI outputs: Optimization weights (α, β, γ)

**Specific Geometry Usage:**
- **Voronoi variance**: Tells xAI about component distribution (clustered vs. uniform)
- **MST length**: Tells xAI about trace routing efficiency
- **Thermal hotspots**: Tells xAI about thermal problems
- **Net crossings**: Tells xAI about routing complexity

**Output:**
```python
{
    "weights": {"alpha": 0.2, "beta": 0.7, "gamma": 0.1},
    "geometry_data": {...}  # Full geometry analysis passed to next agent
}
```

---

## 2. DesignGeneratorAgent (xAI-Powered)

### **Context Window**

**Input Data:**
```python
{
    "description": str,              # Natural language: "Design audio amplifier"
    "board_size": {                  # Optional board dimensions
        "width": float,
        "height": float
    }
}
```

**xAI Model**: `grok-2-1212`
- **Context Window**: ~128K tokens
- **Actual Usage**: ~1000-2000 tokens per request
- **Temperature**: 0.7

### **How It Uses Computational Geometry**

**Currently**: DesignGeneratorAgent doesn't directly use computational geometry (generates initial design).

**Future Enhancement**: Could use geometry to:
- Validate generated design (check component spacing)
- Optimize initial placement (use Voronoi for distribution)
- Ensure manufacturability (check clearance)

**Output:**
```python
{
    "placement": {
        "board": {...},
        "components": [...],
        "nets": [...]
    }
}
```

---

## 3. PlannerAgent (Rule-Based)

### **Context Window**

**Input Data:**
```python
{
    "placement_info": {
        "num_components": int,
        "board_area": float,
        "board_width": float,
        "board_height": float
    },
    "weights": {
        "alpha": float,
        "beta": float,
        "gamma": float
    },
    "optimization_type": str  # "fast" or "quality"
}
```

**No AI Model**: Rule-based heuristics
- **Context Window**: N/A (local computation)
- **Processing Time**: <10ms

### **How It Uses Computational Geometry**

**Indirect Usage:**
- Considers board area (from geometry) to set iteration count
- Adjusts parameters based on component count (affects geometry complexity)
- Sets optimization strategy based on problem size

**Output:**
```python
{
    "plan": {
        "initial_temp": float,
        "final_temp": float,
        "cooling_rate": float,
        "max_iterations": int,
        "strategy": str
    }
}
```

---

## 4. LocalPlacerAgent (Computational Geometry)

### **Context Window**

**Input Data:**
```python
{
    "placement": Placement,          # Full placement object
    "weights": {
        "alpha": float,               # From IntentAgent
        "beta": float,
        "gamma": float
    },
    "max_time_ms": float,            # Time constraint (default: 200ms)
    "random_seed": int               # For deterministic results
}
```

**No AI Model**: Simulated annealing + geometric algorithms
- **Context Window**: N/A (local computation)
- **Processing Time**: <500ms

### **How It Uses Computational Geometry**

**Step 1: Scoring Function Uses Geometry**

**Trace Length (α weight):**
```python
# From scorer.py
def compute_trace_length(self, placement: Placement) -> float:
    # Uses Manhattan/Euclidean distance (geometric)
    for net in placement.nets:
        pin_positions = [get_pin_position(pin) for pin in net.pins]
        # Geometric distance calculation
        total_length += manhattan_distance(pin1, pin2)
```

**Thermal Density (β weight):**
```python
# From scorer.py
def compute_thermal_density(self, placement: Placement) -> float:
    # Gaussian thermal model (computational geometry)
    for comp in high_power_components:
        for other_comp in components:
            dist = euclidean_distance(comp.position, other_comp.position)
            # Gaussian falloff: e^(-dist² / (2σ²))
            thermal_penalty += comp.power * exp(-(dist**2) / (2 * sigma**2))
```

**Clearance Violations (γ weight):**
```python
# From scorer.py
def compute_clearance_violations(self, placement: Placement) -> float:
    # Geometric collision detection
    for comp1, comp2 in component_pairs:
        dist = euclidean_distance(comp1.center, comp2.center)
        min_clearance = (comp1.width + comp2.width) / 2 + clearance_margin
        if dist < min_clearance:
            violations += penalty
```

**Step 2: Optimization Uses Geometry**
- **Perturbations**: Geometric moves (translation, rotation)
- **Distance calculations**: Euclidean/Manhattan for scoring
- **Collision detection**: Geometric overlap checking

**Specific Geometry Usage:**
- **Euclidean distance**: Component-to-component spacing
- **Manhattan distance**: Trace length estimation
- **Gaussian thermal model**: Heat diffusion calculation
- **Collision detection**: Polygon overlap checking

**Output:**
```python
{
    "placement": Placement,          # Optimized placement
    "score": float,                  # Final score
    "stats": {...}                   # Iterations, acceptances, etc.
}
```

---

## 5. GlobalOptimizerAgent (Computational Geometry)

### **Context Window**

**Input Data:**
```python
{
    "placement": Placement,
    "weights": {
        "alpha": float,
        "beta": float,
        "gamma": float
    },
    "plan": {
        "initial_temp": float,
        "final_temp": float,
        "cooling_rate": float,
        "max_iterations": int
    },
    "timeout": float                 # Optional timeout
}
```

**No AI Model**: Simulated annealing + geometric algorithms
- **Context Window**: N/A (local computation)
- **Processing Time**: Minutes (background)

### **How It Uses Computational Geometry**

**Same as LocalPlacerAgent**, but with:
- More iterations (5000+ vs. 200)
- Slower cooling rate (0.95 vs. 0.9)
- Multiple restarts for global optimum
- Full scoring (not incremental)

**Specific Geometry Usage:**
- **Same geometric calculations**: Euclidean distance, Gaussian thermal, collision detection
- **More thorough search**: Explores more of solution space
- **Better quality**: Finds global optimum, not just local

**Output:**
```python
{
    "placement": Placement,          # Best-quality placement
    "score": float,                  # Best score found
    "stats": {...}                   # Full optimization stats
}
```

---

## 6. VerifierAgent (Geometric Validation)

### **Context Window**

**Input Data:**
```python
{
    "placement": Placement           # Placement to verify
}
```

**No AI Model**: Geometric algorithms only
- **Context Window**: N/A (local computation)
- **Processing Time**: <10ms

### **How It Uses Computational Geometry**

**Step 1: Geometric Distance Calculations**
```python
# From verifier_agent.py
for c1, c2 in component_pairs:
    # Euclidean distance (geometric)
    dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    min_dist = board.clearance + (c1.width + c2.width) / 2
    
    if dist < min_dist * 0.8:
        violations.append({
            "type": "clearance",
            "distance": dist,
            "required": min_dist
        })
```

**Step 2: Thermal Spacing (Geometric)**
```python
# From verifier_agent.py
high_power_components = [c for c in components if c.power > 1.0]
for c1, c2 in high_power_pairs:
    dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    if dist < 20.0:  # Minimum thermal spacing
        warnings.append({
            "type": "thermal",
            "distance": dist,
            "required": 20.0
        })
```

**Step 3: Boundary Checking (Geometric)**
```python
# From placement.py
def check_validity(self):
    for comp in components:
        # Geometric bounds checking
        if comp.x < comp.width/2 or comp.x > board.width - comp.width/2:
            errors.append("Component out of bounds")
        if comp.y < comp.height/2 or comp.y > board.height - comp.height/2:
            errors.append("Component out of bounds")
```

**Specific Geometry Usage:**
- **Euclidean distance**: Clearance checking
- **Geometric bounds**: Board boundary validation
- **Distance thresholds**: Thermal spacing validation

**Output:**
```python
{
    "success": bool,
    "violations": [...],              # Design rule violations
    "warnings": [...]                # Thermal/spacing warnings
}
```

---

## 7. ErrorFixerAgent (Geometric Fixing)

### **Context Window**

**Input Data:**
```python
{
    "placement": Placement,           # Placement with errors
    "max_iterations": int             # Max fix attempts (default: 10)
}
```

**No AI Model**: Geometric algorithms only
- **Context Window**: N/A (local computation)
- **Processing Time**: <1s

### **How It Uses Computational Geometry**

**Step 1: Clearance Violation Fixing (Geometric)**
```python
# From error_fixer_agent.py
def _fix_design_rule(self, placement, violation):
    if violation_type == "clearance_violation":
        comp1, comp2 = violation["components"]
        # Calculate required separation (geometric)
        required_dist = violation["required"]
        current_dist = sqrt((comp1.x - comp2.x)**2 + (comp1.y - comp2.y)**2)
        
        # Calculate direction vector (geometric)
        direction = array([comp2.x - comp1.x, comp2.y - comp1.y])
        direction = direction / norm(direction)  # Normalize
        
        # Move components apart (geometric translation)
        move_distance = (required_dist - current_dist) / 2 + 1.0
        comp1.x -= direction[0] * move_distance
        comp1.y -= direction[1] * move_distance
        comp2.x += direction[0] * move_distance
        comp2.y += direction[1] * move_distance
```

**Step 2: Thermal Hotspot Fixing (Geometric)**
```python
# From error_fixer_agent.py
def _fix_thermal(self, placement, issue):
    high_power = [c for c in components if c.power > 1.0]
    min_spacing = 15.0  # Minimum thermal spacing
    
    for c1, c2 in high_power_pairs:
        dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)  # Euclidean distance
        
        if dist < min_spacing:
            # Calculate direction vector (geometric)
            direction = array([c2.x - c1.x, c2.y - c1.y])
            direction = direction / norm(direction)
            
            # Move apart (geometric translation)
            move_distance = (min_spacing - dist) / 2 + 2.0
            c1.x -= direction[0] * move_distance
            c1.y -= direction[1] * move_distance
            c2.x += direction[0] * move_distance
            c2.y += direction[1] * move_distance
```

**Step 3: Signal Integrity Fixing (Geometric)**
```python
# From error_fixer_agent.py
def _fix_signal_integrity(self, placement, issue):
    net_components = get_components_in_net(net)
    positions = array([[c.x, c.y] for c in net_components])
    
    # Calculate centroid (geometric)
    centroid = mean(positions, axis=0)
    
    # Move components closer to centroid (geometric optimization)
    target_dist = 20.0
    for comp in net_components:
        dist_to_centroid = sqrt((comp.x - centroid[0])**2 + (comp.y - centroid[1])**2)
        if dist_to_centroid > target_dist:
            # Calculate direction to centroid (geometric)
            direction = array([centroid[0] - comp.x, centroid[1] - comp.y])
            direction = direction / norm(direction)
            
            # Move closer (geometric translation)
            move_distance = min(dist_to_centroid - target_dist, 10.0)
            comp.x += direction[0] * move_distance
            comp.y += direction[1] * move_distance
```

**Specific Geometry Usage:**
- **Euclidean distance**: Calculate component spacing
- **Direction vectors**: Calculate movement direction
- **Vector normalization**: Ensure unit direction
- **Geometric translation**: Move components
- **Centroid calculation**: Find optimal net placement

**Output:**
```python
{
    "success": bool,
    "placement": Placement,          # Fixed placement
    "fixes_applied": [...],          # List of fixes
    "iterations": int,               # Number of fix iterations
    "quality_improvement": float    # Quality before → after
}
```

---

## 📊 Summary Table

| Agent | Context Window | AI Model | Geometry Usage |
|-------|---------------|----------|----------------|
| **IntentAgent** | User intent + Board context + **Geometry data** | grok-2-1212 | Receives Voronoi/MST/thermal data, reasons over it |
| **DesignGeneratorAgent** | Description + Board size | grok-2-1212 | None (generates initial design) |
| **PlannerAgent** | Placement info + Weights | None (rule-based) | Indirect (considers board area) |
| **LocalPlacerAgent** | Placement + Weights | None (simulated annealing) | **Euclidean distance, Gaussian thermal, collision detection** |
| **GlobalOptimizerAgent** | Placement + Weights + Plan | None (simulated annealing) | **Same as LocalPlacerAgent (more thorough)** |
| **VerifierAgent** | Placement | None (geometric validation) | **Euclidean distance, boundary checking** |
| **ErrorFixerAgent** | Placement + Issues | None (geometric fixing) | **Direction vectors, geometric translation, centroid calculation** |

---

## 🔬 Detailed Geometry Usage by Agent

### **IntentAgent: Geometry → xAI Reasoning**

**Geometry Data Structure:**
```python
geometry_data = {
    "density": 0.05,                 # Voronoi: Component density
    "voronoi_variance": 0.85,          # Voronoi: Distribution uniformity
    "mst_length": 125.5,               # MST: Trace length estimate
    "thermal_hotspots": 3,             # Gaussian: Hotspot count
    "net_crossings": 5,                # Geometric: Routing conflicts
    "convex_hull_area": 4500.0,        # Convex Hull: Board utilization
    "overlap_risk": 0.15               # Geometric: Collision probability
}
```

**xAI Prompt Context:**
```
Computational Geometry Analysis:
- Component density: 0.05 components/mm²
- Voronoi cell variance: 0.85 (high = clustered)
- Minimum spanning tree length: 125.5 mm
- Thermal hotspots: 3 regions
- Net crossing count: 5
- Component overlap risk: 0.15
```

**xAI Reasoning:**
- "High Voronoi variance (0.85) → components clustered → thermal risk"
- "3 thermal hotspots → prioritize thermal optimization"
- "MST length 125.5mm → moderate trace length"
- **Decision**: β = 0.7 (prioritize thermal), α = 0.2, γ = 0.1

---

### **LocalPlacerAgent: Geometry in Scoring**

**Scoring Function Uses Geometry:**

**1. Trace Length (α weight):**
```python
# Manhattan/Euclidean distance (geometric)
for net in nets:
    for pin1, pin2 in net_pairs:
        length += manhattan_distance(pin1.position, pin2.position)
        # OR
        length += euclidean_distance(pin1.position, pin2.position)
```

**2. Thermal Density (β weight):**
```python
# Gaussian thermal model (computational geometry)
for comp in components:
    if comp.power > threshold:
        for other_comp in components:
            dist = euclidean_distance(comp.center, other_comp.center)
            # Gaussian falloff: e^(-dist² / (2σ²))
            thermal_penalty += comp.power * exp(-(dist**2) / (2 * sigma**2))
```

**3. Clearance Violations (γ weight):**
```python
# Geometric collision detection
for comp1, comp2 in component_pairs:
    dist = euclidean_distance(comp1.center, comp2.center)
    min_clearance = (comp1.width + comp2.width) / 2 + clearance_margin
    if dist < min_clearance:
        violations += penalty
```

**Optimization Uses Geometry:**
- **Perturbations**: Geometric moves (translate component by Δx, Δy)
- **Delta scoring**: Only recompute affected nets (geometric locality)
- **Acceptance**: Metropolis criterion based on geometric score

---

### **VerifierAgent: Geometric Validation**

**Clearance Checking:**
```python
# Euclidean distance (geometric)
dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
min_dist = board.clearance + (c1.width + c2.width) / 2
if dist < min_dist:
    violation = True
```

**Thermal Spacing:**
```python
# Euclidean distance for thermal spacing
for high_power_pair:
    dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    if dist < 20.0:  # Minimum thermal spacing
        warning = True
```

**Boundary Checking:**
```python
# Geometric bounds checking
if comp.x < comp.width/2 or comp.x > board.width - comp.width/2:
    violation = "Out of bounds"
```

---

### **ErrorFixerAgent: Geometric Fixing**

**Clearance Fixing:**
```python
# Calculate direction vector (geometric)
direction = array([comp2.x - comp1.x, comp2.y - comp1.y])
direction = direction / norm(direction)  # Normalize to unit vector

# Calculate move distance (geometric)
move_distance = (required_dist - current_dist) / 2 + 1.0

# Apply geometric translation
comp1.x -= direction[0] * move_distance
comp1.y -= direction[1] * move_distance
comp2.x += direction[0] * move_distance
comp2.y += direction[1] * move_distance
```

**Thermal Fixing:**
```python
# Calculate spacing using Euclidean distance
dist = sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

# Calculate direction vector (geometric)
direction = array([c2.x - c1.x, c2.y - c1.y])
direction = direction / norm(direction)

# Move apart (geometric translation)
move_distance = (min_spacing - dist) / 2 + 2.0
c1.x -= direction[0] * move_distance
c1.y -= direction[1] * move_distance
```

**Signal Integrity Fixing:**
```python
# Calculate centroid (geometric)
positions = array([[c.x, c.y] for c in net_components])
centroid = mean(positions, axis=0)

# Move components toward centroid (geometric optimization)
for comp in net_components:
    dist_to_centroid = sqrt((comp.x - centroid[0])**2 + (comp.y - centroid[1])**2)
    direction = array([centroid[0] - comp.x, centroid[1] - comp.y])
    direction = direction / norm(direction)
    comp.x += direction[0] * move_distance
    comp.y += direction[1] * move_distance
```

---

## 🎯 Key Insights

### **1. Context Window Hierarchy**

**AI Agents (xAI):**
- **IntentAgent**: ~500-1000 tokens (geometry data + user intent)
- **DesignGeneratorAgent**: ~1000-2000 tokens (description + examples)

**Non-AI Agents (Geometric):**
- **LocalPlacerAgent**: Full placement object (in-memory)
- **GlobalOptimizerAgent**: Full placement object (in-memory)
- **VerifierAgent**: Full placement object (in-memory)
- **ErrorFixerAgent**: Full placement object (in-memory)

### **2. Geometry Data Flow**

```
Placement
    ↓
GeometryAnalyzer.analyze()
    ↓
geometry_data = {
    voronoi_variance, mst_length, thermal_hotspots, ...
}
    ↓
IntentAgent (xAI reasons over geometry)
    ↓
weights = (α, β, γ)
    ↓
LocalPlacerAgent (uses weights + geometric scoring)
    ↓
Optimized Placement
    ↓
VerifierAgent (geometric validation)
    ↓
ErrorFixerAgent (geometric fixing)
```

### **3. Computational Geometry Usage Patterns**

**Analysis (GeometryAnalyzer):**
- Voronoi diagrams
- MST computation
- Convex hull
- Thermal hotspot detection

**Scoring (LocalPlacerAgent, GlobalOptimizerAgent):**
- Euclidean/Manhattan distance
- Gaussian thermal model
- Collision detection

**Validation (VerifierAgent):**
- Distance checking
- Boundary validation
- Thermal spacing

**Fixing (ErrorFixerAgent):**
- Direction vector calculation
- Geometric translation
- Centroid calculation

---

## 📚 Research Foundation

All geometric operations are based on proven algorithms:

1. **Voronoi Diagrams**: Fortune (1987) - O(n log n)
2. **MST**: Kruskal (1956) - Optimal
3. **Convex Hull**: Graham (1972) - O(n log n)
4. **Gaussian Thermal**: Holman (2010) - Heat transfer equations
5. **Distance Calculations**: Standard computational geometry

**This is mathematically rigorous, not heuristics.**

