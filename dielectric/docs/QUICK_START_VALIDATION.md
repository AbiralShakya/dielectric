# Quick Start: Dataset Collection & Validation

**Date:** 2025-01-XX  
**Status:** Immediate Action Items  
**Priority:** 🔴 **CRITICAL**

---

## Immediate Actions (Next 30 Minutes)

### 1. Collect Initial Dataset

```bash
cd dielectric
python scripts/collect_dataset.py --synthetic 100 --github --max-repos 5
```

This will:
- ✅ Generate 100 synthetic designs
- ✅ Collect real KiCad designs from GitHub
- ✅ Store in `datasets/` directory

**Expected output:**
```
📊 Dielectric Dataset Collection
============================================================
🎨 Generating 100 synthetic designs...
  ✅ Generated 10/100 designs
  ✅ Generated 20/100 designs
  ...
✅ Generated 100 synthetic designs
🔍 Collecting KiCad designs from GitHub...
  📦 Cloning adafruit/Adafruit-PCB-Library...
    ✅ Found 50 PCB files
  ...
✅ Collected 200 PCB files
```

---

### 2. Validate KiCad Exports

```bash
# Validate exports in a directory
python scripts/collect_dataset.py --validate dielectric/exports
```

This will:
- ✅ Check all `.kicad_pcb` files
- ✅ Report validation errors
- ✅ Identify common issues

**Expected output:**
```
🔍 Validating KiCad exports in dielectric/exports...
  ❌ design1.kicad_pcb: ['Missing board outline (Edge.Cuts)']
  ✅ design2.kicad_pcb: Valid
✅ Validation complete: 5 valid, 2 invalid out of 7 files
```

---

### 3. Test Before/After Comparison

**Add this to your optimization code:**

```python
# In LocalPlacerAgent or Orchestrator
import logging
logger = logging.getLogger(__name__)

async def optimize_with_logging(self, placement, weights):
    # Log initial state
    initial_score = self.scorer.score(placement)
    initial_geometry = GeometryAnalyzer(placement).analyze()
    
    logger.info(f"📊 INITIAL STATE:")
    logger.info(f"   Score: {initial_score:.4f}")
    logger.info(f"   MST Length: {initial_geometry['mst_length']:.2f}mm")
    logger.info(f"   Voronoi Variance: {initial_geometry['voronoi_variance']:.2f}")
    logger.info(f"   Thermal Hotspots: {initial_geometry['thermal_hotspots']}")
    
    # Run optimization
    optimized_placement = await self.optimize(placement, weights)
    
    # Log final state
    final_score = self.scorer.score(optimized_placement)
    final_geometry = GeometryAnalyzer(optimized_placement).analyze()
    
    logger.info(f"📊 FINAL STATE:")
    logger.info(f"   Score: {final_score:.4f}")
    logger.info(f"   MST Length: {final_geometry['mst_length']:.2f}mm")
    logger.info(f"   Voronoi Variance: {final_geometry['voronoi_variance']:.2f}")
    logger.info(f"   Thermal Hotspots: {final_geometry['thermal_hotspots']}")
    
    # Calculate improvement
    improvement = {
        "score": initial_score - final_score,
        "mst_length": initial_geometry['mst_length'] - final_geometry['mst_length'],
        "voronoi_variance": initial_geometry['voronoi_variance'] - final_geometry['voronoi_variance'],
        "thermal_hotspots": initial_geometry['thermal_hotspots'] - final_geometry['thermal_hotspots']
    }
    
    logger.info(f"📈 IMPROVEMENT:")
    logger.info(f"   Score: {improvement['score']:.4f}")
    logger.info(f"   MST Length: {improvement['mst_length']:.2f}mm")
    logger.info(f"   Voronoi Variance: {improvement['voronoi_variance']:.2f}")
    logger.info(f"   Thermal Hotspots: {improvement['thermal_hotspots']}")
    
    if abs(improvement['score']) < 0.01:
        logger.warning("⚠️  WARNING: Minimal improvement detected!")
        logger.warning("   This could mean:")
        logger.warning("   1. Initial placement is already optimal")
        logger.warning("   2. Optimization isn't running properly")
        logger.warning("   3. Weights need adjustment")
    
    return optimized_placement
```

---

## Debugging Before/After Issue

### Check 1: Is Optimization Actually Running?

**Add this test:**

```python
# Test script: test_optimization.py
import asyncio
from src.backend.agents.orchestrator import AgentOrchestrator
from src.backend.geometry.placement import Placement
import numpy as np

async def test_optimization():
    # Create intentionally BAD initial placement
    # (all components clustered in one corner)
    placement = create_test_placement()
    
    # Cluster all components
    for comp in placement.components.values():
        comp.x = 10 + np.random.uniform(-5, 5)
        comp.y = 10 + np.random.uniform(-5, 5)
    
    print("📊 BEFORE (Bad Placement):")
    print(f"   Components clustered at (10, 10)")
    
    # Optimize
    orchestrator = AgentOrchestrator()
    result = await orchestrator.optimize_fast(
        placement,
        "Optimize for thermal spreading and trace length"
    )
    
    if result["success"]:
        optimized = result["placement"]
        
        print("\n📊 AFTER (Optimized):")
        # Check if components moved
        movements = []
        for comp_name in placement.components:
            initial = placement.components[comp_name]
            optimized_comp = optimized.components[comp_name]
            
            dx = optimized_comp.x - initial.x
            dy = optimized_comp.y - initial.y
            distance = np.sqrt(dx**2 + dy**2)
            
            movements.append(distance)
        
        avg_movement = np.mean(movements)
        print(f"   Average component movement: {avg_movement:.2f}mm")
        
        if avg_movement < 1.0:
            print("⚠️  WARNING: Components barely moved!")
            print("   Optimization may not be working")
        else:
            print("✅ Components moved significantly - optimization working!")
    else:
        print(f"❌ Optimization failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_optimization())
```

**Run it:**
```bash
python test_optimization.py
```

---

### Check 2: Is Visualization Showing Changes?

**Enhanced visualization function:**

```python
def create_enhanced_comparison(initial_placement, optimized_placement):
    """
    Create before/after comparison with component movement visualization.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Calculate movements
    movements = []
    for comp_name in initial_placement.components:
        initial_comp = initial_placement.components[comp_name]
        optimized_comp = optimized_placement.components[comp_name]
        
        dx = optimized_comp.x - initial_comp.x
        dy = optimized_comp.y - initial_comp.y
        distance = np.sqrt(dx**2 + dy**2)
        
        movements.append({
            "name": comp_name,
            "initial_x": initial_comp.x,
            "initial_y": initial_comp.y,
            "final_x": optimized_comp.x,
            "final_y": optimized_comp.y,
            "dx": dx,
            "dy": dy,
            "distance": distance
        })
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before", "After"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Before
    for m in movements:
        fig.add_trace(
            go.Scatter(
                x=[m["initial_x"]],
                y=[m["initial_y"]],
                mode="markers+text",
                text=[m["name"]],
                name="Before",
                marker=dict(color="red", size=10)
            ),
            row=1, col=1
        )
    
    # After
    for m in movements:
        fig.add_trace(
            go.Scatter(
                x=[m["final_x"]],
                y=[m["final_y"]],
                mode="markers+text",
                text=[m["name"]],
                name="After",
                marker=dict(color="green", size=10)
            ),
            row=1, col=2
        )
    
    # Add movement arrows
    for m in movements:
        if m["distance"] > 0.1:  # Only show if moved significantly
            fig.add_trace(
                go.Scatter(
                    x=[m["initial_x"], m["final_x"]],
                    y=[m["initial_y"], m["final_y"]],
                    mode="lines+markers",
                    line=dict(color="blue", width=2, dash="dash"),
                    name="Movement",
                    showlegend=False
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        title="Before/After Comparison with Component Movement",
        height=600
    )
    
    return fig
```

---

## Next Steps

### This Week

1. ✅ **Run dataset collection** (`python scripts/collect_dataset.py`)
2. ✅ **Add optimization logging** (see code above)
3. ✅ **Test with bad initial placement** (`test_optimization.py`)
4. ✅ **Fix visualization** (enhanced comparison function)

### Next Week

1. **Label 100 designs** (routing, physics, optimization)
2. **Validate all KiCad exports**
3. **Fix export issues** found during validation
4. **Create benchmark suite**

---

## Expected Results

After running these steps, you should see:

1. **Dataset:**
   - ✅ 100+ synthetic designs
   - ✅ 50+ real KiCad designs
   - ✅ Stored in `datasets/` directory

2. **Validation:**
   - ✅ List of export issues
   - ✅ Fixes for common problems

3. **Before/After:**
   - ✅ Clear logging of improvements
   - ✅ Visible component movements
   - ✅ Metrics showing optimization

---

## Troubleshooting

### Issue: "No designs collected"

**Solution:**
- Check internet connection (for GitHub cloning)
- Verify Git is installed
- Try with `--max-repos 1` first

### Issue: "Optimization shows no improvement"

**Solution:**
1. Check logs - is optimization actually running?
2. Test with intentionally bad placement
3. Increase optimization iterations
4. Check weights - are they set correctly?

### Issue: "KiCad exports invalid"

**Solution:**
1. Check validation errors
2. Fix common issues (missing Edge.Cuts, etc.)
3. Use `ValidatedKiCadExporter` class

---

**Run these steps now to start collecting data and fixing issues!**

