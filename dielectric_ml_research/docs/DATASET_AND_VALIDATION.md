# Dataset Collection & Validation Strategy
## Testing Against Real PCB Designs

**Date:** 2025-01-XX  
**Status:** Critical Implementation Guide  
**Priority:** 🔴 **CRITICAL** - Foundation for research validation

---

## Problem Statement

You've identified three critical issues:

1. **No dataset for testing/training** - How do we validate against real designs?
2. **KiCad exports look "false"** - Exported designs may not be valid/usable
3. **Before/after shows little difference** - Optimization may not be working or visualization is broken

This document provides **actionable solutions** for all three.

---

## Part 1: Dataset Collection & Training Strategy

### Current State

**What exists:**
- `dielectric/data/pcb_database.json` - Small database of example designs
- `dielectric/examples/` - A few example JSON files
- No systematic dataset collection
- No training data for ML models

**What's needed:**
- **10,000+ real PCB designs** for training/evaluation
- **Labeled data** (routing paths, physics results, optimization outcomes)
- **Benchmark suite** for validation

---

### Dataset Sources

#### 1. Open Source PCB Repositories

**KiCad Library:**
```bash
# Clone KiCad library (thousands of example boards)
git clone https://gitlab.com/kicad/libraries/kicad-footprints.git
git clone https://gitlab.com/kicad/libraries/kicad-symbols.git

# Extract PCB files
find . -name "*.kicad_pcb" -type f > pcb_files.txt
```

**GitHub PCB Repositories:**
- Search GitHub for `.kicad_pcb` files
- Many open-source hardware projects publish their designs
- Examples: Arduino shields, Raspberry Pi HATs, ESP32 modules

**Script to collect:**
```python
import os
import subprocess
from pathlib import Path

def collect_kicad_designs(output_dir="datasets/kicad_designs"):
    """
    Collect KiCad designs from various sources.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sources = [
        "https://github.com/adafruit/Adafruit-PCB-Library",
        "https://github.com/sparkfun/SparkFun-KiCad-Libraries",
        "https://github.com/raspberrypi/pico-examples",
        # Add more sources
    ]
    
    for source in sources:
        repo_name = source.split("/")[-1]
        repo_path = output_path / repo_name
        
        # Clone repository
        subprocess.run(["git", "clone", source, str(repo_path)])
        
        # Find all .kicad_pcb files
        pcb_files = list(repo_path.rglob("*.kicad_pcb"))
        print(f"Found {len(pcb_files)} PCB files in {repo_name}")
        
        # Copy to dataset directory
        for pcb_file in pcb_files:
            dest = output_path / "pcbs" / pcb_file.name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(pcb_file.read_bytes())
```

---

#### 2. JLCPCB/PCBWay Design Gallery

**JLCPCB Design Gallery:**
- Thousands of user-submitted designs
- Can scrape (with permission) or use API
- Real-world manufacturing data

**PCBWay Gallery:**
- Similar to JLCPCB
- Real designs that were manufactured

---

#### 3. Synthetic Data Generation

**Generate synthetic designs for training:**
```python
import numpy as np
from src.backend.geometry.placement import Placement
from src.backend.geometry.board import Board
from src.backend.geometry.component import Component

def generate_synthetic_design(
    num_components: int = 20,
    board_width: float = 100.0,
    board_height: float = 100.0,
    complexity: str = "medium"
) -> Placement:
    """
    Generate synthetic PCB design for training.
    
    Args:
        num_components: Number of components
        board_width: Board width (mm)
        board_height: Board height (mm)
        complexity: "simple", "medium", "complex"
    
    Returns:
        Placement object
    """
    board = Board(width=board_width, height=board_height)
    components = {}
    nets = {}
    
    # Generate components
    packages = ["SOIC-8", "0805", "QFN-16", "BGA", "LED-5MM"]
    for i in range(num_components):
        comp = Component(
            name=f"U{i+1}",
            package=np.random.choice(packages),
            x=np.random.uniform(10, board_width - 10),
            y=np.random.uniform(10, board_height - 10),
            angle=np.random.choice([0, 90, 180, 270]),
            power=np.random.uniform(0, 2.0)
        )
        components[comp.name] = comp
    
    # Generate nets
    net_id = 1
    for comp_name in list(components.keys())[:-1]:
        # Connect each component to next (simplified)
        net = Net(
            name=f"Net{net_id}",
            pins=[(comp_name, "pin1"), (list(components.keys())[net_id], "pin1")]
        )
        nets[net.name] = net
        net_id += 1
    
    return Placement(board=board, components=components, nets=nets)

def generate_training_dataset(num_designs: int = 10000):
    """
    Generate large training dataset.
    """
    dataset = []
    
    for i in range(num_designs):
        # Vary complexity
        complexity = np.random.choice(["simple", "medium", "complex"])
        num_components = {
            "simple": np.random.randint(5, 15),
            "medium": np.random.randint(15, 50),
            "complex": np.random.randint(50, 200)
        }[complexity]
        
        # Generate design
        placement = generate_synthetic_design(
            num_components=num_components,
            complexity=complexity
        )
        
        # Run optimization (to get "before" and "after")
        # ... optimization code ...
        
        # Store design
        dataset.append({
            "id": i,
            "complexity": complexity,
            "placement_before": placement.to_dict(),
            "placement_after": optimized_placement.to_dict(),
            "optimization_metrics": {
                "score_improvement": score_improvement,
                "trace_length_reduction": trace_length_reduction,
                "thermal_improvement": thermal_improvement
            }
        })
    
    return dataset
```

---

### Dataset Structure

**Recommended structure:**
```
datasets/
├── kicad_designs/          # Real KiCad designs
│   ├── pcb_files/         # .kicad_pcb files
│   ├── metadata.json      # Design metadata
│   └── labels.json        # Labels (routing, physics, etc.)
├── synthetic_designs/      # Generated designs
│   ├── simple/            # Simple designs (5-15 components)
│   ├── medium/            # Medium designs (15-50 components)
│   └── complex/           # Complex designs (50+ components)
├── training/              # Training splits
│   ├── train/             # 80% for training
│   ├── val/               # 10% for validation
│   └── test/              # 10% for testing
└── benchmarks/            # Benchmark suite
    ├── placement_benchmarks/
    ├── routing_benchmarks/
    └── physics_benchmarks/
```

---

### Labeling Strategy

**What to label:**

1. **Routing Labels:**
   - Optimal routing paths (from successful autorouting)
   - Via locations
   - Layer assignments

2. **Physics Labels:**
   - Thermal simulation results (from FDTD/FEM)
   - S-parameters (from EM simulation)
   - Signal integrity metrics

3. **Optimization Labels:**
   - Before/after placements
   - Score improvements
   - Optimization strategies that worked

**Labeling script:**
```python
def label_design(design_path: str) -> Dict:
    """
    Label a PCB design with routing, physics, and optimization data.
    """
    # Load design
    placement = Placement.from_kicad_file(design_path)
    
    # 1. Routing labels (run autorouter)
    routing_result = run_autorouter(placement)
    routing_labels = {
        "paths": routing_result["paths"],
        "vias": routing_result["vias"],
        "layers": routing_result["layers"],
        "length": routing_result["total_length"]
    }
    
    # 2. Physics labels (run simulation)
    physics_labels = {
        "thermal": run_thermal_simulation(placement),
        "si": run_si_simulation(placement),
        "pdn": run_pdn_simulation(placement)
    }
    
    # 3. Optimization labels (run optimization)
    optimized = optimize_placement(placement)
    optimization_labels = {
        "before": placement.to_dict(),
        "after": optimized.to_dict(),
        "improvement": compute_improvement(placement, optimized)
    }
    
    return {
        "design_id": design_path,
        "routing": routing_labels,
        "physics": physics_labels,
        "optimization": optimization_labels
    }
```

---

### Training Data Pipeline

**Complete pipeline:**
```python
class TrainingDataPipeline:
    """
    Complete pipeline for collecting, labeling, and preparing training data.
    """
    def __init__(self):
        self.collector = DesignCollector()
        self.labeler = DesignLabeler()
        self.preprocessor = DataPreprocessor()
    
    def build_dataset(self, num_designs: int = 10000):
        """
        Build complete training dataset.
        """
        # 1. Collect designs
        designs = self.collector.collect(num_designs)
        
        # 2. Label designs
        labeled_designs = []
        for design in designs:
            labels = self.labeler.label(design)
            labeled_designs.append({
                "design": design,
                "labels": labels
            })
        
        # 3. Preprocess
        processed = self.preprocessor.process(labeled_designs)
        
        # 4. Split train/val/test
        train, val, test = self.split_dataset(processed, [0.8, 0.1, 0.1])
        
        # 5. Save
        self.save_dataset(train, "datasets/training/train")
        self.save_dataset(val, "datasets/training/val")
        self.save_dataset(test, "datasets/training/test")
        
        return train, val, test
```

---

## Part 2: KiCad Export Validation

### Problem: Exported Designs Look "False"

**Issues identified:**
1. Footprints may not match KiCad library standards
2. Net connections may be incorrect
3. Board outline may be malformed
4. Design rules may be violated

---

### Validation Strategy

#### 1. KiCad DRC Validation

**Validate exported designs:**
```python
import subprocess
import tempfile
from pathlib import Path

def validate_kicad_export(kicad_file_path: str) -> Dict:
    """
    Validate KiCad export using KiCad's DRC.
    
    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "drc_violations": List[Dict]
        }
    """
    # Create temporary project
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        project_path.mkdir()
        
        # Copy PCB file
        pcb_file = project_path / "test.kicad_pcb"
        pcb_file.write_text(Path(kicad_file_path).read_text())
        
        # Run KiCad DRC via command line
        # Note: Requires KiCad installed
        try:
            result = subprocess.run(
                [
                    "kicad-cli", "pcb", "drc",
                    str(pcb_file)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse DRC output
                violations = parse_drc_output(result.stdout)
                return {
                    "valid": len(violations) == 0,
                    "errors": [],
                    "warnings": [],
                    "drc_violations": violations
                }
            else:
                return {
                    "valid": False,
                    "errors": [result.stderr],
                    "warnings": [],
                    "drc_violations": []
                }
        except FileNotFoundError:
            # KiCad CLI not available
            return {
                "valid": False,
                "errors": ["KiCad CLI not found"],
                "warnings": [],
                "drc_violations": []
            }
```

#### 2. Python pcbnew Validation

**Use KiCad Python API:**
```python
try:
    import pcbnew
    
    def validate_with_pcbnew(kicad_file_path: str) -> Dict:
        """
        Validate using KiCad's pcbnew Python API.
        """
        board = pcbnew.LoadBoard(str(kicad_file_path))
        
        # Check board validity
        errors = []
        warnings = []
        
        # Check footprints
        footprints = board.GetFootprints()
        for fp in footprints:
            # Validate footprint
            if not fp.GetReference():
                errors.append(f"Footprint {fp.GetReference()} missing reference")
            
            # Check pads
            pads = fp.Pads()
            for pad in pads:
                if not pad.GetNet():
                    warnings.append(f"Pad {pad.GetName()} in {fp.GetReference()} not connected")
        
        # Check nets
        nets = board.GetNets()
        for net in nets:
            if net.GetNetClass() is None:
                warnings.append(f"Net {net.GetName()} has no net class")
        
        # Check board outline
        edge_cuts = board.GetLayerID("Edge.Cuts")
        if not board.GetDrawings(edge_cuts):
            errors.append("No board outline (Edge.Cuts) found")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "footprint_count": len(footprints),
            "net_count": len(nets)
        }
except ImportError:
    # pcbnew not available
    pass
```

#### 3. Automated Testing

**Test suite for KiCad exports:**
```python
import pytest
from src.backend.export.kicad_exporter import KiCadExporter
from src.backend.geometry.placement import Placement

class TestKiCadExport:
    """Test suite for KiCad export validation."""
    
    def test_export_valid_kicad_format(self):
        """Test that exported file is valid KiCad format."""
        exporter = KiCadExporter()
        placement = create_test_placement()
        
        kicad_content = exporter.export(placement.to_dict())
        
        # Check KiCad format
        assert "(kicad_pcb" in kicad_content
        assert "(version" in kicad_content
        assert "(layers" in kicad_content
    
    def test_footprints_have_pads(self):
        """Test that all footprints have pads."""
        exporter = KiCadExporter()
        placement = create_test_placement()
        
        kicad_content = exporter.export(placement.to_dict())
        
        # Count footprints and pads
        footprint_count = kicad_content.count("(footprint")
        pad_count = kicad_content.count("(pad")
        
        assert pad_count > 0, "No pads found"
        assert pad_count >= footprint_count * 2, "Footprints missing pads"
    
    def test_nets_are_connected(self):
        """Test that nets are properly connected to pads."""
        exporter = KiCadExporter()
        placement = create_test_placement()
        
        kicad_content = exporter.export(placement.to_dict())
        
        # Check net definitions
        assert "(net" in kicad_content
        
        # Check pad-net connections
        pad_with_net_count = kicad_content.count('(net ')
        assert pad_with_net_count > 0, "No pads connected to nets"
    
    def test_board_outline_exists(self):
        """Test that board outline (Edge.Cuts) exists."""
        exporter = KiCadExporter()
        placement = create_test_placement()
        
        kicad_content = exporter.export(placement.to_dict())
        
        assert 'layer "Edge.Cuts"' in kicad_content
        assert "(gr_line" in kicad_content
    
    def test_export_loads_in_kicad(self):
        """Test that exported file can be loaded in KiCad."""
        exporter = KiCadExporter()
        placement = create_test_placement()
        
        kicad_content = exporter.export(placement.to_dict())
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kicad_pcb', delete=False) as f:
            f.write(kicad_content)
            temp_path = f.name
        
        try:
            # Try to load in KiCad
            validation_result = validate_kicad_export(temp_path)
            assert validation_result["valid"], f"KiCad validation failed: {validation_result['errors']}"
        finally:
            os.unlink(temp_path)
```

---

### Fixing Export Issues

**Common issues and fixes:**

1. **Footprints don't match KiCad library:**
   - **Fix:** Use KiCad's footprint library instead of custom footprints
   - **Implementation:** Query KiCad footprint library via MCP

2. **Net connections incorrect:**
   - **Fix:** Properly map component pins to nets
   - **Implementation:** Improve `comp_pin_to_net` mapping in exporter

3. **Board outline malformed:**
   - **Fix:** Ensure Edge.Cuts layer has closed polygon
   - **Implementation:** Validate board outline geometry

**Improved exporter:**
```python
class ValidatedKiCadExporter(KiCadExporter):
    """
    KiCad exporter with validation.
    """
    def export(self, placement_data: Dict, validate: bool = True) -> str:
        """
        Export with validation.
        """
        # Export
        kicad_content = super().export(placement_data)
        
        if validate:
            # Validate export
            validation_result = self.validate_export(kicad_content, placement_data)
            
            if not validation_result["valid"]:
                # Fix issues
                kicad_content = self.fix_export_issues(kicad_content, validation_result)
        
        return kicad_content
    
    def validate_export(self, kicad_content: str, placement_data: Dict) -> Dict:
        """Validate exported content."""
        errors = []
        warnings = []
        
        # Check format
        if "(kicad_pcb" not in kicad_content:
            errors.append("Invalid KiCad format")
        
        # Check footprints
        footprint_count = kicad_content.count("(footprint")
        if footprint_count != len(placement_data.get("components", [])):
            warnings.append(f"Footprint count mismatch: {footprint_count} vs {len(placement_data.get('components', []))}")
        
        # Check nets
        net_count = kicad_content.count("(net ")
        if net_count != len(placement_data.get("nets", [])):
            warnings.append(f"Net count mismatch: {net_count} vs {len(placement_data.get('nets', []))}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
```

---

## Part 3: Before/After Comparison Fix

### Problem: Little to No Difference

**Possible causes:**
1. Optimization isn't actually running
2. Optimization converges too quickly (already optimal)
3. Visualization isn't showing changes
4. Initial placement is already good

---

### Debugging Strategy

#### 1. Verify Optimization is Running

**Add logging:**
```python
class LocalPlacerAgent:
    async def process(self, placement, weights, ...):
        # Log initial state
        initial_score = self.scorer.score(placement)
        initial_geometry = self.geometry_analyzer.analyze(placement)
        
        logger.info(f"Initial score: {initial_score:.4f}")
        logger.info(f"Initial MST length: {initial_geometry['mst_length']:.2f}mm")
        logger.info(f"Initial Voronoi variance: {initial_geometry['voronoi_variance']:.2f}")
        
        # Run optimization
        optimized_placement = await self.optimize(placement, weights, ...)
        
        # Log final state
        final_score = self.scorer.score(optimized_placement)
        final_geometry = self.geometry_analyzer.analyze(optimized_placement)
        
        logger.info(f"Final score: {final_score:.4f}")
        logger.info(f"Final MST length: {final_geometry['mst_length']:.2f}mm")
        logger.info(f"Final Voronoi variance: {final_geometry['voronoi_variance']:.2f}")
        
        # Calculate improvement
        score_improvement = initial_score - final_score
        mst_improvement = initial_geometry['mst_length'] - final_geometry['mst_length']
        
        logger.info(f"Score improvement: {score_improvement:.4f}")
        logger.info(f"MST improvement: {mst_improvement:.2f}mm")
        
        if abs(score_improvement) < 0.01:
            logger.warning("⚠️  Optimization showed minimal improvement - may need better initial placement or stronger optimization")
        
        return {
            "success": True,
            "placement": optimized_placement,
            "score": final_score,
            "stats": {
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": score_improvement,
                "geometry_improvement": {
                    "mst_length": mst_improvement,
                    "voronoi_variance": initial_geometry['voronoi_variance'] - final_geometry['voronoi_variance']
                }
            }
        }
```

#### 2. Force Initial Bad Placement

**Create intentionally bad initial placement:**
```python
def create_bad_initial_placement(placement: Placement) -> Placement:
    """
    Create intentionally bad initial placement for testing.
    
    Clusters all components in one corner.
    """
    bad_placement = placement.copy()
    
    # Cluster all components in top-left corner
    cluster_x = placement.board.width * 0.1
    cluster_y = placement.board.height * 0.1
    
    for comp in bad_placement.components.values():
        comp.x = cluster_x + np.random.uniform(-5, 5)
        comp.y = cluster_y + np.random.uniform(-5, 5)
    
    return bad_placement
```

#### 3. Improve Visualization

**Enhanced before/after visualization:**
```python
def create_enhanced_comparison(initial_placement, optimized_placement):
    """
    Create enhanced before/after visualization with metrics.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Calculate metrics
    initial_geometry = GeometryAnalyzer(initial_placement).analyze()
    optimized_geometry = GeometryAnalyzer(optimized_placement).analyze()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Before", "After", "Metrics Comparison", "Component Movement"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Before layout
    before_trace = create_layout_trace(initial_placement, "Before")
    fig.add_trace(before_trace, row=1, col=1)
    
    # After layout
    after_trace = create_layout_trace(optimized_placement, "After")
    fig.add_trace(after_trace, row=1, col=2)
    
    # Metrics comparison
    metrics = ["MST Length", "Voronoi Variance", "Thermal Hotspots"]
    before_values = [
        initial_geometry["mst_length"],
        initial_geometry["voronoi_variance"],
        initial_geometry["thermal_hotspots"]
    ]
    after_values = [
        optimized_geometry["mst_length"],
        optimized_geometry["voronoi_variance"],
        optimized_geometry["thermal_hotspots"]
    ]
    
    fig.add_trace(go.Bar(name="Before", x=metrics, y=before_values), row=2, col=1)
    fig.add_trace(go.Bar(name="After", x=metrics, y=after_values), row=2, col=1)
    
    # Component movement
    movements = []
    for comp_name in initial_placement.components:
        initial_comp = initial_placement.components[comp_name]
        optimized_comp = optimized_placement.components[comp_name]
        
        dx = optimized_comp.x - initial_comp.x
        dy = optimized_comp.y - initial_comp.y
        distance = np.sqrt(dx**2 + dy**2)
        
        movements.append({
            "component": comp_name,
            "distance": distance,
            "dx": dx,
            "dy": dy
        })
    
    movement_trace = go.Scatter(
        x=[m["dx"] for m in movements],
        y=[m["dy"] for m in movements],
        mode="markers+text",
        text=[m["component"] for m in movements],
        name="Movement"
    )
    fig.add_trace(movement_trace, row=2, col=2)
    
    return fig
```

#### 4. Increase Optimization Strength

**Make optimization more aggressive:**
```python
class AggressiveOptimizer:
    """
    More aggressive optimization for visible improvements.
    """
    def optimize(self, placement, weights, max_iterations=5000):
        """
        Run more aggressive optimization.
        """
        # Use lower temperature (more exploration)
        optimizer = EnhancedSimulatedAnnealing(
            initial_temperature=100.0,  # Lower = more exploration
            cooling_rate=0.95,  # Slower cooling
            min_temperature=0.1
        )
        
        # More iterations
        optimized = optimizer.optimize(
            placement,
            weights,
            max_iterations=max_iterations,
            convergence_threshold=0.0001  # Stricter convergence
        )
        
        return optimized
```

---

## Implementation Plan

### Week 1: Dataset Collection

1. **Collect 1,000 real KiCad designs**
   - Clone open-source repositories
   - Extract `.kicad_pcb` files
   - Store in `datasets/kicad_designs/`

2. **Generate 5,000 synthetic designs**
   - Use `generate_synthetic_design()` function
   - Vary complexity (simple, medium, complex)
   - Store in `datasets/synthetic_designs/`

3. **Label 100 designs**
   - Run routing (autorouter)
   - Run physics simulation
   - Store labels in `datasets/labels/`

### Week 2: KiCad Validation

1. **Implement validation suite**
   - `validate_kicad_export()` function
   - `TestKiCadExport` test suite
   - Fix export issues

2. **Test on collected designs**
   - Export each design
   - Validate export
   - Fix issues found

### Week 3: Before/After Fix

1. **Add logging**
   - Log initial/final scores
   - Log geometry metrics
   - Log component movements

2. **Improve visualization**
   - Enhanced comparison view
   - Metrics comparison
   - Component movement visualization

3. **Test with bad initial placement**
   - Create intentionally bad placements
   - Verify optimization improves them

---

## Expected Results

### Dataset

- ✅ **6,000 designs** collected (1,000 real + 5,000 synthetic)
- ✅ **100 designs** labeled (routing, physics, optimization)
- ✅ **Benchmark suite** for validation

### KiCad Validation

- ✅ **100% of exports** pass KiCad DRC
- ✅ **All footprints** match KiCad library standards
- ✅ **All nets** properly connected

### Before/After Comparison

- ✅ **Visible improvements** in before/after comparison
- ✅ **Metrics show** clear optimization (MST length reduction, etc.)
- ✅ **Component movements** clearly visible

---

## Next Steps

1. **Implement dataset collection script** (Week 1)
2. **Implement KiCad validation** (Week 2)
3. **Fix before/after visualization** (Week 3)
4. **Run validation on collected designs** (Week 4)
5. **Iterate based on results** (Ongoing)

---

**This addresses all three concerns with actionable, implementable solutions.**

