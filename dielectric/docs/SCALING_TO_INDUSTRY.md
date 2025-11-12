# Scaling to Industry-Scale PCBs

**Guide for testing and optimizing large, industry-scale PCB designs**

---

## 🎯 Overview

Dielectric is designed to scale from small boards (10 components) to industry-scale PCBs (200+ components). This guide shows how to test with real-world designs.

---

## 📊 Industry-Scale PCB Characteristics

### Size Categories

| Category | Components | Examples | Optimization Time |
|----------|-----------|----------|-------------------|
| **Small** | < 50 | Arduino Uno, simple sensors | < 5s |
| **Medium** | 50-100 | Raspberry Pi Pico, ESP32 DevKit | 5-20s |
| **Large** | 100-200 | Raspberry Pi 4, BeagleBone | 20-60s |
| **Very Large** | 200+ | Server motherboards, complex industrial | 60-180s |

### Typical Industry PCBs

- **Development Boards:** 50-150 components
- **IoT Devices:** 30-100 components
- **Industrial Controllers:** 100-200 components
- **Server Motherboards:** 200-500+ components
- **High-Speed Digital:** 150-300 components

---

## 🚀 Quick Start: Download Real PCBs

### Method 1: Quick Download Script

```bash
cd dielectric
./scripts/quick_download_pcbs.sh
```

This downloads KiCad examples (small-medium PCBs) for quick testing.

### Method 2: Full Download Script

```bash
cd dielectric
python scripts/download_pcb_examples.py --output-dir examples/real_pcbs --create-manifest
```

This downloads multiple PCB projects from GitHub.

### Method 3: Manual Download

1. Go to https://github.com/KiCad/kicad-examples
2. Click "Code" → "Download ZIP"
3. Extract and find `.kicad_pcb` files
4. Upload via frontend

---

## 📤 Uploading Large PCBs

### Via Frontend

1. **Single File (< 200MB):**
   - Go to "Optimize Design"
   - Select "Single File"
   - Upload `.kicad_pcb` file
   - Click "Optimize"

2. **Folder/Zip (Recommended for large projects):**
   - Select "Folder/Zip"
   - Upload zip file or folder
   - Click "🚀 Process Files"
   - System auto-detects PCB files

### Via API

```python
import requests

# Single file
with open("large_board.kicad_pcb", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/pcb",
        files={"file": f},
        params={"optimization_intent": "Optimize for thermal management"}
    )

# Folder/Zip (better for large projects)
with open("project.zip", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/folder",
        files={"files": f},
        params={"optimization_intent": "Minimize trace length"}
    )
```

---

## ⚙️ Optimization Settings for Large PCBs

### For 100-200 Component PCBs

**Recommended Settings:**
- **Optimization Type:** "Quality" (not "Fast")
- **Max Time:** 60-120 seconds
- **Parallel Chains:** 4-8 (automatic)
- **Iterations:** 1000-2000

**Intent Examples:**
- "Optimize for thermal management with 100+ components"
- "Minimize trace length while maintaining signal integrity"
- "Balance thermal spreading and routing efficiency"

### For 200+ Component PCBs

**Recommended Settings:**
- **Optimization Type:** "Quality"
- **Max Time:** 120-300 seconds
- **Use Hierarchical Optimization:** Yes (automatic)
- **Module-Based:** Yes (automatic)

**Intent Examples:**
- "Optimize module placement first, then component placement"
- "Focus on power distribution network optimization"
- "Prioritize manufacturability for large-scale production"

---

## 🔧 Performance Optimization

### Current Capabilities

✅ **Scalable Algorithms:**
- Incremental Voronoi (O(log n) updates)
- Parallel Simulated Annealing (4-8x speedup)
- Scalable Thermal FDM (sparse matrices)

✅ **Handles:**
- 100+ component PCBs efficiently
- Multi-layer designs
- Complex routing

### Performance Tips

1. **Use Folder Upload:** Better for large projects with multiple files
2. **Enable Parallel Processing:** Automatic (uses all CPU cores)
3. **Adjust Timeout:** Increase for very large boards (200+ components)
4. **Use Quality Mode:** Better results for complex designs

---

## 📈 Testing Workflow

### Step 1: Download Test PCBs

```bash
cd dielectric
python scripts/download_pcb_examples.py
```

### Step 2: Upload via Frontend

1. Open http://localhost:8501
2. Go to "Optimize Design"
3. Upload PCB file or folder
4. Enter optimization intent
5. Click "Optimize"

### Step 3: Analyze Results

- **Before/After Metrics:**
  - Voronoi variance (distribution uniformity)
  - MST length (trace length estimate)
  - Thermal hotspots
  - Score improvement

- **Visualizations:**
  - PCB layout comparison
  - Thermal heatmap
  - Component distribution

### Step 4: Export Results

- Export to KiCad format
- Download optimized design
- Compare with original

---

## 🐛 Troubleshooting Large PCBs

### Upload Fails

**Issue:** File too large (> 200MB)
**Solution:** 
- Use folder/zip upload (handles large files better)
- Compress files before upload
- Split into modules if possible

### Parsing Errors

**Issue:** Parser can't handle certain KiCad features
**Solution:**
- Check parser logs for specific errors
- Try simpler boards first
- Report unsupported features

### Optimization Timeout

**Issue:** Optimization takes too long
**Solution:**
- Use "Fast" mode for initial testing
- Reduce max iterations
- Use hierarchical optimization (automatic)

### Memory Issues

**Issue:** Out of memory with very large PCBs
**Solution:**
- Use sparse matrix solvers (automatic)
- Enable incremental updates (automatic)
- Process in modules (automatic)

---

## 📚 Example Test Cases

### Test Case 1: Medium Board (50-100 components)

**Board:** Raspberry Pi Pico
**Goal:** Thermal optimization
**Expected Time:** 10-20 seconds
**Expected Improvement:** 15-25% score reduction

### Test Case 2: Large Board (100-200 components)

**Board:** Raspberry Pi 4 or BeagleBone
**Goal:** Signal integrity + thermal
**Expected Time:** 30-60 seconds
**Expected Improvement:** 20-30% score reduction

### Test Case 3: Very Large Board (200+ components)

**Board:** Complex industrial controller
**Goal:** Multi-objective optimization
**Expected Time:** 60-180 seconds
**Expected Improvement:** 25-35% score reduction

---

## 🎯 Success Metrics

### For Industry-Scale PCBs

✅ **Performance:**
- Optimization completes in < 3 minutes for 200+ components
- Memory usage < 2GB for 200+ components
- No crashes or timeouts

✅ **Quality:**
- 20%+ score improvement
- Thermal hotspots reduced
- Trace length minimized
- No design rule violations

✅ **Scalability:**
- Handles 200+ components
- Processes multi-layer designs
- Works with complex routing

---

## 📖 Next Steps

1. **Download Test PCBs:** Use the download scripts
2. **Upload and Test:** Try with real industry designs
3. **Compare Results:** Analyze before/after metrics
4. **Report Issues:** Document any problems or limitations
5. **Scale Up:** Test with progressively larger boards

---

**Ready to test industry-scale PCBs! 🚀**

