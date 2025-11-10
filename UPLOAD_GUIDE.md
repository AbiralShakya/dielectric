# 📤 Upload Your Own PCB Design - Quick Guide

## How to Upload and Optimize Your Design

### Step 1: Download Template
1. Open the frontend: `http://127.0.0.1:8501`
2. In the sidebar, click **"📥 Download JSON Template"**
3. Save the template file (`pcb_design_template.json`)

### Step 2: Create Your Design
Edit the template JSON file with your components:

```json
{
  "board": {
    "width": 100,      // Board width in mm
    "height": 100,     // Board height in mm
    "clearance": 0.5   // Minimum clearance in mm
  },
  "components": [
    {
      "name": "U1",           // Component name
      "package": "SOIC-8",    // Package type
      "width": 5,             // Width in mm
      "height": 4,            // Height in mm
      "power": 0.5,           // Power dissipation in W
      "x": 20,                // X position in mm
      "y": 20,                // Y position in mm
      "angle": 0,             // Rotation angle (degrees)
      "placed": true,         // Whether component is placed
      "pins": [...]           // Optional: pin definitions
    }
  ],
  "nets": [
    {
      "name": "VCC",
      "pins": [["U1", "pin1"], ["U2", "pin1"]]  // Connections
    }
  ]
}
```

### Step 3: Upload Your Design
1. In the sidebar, click **"Choose File"** under "Upload PCB Design"
2. Select your JSON file
3. Your design will be loaded and previewed

### Step 4: Computational Geometry Analysis
When you upload a design, the system automatically:
1. **Analyzes Voronoi Diagrams**: Component distribution uniformity
2. **Computes Minimum Spanning Tree**: Trace length estimation
3. **Calculates Convex Hull**: Board utilization
4. **Detects Thermal Hotspots**: High-power component regions
5. **Analyzes Net Crossings**: Routing conflict estimation

### Step 5: xAI Understanding
The computational geometry data is passed to xAI Grok, which:
- Understands your design's geometric properties
- Reasons about optimization priorities
- Generates optimization weights (α, β, γ)

### Step 6: AI Optimization
Click **"🚀 Generate AI-Optimized Layout"** to:
1. **IntentAgent**: Converts natural language + geometry → weights
2. **LocalPlacerAgent**: Optimizes placement using simulated annealing
3. **VerifierAgent**: Checks design rules

### Step 7: View Results
- **Before/After Comparison**: See your design vs optimized
- **Computational Geometry Metrics**: MST, Voronoi, thermal hotspots
- **Multi-Agent Status**: See which agents contributed
- **Export to KiCad**: Download for simulation

## Example Workflow

```
1. Upload your PCB design JSON
   ↓
2. System analyzes computational geometry
   - Voronoi: Component distribution
   - MST: Trace length estimate
   - Thermal: Hotspot detection
   ↓
3. xAI reasons over geometry data
   - Understands design properties
   - Maps to optimization priorities
   ↓
4. AI agents optimize
   - IntentAgent: Sets weights
   - LocalPlacerAgent: Optimizes placement
   - VerifierAgent: Validates design
   ↓
5. View professional visualization
   - Interactive Plotly charts
   - Thermal heatmaps
   - Net routing
   ↓
6. Export to KiCad
   - Ready for simulation
   - Manufacturing-ready
```

## Technical Details

See `TECHNICAL_DOCUMENTATION.md` for:
- Computational geometry algorithms
- xAI reasoning process
- Multi-agent architecture
- Research papers and foundations

## Tips

- **Component Names**: Use unique names (U1, R1, C1, etc.)
- **Power Values**: Set realistic power dissipation for thermal analysis
- **Nets**: Define connections between components for routing analysis
- **Clearance**: Set appropriate minimum clearance for your manufacturing process

## Troubleshooting

**"Invalid design format" error?**
- Ensure your JSON has `board`, `components`, and optionally `nets` fields
- Check JSON syntax (use a JSON validator)

**No visualization?**
- Make sure components have valid x, y coordinates
- Check that board dimensions are positive

**Optimization not working?**
- Ensure backend is running: `./venv/bin/python deploy_simple.py`
- Check that XAI_API_KEY is set in `.env` file

