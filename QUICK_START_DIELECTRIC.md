# Dielectric: Quick Start Guide

## What is Dielectric?

**Dielectric** is an AI-powered PCB design platform that:
- **Generates** PCB designs from natural language
- **Optimizes** designs using computational geometry + xAI
- **Fixes errors** automatically (agentic)
- **Validates quality** before manufacturing

## Key Features

1. **Natural Language Design Generation**: Describe what you want, get a complete PCB
2. **Agentic Error Fixing**: System automatically fixes design issues
3. **Computational Geometry**: Voronoi, MST, Convex Hull analysis
4. **Multi-Agent Architecture**: Specialized AI agents for each task
5. **Quality Validation**: Automated checks ensure manufacturing-ready designs

## How to Run

### Start Backend
```bash
cd neuro-geometric-placer
./venv/bin/python deploy_simple.py
```

### Start Frontend
```bash
./venv/bin/streamlit run frontend/app_dielectric.py --server.port 8501
```

Then open: **http://127.0.0.1:8501**

## Workflows

### 1. Generate New Design

1. Select **"Generate Design"** workflow
2. Enter natural language: "Design an audio amplifier with op-amp, resistors, and capacitors"
3. Set board size (optional)
4. Click **"Generate Design"**
5. System creates complete PCB design

### 2. Optimize Existing Design

1. Select **"Optimize Design"** workflow
2. Upload JSON file or load example
3. Enter optimization intent: "Optimize for thermal management"
4. Click **"Run Optimization"**
5. System optimizes and **automatically fixes errors**
6. Export to KiCad

## What Makes It Agentic

**Error Fixer Agent** automatically:
- Fixes design rule violations
- Resolves thermal hotspots
- Optimizes signal integrity
- Fixes manufacturability issues

**No manual intervention needed** - system fixes errors until design passes!

## API Endpoints

- `POST /generate` - Generate design from natural language
- `POST /optimize` - Optimize existing design
- `POST /export/kicad` - Export to KiCad format
- `POST /simulate` - Run simulation suite
- `GET /health` - Health check

## Example Usage

### Generate Design
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Audio amplifier with power supply",
    "board_size": {"width": 120, "height": 80}
  }'
```

### Optimize Design
```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "board": {...},
    "components": [...],
    "nets": [...],
    "intent": "Optimize for thermal management"
  }'
```

## Key Improvements

✅ **Natural Language Design**: Create PCBs from descriptions
✅ **Agentic Error Fixing**: Automatically resolves issues
✅ **Separated Workflows**: Clear distinction between generation and optimization
✅ **Better Error Handling**: KiCad export fixed
✅ **Research-Backed**: Computational geometry papers documented

---

**Dielectric**: The future of PCB design automation.

