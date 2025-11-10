# How to Run Dielectric

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/neuro-geometric-placer

# Activate virtual environment
source venv/bin/activate  # or: ./venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. **Set Environment Variables**

Create `.env` file in project root:

```bash
XAI_API_KEY=your_xai_api_key_here
```

### 3. **Start Backend**

```bash
# Terminal 1: Start FastAPI backend
./venv/bin/python deploy_simple.py
```

Backend will start on: `http://localhost:8000`

### 4. **Start Frontend**

```bash
# Terminal 2: Start Streamlit frontend
./venv/bin/streamlit run frontend/app_dielectric.py --server.port 8501
```

Frontend will open in browser: `http://localhost:8501`

## 📋 Usage Workflows

### **Workflow 1: Generate New Design**

1. Open frontend: `http://localhost:8501`
2. Select **"Generate Design"** workflow
3. Enter natural language prompt (see `COMPLEX_PCB_PROMPTS.md`)
4. Click **"Generate Design"**
5. System creates complete PCB design
6. View visualization and export to KiCad

### **Workflow 2: Optimize Existing Design**

1. Select **"Optimize Design"** workflow
2. **Upload JSON file** or **Load Example**
3. Enter optimization intent (e.g., "minimize trace length", "optimize thermal")
4. Click **"Run Optimization"**
5. View before/after comparison
6. Check quality metrics
7. Export optimized design to KiCad

### **Workflow 3: Complex/Large PCB**

1. Use prompts from `COMPLEX_PCB_PROMPTS.md`
2. System automatically:
   - Creates knowledge graph
   - Identifies modules
   - Applies fabrication constraints
   - Optimizes hierarchically
3. View module structure in visualization
4. Export multi-layer KiCad file

## 🎯 Example Commands

### **Generate Design via API**

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design a multi-module audio amplifier with power management, analog section, and digital control",
    "board_width": 150,
    "board_height": 100
  }'
```

### **Optimize Design via API**

```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {...},  # Your placement JSON
    "user_intent": "Optimize for thermal management and minimize trace length"
  }'
```

### **Export to KiCad via API**

```bash
curl -X POST "http://localhost:8000/export/kicad" \
  -H "Content-Type: application/json" \
  -d '{
    "placement_data": {...}  # Your placement JSON
  }'
```

## 📚 Documentation Files

### **Core Documentation:**
- `QUICK_REFERENCE.md` - Quick reference guide
- `COMPLETE_FIXES_SUMMARY.md` - All fixes and features
- `LARGE_PCB_COMPUTATIONAL_GEOMETRY.md` - Large PCB design guide
- `COMPLEX_PCB_PROMPTS.md` - Example prompts for complex designs

### **Technical Documentation:**
- `TECHNICAL_DOCUMENTATION.md` - Architecture and algorithms
- `RESEARCH_PAPERS.md` - Academic papers on computational geometry
- `ENTERPRISE_AI_TRACK.md` - Enterprise positioning
- `PCB_ENGINEER_WORKFLOW.md` - What engineers actually do

### **Setup & Deployment:**
- `DEDALUS_SETUP.md` - Dedalus Labs setup
- `DEDALUS_DEPLOYMENT_STEPS.md` - Step-by-step deployment
- `QUICK_START_DIELECTRIC.md` - Quick start guide

### **Competitive & Strategy:**
- `COMPETITIVE_ANALYSIS.md` - vs. JITX, Altium, KiCad
- `JUDGES_PITCH.md` - HackPrinceton pitch
- `MAKING_IT_THE_BEST.md` - How to impress judges

## 🔧 Advanced Features

### **Deterministic Optimization**

Same intent always gives same result:

```python
# In orchestrator, seed is derived from user_intent
seed = int(hashlib.md5(user_intent.encode()).hexdigest()[:8], 16) % (2**31)
```

### **Knowledge Graph**

```python
from src.backend.knowledge.component_graph import ComponentKnowledgeGraph

kg = ComponentKnowledgeGraph.from_placement(placement)
modules = kg.identify_modules(placement)  # Auto-identifies modules
hints = kg.get_placement_hints("U1")  # Get placement suggestions
```

### **Fabrication Constraints**

```python
from src.backend.constraints.pcb_fabrication import FabricationConstraints, ConstraintValidator

constraints = FabricationConstraints(
    min_trace_width=0.15,  # mm (6 mil)
    min_trace_spacing=0.15,  # mm
    min_pad_to_pad_clearance=0.2  # mm (8 mil)
)

validator = ConstraintValidator(constraints)
result = validator.validate_placement(placement, knowledge_graph=kg)
```

## 🐛 Troubleshooting

### **Backend won't start:**
- Check if port 8000 is available
- Verify `XAI_API_KEY` is set in `.env`
- Check Python version: `python --version` (needs 3.12+)

### **Frontend won't start:**
- Check if port 8501 is available
- Verify backend is running on port 8000
- Check Streamlit: `pip install streamlit`

### **Export fails:**
- Check placement data has `board`, `components`, `nets`
- Verify all components have valid `package` types
- Check nets have proper `pins` array

### **Optimization not deterministic:**
- Ensure using latest code (seed support added)
- Check that `user_intent` is same for both runs
- Verify `random_seed` is being passed to optimizer

## 📊 API Endpoints

### **Generate Design**
- `POST /generate` - Generate PCB from natural language
- Body: `{"description": "...", "board_width": 150, "board_height": 100}`

### **Optimize Design**
- `POST /optimize` - Optimize existing placement
- Body: `{"placement": {...}, "user_intent": "..."}`

### **Export KiCad**
- `POST /export/kicad` - Export to KiCad format
- Body: `{"placement_data": {...}}`

### **Health Check**
- `GET /health` - Check if backend is running

## 🎓 Learning Resources

1. **Start Simple**: Use basic prompts first
2. **Add Complexity**: Gradually add more requirements
3. **Check Examples**: See `examples/` directory
4. **Read Docs**: Understand computational geometry features
5. **Experiment**: Try different optimization intents

## 🚀 Production Deployment

### **With Dedalus Labs:**
1. Commit `dedalus.json` and `dedalus_entrypoint.py` to GitHub
2. Set environment variables in Dedalus dashboard
3. Redeploy server

### **Without Dedalus:**
- System works perfectly locally
- All agents run in-process
- No external dependencies needed

---

**Dielectric**: Enterprise AI for PCB Design

