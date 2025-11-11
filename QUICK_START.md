# 🚀 Quick Start Guide - Dielectric

## One-Command Setup & Run

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# 1. Setup (first time only)
./setup.sh

# 2. Set API key
export XAI_API_KEY="your_xai_api_key_here"

# 3. Run everything
./run_complete_system.sh
```

Then open: **http://localhost:8501**

---

## Manual Setup (Step by Step)

### Step 1: Setup Environment

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set API Key

```bash
export XAI_API_KEY="your_xai_api_key_here"
```

Or create `.env` file:
```bash
echo "XAI_API_KEY=your_xai_api_key_here" > .env
```

### Step 3: Start Backend (Terminal 1)

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Backend runs on:** http://localhost:8000

### Step 4: Start Frontend (Terminal 2)

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
streamlit run frontend/app_dielectric.py --server.port 8501
```

**Frontend runs on:** http://localhost:8501

---

## Quick Test

```bash
# Test xAI API directly (matching your format)
export XAI_API_KEY="your_key"
./test_xai_api.sh

# Or test backend
curl http://localhost:8000/health

# Or use test script
./test_system.sh
```

---

## What to Do in Frontend

1. **Generate Design:**
   - Select "Generate Design" workflow
   - Enter description: `"Design an audio amplifier with op-amp, resistors, and capacitors"`
   - Set board size: 120mm x 80mm
   - Click "Generate Design"
   - **View tabs:** PCB Layout | Schematic | Thermal View

2. **Optimize Design:**
   - Switch to "Optimize Design" workflow
   - Upload design or use example
   - Enter intent: `"Optimize for thermal management"`
   - Click "Run Optimization"
   - **View comparisons:** PCB Layout | Schematic | Thermal

3. **Export to KiCad:**
   - After optimization, scroll to "Export" section
   - Click "Export to KiCad"
   - Download `.kicad_pcb` file
   - Open in KiCad to verify

---

## Troubleshooting

**Backend won't start:**
```bash
# Check if port 8000 is in use
lsof -ti:8000 | xargs kill -9

# Check dependencies
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend won't start:**
```bash
# Make sure you're in dielectric directory
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# Check if app_dielectric.py exists
ls frontend/app_dielectric.py
```

**xAI not working:**
```bash
# Check API key is set
echo $XAI_API_KEY

# Check backend logs for xAI calls
# Should see: 🔌 xAI API Call #1, #2, etc.
```

**KiCad export fails:**
```bash
# Install KiCad MCP server (optional)
git clone https://github.com/lamaalrajih/kicad-mcp.git ~/kicad-mcp
export KICAD_MCP_PATH=~/kicad-mcp
```

---

## Expected Output

### Backend Logs:
```
✅ xAI Client initialized
INFO:     Uvicorn running on http://0.0.0.0:8000
🔌 xAI API Call #1: ...
✅ xAI API call #1 successful
```

### Frontend:
- Streamlit interface at http://localhost:8501
- "Dielectric" title
- Workflow tabs: Generate Design | Optimize Design
- Visualization tabs: PCB Layout | Schematic | Thermal View

---

## Full Documentation

- **Setup Guide:** `HOW_TO_TEST_AND_RUN.md`
- **KiCad MCP Setup:** `KICAD_MCP_SETUP.md`
- **Enhanced xAI Integration:** `ENHANCED_XAI_INTEGRATION.md`

---

## Quick Commands Reference

```bash
# Setup
./setup.sh

# Run everything
./run_complete_system.sh

# Run backend only
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000

# Run frontend only
./run_frontend.sh

# Test system
./test_system.sh

# Check health
curl http://localhost:8000/health
```

---

**Ready to design!** 🎨 Open http://localhost:8501 and start creating PCBs!

