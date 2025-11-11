# How to Test and Run Dielectric

## Quick Start Guide

### Prerequisites

1. **Python 3.10+** installed
2. **KiCad 9.0+** installed (optional but recommended)
3. **xAI API Key** (required for AI features)

### Step 1: Setup Environment

```bash
# Navigate to dielectric directory
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Environment Variables

```bash
# Set xAI API key (REQUIRED)
export XAI_API_KEY="your_xai_api_key_here"

# Optional: Set KiCad MCP path (if you installed KiCad MCP server)
export KICAD_MCP_PATH=~/kicad-mcp

# Optional: Set API port (default: 8000)
export API_PORT=8000
```

**Or create a `.env` file:**
```bash
# Create .env file
cat > .env << EOF
XAI_API_KEY=your_xai_api_key_here
KICAD_MCP_PATH=~/kicad-mcp
API_PORT=8000
EOF
```

### Step 3: Start Backend Server

**Option A: Using Python directly**
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
python3 src/backend/api/main.py
```

**Option B: Using uvicorn**
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using the run script**
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
./run_complete_system.sh
```

The backend will start on `http://localhost:8000`

### Step 4: Start Frontend (in a new terminal)

```bash
# Open new terminal
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate

# Start frontend
streamlit run frontend/app_dielectric.py --server.port 8501
```

**Or use the script:**
```bash
./run_frontend.sh
```

The frontend will open at `http://localhost:8501`

### Step 5: Test the System

#### Test 1: Health Check

```bash
# Check if backend is running
curl http://localhost:8000/health

# Should return: {"status":"healthy","service":"Dielectric API"}
```

#### Test 2: Generate Design

1. Open frontend: `http://localhost:8501`
2. Select "Generate Design" workflow
3. Enter a description: `"Design an audio amplifier with op-amp, resistors, and capacitors"`
4. Set board size: 120mm x 80mm
5. Click "Generate Design"
6. Check the visualizations:
   - **PCB Layout** tab: Should show components, pads, traces
   - **Schematic** tab: Should show component symbols and connections
   - **Thermal View** tab: Should show thermal heatmap

#### Test 3: Optimize Design

1. After generating a design, switch to "Optimize Design" workflow
2. Upload the generated design (or use example)
3. Enter optimization intent: `"Optimize for thermal management and signal integrity"`
4. Click "Run Optimization"
5. Check results:
   - **PCB Layout Comparison**: Before/after PCB layouts
   - **Schematic Comparison**: Before/after schematics
   - **Thermal Comparison**: Before/after thermal maps
   - **Computational Geometry Analysis**: Voronoi, MST, Convex Hull

#### Test 4: Export to KiCad

1. After optimization, scroll to "Export" section
2. Click "Export to KiCad"
3. Download the `.kicad_pcb` file
4. Open in KiCad to verify format

### Step 6: Verify xAI Usage

Check backend logs for xAI API calls:
- You should see: `🔌 xAI API Call #1`, `#2`, etc.
- Multiple calls during optimization (design generation, intent processing, optimization reasoning)

### Testing via API (Command Line)

#### Test xAI API Directly

**Using curl (matching your format):**
```bash
export XAI_API_KEY="your_key"

curl https://api.x.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a test assistant."
      },
      {
        "role": "user",
        "content": "Testing. Just say hi and hello world and nothing else."
      }
    ],
    "model": "grok-4-latest",
    "stream": false,
    "temperature": 0
  }'
```

**Or use the test script:**
```bash
# Bash version
./test_xai_api.sh

# Python version
python3 test_xai_api.py
```

#### Generate Design
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Design an audio amplifier",
    "board_size": {"width": 120, "height": 80, "clearance": 0.5}
  }' | python3 -m json.tool
```

#### Optimize Design
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "board": {"width": 100, "height": 100, "clearance": 0.5},
    "components": [
      {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 50, "y": 50, "angle": 0, "placed": true}
    ],
    "nets": [],
    "intent": "Optimize for thermal management"
  }' | python3 -m json.tool
```

#### Export to KiCad
```bash
curl -X POST http://localhost:8000/export/kicad \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {
      "board": {"width": 100, "height": 100, "clearance": 0.5},
      "components": [
        {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 50, "y": 50, "angle": 0, "placed": true}
      ],
      "nets": []
    }
  }' | python3 -m json.tool
```

## Troubleshooting

### Backend Won't Start

**Error: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Error: `XAI_API_KEY not found`**
```bash
# Set the API key
export XAI_API_KEY="your_key_here"

# Or add to .env file
echo "XAI_API_KEY=your_key_here" >> .env
```

**Error: Port 8000 already in use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
export API_PORT=8001
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8001
```

### Frontend Won't Start

**Error: `File does not exist: frontend/app_dielectric.py`**
```bash
# Make sure you're in the dielectric directory
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# Check if file exists
ls -la frontend/app_dielectric.py
```

**Error: `ModuleNotFoundError: No module named 'streamlit'`**
```bash
# Activate venv and install
source venv/bin/activate
pip install streamlit plotly
```

### xAI Not Being Called

**Check logs for:**
- `🔌 xAI API Call #1`, `#2`, etc.
- If you don't see these, check:
  1. `XAI_API_KEY` is set correctly
  2. Backend is using enhanced xAI client
  3. Check backend logs for errors

**Enable verbose logging:**
```bash
# Set environment variable
export DEBUG=1

# Restart backend
python3 src/backend/api/main.py
```

### KiCad Export Fails

**Error: `KiCad MCP Server not available`**
```bash
# Option 1: Install KiCad MCP server (recommended)
git clone https://github.com/lamaalrajih/kicad-mcp.git ~/kicad-mcp
export KICAD_MCP_PATH=~/kicad-mcp

# Option 2: Install KiCad with Python support
# macOS: Install KiCad.app from kicad.org
# Linux: sudo apt-get install kicad kicad-python3
```

## Quick Test Script

Create `test_system.sh`:

```bash
#!/bin/bash

echo "🧪 Testing Dielectric System"
echo "=============================="

# Test 1: Health check
echo ""
echo "1. Testing backend health..."
curl -s http://localhost:8000/health | python3 -m json.tool

# Test 2: Generate design
echo ""
echo "2. Testing design generation..."
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Test audio amplifier",
    "board_size": {"width": 100, "height": 100}
  }' | python3 -m json.tool | head -20

echo ""
echo "✅ Tests complete!"
echo "Open http://localhost:8501 to use the frontend"
```

Make it executable:
```bash
chmod +x test_system.sh
./test_system.sh
```

## Expected Output

### Backend Logs (should show):
```
✅ xAI Client initialized (endpoint: https://api.x.ai/v1/chat/completions)
INFO:     Started server process [12345]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Frontend (should show):
- Streamlit interface at `http://localhost:8501`
- "Dielectric" title
- Workflow selection (Generate Design / Optimize Design)
- Design examples
- Visualization tabs (PCB Layout, Schematic, Thermal)

### During Optimization (should show):
```
🧠 IntentAgent: Processing intent 'Optimize for thermal management'
   📐 Computing computational geometry analysis...
   ✅ Geometry analysis complete: 8 metrics
   🤖 Calling xAI API for weight reasoning...
🔌 xAI API Call #1: https://api.x.ai/v1/chat/completions
   ✅ xAI API call #1 successful
   ✅ xAI returned weights: α=0.300, β=0.600, γ=0.100
🔧 LocalPlacerAgent: Running optimization...
   🚀 Using Enhanced Simulated Annealing with xAI reasoning
   🤖 xAI Reasoning (iter 25): thermal priority
   🤖 xAI Reasoning (iter 50): thermal priority
   ✅ LocalPlacerAgent: Score = 0.4523
```

## Next Steps

1. **Generate a design** using natural language
2. **View circuit visualizations** (PCB Layout, Schematic, Thermal)
3. **Optimize the design** and see before/after comparisons
4. **Export to KiCad** and open in KiCad to verify
5. **Check xAI usage** in backend logs (should see multiple API calls)

## Full System Test

Run everything together:

```bash
# Terminal 1: Backend
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
export XAI_API_KEY="your_key"
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
streamlit run frontend/app_dielectric.py --server.port 8501

# Terminal 3: Test API
curl http://localhost:8000/health
```

Open browser: `http://localhost:8501` and start designing!

