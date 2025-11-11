# 🚀 How to Run Dielectric - Complete Guide

## Quick Start (Easiest Way)

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# 1. First time setup (only needed once)
./setup.sh

# 2. Set your xAI API key
export XAI_API_KEY="your_xai_api_key_here"

# 3. Run everything (backend + frontend)
./run_complete_system.sh
```

Then open: **http://localhost:8501**

---

## Step-by-Step Instructions

### Step 1: Navigate to Directory

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
```

### Step 2: Setup (First Time Only)

```bash
# Create virtual environment and install dependencies
./setup.sh
```

### Step 3: Set API Key

```bash
export XAI_API_KEY="your_xai_api_key_here"
```

**Or create `.env` file:**
```bash
echo "XAI_API_KEY=your_xai_api_key_here" > .env
```

### Step 4: Run the System

**Option A: Run Everything Together (Recommended)**
```bash
./run_complete_system.sh
```

This starts:
- Backend server on http://localhost:8000
- Frontend UI on http://localhost:8501

**Option B: Run Separately (Two Terminals)**

Terminal 1 - Backend:
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
export XAI_API_KEY="your_key"
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Frontend:
```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
./run_frontend.sh
```

---

## Testing

### Test xAI API First

```bash
# Test your xAI API key
export XAI_API_KEY="your_key"
./test_xai_api.sh
```

### Test Backend

```bash
# Test backend health
curl http://localhost:8000/health

# Or use test script
./test_system.sh
```

---

## What You'll See

### Backend (Terminal Output)
```
✅ xAI Client initialized (endpoint: https://api.x.ai/v1/chat/completions)
INFO:     Started server process [12345]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Frontend (Browser)
- Open http://localhost:8501
- You'll see the Dielectric interface
- Two workflows: "Generate Design" and "Optimize Design"
- Visualization tabs: PCB Layout | Schematic | Thermal View

---

## Quick Commands Cheat Sheet

```bash
# Setup (first time)
./setup.sh

# Run everything
./run_complete_system.sh

# Run backend only
source venv/bin/activate
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000

# Run frontend only
./run_frontend.sh

# Test xAI API
./test_xai_api.sh

# Test system
./test_system.sh

# Check health
curl http://localhost:8000/health
```

---

## Troubleshooting

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

**Virtual environment not found:**
```bash
# Create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**API key not set:**
```bash
export XAI_API_KEY="your_key"
```

---

## Next Steps After Running

1. **Open Frontend:** http://localhost:8501
2. **Generate Design:** Enter natural language description
3. **View Visualizations:** Check PCB Layout, Schematic, Thermal tabs
4. **Optimize:** Switch to Optimize Design workflow
5. **Export:** Export to KiCad format

---

**That's it!** Run `./run_complete_system.sh` and open http://localhost:8501 🚀

