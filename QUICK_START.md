# 🚀 Quick Start Guide - Dielectric

## Start the Complete System

### Option 1: Using the Startup Script (Recommended)

```bash
cd dielectric
chmod +x run_complete_system.sh
./run_complete_system.sh
```

This will:
- ✅ Start the Backend API server on `http://localhost:8000`
- ✅ Start the Frontend UI on `http://localhost:8501`
- ✅ Handle cleanup on exit (Ctrl+C)

### Option 2: Manual Start

#### Start Backend API:
```bash
cd dielectric
source venv/bin/activate  # If using virtual environment
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend (in another terminal):
```bash
cd dielectric
source venv/bin/activate  # If using virtual environment
streamlit run frontend/app_dielectric.py --server.port 8501
```

## Access Points

Once started:
- 🌐 **Frontend UI:** http://localhost:8501
- 🔧 **API Docs:** http://localhost:8000/docs
- 🏥 **Health Check:** http://localhost:8000/health

## Prerequisites

1. **Virtual Environment:** Make sure you have a venv set up
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables:** Set `XAI_API_KEY` (REQUIRED for natural language processing)
   ```bash
   export XAI_API_KEY=your_key_here
   # Or create a .env file
   ```
   
   **⚠️ IMPORTANT:** XAI_API_KEY is REQUIRED for Dielectric to work properly. Without it:
   - Natural language intent parsing will use basic keyword matching (much less accurate)
   - No intelligent reasoning about geometry data
   - No sophisticated optimization weight generation
   - The core value proposition (natural language PCB design) won't work
   
   Get your API key from: https://console.x.ai/

3. **KiCad (Optional):** For full KiCad integration
   - Install KiCad 9.0+ with Python support
   - Agents will auto-detect and use KiCad if available

## Quick Test

Test the API is running:
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy","service":"Dielectric API"}`

## Stop the System

Press `Ctrl+C` in the terminal where you ran the startup script, or:
```bash
pkill -f "uvicorn.*main"
pkill -f "streamlit"
```

