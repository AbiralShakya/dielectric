# 🚀 Quick Start Guide

## Option 1: Run Everything at Once (Recommended)

```bash
cd neuro-geometric-placer
./run_complete_system.sh
```

This will start:
- ✅ Backend API server (http://127.0.0.1:8000)
- ✅ Frontend UI (http://127.0.0.1:8501)

Then open your browser to: **http://127.0.0.1:8501**

---

## Option 2: Run Separately

### Terminal 1: Backend Server
```bash
cd neuro-geometric-placer
./venv/bin/python deploy_simple.py
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2: Frontend UI
```bash
cd neuro-geometric-placer
./venv/bin/streamlit run frontend/app.py --server.port 8501
```

Then open: **http://127.0.0.1:8501**

---

## 🎯 How to Use

### 1. **Upload Your Design**
- Click "📥 Download JSON Template" in sidebar
- Edit the template with your components
- Upload your JSON file

### 2. **Natural Language Optimization**
- Type your intent: "Optimize for thermal management and minimize trace length"
- Click "🚀 Generate AI-Optimized Layout"

### 3. **View Results**
- See computational geometry analysis (Voronoi, MST, thermal hotspots)
- View before/after comparison
- Check multi-agent status

### 4. **Export to KiCad**
- Go to "📤 Export" tab
- Click "📥 Export KiCad File"
- Open in KiCad for simulation

---

## 🔧 Troubleshooting

### Backend won't start?
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
pkill -f "deploy_simple.py"
```

### Frontend won't start?
```bash
# Check if port 8501 is in use
lsof -i :8501

# Kill existing process
pkill -f "streamlit"
```

### API Key Issues?
```bash
# Make sure .env file exists
cat .env

# Should have:
# XAI_API_KEY=your_key
# DEDALUS_API_KEY=your_key (optional)
```

### Import Errors?
```bash
# Make sure you're in the right directory
cd neuro-geometric-placer

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 API Endpoints

- **Health Check**: http://127.0.0.1:8000/health
- **API Docs**: http://127.0.0.1:8000/docs
- **Optimize**: POST http://127.0.0.1:8000/optimize
- **Export KiCad**: POST http://127.0.0.1:8000/export/kicad

---

## 🎤 Demo Flow for HackPrinceton

1. **Upload Design**: Show file upload
2. **Natural Language**: "Optimize for thermal management"
3. **Show Geometry**: Voronoi, MST, thermal hotspots
4. **Multi-Agent**: IntentAgent → LocalPlacerAgent → VerifierAgent
5. **Export**: KiCad file opens correctly

---

## ✅ Quick Test

```bash
# Test backend
curl http://127.0.0.1:8000/health

# Should return: {"status": "healthy"}
```

---

**Ready to optimize PCBs! 🎉**

