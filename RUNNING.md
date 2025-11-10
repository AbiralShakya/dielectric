# 🚀 Neuro-Geometric Placer - AI Agent System

## Quick Start

### 1. Run Locally
```bash
# Make sure you're in the virtual environment
cd /Users/abiralshakya/Documents/hackprinceton2025/neuro-geometric-placer

# Start the server
./venv/bin/python deploy_simple.py

# Server runs on http://127.0.0.1:8000
```

### 2. Test the System
```bash
# Run automated test
./test_ai_agents.sh

# Or test manually
curl "http://127.0.0.1:8000/health"
curl -X POST "http://127.0.0.1:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{"board":{"width":100,"height":100},"components":[{"name":"U1","package":"BGA","width":10,"height":10,"power":2.0,"x":20,"y":20,"angle":0,"placed":true}],"nets":[],"intent":"minimize thermal issues"}'
```

### 3. View API Documentation
Open: http://127.0.0.1:8000/docs

## How It Works

The system uses **3 AI Agents** powered by xAI's Grok API:

1. **IntentAgent**: Converts natural language ("minimize thermal issues") → optimization weights (α, β, γ)
2. **LocalPlacerAgent**: Runs computational geometry optimization using the weights
3. **VerifierAgent**: Checks design rule compliance and manufacturability

## Demo for HackPrinceton

**Your pitch:**
"Our AI agent system features IntentAgent for natural language processing, LocalPlacerAgent for geometry optimization, and VerifierAgent for validation - all powered by xAI's Grok API for intelligent intent understanding."

## Deployment

### Railway.app (Recommended)
1. Go to railway.app
2. Connect your GitHub repo: `AbiralShakya/hackprincetonfall2025`
3. Set environment variables:
   - `XAI_API_KEY=your_key_here`
4. Deploy automatically

### Other Platforms
- **Render**: Build command `pip install -r requirements.txt`, Start `python deploy_simple.py`
- **Heroku**: Set `PYTHONPATH=/app`, Command `python deploy_simple.py`

## Files You Need

For deployment, make sure these files are included:
- `deploy_simple.py` (main server)
- `requirements.txt` 
- `src/backend/` (all AI agent code)
- `.env` (with XAI_API_KEY)

## API Endpoints

- `GET /` - System info
- `GET /health` - Health check  
- `POST /optimize` - AI-powered optimization
- `GET /docs` - Interactive API documentation

That's it! Your AI agent system is ready for HackPrinceton! 🎉
