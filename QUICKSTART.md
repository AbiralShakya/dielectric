# Quick Start Guide

## 🚀 Setup (5 minutes)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your XAI_API_KEY
```

## 🧪 Test the Stack

```bash
# Test full pipeline
python test_full_stack.py

# Run unit tests
pytest tests/ -v
```

## 🎯 Run the Application

### Terminal 1: Backend API
```bash
python -m backend.api.main
# API runs on http://localhost:8000
```

### Terminal 2: Frontend UI
```bash
streamlit run frontend/app.py
# UI runs on http://localhost:8501
```

## 📊 Example Usage

1. Open UI in browser: http://localhost:8501
2. Click "Load Example Board"
3. Enter optimization intent: "Optimize for minimal trace length"
4. Click "Optimize (Fast Path)"
5. View results!

## 🔑 API Keys

- **XAI_API_KEY**: Required for Grok reasoning
  - Get from: https://x.ai/api
  
- **DEDALUS_API_KEY**: Optional, for MCP hosting
  - Get from: https://www.dedaluslabs.ai/

## 🐛 Troubleshooting

- **Import errors**: Make sure venv is activated
- **API key errors**: Check .env file exists and has XAI_API_KEY
- **Port conflicts**: Change ports in .env (API_PORT, STREAMLIT_PORT)

## 📚 Next Steps

- Read `README.md` for architecture details
- Check `examples/` for sample boards
- Run `pytest tests/` to verify everything works

