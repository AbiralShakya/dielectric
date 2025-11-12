#!/bin/bash
# Quick script to start just the backend server

echo "🚀 Starting Dielectric Backend Server"
echo "======================================"
echo ""

cd "$(dirname "$0")"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup.sh first"
    exit 1
fi

# Check for XAI_API_KEY
if [ -z "$XAI_API_KEY" ]; then
    echo "⚠️  XAI_API_KEY not set (optional but recommended)"
    echo "   Set it with: export XAI_API_KEY=your_key"
    echo ""
fi

# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

echo ""
echo "📡 Starting Backend Server on http://0.0.0.0:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start server
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload

