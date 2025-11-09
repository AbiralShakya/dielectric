#!/bin/bash
echo "🚀 Starting Complete Neuro-Geometric Placer System"
echo "=================================================="
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    pkill -f "deploy_simple.py" 2>/dev/null
    pkill -f "streamlit" 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

echo "📡 Starting AI Agent Backend Server..."
./venv/bin/python deploy_simple.py &
BACKEND_PID=$!
sleep 3

# Check if backend is running
if curl -s "http://127.0.0.1:8000/health" > /dev/null; then
    echo "✅ Backend server running on http://127.0.0.1:8000"
else
    echo "❌ Backend server failed to start"
    exit 1
fi

echo ""
echo "🎨 Starting Frontend Interface..."
./venv/bin/streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 &
FRONTEND_PID=$!
sleep 3

echo "✅ Frontend running on http://127.0.0.1:8501"
echo ""
echo "🌟 SYSTEM READY!"
echo "==============="
echo ""
echo "🎯 Frontend UI: http://127.0.0.1:8501"
echo "🔧 API Docs:    http://127.0.0.1:8000/docs"
echo "🏥 Health Check: http://127.0.0.1:8000/health"
echo ""
echo "🤖 Workflow:"
echo "  1. Open Frontend UI in browser"
echo "  2. Select an example design or write natural language"
echo "  3. Click '🚀 Generate AI-Optimized Layout'"
echo "  4. View optimization results and export to KiCad"
echo "  5. Open exported KiCad file in KiCad for simulation"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
