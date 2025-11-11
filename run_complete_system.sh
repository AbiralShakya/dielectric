#!/bin/bash
echo "🚀 Starting Complete Dielectric System"
echo "=================================================="
echo ""

# Check if in correct directory
if [ ! -f "src/backend/api/main.py" ]; then
    echo "❌ Please run from dielectric directory"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    pkill -f "uvicorn.*main" 2>/dev/null
    pkill -f "streamlit" 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

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
    echo "⚠️  XAI_API_KEY not set. Set it with: export XAI_API_KEY=your_key"
    echo "   Or add to .env file"
fi

echo ""
echo "📡 Starting Backend Server..."
# Kill any existing processes on these ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
sleep 2

uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/dielectric_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready (max 15 seconds)
echo "⏳ Waiting for backend to start..."
for i in {1..15}; do
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "✅ Backend server running on http://localhost:8000"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Backend server failed to start after 15 seconds"
        echo "Check logs: tail -f /tmp/dielectric_backend.log"
        exit 1
    fi
    sleep 1
done

echo ""
echo "🎨 Starting Frontend Interface..."
streamlit run frontend/app_dielectric.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
sleep 3

echo "✅ Frontend running on http://localhost:8501"
echo ""
echo "🌟 SYSTEM READY!"
echo "==============="
echo ""
echo "🌐 Frontend UI: http://localhost:8501"
echo "🔧 API Docs:    http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo ""
echo "🤖 Workflow:"
echo "  1. Open Frontend UI in browser"
echo "  2. Select 'Generate Design' workflow"
echo "  3. Enter natural language description"
echo "  4. View PCB Layout, Schematic, and Thermal visualizations"
echo "  5. Switch to 'Optimize Design' to optimize"
echo "  6. Export to KiCad"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
