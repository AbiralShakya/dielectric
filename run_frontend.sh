#!/bin/bash

# Run Neuro-Geometric Placer Frontend
# Separate script for frontend to run alongside backend

set -e

echo "🌐 Neuro-Geometric Placer Frontend"
echo "==================================="
echo ""

# Check if in correct directory
if [ ! -f "frontend/app.py" ]; then
    echo "❌ Please run from neuro-geometric-placer directory"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup.sh first"
    exit 1
fi

echo "🎨 Starting Streamlit Frontend..."
echo "Frontend will run on http://localhost:8501"
echo "Open browser to view the interface"
echo ""

# Start frontend
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
