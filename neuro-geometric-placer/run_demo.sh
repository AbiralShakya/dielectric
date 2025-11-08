#!/bin/bash

# Neuro-Geometric Placer Demo Script
# Runs the complete system for HackPrinceton demo

set -e

echo "🔌 Neuro-Geometric Placer Demo"
echo "================================"
echo ""

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
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

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found. Copy .env.example to .env"
    exit 1
fi

echo ""
echo "🧪 Running System Tests..."
echo ""

# Test MCP servers
echo "Testing MCP Servers..."
python test_mcp_servers.py
echo ""

# Test geometry
echo "Testing Geometry Engine..."
python -m pytest tests/test_geometry.py -v --tb=short
echo ""

# Test scoring
echo "Testing Scoring Engine..."
python -m pytest tests/test_scoring.py -v --tb=short
echo ""

# Test Dedalus integration
echo "Testing Dedalus Labs Integration..."
python -c "
from backend.ai.dedalus_client import DedalusClient
client = DedalusClient()
print('✅ Dedalus client initialized')
"
echo ""

echo "🚀 Starting Backend API..."
echo "API will run on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start backend (this will run until interrupted)
python -m backend.api.main
