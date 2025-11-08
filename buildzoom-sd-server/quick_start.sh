#!/bin/bash

echo "🚀 BuildZoom AI - Stable Diffusion Quick Start"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run setup script
echo "📁 Setting up models..."
python setup_models.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the server:"
echo "1. source venv/bin/activate"
echo "2. python sd_server.py"
echo ""
echo "Server will be available at: http://localhost:8000"
echo "Test with: curl http://localhost:8000/health"
