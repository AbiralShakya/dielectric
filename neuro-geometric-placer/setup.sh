#!/bin/bash

# Setup script for Neuro-Geometric Placer

set -e

echo "🔌 Neuro-Geometric Placer Setup"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# xAI API Key (REQUIRED)
XAI_API_KEY=your_xai_api_key_here

# Dedalus Labs API Key (optional)
DEDALUS_API_KEY=your_dedalus_api_key_here

# Server config
API_PORT=8000
STREAMLIT_PORT=8501
EOF
    echo "⚠️  Please edit .env and add your XAI_API_KEY"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your XAI_API_KEY"
echo "2. Run tests: python test_full_stack.py"
echo "3. Start backend: python -m backend.api.main"
echo "4. Start frontend: streamlit run frontend/app.py"

