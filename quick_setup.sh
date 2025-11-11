#!/bin/bash
# Quick Setup Script for Dielectric

echo "🔌 Dielectric Setup"
echo "==================="
echo ""

# Navigate to correct directory
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric

# Activate venv
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set API key (user should set this themselves)
echo ""
echo "🔑 Please set XAI_API_KEY environment variable:"
echo "   export XAI_API_KEY=\"your_key_here\""

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start backend:"
echo "  python3 src/backend/api/main.py"
echo ""
echo "To start frontend (in new terminal):"
echo "  cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric"
echo "  source venv/bin/activate"
echo "  streamlit run frontend/app_dielectric.py --server.port 8501"
echo ""
echo "Or use the run script:"
echo "  ./run_complete_system.sh"

