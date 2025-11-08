#!/bin/bash

echo "🍎 BuildZoom AI - Mac Setup for Stable Diffusion"
echo "================================================"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"

# Install Python if not present
if ! command -v python3 &> /dev/null; then
    echo "📦 Installing Python 3..."
    brew install python@3.11
else
    echo "✅ Python 3 found"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Mac (with MPS support)
echo "🔥 Installing PyTorch for Mac (with Metal/MPS support)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Test PyTorch installation
echo "🧪 Testing PyTorch..."
python3 -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
else:
    print('MPS not supported')
sys.exit(0)
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch test failed"
    exit 1
fi

echo "✅ PyTorch working!"

# Install Stable Diffusion XL Turbo dependencies
echo "🎨 Installing Stable Diffusion XL Turbo..."
pip install diffusers transformers accelerate safetensors

# Install web server dependencies
echo "🌐 Installing web server dependencies..."
pip install fastapi uvicorn python-multipart pillow

# Test FastAPI
echo "🧪 Testing FastAPI..."
python3 -c "
from fastapi import FastAPI
print('✅ FastAPI imported successfully')
"

if [ $? -ne 0 ]; then
    echo "❌ FastAPI test failed"
    exit 1
fi

echo ""
echo "🎉 Setup complete! You now have SDXL Turbo ready."
echo ""
echo "🚀 Next steps:"
echo "1. python sd_server.py  # Downloads ~7GB SDXL Turbo model on first run"
echo "2. Test with: curl -X POST http://localhost:8000/generate -F \"image=@test.jpg\" -F \"prompt=modern kitchen\""
echo ""
echo "💡 SDXL Turbo is super fast - only 4 inference steps!"
echo "📍 Server will run on http://localhost:8000"
