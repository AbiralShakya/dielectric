#!/usr/bin/env python3
"""
Quick test setup for BuildZoom AI Stable Diffusion on Mac
Tests basic functionality without downloading large models
"""

import torch
import sys

def test_basic_setup():
    """Test basic PyTorch and hardware setup"""
    print("🧪 Testing BuildZoom AI Stable Diffusion Setup")
    print("=" * 50)

    # Test Python version
    print(f"✅ Python version: {sys.version}")

    # Test PyTorch installation
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        # Test MPS (Metal Performance Shaders) for Mac
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            print(f"✅ MPS (Metal) available: {mps_available}")
            if mps_available:
                print("🎉 Great! Your Mac supports GPU acceleration!")
            else:
                print("⚠️  MPS not available - will use CPU (slower)")
        else:
            print("⚠️  MPS not supported - will use CPU")

    except ImportError:
        print("❌ PyTorch not installed")
        return False

    # Test basic tensor operations
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("🎯 Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🎯 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("🐌 Using CPU (will be slower)")

        # Test tensor creation
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.mm(x, y)
        print("✅ Tensor operations work!")

        print(f"📊 Device: {device}")
        print(f"📊 Tensor shape test: {z.shape}")

    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

    print("\n🎉 Basic setup test PASSED!")
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run: python setup_models.py (downloads ~12GB models)")
    print("3. Run: python sd_server.py")
    print("4. Test: curl http://localhost:8000/health")

    return True

if __name__ == "__main__":
    success = test_basic_setup()
    sys.exit(0 if success else 1)
