#!/usr/bin/env python3
"""
Root entry point - delegates to src/main.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main module
from main import main

if __name__ == "__main__":
    main()
