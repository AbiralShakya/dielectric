#!/bin/bash
# Deploy anywhere - Railway, Render, Heroku, etc.

echo "🚀 Deploying Dielectric..."

# Install dependencies
pip install -r requirements.txt

# Run the simple API server
python deploy_simple.py
