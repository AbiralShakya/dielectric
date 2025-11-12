#!/bin/bash
# Setup script for Dedalus Labs integration

echo "🚀 Setting up Dedalus Labs integration for Dielectric..."

# Load .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  .env file not found. Creating one..."
    touch .env
fi

# Check for API key
if [ -z "$DEDALUS_API_KEY" ]; then
    echo "⚠️  DEDALUS_API_KEY not set in environment or .env file."
    echo ""
    echo "Please add it to .env file:"
    echo "   echo 'DEDALUS_API_KEY=your_key_here' >> .env"
    echo ""
    echo "Or export it:"
    echo "   export DEDALUS_API_KEY=your_key_here"
    exit 1
fi

echo "✅ Found DEDALUS_API_KEY"

# Install dependencies
echo "📦 Installing dependencies..."
pip install openmcp

# Create data directory for database
echo "📁 Creating data directories..."
mkdir -p data
mkdir -p exports

# Test Dedalus connection
echo "🔌 Testing Dedalus connection..."
python3 -c "
from src.backend.agents.dedalus_integration import get_dedalus_deployment
deployment = get_dedalus_deployment()
result = deployment.deploy_to_dedalus()
print(f'✅ Dedalus deployment: {result}')
"

echo "✅ Dedalus Labs setup complete!"
echo ""
echo "To deploy agents to Dedalus:"
echo "  python3 -c 'from src.backend.agents.dedalus_integration import get_dedalus_deployment; get_dedalus_deployment().deploy_to_dedalus()'"

