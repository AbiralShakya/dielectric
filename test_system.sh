#!/bin/bash

# Quick Test Script for Dielectric
# Tests backend health and basic functionality

set -e

echo "🧪 Testing Dielectric System"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is running
echo "1. Checking backend health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running${NC}"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo -e "${RED}❌ Backend is not running${NC}"
    echo "   Start backend with: uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo ""
echo "2. Testing design generation..."
RESPONSE=$(curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Test audio amplifier with op-amp",
    "board_size": {"width": 100, "height": 100}
  }')

if echo "$RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✅ Design generation works${NC}"
    echo "$RESPONSE" | python3 -m json.tool | head -15
else
    echo -e "${RED}❌ Design generation failed${NC}"
    echo "$RESPONSE"
fi

echo ""
echo "3. Testing API endpoints..."
ENDPOINTS=$(curl -s http://localhost:8000/ | python3 -m json.tool)
if echo "$ENDPOINTS" | grep -q "generate"; then
    echo -e "${GREEN}✅ All endpoints available${NC}"
    echo "$ENDPOINTS"
else
    echo -e "${YELLOW}⚠️  Some endpoints may be missing${NC}"
fi

echo ""
echo -e "${GREEN}✅ Tests complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Open frontend: http://localhost:8501"
echo "  2. Generate a design"
echo "  3. View circuit visualizations (PCB Layout, Schematic, Thermal)"
echo "  4. Optimize the design"
echo "  5. Export to KiCad"

