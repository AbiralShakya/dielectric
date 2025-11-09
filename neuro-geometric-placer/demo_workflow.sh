#!/bin/bash
echo "🎯 Neuro-Geometric Placer - Complete Demo Workflow"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔬 DEMONSTRATING: Natural Language → AI Agents → PCB Layout → Simulator Export${NC}"
echo ""

# Step 1: Start the system
echo -e "${YELLOW}Step 1: Starting AI Agent System${NC}"
echo "----------------------------------"
./run_complete_system.sh &
SYSTEM_PID=$!
sleep 5

# Check if both services are running
if curl -s "http://127.0.0.1:8000/health" > /dev/null && curl -s "http://127.0.0.1:8501" > /dev/null; then
    echo -e "${GREEN}✅ Backend API: http://127.0.0.1:8000${NC}"
    echo -e "${GREEN}✅ Frontend UI: http://127.0.0.1:8501${NC}"
else
    echo -e "${RED}❌ System failed to start${NC}"
    exit 1
fi

echo ""

# Step 2: Test natural language optimization
echo -e "${YELLOW}Step 2: Testing Natural Language → AI Optimization${NC}"
echo "---------------------------------------------------"

echo "📝 Natural Language Input: 'Design a thermal-managed LED circuit'"
OPTIMIZE_RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "board": {"width": 80, "height": 60},
    "components": [
      {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 20, "y": 20, "angle": 0, "placed": true},
      {"name": "LED1", "package": "LED-5MM", "width": 5, "height": 5, "power": 0.1, "x": 50, "y": 30, "angle": 0, "placed": true},
      {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 35, "y": 25, "angle": 0, "placed": true}
    ],
    "nets": [
      {"name": "VCC", "pins": [["U1", "pin8"], ["LED1", "anode"]]},
      {"name": "GND", "pins": [["U1", "pin4"], ["LED1", "cathode"]]}
    ],
    "intent": "Design a thermal-managed LED circuit - prioritize cooling over trace length"
  }')

# Extract key results
SUCCESS=$(echo "$OPTIMIZE_RESPONSE" | jq -r '.success // false')
SCORE=$(echo "$OPTIMIZE_RESPONSE" | jq -r '.score // "N/A"')
METHOD=$(echo "$OPTIMIZE_RESPONSE" | jq -r '.method // "unknown"')
AGENTS=$(echo "$OPTIMIZE_RESPONSE" | jq -r '.agents_used | join(", ") // "none"')

if [ "$SUCCESS" = "true" ]; then
    echo -e "${GREEN}✅ AI Optimization Successful!${NC}"
    echo "   📊 Score: $SCORE"
    echo "   🎨 Method: $METHOD"
    echo "   🤖 Agents: $AGENTS"
else
    echo -e "${RED}❌ AI Optimization Failed${NC}"
    echo "Response: $OPTIMIZE_RESPONSE"
fi

echo ""

# Step 3: Test KiCad export
echo -e "${YELLOW}Step 3: Testing Simulator Export (KiCad)${NC}"
echo "--------------------------------------------"

PLACEMENT_DATA=$(echo "$OPTIMIZE_RESPONSE" | jq '.placement // {"board": {"width": 100, "height": 100}, "components": []}')

KICAD_RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/export/kicad" \
  -H "Content-Type: application/json" \
  -d "{\"placement\": $PLACEMENT_DATA}")

EXPORT_SUCCESS=$(echo "$KICAD_RESPONSE" | jq -r '.success // false')
FILE_SIZE=$(echo "$KICAD_RESPONSE" | jq -r '.size_bytes // 0')

if [ "$EXPORT_SUCCESS" = "true" ]; then
    echo -e "${GREEN}✅ KiCad Export Successful!${NC}"
    echo "   📁 File: optimized_layout.kicad_pcb"
    echo "   📊 Size: $FILE_SIZE bytes"
    echo "   🛠️  Compatible with KiCad EDA Suite"
else
    echo -e "${RED}❌ KiCad Export Failed${NC}"
fi

echo ""

# Step 4: Show workflow summary
echo -e "${YELLOW}Step 4: Complete Workflow Summary${NC}"
echo "-------------------------------------"
echo -e "${BLUE}🎯 Input:${NC} Natural language design requirements"
echo -e "${BLUE}🤖 Processing:${NC} xAI-powered multi-agent optimization"
echo -e "${BLUE}🔧 Output:${NC} Optimized PCB layout with thermal management"
echo -e "${BLUE}📤 Export:${NC} Industry-standard KiCad files for simulation"
echo ""

echo -e "${GREEN}🚀 WORKFLOW COMPLETE!${NC}"
echo ""
echo "🎮 Next Steps for Simulator Integration:"
echo "  1. Download the KiCad file from the frontend"
echo "  2. Open in KiCad: File → Open → optimized_layout.kicad_pcb"
echo "  3. Add schematics and run SPICE simulation: Tools → Simulator"
echo "  4. For thermal analysis: Use OpenFOAM or KiCad plugins"
echo ""

echo -e "${YELLOW}🔗 Access Points:${NC}"
echo "  Frontend UI: http://127.0.0.1:8501"
echo "  API Docs:    http://127.0.0.1:8000/docs"
echo "  Health Check: http://127.0.0.1:8000/health"
echo ""

echo -e "${BLUE}Press Ctrl+C to stop the demo${NC}"

# Wait for user to stop
wait $SYSTEM_PID
