#!/bin/bash
echo "🧪 Testing Dielectric AI Agent System"
echo "================================================="

# Start server in background
echo "🚀 Starting server..."
./venv/bin/python deploy_simple.py &
SERVER_PID=$!
sleep 3

# Test health
echo -e "\n🏥 Testing health endpoint..."
curl -s "http://127.0.0.1:8000/health" | jq . 2>/dev/null || curl -s "http://127.0.0.1:8000/health"

# Test AI optimization
echo -e "\n🤖 Testing AI optimization..."
RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "board": {"width": 100, "height": 100},
    "components": [
      {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": true}
    ],
    "nets": [],
    "intent": "minimize thermal issues"
  }')

# Parse and display results
echo "Method: $(echo $RESPONSE | jq -r '.method' 2>/dev/null || echo 'N/A')"
echo "AI Driven: $(echo $RESPONSE | jq -r '.ai_driven' 2>/dev/null || echo 'N/A')"  
echo "Agents Used: $(echo $RESPONSE | jq -r '.agents_used[0]' 2>/dev/null || echo 'N/A'), $(echo $RESPONSE | jq -r '.agents_used[1]' 2>/dev/null || echo 'N/A'), $(echo $RESPONSE | jq -r '.agents_used[2]' 2>/dev/null || echo 'N/A')"
echo "Score: $(echo $RESPONSE | jq -r '.score' 2>/dev/null || echo 'N/A')"
echo "Weights - α: $(echo $RESPONSE | jq -r '.weights_used.alpha' 2>/dev/null || echo 'N/A'), β: $(echo $RESPONSE | jq -r '.weights_used.beta' 2>/dev/null || echo 'N/A'), γ: $(echo $RESPONSE | jq -r '.weights_used.gamma' 2>/dev/null || echo 'N/A')"

# Stop server
echo -e "\n🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo -e "\n✅ Test complete!"
echo "🌐 API docs available at: http://127.0.0.1:8000/docs"
