#!/bin/bash

# Test xAI API Directly
# Tests the xAI (Grok) API with a simple request

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for API key
if [ -z "$XAI_API_KEY" ]; then
    echo -e "${RED}❌ XAI_API_KEY not set${NC}"
    echo "Set it with: export XAI_API_KEY=your_key"
    exit 1
fi

echo "🧪 Testing xAI API Directly"
echo "=============================="
echo ""

# Test 1: Simple test (matching your format)
echo "1. Simple Test (grok-4-latest)..."
RESPONSE=$(curl -s -X POST https://api.x.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a test assistant."
      },
      {
        "role": "user",
        "content": "Testing. Just say hi and hello world and nothing else."
      }
    ],
    "model": "grok-4-latest",
    "stream": false,
    "temperature": 0
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✅ xAI API is working!${NC}"
    echo "$RESPONSE" | python3 -m json.tool | head -30
else
    echo -e "${RED}❌ xAI API test failed${NC}"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo "2. Testing with grok-2-1212 (used by Dielectric)..."
RESPONSE2=$(curl -s -X POST https://api.x.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a PCB design expert."
      },
      {
        "role": "user",
        "content": "What is a PCB?"
      }
    ],
    "model": "grok-2-1212",
    "stream": false,
    "temperature": 0.7
  }')

if echo "$RESPONSE2" | grep -q "choices"; then
    echo -e "${GREEN}✅ grok-2-1212 is working!${NC}"
    echo "$RESPONSE2" | python3 -m json.tool | head -30
else
    echo -e "${YELLOW}⚠️  grok-2-1212 test failed, but grok-4-latest works${NC}"
    echo "$RESPONSE2"
fi

echo ""
echo -e "${GREEN}✅ xAI API tests complete!${NC}"
echo ""
echo "Your API key is working. Dielectric can now use xAI for:"
echo "  - Design generation"
echo "  - Intent processing"
echo "  - Optimization reasoning"
echo "  - Post-optimization analysis"
