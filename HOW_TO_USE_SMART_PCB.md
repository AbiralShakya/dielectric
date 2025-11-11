# 🚀 Quick Start: Smart PCB Parsing & Simulation

## 1. Start Backend Server

```bash
cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
source venv/bin/activate
export XAI_API_KEY="your_xai_api_key_here"
uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the complete system script:
```bash
./run_complete_system.sh
```

## 2. Upload & Analyze PCB File

### Using curl:

```bash
# Upload PCB file and analyze (no optimization)
curl -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@/path/to/NFCREAD-001-RevA.kicad_pcb"

# Upload PCB file, analyze, AND optimize
curl -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@/path/to/NFCREAD-001-RevA.kicad_pcb" \
  -F "optimization_intent=Optimize for thermal management and reduce EMI"
```

### Using Python:

```python
import requests

# Upload and analyze
with open("NFCREAD-001-RevA.kicad_pcb", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/upload/pcb",
        files=files
    )
    print(response.json())

# Upload, analyze, and optimize
with open("NFCREAD-001-RevA.kicad_pcb", "rb") as f:
    files = {"file": f}
    data = {"optimization_intent": "Optimize for thermal management"}
    response = requests.post(
        "http://localhost:8000/upload/pcb",
        files=files,
        data=data
    )
    print(response.json())
```

## 3. Run Simulations

### Thermal Simulation:

```bash
curl -X POST "http://localhost:8000/simulate/thermal" \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {
      "board": {"width": 100, "height": 100, "clearance": 0.5},
      "components": [
        {"name": "U1", "package": "SOIC-8", "x": 50, "y": 50, "width": 5, "height": 4, "power": 1.5, "placed": true, "angle": 0}
      ],
      "nets": []
    },
    "ambient_temp": 25.0,
    "board_material": "FR4"
  }'
```

### Signal Integrity Analysis:

```bash
curl -X POST "http://localhost:8000/simulate/signal-integrity" \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {
      "board": {"width": 100, "height": 100, "clearance": 0.5},
      "components": [...],
      "nets": [...]
    },
    "frequency": 100000000
  }'
```

### Power Distribution Network Analysis:

```bash
curl -X POST "http://localhost:8000/simulate/pdn" \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {
      "board": {"width": 100, "height": 100, "clearance": 0.5},
      "components": [...],
      "nets": [...]
    },
    "supply_voltage": 5.0
  }'
```

## 4. Complete Workflow Example

```bash
#!/bin/bash

# 1. Start backend (if not running)
# cd /Users/abiralshakya/Documents/hackprinceton2025/dielectric
# source venv/bin/activate
# export XAI_API_KEY="your_key"
# uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload &

# 2. Upload PCB file and optimize
echo "Uploading PCB file..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@NFCREAD-001-RevA.kicad_pcb" \
  -F "optimization_intent=Optimize for thermal management")

echo "Response:"
echo $RESPONSE | jq .

# 3. Extract optimized placement
OPTIMIZED_PLACEMENT=$(echo $RESPONSE | jq -r '.optimized_placement')

# 4. Run thermal simulation on optimized design
echo "Running thermal simulation..."
curl -X POST "http://localhost:8000/simulate/thermal" \
  -H "Content-Type: application/json" \
  -d "{\"placement\": $OPTIMIZED_PLACEMENT}" | jq .
```

## 5. Python Script Example

```python
#!/usr/bin/env python3
"""
Complete workflow: Upload PCB, optimize, simulate
"""

import requests
import json

API_BASE = "http://localhost:8000"

# 1. Upload PCB file and optimize
print("📤 Uploading PCB file...")
with open("NFCREAD-001-RevA.kicad_pcb", "rb") as f:
    files = {"file": f}
    data = {
        "optimization_intent": "Optimize for thermal management and reduce EMI"
    }
    response = requests.post(
        f"{API_BASE}/upload/pcb",
        files=files,
        data=data
    )

if response.status_code == 200:
    result = response.json()
    print("✅ Upload successful!")
    print(f"   Modules identified: {len(result.get('knowledge_graph', {}).get('modules', {}))}")
    print(f"   Optimization insights: {len(result.get('optimization_insights', []))}")
    
    # 2. Run thermal simulation
    if "optimized_placement" in result:
        print("\n🔥 Running thermal simulation...")
        sim_response = requests.post(
            f"{API_BASE}/simulate/thermal",
            json={
                "placement": result["optimized_placement"],
                "ambient_temp": 25.0,
                "board_material": "FR4"
            }
        )
        
        if sim_response.status_code == 200:
            sim_result = sim_response.json()
            print(f"✅ Max temperature: {sim_result['max_temperature']:.1f}°C")
            print(f"   Hotspots found: {len(sim_result['hotspots'])}")
            print(f"   Recommendations: {sim_result['recommendations']}")
        
        # 3. Run signal integrity analysis
        print("\n📡 Running signal integrity analysis...")
        si_response = requests.post(
            f"{API_BASE}/simulate/signal-integrity",
            json={
                "placement": result["optimized_placement"],
                "frequency": 100e6
            }
        )
        
        if si_response.status_code == 200:
            si_result = si_response.json()
            print(f"✅ Crosstalk risks: {len(si_result['crosstalk_risks'])}")
            print(f"   Reflection risks: {len(si_result['reflection_risks'])}")
            print(f"   Recommendations: {si_result['recommendations']}")
        
        # 4. Run PDN analysis
        print("\n⚡ Running PDN analysis...")
        pdn_response = requests.post(
            f"{API_BASE}/simulate/pdn",
            json={
                "placement": result["optimized_placement"],
                "supply_voltage": 5.0
            }
        )
        
        if pdn_response.status_code == 200:
            pdn_result = pdn_response.json()
            print(f"✅ Power loss: {pdn_result['power_loss']:.3f}W")
            print(f"   Recommendations: {pdn_result['recommendations']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
```

## 6. Quick Test Script

Save as `test_smart_pcb.sh`:

```bash
#!/bin/bash

API_BASE="http://localhost:8000"

echo "🧪 Testing Smart PCB Features"
echo "=============================="

# Check if backend is running
if ! curl -s "$API_BASE/health" > /dev/null; then
    echo "❌ Backend not running. Start it with:"
    echo "   cd dielectric && ./run_complete_system.sh"
    exit 1
fi

echo "✅ Backend is running"

# Test upload endpoint (with example JSON)
echo ""
echo "📤 Testing PCB upload..."
curl -X POST "$API_BASE/upload/pcb" \
  -F "file=@examples/simple_board.json" \
  -F "optimization_intent=Optimize for thermal management" \
  | jq '.success, .knowledge_graph.modules, .optimization_insights'

echo ""
echo "🔥 Testing thermal simulation..."
curl -X POST "$API_BASE/simulate/thermal" \
  -H "Content-Type: application/json" \
  -d '{
    "placement": {
      "board": {"width": 100, "height": 100, "clearance": 0.5},
      "components": [
        {"name": "U1", "package": "SOIC-8", "x": 50, "y": 50, "width": 5, "height": 4, "power": 1.5, "placed": true, "angle": 0}
      ],
      "nets": []
    }
  }' | jq '.success, .max_temperature, .recommendations'

echo ""
echo "✅ All tests complete!"
```

Make it executable:
```bash
chmod +x test_smart_pcb.sh
./test_smart_pcb.sh
```

## 7. Frontend Integration (Coming Soon)

The frontend will have:
- File upload button for `.kicad_pcb` files
- Simulation tabs (Thermal, Signal Integrity, PDN)
- Visualization of simulation results
- Optimization with natural language

## Summary

**Quick Commands:**

```bash
# 1. Start backend
cd dielectric && ./run_complete_system.sh

# 2. Upload PCB file
curl -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@your_file.kicad_pcb" \
  -F "optimization_intent=Optimize for thermal management"

# 3. Simulate thermal
curl -X POST "http://localhost:8000/simulate/thermal" \
  -H "Content-Type: application/json" \
  -d '{"placement": {...}}'
```

That's it! 🚀

