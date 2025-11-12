# API Endpoints - Complete ✅

**Date:** 2025-01-XX  
**Status:** ✅ **ALL ENDPOINTS IMPLEMENTED**

---

## New API Endpoints Added

All Phase 1 & Phase 2 features are now accessible via REST API endpoints.

### Routing Endpoints

#### `POST /routing/auto`
Auto-route PCB design using autorouter.

**Request:**
```json
{
  "placement": {...},
  "backend": "auto",  // "freerouting", "kicad", "mst", "auto"
  "nets": ["NET1", "NET2"]  // Optional: specific nets to route
}
```

**Response:**
```json
{
  "success": true,
  "routed_nets": 10,
  "total_traces": 25,
  "total_vias": 5,
  "total_length": 150.5,
  "traces": [...],
  "vias": [...],
  "backend": "mst"
}
```

#### `POST /routing/differential-pairs`
Route differential pairs with impedance control.

**Request:**
```json
{
  "placement": {...}
}
```

**Response:**
```json
{
  "success": true,
  "pairs_routed": 2,
  "results": [
    {
      "positive_net": "USB_D+",
      "negative_net": "USB_D-",
      "impedance": 100.0,
      "length_difference": 0.05
    }
  ]
}
```

---

### Manufacturing Endpoints

#### `POST /manufacturing/gerber`
Generate Gerber files for all layers.

**Request:**
```json
{
  "placement": {...},
  "output_dir": "/tmp/gerber",
  "board_name": "my_board"
}
```

**Response:**
```json
{
  "success": true,
  "files": {
    "F.Cu": "/tmp/gerber/my_board-F_Cu.gbr",
    "B.Cu": "/tmp/gerber/my_board-B_Cu.gbr",
    "F.Mask": "/tmp/gerber/my_board-F_Mask.gbr",
    ...
  },
  "output_dir": "/tmp/gerber"
}
```

#### `POST /manufacturing/drill`
Generate Excellon drill file.

**Request:**
```json
{
  "placement": {...},
  "output_path": "/tmp/drill.drl",
  "board_name": "my_board"
}
```

#### `POST /manufacturing/pick-place`
Generate pick-and-place file.

**Request:**
```json
{
  "placement": {...},
  "output_path": "/tmp/pick-place.csv",
  "side": "top",  // "top" or "bottom"
  "format": "csv"  // "csv" or "json"
}
```

#### `POST /manufacturing/jlcpcb/upload`
Upload to JLCPCB for quote.

**Request:**
```json
{
  "gerber_files": {
    "F.Cu": "/path/to/F_Cu.gbr",
    "B.Cu": "/path/to/B_Cu.gbr",
    ...
  },
  "drill_file": "/path/to/drill.drl",
  "bom_file": "/path/to/bom.csv",
  "cpl_file": "/path/to/cpl.csv",
  "board_parameters": {
    "thickness": 1.6,
    "color": "green",
    "quantity": 5
  }
}
```

**Response:**
```json
{
  "success": true,
  "zip_file": "/tmp/jlcpcb_package.zip",
  "quote": {
    "price": 25.50,
    "currency": "USD"
  },
  "order_id": "JLCPCB-12345"
}
```

---

### DRC Endpoints

#### `POST /drc/advanced`
Run advanced Design Rule Checking.

**Request:**
```json
{
  "placement": {...}
}
```

**Response:**
```json
{
  "success": true,
  "violations": [
    {
      "type": "trace_width",
      "severity": "error",
      "message": "Trace width 0.05mm < minimum 0.1mm",
      "trace": {...}
    }
  ],
  "summary": {
    "total_violations": 5,
    "errors": 2,
    "warnings": 3,
    "passed": false
  }
}
```

---

### Simulation Endpoints

#### `POST /simulation/signal-integrity/analyze`
Analyze signal integrity.

**Request:**
```json
{
  "placement": {...},
  "net": "CLK",  // Optional: specific net, or analyze all
  "frequency": 1e9  // Hz
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "net": "CLK",
    "impedance": 50.2,
    "propagation_delay": 0.5,
    "requires_termination": false
  },
  "violations": [...]
}
```

#### `POST /simulation/power-integrity/analyze`
Analyze power integrity.

**Request:**
```json
{
  "placement": {...},
  "power_net": "VCC",  // Optional: specific net
  "current": 1.0  // Amperes
}
```

**Response:**
```json
{
  "success": true,
  "ir_drop": {
    "voltage_drop": 0.05,
    "drop_percent": 1.0,
    "acceptable": true
  },
  "pdn_impedance": {...},
  "current_density": {...}
}
```

#### `POST /simulation/thermal/heatmap`
Generate thermal heat map.

**Request:**
```json
{
  "placement": {...},
  "resolution": 50,
  "ambient_temp": 25.0
}
```

**Response:**
```json
{
  "success": true,
  "heat_map": {
    "x_grid": [...],
    "y_grid": [...],
    "temperature": [[...]],
    "max_temperature": 45.2,
    "hotspots": [...]
  }
}
```

---

### BOM Endpoints

#### `POST /bom/generate`
Generate Bill of Materials.

**Request:**
```json
{
  "placement": {...},
  "include_pricing": true
}
```

**Response:**
```json
{
  "success": true,
  "board_name": "board",
  "items": [
    {
      "part_number": "R0805",
      "designator": "R1,R2,R3",
      "value": "10k",
      "quantity": 3,
      "unit_price": 0.001,
      "supplier": "JLCPCB"
    }
  ],
  "summary": {
    "total_components": 15,
    "total_quantity": 25,
    "total_cost": 5.50
  }
}
```

#### `POST /bom/export`
Export BOM to file.

**Request:**
```json
{
  "bom": {...},
  "output_path": "/tmp/bom.csv",
  "format": "csv"  // "csv" or "json"
}
```

#### `POST /bom/check-availability`
Check component availability.

**Request:**
```json
{
  "bom": {...}
}
```

**Response:**
```json
{
  "all_available": true,
  "unavailable_items": [],
  "low_stock_items": [],
  "total_unavailable": 0
}
```

---

### Variant Endpoints

#### `POST /variants/create`
Create design variant.

**Request:**
```json
{
  "placement": {...},
  "variant_name": "variant_1",
  "modifications": {
    "component_values": {
      "R1": "10k",
      "R2": "20k"
    },
    "populated_components": ["R1", "R2", "C1"]
  }
}
```

---

### Schematic Endpoints

#### `POST /schematic/netlist`
Generate netlist from placement.

**Request:**
```json
{
  "placement": {...},
  "format": "kicad"  // "kicad" or "spice"
}
```

**Response:**
```json
{
  "success": true,
  "format": "kicad",
  "netlist": "(export (version D)..."
}
```

---

## API Usage Example

```python
import requests

API_BASE = "http://localhost:8000"

# 1. Generate design
response = requests.post(f"{API_BASE}/generate", json={
    "description": "LED driver circuit",
    "board_size": {"width": 50, "height": 50}
})
design = response.json()

# 2. Optimize
response = requests.post(f"{API_BASE}/optimize", json={
    "board": design["placement"]["board"],
    "components": design["placement"]["components"],
    "nets": design["placement"]["nets"],
    "intent": "Optimize for thermal management"
})
optimized = response.json()

# 3. Route
response = requests.post(f"{API_BASE}/routing/auto", json={
    "placement": optimized["placement"]
})
routed = response.json()

# 4. Run DRC
response = requests.post(f"{API_BASE}/drc/advanced", json={
    "placement": routed["placement"]
})
drc = response.json()

# 5. Generate BOM
response = requests.post(f"{API_BASE}/bom/generate", json={
    "placement": routed["placement"]
})
bom = response.json()

# 6. Generate manufacturing files
response = requests.post(f"{API_BASE}/manufacturing/gerber", json={
    "placement": routed["placement"],
    "board_name": "led_driver"
})
gerber = response.json()

# 7. Upload to JLCPCB
response = requests.post(f"{API_BASE}/manufacturing/jlcpcb/upload", json={
    "gerber_files": gerber["files"],
    "drill_file": "/tmp/drill.drl",
    "board_parameters": {"thickness": 1.6, "quantity": 5}
})
quote = response.json()
```

---

## Status: ✅ COMPLETE

All endpoints are implemented and ready to use. The xAI API is already integrated - no additional setup needed!

