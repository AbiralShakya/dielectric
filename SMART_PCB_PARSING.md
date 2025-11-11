# 🚀 Smart PCB File Parsing & Context-Aware Optimization

## Overview

Dielectric now intelligently parses **real PCB files** (KiCad, JSON) and uses:
1. **Knowledge Graphs** - Hierarchical understanding of design
2. **Computational Geometry** - Mathematical analysis
3. **xAI Reasoning** - Context understanding
4. **ML Engineering** - Proper abstraction and hierarchy

## Features

### 1. Smart PCB File Parser

**Location:** `src/backend/parsers/smart_pcb_parser.py`

**Capabilities:**
- Parses `.kicad_pcb` files (real KiCad designs)
- Parses JSON placement files
- Extracts components, nets, board geometry
- Builds rich context automatically

**Example:**
```python
from src.backend.parsers.smart_pcb_parser import SmartPCBParser

parser = SmartPCBParser()
context = parser.parse_pcb_file("NFCREAD-001-RevA.kicad_pcb")

# Returns:
# - placement: Component/net data
# - knowledge_graph: Hierarchical modules
# - geometry_analysis: Computational geometry metrics
# - design_context: xAI-understood intent
# - optimization_insights: Recommendations
```

### 2. Knowledge Graph Integration

**What it does:**
- Identifies functional modules (power supply, analog, digital, RF, etc.)
- Builds hierarchy (components → modules → system blocks)
- Assigns thermal zones
- Understands component relationships

**Example output:**
```json
{
  "knowledge_graph": {
    "modules": {
      "Power_Supply_Module": {
        "type": "power_supply",
        "components": ["U1", "C1", "L1"],
        "thermal_zone": "high_power"
      }
    },
    "hierarchy_levels": {
      "0": ["U1", "U2", "R1", ...],  // Components
      "1": ["Power_Supply_Module", ...],  // Modules
      "2": ["Power_Block", ...]  // System blocks
    }
  }
}
```

### 3. xAI Context Understanding

**What it does:**
- Analyzes design intent ("What is this PCB for?")
- Identifies optimization opportunities
- Understands thermal concerns
- Considers signal integrity
- Provides manufacturing insights

**Example:**
```python
design_context = {
  "design_intent": "NFC reader board with power management",
  "optimization_opportunities": [
    "thermal_management",
    "component_spacing",
    "power_distribution"
  ],
  "thermal_concerns": ["high_power_components"],
  "signal_integrity": ["rf_isolation", "trace_length"]
}
```

### 4. Context-Aware Optimization

**How it works:**
1. Parse PCB file → Build knowledge graph
2. Analyze geometry → Extract metrics
3. Use xAI → Understand context
4. Optimize with intent → Natural language prompt

**Example:**
```python
# Upload PCB file and optimize
result = upload_and_optimize_pcb(
    "NFCREAD-001-RevA.kicad_pcb",
    optimization_intent="Optimize for thermal management and reduce EMI"
)

# Returns optimized design with:
# - Original context
# - Optimized placement
# - Optimization metrics
# - Insights
```

## API Endpoints

### Upload & Analyze PCB File

```bash
POST /upload/pcb
Content-Type: multipart/form-data

Parameters:
- file: PCB file (.kicad_pcb or .json)
- optimization_intent: Optional natural language optimization request

Response:
{
  "success": true,
  "parsed_placement": {...},
  "knowledge_graph": {...},
  "geometry_analysis": {...},
  "design_context": {...},
  "optimization_insights": [...],
  "optimized_placement": {...}  // If optimization_intent provided
}
```

### Example Usage

```bash
# Upload and analyze only
curl -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@NFCREAD-001-RevA.kicad_pcb"

# Upload, analyze, and optimize
curl -X POST "http://localhost:8000/upload/pcb" \
  -F "file=@NFCREAD-001-RevA.kicad_pcb" \
  -F "optimization_intent=Optimize for thermal management and reduce component spacing"
```

## How It's Different from JSON Files

### Before (JSON files):
- Just component positions
- No context understanding
- No hierarchy
- No design intent

### Now (Smart parsing):
- **Real PCB files** - Works with actual KiCad designs
- **Knowledge graph** - Understands modules and hierarchy
- **Computational geometry** - Mathematical analysis
- **xAI reasoning** - Understands design context
- **Context-aware** - Optimizes based on understanding

## Integration with Optimization Pipeline

```
PCB File (.kicad_pcb)
    ↓
Smart Parser
    ├─ Parse components/nets
    ├─ Build knowledge graph
    ├─ Analyze geometry
    └─ Use xAI for context
    ↓
Rich Context
    ├─ Knowledge graph (hierarchy)
    ├─ Geometry metrics
    ├─ Design context
    └─ Optimization insights
    ↓
Natural Language Intent
    "Optimize for thermal management"
    ↓
Context-Aware Optimization
    ├─ Uses knowledge graph for module-level optimization
    ├─ Uses geometry for placement
    ├─ Uses xAI for reasoning
    └─ Simulated annealing with context
    ↓
Optimized Design
```

## Example: NFC Reader Board

```python
# Parse NFC reader board
parser = SmartPCBParser()
context = parser.parse_pcb_file("NFCREAD-001-RevA.kicad_pcb")

# Knowledge graph identifies:
# - Power supply module (U1, C1, L1)
# - NFC reader module (U2, antenna components)
# - Control logic module (U3, resistors, caps)

# Geometry analysis finds:
# - Thermal hotspots near power components
# - High component density in power section
# - Long traces for antenna connections

# xAI understands:
# - Design intent: NFC reader with power management
# - Optimization opportunities: Thermal management, EMI reduction
# - Concerns: Power dissipation, RF isolation

# Optimize with natural language:
result = optimize_with_intent(
    context,
    "Optimize for thermal management and reduce EMI"
)

# System automatically:
# - Spreads power components for thermal
# - Isolates RF section
# - Optimizes trace lengths
# - Uses knowledge graph hierarchy
```

## Benefits

1. **Works with Real Designs** - Not just JSON, actual KiCad files
2. **Understands Context** - Knows what the design is for
3. **Hierarchical** - Optimizes at module level, not just components
4. **Intelligent** - Uses xAI to reason about design
5. **Mathematical** - Grounded in computational geometry

## Next Steps

- Add Altium file support
- Add Gerber file parsing
- Enhance knowledge graph with component libraries
- Add more simulation capabilities
- Integrate with manufacturing analysis

