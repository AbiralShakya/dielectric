# KiCad MCP Server Integration Guide

## Overview

Dielectric now uses the **proper KiCad MCP Server** from [https://github.com/lamaalrajih/kicad-mcp](https://github.com/lamaalrajih/kicad-mcp) to create correct PCB designs with proper KiCad project files.

## Installation

### 1. Install KiCad MCP Server

```bash
# Clone the KiCad MCP server
cd ~
git clone https://github.com/lamaalrajih/kicad-mcp.git
cd kicad-mcp

# Install dependencies
pip install -r requirements.txt

# Or use uv (recommended)
uv pip install -r requirements.txt
```

### 2. Configure Environment

Set the `KICAD_MCP_PATH` environment variable:

```bash
export KICAD_MCP_PATH=~/kicad-mcp
```

Or add to your `.env` file:

```
KICAD_MCP_PATH=~/kicad-mcp
```

### 3. Install KiCad

Make sure KiCad 9.0+ is installed:

- **macOS**: Download from [kicad.org](https://www.kicad.org/download/)
- **Linux**: `sudo apt-get install kicad kicad-python3`
- **Windows**: Download installer from kicad.org

### 4. Verify Installation

```bash
# Check KiCad installation
kicad-cli --version

# Check Python KiCad API
python3 -c "import pcbnew; print('KiCad Python API available')"
```

## Usage

The KiCad MCP client is automatically used when exporting designs:

```python
from src.backend.export.kicad_mcp_exporter import KiCadMCPExporter

exporter = KiCadMCPExporter()
output_path = exporter.export(placement_data)
kicad_content = exporter.get_file_content(output_path)
```

## Features

### Proper PCB Files

The KiCad MCP client creates:
- **`.kicad_pcb`** files with proper format
- **`.kicad_sch`** schematic files
- **`.kicad_pro`** project files
- Proper layer definitions (F.Cu, B.Cu, Edge.Cuts, etc.)
- Correct footprint definitions
- Proper net assignments

### Circuit Visualizations

The frontend now shows:
- **PCB Layout View**: Proper PCB layout with components, pads, traces, board outline
- **Schematic View**: Schematic representation with component symbols and connections
- **Thermal View**: Thermal heatmap overlay

## Integration with xAI

The KiCad MCP server can be used with xAI API key for enhanced design generation:

```python
# Set xAI API key
export XAI_API_KEY=your_key_here

# The enhanced xAI client will use KiCad MCP for proper design generation
```

## Troubleshooting

### KiCad MCP Server Not Found

If you get "KiCad MCP Server not available":

1. Check that `KICAD_MCP_PATH` is set correctly
2. Verify the path exists: `ls $KICAD_MCP_PATH`
3. Check that `main.py` exists in the KiCad MCP directory

### KiCad Python API Not Available

If pcbnew import fails:

1. Make sure KiCad is installed
2. Check Python path includes KiCad's Python modules
3. On macOS: KiCad Python modules are in `/Applications/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.x/lib/python3.x/site-packages/`

### Export Fails

If export fails:

1. Check KiCad MCP server logs
2. Verify placement data has required fields (board, components, nets)
3. Check file permissions for output directory

## Next Steps

1. **Install KiCad MCP Server**: Follow installation steps above
2. **Set Environment Variables**: Configure `KICAD_MCP_PATH`
3. **Test Export**: Try exporting a design from the frontend
4. **View Circuit Visualizations**: Check PCB Layout and Schematic views

## References

- KiCad MCP Server: https://github.com/lamaalrajih/kicad-mcp
- KiCad Documentation: https://docs.kicad.org/
- MCP Protocol: https://modelcontextprotocol.io/

