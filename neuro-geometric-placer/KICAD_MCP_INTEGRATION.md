# KiCAD MCP Server Integration

## Overview

Dielectric now uses the [KiCAD-MCP-Server](https://github.com/mixelpixx/KiCAD-MCP-Server) to generate professional PCB designs using KiCAD's Python API (`pcbnew`) instead of manually generating `.kicad_pcb` files.

## Why This Is Better

### Before (Manual File Generation)
- ❌ Manually constructing KiCad file format strings
- ❌ Limited footprint support
- ❌ No access to KiCAD's design rules
- ❌ Components often don't connect properly
- ❌ Poor quality exports

### After (KiCAD Python API)
- ✅ Uses KiCAD's native Python API (`pcbnew`)
- ✅ Proper footprint library integration
- ✅ Real design rules and DRC support
- ✅ Correct net connections
- ✅ Professional-quality exports
- ✅ Access to 52+ KiCAD tools via MCP

## Installation

### 1. Install KiCAD

**macOS:**
```bash
# Download from https://www.kicad.org/download/macos/
# Or use Homebrew:
brew install --cask kicad
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install kicad kicad-python3
```

**Windows:**
- Download from https://www.kicad.org/download/windows/
- Ensure Python support is enabled during installation

### 2. Verify KiCAD Python API

```bash
python3 -c "import pcbnew; print('KiCAD Python API available!')"
```

If this fails, you need to:
- Add KiCAD's Python path to `PYTHONPATH`
- Or install KiCAD with Python support

### 3. KiCAD-MCP-Server

The KiCAD-MCP-Server is already cloned in `kicad-mcp-server/`. No additional setup needed.

## How It Works

### Architecture

```
Dielectric Placement Data
    ↓
KiCadMCPExporter
    ↓
KiCAD Python API (pcbnew)
    ↓
Professional .kicad_pcb File
```

### Export Flow

1. **Create Board**: Uses `pcbnew.CreateEmptyBoard()`
2. **Set Board Size**: Configures dimensions and outline
3. **Place Components**: 
   - Tries to load footprints from KiCAD libraries
   - Falls back to generic footprints if not found
4. **Create Nets**: Properly connects component pads to nets
5. **Save File**: Uses `board.Save()` for proper file format

### Fallback Behavior

If KiCAD Python API is not available:
- Automatically falls back to manual file generation
- System continues to work (with lower quality exports)
- Warning message is logged

## Usage

The integration is automatic! Just export as normal:

```python
# In deploy_simple.py
kicad_content = generate_kicad_pcb(placement_data)
```

The system will:
1. Try KiCAD MCP exporter first
2. Fall back to manual exporter if needed
3. Return the generated file content

## Available Features

### Component Placement
- ✅ Library footprint loading
- ✅ Generic footprint generation
- ✅ Proper pad placement
- ✅ Component rotation and positioning

### Net Connectivity
- ✅ Net creation and management
- ✅ Pad-to-net connections
- ✅ Multi-pin component support

### Board Setup
- ✅ Board size configuration
- ✅ Edge.Cuts outline generation
- ✅ Design rules (clearance, trace width)
- ✅ Layer management

## Future Enhancements

With KiCAD MCP Server, we can add:
- **Routing**: Automatic trace routing
- **DRC**: Design rule checking
- **3D Export**: 3D model generation
- **Gerber Export**: Manufacturing file generation
- **Schematic Integration**: Link to schematics
- **Library Management**: Search and use footprint libraries

## Troubleshooting

### "KiCAD Python API not available"

**Solution:**
1. Install KiCAD with Python support
2. Add KiCAD Python path to `PYTHONPATH`:
   ```bash
   export PYTHONPATH="/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages:$PYTHONPATH"
   ```
3. Verify: `python3 -c "import pcbnew"`

### "Footprint not found"

**Solution:**
- The system will create generic footprints automatically
- For better results, ensure KiCAD footprint libraries are installed
- Check `kicad-mcp-server/python/commands/library.py` for library management

### Export Quality Issues

**If exports are still poor:**
1. Check KiCAD version (9.0+ recommended)
2. Verify `pcbnew` import works
3. Check logs for errors
4. System will fall back to manual exporter if needed

## References

- [KiCAD-MCP-Server](https://github.com/mixelpixx/KiCAD-MCP-Server)
- [KiCAD Python API Documentation](https://dev-docs.kicad.org/en/python-api/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

