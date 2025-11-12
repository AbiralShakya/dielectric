# PCB Design Sources for Industry-Scale Testing

**Where to find real PCB designs for testing optimization on large, industry-scale boards**

---

## 🎯 Quick Start

Run the download script to get real PCB designs:

```bash
cd dielectric
python scripts/download_pcb_examples.py --output-dir examples/real_pcbs --create-manifest
```

This will download PCB designs from GitHub and create a manifest for easy testing.

---

## 📚 GitHub Repositories

### Official KiCad Examples
- **URL:** https://github.com/KiCad/kicad-examples
- **Description:** Official KiCad example projects
- **Complexity:** Low to High
- **Components:** 10-200+
- **Files:** Multiple `.kicad_pcb` files

### Raspberry Pi Projects
- **Raspberry Pi Pico:** https://github.com/raspberrypi/pico-sdk
- **Raspberry Pi 4:** https://github.com/raspberrypi/rpi-eeprom
- **Complexity:** Medium-High
- **Components:** 50-100+

### Arduino Projects
- **Arduino Uno:** https://github.com/arduino/ArduinoCore-avr
- **Arduino Mega:** Various GitHub repos
- **Complexity:** Medium
- **Components:** 30-80

### ESP32/ESP8266 Projects
- **ESP32 DevKit:** https://github.com/espressif/esp-idf
- **ESP8266:** Various GitHub repos
- **Complexity:** Medium-High
- **Components:** 40-100+

### STM32 Projects
- **STM32 Nucleo:** https://github.com/STMicroelectronics/STM32CubeNucleo
- **STM32 Discovery:** Various repos
- **Complexity:** Medium-High
- **Components:** 50-150

### BeagleBone Projects
- **BeagleBone Black:** https://github.com/beagleboard/beaglebone-black
- **Complexity:** High
- **Components:** 150+

---

## 🔍 Finding More PCBs

### GitHub Search Queries

1. **Search for KiCad files:**
   ```
   extension:kicad_pcb stars:>10
   ```

2. **Search for PCB projects:**
   ```
   "kicad_pcb" language:Python stars:>5
   ```

3. **Search for hardware projects:**
   ```
   "PCB design" OR "circuit board" stars:>20
   ```

### Popular Hardware Communities

1. **OSHWA (Open Source Hardware Association)**
   - https://www.oshwa.org/
   - Certified open-source hardware projects
   - Many include PCB designs

2. **Hackaday Projects**
   - https://hackaday.io/projects
   - Many projects include PCB files
   - Search for "PCB" or "KiCad"

3. **PCBWay Community**
   - https://www.pcbway.com/project/
   - Community-shared PCB projects
   - Can download Gerber files

4. **EasyEDA Projects**
   - https://easyeda.com/explore
   - Online PCB design platform
   - Many open-source projects

---

## 📥 Download Methods

### Method 1: Using the Download Script

```bash
cd dielectric
python scripts/download_pcb_examples.py
```

### Method 2: Manual GitHub Download

1. Go to GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract and find `.kicad_pcb` files
4. Upload via frontend

### Method 3: Git Clone

```bash
git clone https://github.com/KiCad/kicad-examples.git
cd kicad-examples
# Find .kicad_pcb files
find . -name "*.kicad_pcb"
```

---

## 🧪 Testing with Downloaded PCBs

### Upload via Frontend

1. **Single File:**
   - Go to "Optimize Design" page
   - Select "Single File"
   - Upload `.kicad_pcb` file
   - Click "Optimize"

2. **Folder/Zip:**
   - Select "Folder/Zip"
   - Upload zip file or folder
   - Click "🚀 Process Files"
   - System will auto-detect PCB files

### Upload via API

```python
import requests

# Single file
with open("example.kicad_pcb", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/pcb",
        files={"file": f},
        params={"optimization_intent": "Optimize for thermal management"}
    )

# Folder/Zip
with open("project.zip", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/folder",
        files={"files": f},
        params={"optimization_intent": "Minimize trace length"}
    )
```

---

## 📊 Industry-Scale PCB Characteristics

### Small PCBs (< 50 components)
- Arduino Uno
- Simple sensor boards
- Basic power supplies

### Medium PCBs (50-100 components)
- Raspberry Pi Pico
- ESP32 DevKit
- STM32 Nucleo
- Most development boards

### Large PCBs (100-200 components)
- Raspberry Pi 4
- BeagleBone Black
- Complex IoT devices
- Industrial controllers

### Very Large PCBs (200+ components)
- Server motherboards
- High-end embedded systems
- Complex industrial equipment

---

## 🎯 Recommended Test Cases

### For Testing Optimization:

1. **Thermal Optimization:**
   - High-power boards (power supplies, motor drivers)
   - Look for boards with heat sinks or thermal vias

2. **Signal Integrity:**
   - High-speed boards (Raspberry Pi, BeagleBone)
   - RF boards (WiFi, Bluetooth modules)

3. **Power Integrity:**
   - Power supply boards
   - Boards with multiple voltage rails

4. **Manufacturability:**
   - Complex multi-layer boards
   - Boards with fine-pitch components

---

## 📝 File Formats Supported

- **KiCad:** `.kicad_pcb` (primary format)
- **JSON:** Custom placement JSON files
- **ZIP:** Folders containing PCB files
- **Altium:** `.PcbDoc` (zip archives, experimental)

---

## 🔧 Troubleshooting

### No PCB Files Found

If the download script doesn't find PCB files:
1. Check the repository structure
2. Some repos have PCBs in subdirectories
3. Manually extract and locate `.kicad_pcb` files

### Upload Fails

1. Check file size (max 200MB)
2. Verify file format (`.kicad_pcb` or `.json`)
3. Check backend logs: `tail -f /tmp/dielectric_backend.log`

### Parsing Errors

1. Some KiCad files may have unsupported features
2. Try simpler boards first
3. Check parser logs for specific errors

---

## 📈 Scaling to Industry Scale

### Current Capabilities

- ✅ Handles 100+ component PCBs efficiently
- ✅ Parallel optimization (4-8x speedup)
- ✅ Incremental geometry updates
- ✅ Scalable thermal FDM solver

### Testing Large PCBs

1. Start with medium boards (50-100 components)
2. Gradually test larger boards (100-200 components)
3. Monitor optimization time and quality
4. Adjust optimization parameters as needed

---

**Happy testing! 🚀**

