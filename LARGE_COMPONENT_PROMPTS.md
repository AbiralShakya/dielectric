# Large Component Count Prompts

## How to Design PCBs with 50+ Components

### Strategy: Use Hierarchical Descriptions

Instead of listing every component, describe **modules** and **subsystems**. The system will automatically:
1. Create knowledge graph of relationships
2. Identify modules using Voronoi clustering
3. Optimize hierarchically (modules → components)

## Example Prompts

### 1. **High-Density Digital Board (100+ components)**

```
Design a high-density digital processing board with:

**Main Processor Module:**
- ARM Cortex-A78 SoC (BGA-676, 0.8mm pitch)
- Power: 1.0V core @ 15A, 1.8V I/O @ 5A, 3.3V @ 2A
- Decoupling: 50x 100nF ceramic (0402) + 10x 10uF tantalum (0805) near power pins
- Thermal: Heatsink with 20x thermal vias (0.3mm drill)

**Memory Module:**
- 2x DDR4 SDRAM (BGA-96 each, 8GB)
- Memory controller: Integrated in SoC
- Termination: On-die termination (ODT)
- Decoupling: 20x 100nF (0402) per memory module
- Length matching: ±0.1mm for data lines

**Storage Module:**
- 2x eMMC Flash (BGA-153, 64GB each)
- SPI Flash: 1x 256MB (SOIC-8)
- Decoupling: 5x 100nF (0402) per storage device

**Interface Module:**
- USB 3.0: 2x Type-C connectors with ESD protection
- Ethernet: Gigabit PHY (QFN-48) + RJ45 connector
- HDMI: 1x HDMI 2.0 output
- PCIe: 1x Gen3 x4 connector
- Each interface needs: ESD protection, decoupling, termination

**Power Module:**
- Multi-phase buck: 12V → 1.0V @ 15A (6 phases)
- LDOs: 1.8V @ 2A, 3.3V @ 2A
- Power sequencing: Controlled startup
- Current sensing: Sense resistors + op-amps
- Decoupling: 10x 100uF + 20x 100nF per rail

**Clock Module:**
- Main crystal: 24MHz (HC-49) with load capacitors
- RTC crystal: 32.768kHz (SMD) with load capacitors
- Clock distribution: Clock buffers (SOIC-8)

**Total Components: ~150**
- 1x SoC
- 2x DDR4
- 3x Flash
- 5x Connectors
- 3x Power ICs
- 2x Crystals
- 130+ Passives (resistors, capacitors, inductors)

**Constraints:**
- Board: 120mm x 80mm, 8-layer
- High-speed: Controlled impedance 50Ω/100Ω
- Power: Low-impedance power planes
- Thermal: Maximum 85°C junction
- Manufacturing: HDI process, microvias

**Optimize for:**
- Minimize signal path length
- Optimize power delivery network
- Maximize thermal dissipation
- Minimize crosstalk
```

### 2. **IoT Sensor Array (80+ components)**

```
Design an IoT sensor array board with multiple sensor modules:

**Sensor Modules (5x identical):**
Each sensor module contains:
- Temperature sensor (SOIC-8)
- Humidity sensor (QFN-16)
- Accelerometer (LGA-14)
- Magnetometer (QFN-16)
- I2C bus: Pull-up resistors (2.2kΩ, 0805)
- Decoupling: 2x 100nF (0402) per sensor
- ESD protection: TVS diodes (SOD-123)

**Communication Module:**
- WiFi/BLE combo chip (QFN-48)
- Antenna: PCB trace antenna
- RF matching: 50Ω network (resistors, capacitors, inductors)
- RF filtering: Bandpass filters
- Keep-out: 5mm clearance around antenna

**Power Module:**
- Battery: 3.7V Li-ion, 2000mAh
- Charging IC: USB-C with battery management (QFN-20)
- Power path: Battery → LDO → 3.3V rail
- Current sensing: Sense resistor + op-amp
- Low-power mode: <5mA sleep current
- Decoupling: 10x 100uF + 20x 100nF

**MCU Module:**
- Low-power MCU (QFN-32)
- 32kHz crystal for RTC
- 16MHz crystal for main clock
- Flash: 4MB SPI Flash (SOIC-8)
- Debug: SWD connector
- Decoupling: 5x 100nF (0402) near power pins

**Total Components: ~85**
- 5x sensor modules (25 components)
- 1x communication module (15 components)
- 1x power module (20 components)
- 1x MCU module (10 components)
- 15x shared passives

**Constraints:**
- Board: 60mm x 40mm, 4-layer
- RF section: Top layer only, no components below
- Battery: Accessible for replacement
- Antenna clearance: 5mm minimum
- Low power: <10mA average consumption

**Optimize for:**
- Minimize board size
- Maximize sensor accuracy (minimize noise)
- Optimize battery life
- Ensure RF compliance
```

### 3. **Audio Processing Board (60+ components)**

```
Design a professional audio processing board:

**Audio Input Module:**
- 4x ADC (24-bit, 192kHz, SOIC-28)
- Input protection: TVS diodes (SOD-123)
- Anti-aliasing filters: RC networks (resistors, capacitors)
- Decoupling: 3x 100nF (0402) per ADC
- Clock: 24.576MHz crystal with load capacitors

**Audio Processing Module:**
- DSP chip (BGA-256)
- Power: 1.2V @ 3A, 3.3V @ 1A
- Decoupling: 30x 100nF (0402) + 5x 10uF (0805)
- Memory: 2x DDR3 (BGA-96, 512MB each)
- Clock: 100MHz crystal with load capacitors

**Audio Output Module:**
- 4x DAC (24-bit, 192kHz, SOIC-28)
- Reconstruction filters: RC networks
- Output drivers: Op-amps (SOIC-8)
- Decoupling: 3x 100nF (0402) per DAC
- Output protection: Series resistors

**Power Module:**
- Switching regulator: 12V → 1.2V @ 3A (QFN-16)
- LDOs: 3.3V @ 1A (SOIC-8)
- Power sequencing: Controlled startup
- Decoupling: 10x 100uF + 20x 100nF
- Thermal: Vias under power IC

**Interface Module:**
- USB 2.0: Type-B connector
- I2S: 4x connectors for external modules
- Control: I2C bus with pull-ups

**Total Components: ~65**
- 4x ADC (28 components)
- 1x DSP (35 components)
- 4x DAC (28 components)
- 1x Power (15 components)
- 1x Interface (10 components)

**Constraints:**
- Board: 100mm x 80mm, 6-layer
- Analog section: Isolated from digital (3mm clearance)
- Power traces: 0.5mm minimum for 3A
- Signal integrity: Controlled impedance 50Ω
- Thermal: Maximum 70°C

**Optimize for:**
- Minimize noise (analog isolation)
- Maximize signal quality
- Optimize thermal management
- Minimize crosstalk
```

## Key Principles for Large Designs

### 1. **Describe Modules, Not Components**
❌ Bad: "Add 50 capacitors, 30 resistors, 20 ICs..."
✅ Good: "Power module with 10x decoupling capacitors, 5x bulk capacitors, and 2x LDOs"

### 2. **Use Hierarchical Structure**
- Level 1: Modules (Power, Digital, Analog, RF)
- Level 2: Sub-modules (Regulator, Filter, Interface)
- Level 3: Components (ICs, passives, connectors)

### 3. **Specify Relationships**
- "Decoupling capacitors near power pins"
- "Pull-up resistors on I2C bus"
- "Load capacitors for crystal"

### 4. **Include Constraints**
- Board size, layer count
- Trace width, spacing
- Thermal requirements
- Signal integrity needs

## System Behavior

When you use these prompts, the system:

1. **Creates Knowledge Graph**: Understands component relationships
2. **Identifies Modules**: Uses Voronoi clustering to group related components
3. **Optimizes Hierarchically**: 
   - First: Module placement
   - Then: Component placement within modules
4. **Validates Constraints**: Checks spacing, thermal, signal integrity
5. **Exports Properly**: Multi-layer KiCad with correct net connections

## Example: What Happens Internally

```
Input: "Design audio board with 4x ADC, DSP, 4x DAC, power module"

1. DesignGeneratorAgent:
   - Creates 4x ADC modules (each: ADC + decoupling + protection)
   - Creates DSP module (DSP + memory + decoupling)
   - Creates 4x DAC modules (each: DAC + filters + drivers)
   - Creates power module (regulator + LDOs + decoupling)
   → Total: ~65 components

2. Knowledge Graph:
   - Groups: ADC modules, DAC modules, DSP module, Power module
   - Relationships: Power → All modules, DSP → ADC/DAC

3. Module Identification (Voronoi):
   - Automatically clusters: ADC group, DAC group, DSP group, Power group

4. Hierarchical Optimization:
   - Level 1: Place modules (keep analog isolated from digital)
   - Level 2: Place components within each module

5. Validation:
   - Spacing: 3mm between analog and digital
   - Thermal: Vias under power IC
   - Signal integrity: Controlled impedance

6. Export:
   - Multi-layer KiCad with proper net connections
```

---

**Tip**: Start with module descriptions, then add detail. The system handles the complexity!

