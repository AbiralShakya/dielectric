# Complex PCB Design Prompts

## 🎯 Example Prompts for Large/Complex PCBs

### 1. **Multi-Module Audio Amplifier with Power Management**

```
Design a professional audio amplifier PCB with the following requirements:

**Power Module:**
- 12V input power supply with switching regulator (QFN-16 package)
- Output: 3.3V @ 2A for digital section, 5V @ 1A for analog section
- Input filtering: 100uF capacitor, 10mH inductor
- Output filtering: 47uF capacitors on each rail
- Thermal vias under power IC for heat dissipation

**Analog Module:**
- High-performance op-amp (SOIC-8) for audio amplification
- Input impedance: 10kΩ, feedback network with 0.1% tolerance resistors
- Decoupling capacitors: 100nF ceramic + 10uF tantalum near op-amp
- Keep analog section isolated from digital noise
- Minimum 3mm clearance from digital components

**Digital Module:**
- ARM Cortex-M4 MCU (BGA-256 package)
- 32MB SDRAM, 16MB Flash memory
- I2S audio interface
- USB-C connector for programming
- Crystal oscillator: 24MHz with load capacitors
- Decoupling: 10uF + 100nF near each power pin

**Constraints:**
- Board size: 100mm x 80mm, 4-layer stackup
- Minimum trace width: 0.15mm (6 mil)
- Power traces: 0.5mm minimum for 2A current
- Analog traces: Controlled impedance 50Ω
- Keep power and analog sections separated
- Thermal management: Power IC must have thermal vias
- Manufacturing: Standard FR-4, 1.6mm thickness

**Optimization Goals:**
- Minimize trace length for high-speed signals
- Maximize thermal dissipation for power components
- Minimize crosstalk between analog and digital sections
- Optimize for manufacturability (DFM)
```

### 2. **IoT Sensor Node with RF Module**

```
Design an IoT sensor node PCB with integrated RF communication:

**RF Module:**
- 2.4GHz WiFi/BLE module (QFN-48 package)
- Antenna: PCB trace antenna on top layer
- RF matching network: 50Ω impedance
- Keep-out area: 5mm clearance around antenna
- Ground plane: Continuous under RF section

**Sensor Module:**
- Temperature sensor (SOIC-8)
- Humidity sensor (QFN-16)
- Accelerometer (LGA-14)
- I2C bus: 3.3V, 400kHz, pull-up resistors 2.2kΩ
- Sensor placement: Top-left corner, isolated from RF

**Power Module:**
- Battery: 3.7V Li-ion, 500mAh
- Charging IC: USB-C with battery management
- Power path: Battery → LDO → 3.3V rail
- Low-power mode: <10mA sleep current
- Power switch: MOSFET for battery disconnect

**MCU Module:**
- Low-power MCU (QFN-32)
- 32kHz crystal for RTC
- 16MHz crystal for main clock
- Flash: 4MB SPI Flash
- Debug: SWD connector

**Constraints:**
- Board size: 50mm x 30mm, 2-layer (cost-optimized)
- RF section: Must be on top layer, no components below
- Battery placement: Bottom-left, accessible for replacement
- Antenna clearance: 5mm minimum from all components
- Power traces: 0.3mm for 500mA max current
- Signal traces: 0.1mm minimum (HDI process)

**Optimization Goals:**
- Minimize board size while maintaining RF performance
- Optimize battery life (minimize power consumption)
- Maximize sensor accuracy (minimize noise)
- Ensure RF compliance (antenna tuning)
```

### 3. **High-Speed Digital Board with DDR Memory**

```
Design a high-speed digital processing board:

**Processor Module:**
- FPGA/SoC (BGA-676 package, 0.8mm pitch)
- Power: Multiple rails (1.0V core, 1.8V I/O, 3.3V)
- Decoupling: 100+ capacitors (0402, 0603, 0805)
- Thermal: Heatsink mounting holes, thermal vias

**Memory Module:**
- DDR4 SDRAM: 2x 8GB modules (BGA-96 each)
- Memory controller: Integrated in SoC
- Signal integrity: Controlled impedance 50Ω single-ended, 100Ω differential
- Length matching: ±0.1mm for data lines, ±0.05mm for clock
- Termination: On-die termination (ODT)

**Interface Module:**
- PCIe Gen3 x4 connector
- USB 3.0 Type-C connector
- Gigabit Ethernet (RJ45)
- HDMI 2.0 output
- SATA 3.0 connector

**Power Module:**
- Multi-phase buck converter: 12V → 1.0V @ 20A
- LDOs: 1.8V, 3.3V rails
- Power sequencing: Controlled startup/shutdown
- Current monitoring: Sense resistors

**Constraints:**
- Board size: 150mm x 100mm, 8-layer stackup
- High-speed signals: Stripline routing, length matching
- Power delivery: Low impedance power planes
- Thermal: Maximum junction temperature 85°C
- Manufacturing: HDI process, microvias (0.1mm)
- Impedance control: ±5% tolerance

**Optimization Goals:**
- Minimize signal path length for high-speed signals
- Optimize power delivery network (PDN) impedance
- Maximize thermal dissipation
- Minimize crosstalk between high-speed signals
- Ensure signal integrity (SI) compliance
```

### 4. **Motor Control Board with Isolated Power**

```
Design a motor control board with safety isolation:

**Power Module:**
- Mains input: 120VAC/240VAC (connector with safety clearance)
- Isolation: 2kV isolation barrier (creepage: 8mm, clearance: 5mm)
- Power supply: Isolated DC-DC converter (12V output)
- Fuse: 5A fast-blow fuse
- EMI filtering: Common-mode choke, X/Y capacitors

**Control Module:**
- MCU: ARM Cortex-M7 (LQFP-100)
- Gate drivers: 3x isolated gate drivers (SOIC-8)
- Current sensing: 3x shunt resistors + op-amps
- Encoder interface: Differential receiver
- Communication: Isolated CAN transceiver

**Motor Drive Module:**
- Inverter: 3-phase bridge (6x MOSFETs, TO-220 packages)
- Gate resistors: 10Ω for switching control
- Bootstrap capacitors: 10uF for high-side drive
- Snubber circuits: RC networks for EMI reduction
- Thermal: Heatsink with thermal interface material

**Feedback Module:**
- Current sensors: 3x Hall-effect sensors
- Position encoder: Optical encoder interface
- Temperature: NTC thermistor on heatsink
- Overcurrent protection: Comparator circuit

**Constraints:**
- Board size: 120mm x 80mm, 4-layer
- High-voltage section: 8mm minimum clearance from low-voltage
- Isolation barrier: No copper crossing, slot in PCB
- Power traces: 2mm minimum for 10A motor current
- Thermal: Heatsink mounting, thermal vias
- Safety: UL/IEC compliance for isolation

**Optimization Goals:**
- Maximize isolation distance (safety)
- Minimize EMI emissions
- Optimize thermal management
- Minimize switching losses
- Ensure safety compliance
```

### 5. **RF Transceiver with Antenna Array**

```
Design a 5G mmWave RF transceiver board:

**RF Module:**
- Transceiver IC: QFN-64, 28GHz operation
- RF matching: 50Ω impedance, balun networks
- Filters: Bandpass filters for TX/RX
- Switches: RF switches for TDD operation
- Amplifiers: LNA (low-noise) and PA (power)

**Antenna Module:**
- Antenna array: 4x4 patch antenna array
- Antenna spacing: λ/2 at 28GHz (5.4mm)
- Feed network: Corporate feed with power dividers
- Phase shifters: For beam steering
- Antenna placement: Top layer, no components below

**Digital Module:**
- Baseband processor: FPGA (BGA-484)
- ADC/DAC: 14-bit, 500MSPS
- Memory: DDR3 for buffering
- Interface: PCIe Gen2 x4

**Power Module:**
- Multiple rails: 1.2V, 1.8V, 3.3V, 5V
- LDOs: Low-noise for RF sections
- Power sequencing: Controlled startup
- Current: 5A total consumption

**Constraints:**
- Board size: 80mm x 80mm, 6-layer Rogers 4350B
- RF layer: Top layer, continuous ground plane
- Impedance: 50Ω single-ended, 100Ω differential
- Antenna clearance: 10mm minimum from all components
- Manufacturing: HDI, 0.075mm trace/space
- Thermal: Maximum 70°C operating temperature

**Optimization Goals:**
- Maximize antenna gain (optimize array geometry)
- Minimize RF losses (short traces, proper matching)
- Optimize beam pattern (antenna placement)
- Minimize interference (isolation between sections)
- Ensure RF performance (impedance matching)
```

## 🎨 Natural Language Prompts (Simplified)

### For Quick Testing:

```
Design a multi-module PCB with:
- Power supply module: 12V to 3.3V converter with filtering
- Digital module: MCU with memory and crystal
- Analog module: Op-amp circuit with feedback network
- Keep modules separated for noise isolation
- Optimize for thermal management and manufacturability
```

```
Create a complex IoT sensor board with:
- RF communication module (2.4GHz)
- Multiple sensors (temperature, humidity, motion)
- Battery power management
- Low-power MCU
- Optimize for minimal board size and power consumption
```

## 📝 How to Use These Prompts

1. **Copy the prompt** into the "Design Intent" or "Optimization Intent" field
2. **Upload your design** (if optimizing existing) or use "Generate Design"
3. **System will:**
   - Parse the requirements
   - Create knowledge graph of components
   - Identify modules automatically
   - Apply fabrication constraints
   - Optimize hierarchically
   - Validate against constraints
   - Export to KiCad

## 🔧 Advanced Features Demonstrated

- ✅ **Knowledge Graph**: Component relationships and categories
- ✅ **Module Identification**: Automatic clustering
- ✅ **Hierarchical Optimization**: Modules → Components
- ✅ **Fabrication Constraints**: Real-world limits
- ✅ **Computational Geometry**: Voronoi, MST, thermal analysis
- ✅ **Multi-Agent Workflow**: Design → Optimize → Validate → Export

---

**Tip**: Start with simpler prompts and add complexity as needed!

