# 🎯 Natural Language Design & Optimization Prompts for Dielectric

## Overview

Dielectric understands natural language prompts for both **designing new PCBs** and **optimizing existing designs**. This guide provides effective prompts organized by use case.

---

## 📐 Design Prompts (Creating New PCBs)

### Simple Designs

#### Basic LED Circuit
```
Design a simple LED driver circuit with:
- One LED (5mm through-hole)
- Current limiting resistor (1kΩ)
- Power supply connector
- Board size: 50mm x 30mm
- Optimize for easy assembly
```

#### Power Supply Module
```
Create a 12V to 5V step-down converter:
- Input: 12V DC jack
- Output: 5V USB connector
- Include filtering capacitors
- Add status LED
- Keep components spaced for thermal management
```

#### Sensor Board
```
Design a temperature sensor board:
- Microcontroller (ATmega328P)
- Temperature sensor (DS18B20)
- Power indicator LED
- 6-pin programming header
- Board size: 40mm x 40mm
```

### Intermediate Designs

#### Audio Amplifier
```
Design a multi-module audio amplifier:
- Power supply: 12V to ±15V dual rail converter
- Analog section: High-performance op-amp with feedback network
- Input: 3.5mm audio jack
- Output: Speaker terminals
- Keep analog and power sections separated for noise isolation
- Optimize for thermal management
```

#### Motor Controller
```
Create a DC motor controller board:
- MCU: ESP32
- Motor driver: H-bridge with current sensing
- Power input: 12V barrel jack
- Control: PWM input and direction pins
- Protection: Fuses and overcurrent detection
- Separate power and control sections
- Optimize trace width for high current paths
```

#### IoT Sensor Node
```
Design an IoT sensor node:
- MCU: ESP8266
- Sensors: Temperature, humidity, light
- Power: Battery connector with charging circuit
- Communication: WiFi antenna
- Status: RGB LED
- Keep RF section isolated from digital circuits
- Optimize for low power consumption
```

### Complex Designs

#### Multi-Module System
```
Design a complete embedded system with:
- Power management: 12V to 3.3V/5V dual output
- Processing: Raspberry Pi Compute Module
- Communication: Ethernet PHY and WiFi module
- Storage: SD card slot
- I/O: GPIO header and USB ports
- Keep modules separated: power, digital, analog, RF
- Optimize for signal integrity and thermal management
- Board size: 100mm x 80mm
```

#### High-Speed Digital Board
```
Create a high-speed digital processing board:
- Processor: ARM Cortex-A53
- Memory: DDR4 RAM with proper routing
- Storage: eMMC and SD card
- Interfaces: USB 3.0, HDMI, Ethernet
- Power: Multiple voltage rails (1.8V, 3.3V, 5V)
- Optimize for signal integrity:
  - Controlled impedance traces
  - Proper ground planes
  - Keep high-speed signals away from noise sources
- Thermal: Add thermal vias under processor
```

#### RF Transceiver Board
```
Design an RF transceiver module:
- RF IC: 2.4GHz transceiver
- Matching network: LC filters and balun
- Antenna: PCB trace antenna
- MCU: Low-power microcontroller
- Power: Battery management circuit
- Keep RF section isolated:
  - Separate ground planes
  - No digital signals near RF
  - Proper shielding considerations
- Optimize for RF performance and low power
```

---

## 🔧 Optimization Prompts (Improving Existing Designs)

### Thermal Management

#### Basic Thermal Optimization
```
Optimize this design for thermal management:
- Spread high-power components evenly
- Increase spacing around hot components
- Minimize thermal hotspots
```

#### Advanced Thermal Optimization
```
Optimize thermal performance:
- Identify and separate power-dense components
- Maximize spacing for components >2W
- Create thermal zones with proper spacing
- Prioritize thermal spreading over trace length
- Add thermal relief areas around hot components
```

### Trace Length Optimization

#### Minimize Trace Length
```
Optimize for minimal trace length:
- Minimize total routing distance
- Keep connected components close together
- Prioritize trace length over other factors
```

#### Balanced Routing
```
Optimize trace routing:
- Minimize trace length while maintaining thermal spacing
- Keep high-speed signals short
- Balance trace length with component spacing
```

### Signal Integrity

#### High-Speed Signal Optimization
```
Optimize for signal integrity:
- Keep high-speed signals away from noise sources
- Minimize trace length for clock signals
- Maintain proper spacing between signal layers
- Reduce crosstalk between nets
```

#### Mixed-Signal Design
```
Optimize mixed-signal layout:
- Separate analog and digital sections
- Keep analog components away from digital noise
- Minimize trace length for analog signals
- Maintain proper grounding
```

### Manufacturing & Assembly

#### DFM Optimization
```
Optimize for manufacturability:
- Ensure minimum clearance between components
- Avoid components too close to board edges
- Optimize for pick-and-place assembly
- Maintain proper component spacing
```

#### Cost Optimization
```
Optimize for cost:
- Minimize board area while maintaining functionality
- Keep components in standard packages
- Optimize for panelization
- Reduce number of layers if possible
```

### Multi-Objective Optimization

#### Balanced Optimization
```
Optimize this design balancing:
- Thermal management (high priority)
- Trace length minimization (medium priority)
- Component clearance (medium priority)
- Keep high-power components cool
- Minimize routing distance
- Maintain proper spacing
```

#### Performance-Focused
```
Optimize for performance:
- Minimize trace length for signal integrity
- Optimize thermal management for reliability
- Maintain proper clearances
- Prioritize signal paths over power distribution
```

#### Reliability-Focused
```
Optimize for reliability:
- Maximize thermal spacing
- Ensure proper component clearances
- Minimize thermal hotspots
- Optimize for long-term operation
- Prioritize thermal management over trace length
```

---

## 🎨 Advanced Prompt Patterns

### Pattern 1: Specify Priorities
```
Optimize this design with priorities:
1. Thermal management (70%)
2. Trace length (20%)
3. Clearance (10%)

Keep high-power components well-spaced and cool.
```

### Pattern 2: Specify Constraints
```
Design a PCB with these constraints:
- Board size: 80mm x 60mm maximum
- Minimum clearance: 0.5mm
- Power components: Keep >5mm apart
- High-speed signals: Keep <20mm length
- Optimize within these constraints
```

### Pattern 3: Specify Modules
```
Design a multi-module system:
- Module 1: Power supply (top-left)
- Module 2: Digital processing (center)
- Module 3: Analog I/O (bottom-right)
- Keep modules separated with clear boundaries
- Optimize each module independently
```

### Pattern 4: Specify Performance Goals
```
Optimize for these performance goals:
- Maximum operating temperature: 70°C
- Signal rise time: <1ns
- Power consumption: Minimize
- Board area: Minimize while meeting other goals
```

---

## 💡 Prompt Writing Tips

### ✅ Good Prompts

1. **Be Specific**
   ```
   ✅ "Optimize for thermal management with high-power components spaced at least 10mm apart"
   ❌ "Make it better"
   ```

2. **Specify Priorities**
   ```
   ✅ "Prioritize thermal management (70%) over trace length (30%)"
   ❌ "Optimize everything"
   ```

3. **Include Constraints**
   ```
   ✅ "Board size: 100mm x 80mm, minimum clearance: 0.5mm"
   ❌ "Make it fit"
   ```

4. **Mention Component Types**
   ```
   ✅ "Keep BGA components cool and well-spaced"
   ❌ "Fix the components"
   ```

5. **Specify Use Case**
   ```
   ✅ "Optimize for automotive environment with high temperature operation"
   ❌ "Make it work"
   ```

### ❌ Avoid These

- Vague requests: "Make it good"
- Conflicting priorities: "Minimize everything"
- Missing context: "Optimize" (without saying what)
- Too many priorities: "Optimize thermal, routing, cost, size, assembly..."

---

## 🔬 Example Workflows

### Workflow 1: Design → Optimize
```
Step 1 (Design):
"Design a simple LED driver with one LED, resistor, and power connector"

Step 2 (Optimize):
"Optimize the design for minimal trace length while maintaining proper spacing"
```

### Workflow 2: Iterative Optimization
```
Iteration 1:
"Optimize for thermal management"

Iteration 2:
"Now optimize trace length while keeping thermal improvements"

Iteration 3:
"Fine-tune clearances and verify all constraints are met"
```

### Workflow 3: Multi-Stage Design
```
Stage 1:
"Design the power supply module with 12V input and 5V output"

Stage 2:
"Add the digital processing section with MCU and memory"

Stage 3:
"Add analog I/O section with proper isolation"

Stage 4:
"Optimize the complete system for thermal and signal integrity"
```

---

## 📊 Prompt Templates

### Design Template
```
Design a [TYPE] PCB with:
- Components: [LIST]
- Board size: [SIZE]
- Requirements: [REQUIREMENTS]
- Constraints: [CONSTRAINTS]
- Optimize for: [PRIORITIES]
```

### Optimization Template
```
Optimize this design for:
- Priority 1: [GOAL] ([PERCENTAGE]%)
- Priority 2: [GOAL] ([PERCENTAGE]%)
- Priority 3: [GOAL] ([PERCENTAGE]%)

Constraints:
- [CONSTRAINT 1]
- [CONSTRAINT 2]

Goals:
- [GOAL 1]
- [GOAL 2]
```

---

## 🎯 Real-World Examples

### Example 1: Smart Home Device
```
Design a smart home sensor hub:
- MCU: ESP32 with WiFi
- Sensors: Temperature, humidity, motion, light
- Power: USB-C with battery backup
- Communication: WiFi antenna
- Status: RGB LED and buzzer
- Keep RF section isolated from sensors
- Optimize for low power consumption
- Board size: 60mm x 40mm
```

### Example 2: Industrial Controller
```
Create an industrial control board:
- Processor: ARM Cortex-M4
- I/O: 16 digital inputs, 8 analog inputs, 8 relay outputs
- Communication: RS485, Ethernet, CAN bus
- Power: 24V input with multiple voltage rails
- Protection: Fuses, TVS diodes, optocouplers
- Separate power, digital, and I/O sections
- Optimize for EMI/EMC compliance
- Thermal: Add heatsinks for power components
```

### Example 3: Medical Device
```
Design a medical monitoring device:
- MCU: Low-power ARM processor
- Sensors: ECG, pulse oximeter, temperature
- Display: Small OLED screen
- Power: Rechargeable battery with charging
- Communication: Bluetooth Low Energy
- Safety: Isolation barriers, proper grounding
- Optimize for reliability and safety
- Maintain medical device standards compliance
```

---

## 🚀 Quick Reference

### Common Optimization Goals
- **Thermal**: "Optimize for thermal management"
- **Routing**: "Minimize trace length"
- **Clearance**: "Ensure proper component spacing"
- **Signal**: "Optimize for signal integrity"
- **Cost**: "Minimize board area"
- **Assembly**: "Optimize for manufacturability"

### Common Design Elements
- **Power**: Power supply, voltage regulators, filtering
- **Processing**: MCU, processor, memory
- **I/O**: Connectors, headers, interfaces
- **Sensors**: Temperature, pressure, motion, etc.
- **Communication**: WiFi, Bluetooth, Ethernet, serial
- **Status**: LEDs, displays, indicators

---

## 📝 Notes

- **Be specific**: More details = better results
- **Set priorities**: Tell Dielectric what matters most
- **Include constraints**: Board size, clearances, etc.
- **Iterate**: Refine designs through multiple optimization passes
- **Use examples**: Reference similar designs if helpful

---

**Dielectric** understands natural language - describe what you want, and it will create optimized PCB designs!

