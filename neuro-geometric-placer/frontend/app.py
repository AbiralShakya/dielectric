"""
Neuro-Geometric Placer - Complete PCB Design Flow

Natural Language → AI Optimization → Visualization → Simulator Export
"""

import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import time

# Configuration
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Neuro-Geometric Placer",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 Neuro-Geometric Placer")
st.markdown("**Natural Language → AI PCB Design → Visualization → Simulator Export**")
st.markdown("*Powered by xAI Agents + Computational Geometry*")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "board_data" not in st.session_state:
    st.session_state.board_data = None

# Sidebar with natural language input
with st.sidebar:
    st.header("🎯 Natural Language Design")

    user_intent = st.text_area(
        "Describe Your PCB Design",
        value="Design a simple LED circuit with thermal management - minimize trace length but keep components cool",
        height=100,
        help="Describe your circuit requirements, optimization goals, and constraints in natural language"
    )

    st.header("🎛️ Board Configuration")

    col1, col2 = st.columns(2)
    with col1:
        board_width = st.slider("Board Width (mm)", 50, 300, 100, 10)
    with col2:
        board_height = st.slider("Board Height (mm)", 50, 300, 100, 10)

    # Load example designs
    st.header("📋 Example Designs")
    example_choice = st.selectbox(
        "Load Example:",
        ["", "Simple LED Circuit", "Power Supply Board", "Audio Amplifier", "Sensor Module"]
    )

    if example_choice == "Simple LED Circuit":
        user_intent = "Design a simple LED driver circuit with thermal management - minimize trace length but keep the LED driver cool"
        st.session_state.board_data = {
            "board": {"width": 80, "height": 60, "clearance": 0.5},
            "components": [
                {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 20, "y": 20, "angle": 0, "placed": True},
                {"name": "LED1", "package": "LED-5MM", "width": 5, "height": 5, "power": 0.1, "x": 50, "y": 30, "angle": 0, "placed": True},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 35, "y": 25, "angle": 0, "placed": True},
                {"name": "C1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 35, "y": 35, "angle": 0, "placed": True},
            ],
            "nets": [
                {"name": "VCC", "pins": [["U1", "pin8"], ["LED1", "anode"], ["C1", "pin1"]]},
                {"name": "GND", "pins": [["U1", "pin4"], ["LED1", "cathode"], ["C1", "pin2"]]},
                {"name": "SIGNAL", "pins": [["U1", "pin1"], ["R1", "pin1"], ["LED1", "anode"]]},
            ]
        }
        st.success("LED circuit loaded!")

    elif example_choice == "Power Supply Board":
        user_intent = "Design a DC-DC converter with excellent thermal management - prioritize cooling over trace length"
        st.session_state.board_data = {
            "board": {"width": 120, "height": 80, "clearance": 0.5},
            "components": [
                {"name": "U1", "package": "QFN-16", "width": 4, "height": 4, "power": 2.5, "x": 30, "y": 25, "angle": 0, "placed": True},
                {"name": "L1", "package": "INDUCTOR-10MM", "width": 10, "height": 10, "power": 0.0, "x": 60, "y": 30, "angle": 0, "placed": True},
                {"name": "C1", "package": "CAP-10MM", "width": 10, "height": 10, "power": 0.0, "x": 90, "y": 25, "angle": 0, "placed": True},
                {"name": "C2", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 20, "angle": 0, "placed": True},
            ],
            "nets": [
                {"name": "VIN", "pins": [["U1", "pin1"], ["L1", "pin1"]]},
                {"name": "VOUT", "pins": [["U1", "pin2"], ["L1", "pin2"], ["C1", "pin1"]]},
                {"name": "GND", "pins": [["U1", "pin3"], ["C1", "pin2"], ["C2", "pin2"]]},
            ]
        }
        st.success("Power supply board loaded!")

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Design", "🔧 Optimize", "📤 Export"])

with tab1:
    st.header("🎯 Natural Language PCB Design")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Current Design Intent:")
        st.info(f"📝 {user_intent}")

        if st.button("🚀 Generate AI-Optimized Layout", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Agents working... Converting natural language to PCB layout..."):
                try:
                    # Prepare request data
                    request_data = {
                        "board": {
                            "width": board_width,
                            "height": board_height
                        },
                        "components": st.session_state.board_data["components"] if st.session_state.board_data else [
                            {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": True}
                        ],
                        "nets": st.session_state.board_data["nets"] if st.session_state.board_data else [],
                        "intent": user_intent
                    }

                    # Call AI optimization API
                    response = requests.post(
                        f"{API_BASE}/optimize",
                        json=request_data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.results = results
                        st.success("🎉 AI optimization completed!")
                        st.balloons()
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)

                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out. Try a simpler design or check your connection.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to AI server. Make sure it's running: `./venv/bin/python deploy_simple.py`")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with col2:
        st.markdown("### AI Agent Status")
        if st.session_state.results:
            st.success("✅ IntentAgent: Active")
            st.success("✅ LocalPlacerAgent: Active")
            st.success("✅ VerifierAgent: Active")
        else:
            st.info("🤖 Agents ready to optimize")

        st.markdown("### Design Metrics")
        if st.session_state.board_data:
            components = len(st.session_state.board_data["components"])
            nets = len(st.session_state.board_data["nets"])
            st.metric("Components", components)
            st.metric("Nets", nets)
            st.metric("Board Size", f"{board_width}×{board_height}mm")

with tab2:
    st.header("🔧 AI Optimization Results")

    if not st.session_state.results:
        st.info("👆 Generate an optimized layout first in the Design tab")
    else:
        results = st.session_state.results

        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Success", "✅" if results.get("success") else "❌")
        with col2:
            st.metric("🎨 Method", results.get("method", "unknown"))
        with col3:
            st.metric("🤖 AI Driven", "✅" if results.get("ai_driven") else "❌")
        with col4:
            st.metric("📊 Score", f"{results.get('score', 0):.3f}")

        # AI Agent breakdown
        st.subheader("🤖 AI Agent Performance")
        agents = results.get("agents_used", [])
        cols = st.columns(len(agents))
        for i, agent in enumerate(agents):
            with cols[i]:
                st.success(f"✅ {agent}")

        # Optimization details
        with st.expander("🔍 Optimization Details", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 Intent Understanding")
                intent_expl = results.get("intent", "N/A")
                st.info(intent_expl)

                st.markdown("### ⚖️ Optimization Weights")
                weights = results.get("weights_used", {})
                st.write(f"**Trace Length:** {weights.get('alpha', 0)*100:.0f}%")
                st.write(f"**Thermal:** {weights.get('beta', 0)*100:.0f}%")
                st.write(f"**Clearance:** {weights.get('gamma', 0)*100:.0f}%")

            with col2:
                st.markdown("### 📊 Performance Stats")
                stats = results.get("stats", {})
                if stats:
                    for key, value in stats.items():
                        # Only show numeric metrics, skip lists/arrays
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            st.metric(key.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, float) else value)
                else:
                    st.info("Detailed stats not available")

        # Visual comparison
        st.subheader("📊 Before vs After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Before (Initial)")
            # Show original placement
            if st.session_state.board_data:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.set_xlim(0, board_width)
                ax.set_ylim(0, board_height)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title("Initial Placement")

                for comp in st.session_state.board_data["components"]:
                    x, y = comp["x"], comp["y"]
                    w, h = comp["width"], comp["height"]
                    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                                        fill=True, alpha=0.5, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x, y, comp["name"], ha='center', va='center', fontsize=8, fontweight='bold')

                st.pyplot(fig)

        with col2:
            st.markdown("### After (AI Optimized)")
            # Show optimized placement
            if results.get("placement"):
                fig, ax = plt.subplots(figsize=(6, 6))
                board = results["placement"]["board"]
                ax.set_xlim(0, board["width"])
                ax.set_ylim(0, board["height"])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title("AI Optimized Placement")

                for comp in results["placement"]["components"]:
                    x, y = comp["x"], comp["y"]
                    w, h = comp["width"], comp["height"]
                    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                                        fill=True, alpha=0.5, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, y, comp["name"], ha='center', va='center', fontsize=8, fontweight='bold')

                st.pyplot(fig)

with tab3:
    st.header("📤 Export for Simulators")

    if not st.session_state.results:
        st.info("👆 Generate and optimize a design first")
    else:
        results = st.session_state.results

        st.success("🎉 Your optimized PCB design is ready for export!")

        # Export options
        st.subheader("📋 Export Formats")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🛠️ KiCad (.kicad_pcb)")
            st.markdown("**Compatible with:** KiCad EDA Suite")
            st.markdown("**Use for:** Full PCB design, manufacturing")

            if st.button("📥 Export KiCad File", use_container_width=True):
                try:
                    # Call KiCad export API
                    export_response = requests.post(
                        f"{API_BASE}/export/kicad",
                        json={"placement": results["placement"]},
                        timeout=10
                    )

                    if export_response.status_code == 200:
                        export_data = export_response.json()
                        kicad_content = export_data["content"]
                        filename = export_data["filename"]

                        st.download_button(
                            label="💾 Download KiCad PCB File",
                            data=kicad_content,
                            file_name=filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                        st.success(f"✅ KiCad file generated ({export_data['size_bytes']} bytes)")
                    else:
                        st.error(f"Export failed: {export_response.status_code}")

                except Exception as e:
                    st.error(f"Export error: {str(e)}")

        with col2:
            st.markdown("### 🔧 Altium Designer")
            st.markdown("**Compatible with:** Altium Designer")
            st.markdown("**Use for:** Professional PCB design")
            if st.button("📥 Download Altium File", use_container_width=True):
                st.info("Altium export coming soon...")

        with col3:
            st.markdown("### 📄 JSON Format")
            st.markdown("**Compatible with:** Custom tools, analysis")
            st.markdown("**Use for:** Data analysis, further processing")
            if st.button("📥 Download JSON", use_container_width=True):
                json_data = json.dumps(results, indent=2)
                st.download_button(
                    label="💾 Save JSON File",
                    data=json_data,
                    file_name="optimized_pcb_layout.json",
                    mime="application/json",
                    use_container_width=True
                )

        # Simulator integration guide
        st.subheader("🎮 Simulator Integration Guide")

        with st.expander("🛠️ KiCad + Circuit Simulators", expanded=True):
            st.markdown("""
            ### Step-by-Step Integration:

            1. **Download KiCad file** (above)
            2. **Open in KiCad:**
               - File → Open → Select your .kicad_pcb file
               - The optimized component placement will be loaded

            3. **Add Schematics:**
               - Create or import your circuit schematic
               - Use Annotate → Annotate to match components

            4. **Run Simulations:**
               - **SPICE Simulation:** Tools → Simulator
               - **Signal Integrity:** Tools → External Plugins → Signal Integrity
               - **Thermal Analysis:** Use external tools like OpenFOAM

            5. **Export for Manufacturing:**
               - File → Fabrication Outputs → Gerber Files
               - File → Fabrication Outputs → Drill Files
            """)

        with st.expander("🔬 Professional EDA Tools"):
            st.markdown("""
            ### Altium Designer:
            - Import JSON → Convert to .PcbDoc format
            - Run DRC (Design Rule Check)
            - Use Signal Integrity analysis
            - Export to ODB++ for manufacturing

            ### Cadence/Allegro:
            - Convert JSON to ASCII format
            - Import via File → Import → ASCII
            - Run constraint-driven optimization
            - Use Clarity for signal analysis

            ### Mentor Graphics PADS:
            - Use JSON data for automated placement
            - Run hyperLynx for SI analysis
            - Export to manufacturing formats
            """)

        with st.expander("🧪 Open-Source Simulators"):
            st.markdown("""
            ### ngspice (SPICE):
            ```bash
            # 1. Export netlist from your design
            # 2. Create SPICE deck
            # 3. Run: ngspice your_circuit.cir
            ```

            ### Qucs (Quite Universal Circuit Simulator):
            - Import netlist data
            - Run time-domain and frequency analysis
            - Export results for further processing

            ### OpenFOAM (Thermal):
            - Use component placement data as boundary conditions
            - Run CFD simulations for thermal analysis
            - Visualize temperature distributions
            """)

# Footer
st.markdown("---")
st.markdown("**Neuro-Geometric Placer** - Complete AI PCB Design Flow")
st.markdown("*Natural Language → AI Agents → Computational Geometry → Simulator Export*")
st.markdown("**Built for HackPrinceton 2025** | *xAI Grok + Multi-Agent Architecture*")

