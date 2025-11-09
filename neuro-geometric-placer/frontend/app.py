"""
Neuro-Geometric Placer - Complete PCB Design Flow

Natural Language → AI Optimization → Visualization → Simulator Export
"""

import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
import time

# Configuration
API_BASE = "http://localhost:8000"


def create_professional_pcb_visualization(
    placement_data: Dict,
    title: str = "PCB Layout",
    show_nets: bool = True,
    show_thermal: bool = True,
    highlight_moved: List[str] = None
) -> go.Figure:
    """
    Create industry-standard PCB visualization like JITX.
    
    Shows: Component footprints, nets/traces, thermal heatmap, design rules.
    """
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])
    nets = placement_data.get("nets", [])
    
    board_width = board.get("width", 100)
    board_height = board.get("height", 100)
    
    fig = go.Figure()
    
    # Draw board outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="black", width=2),
        fillcolor="rgba(245, 245, 245, 0.1)",
        layer="below"
    )
    
    # Draw thermal heatmap if enabled
    if show_thermal and components:
        # Create thermal grid
        grid_size = 50
        x_grid = np.linspace(0, board_width, grid_size)
        y_grid = np.linspace(0, board_height, grid_size)
        thermal_map = np.zeros((grid_size, grid_size))
        
        for comp in components:
            if comp.get("power", 0) > 0:
                x, y = comp.get("x", 0), comp.get("y", 0)
                power = comp.get("power", 0)
                # Gaussian thermal distribution
                for i, gx in enumerate(x_grid):
                    for j, gy in enumerate(y_grid):
                        dist = np.sqrt((gx - x)**2 + (gy - y)**2)
                        thermal_map[j, i] += power * np.exp(-(dist**2) / (2 * 10**2))
        
        # Add thermal heatmap
        fig.add_trace(go.Contour(
            x=x_grid,
            y=y_grid,
            z=thermal_map,
            colorscale="Hot",
            showscale=True,
            name="Thermal",
            opacity=0.3,
            hovertemplate="Temperature: %{z:.2f}°C<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<extra></extra>"
        ))
    
    # Draw nets/traces
    if show_nets and nets:
        for net in nets:
            net_pins = net.get("pins", [])
            if len(net_pins) >= 2:
                # Get component positions for this net
                positions = []
                for pin_ref in net_pins:
                    comp_name = pin_ref[0]
                    comp = next((c for c in components if c.get("name") == comp_name), None)
                    if comp:
                        positions.append([comp.get("x", 0), comp.get("y", 0)])
                
                # Draw trace (Manhattan routing approximation)
                if len(positions) >= 2:
                    for i in range(len(positions) - 1):
                        x0, y0 = positions[i]
                        x1, y1 = positions[i+1]
                        # Manhattan routing
                        fig.add_trace(go.Scatter(
                            x=[x0, x1, x1], y=[y0, y0, y1],
                            mode="lines",
                            line=dict(color="blue", width=2, dash="dash"),
                            name=f"Net: {net.get('name', 'Unknown')}",
                            showlegend=False,
                            hoverinfo="skip"
                        ))
    
    # Draw components as professional footprints
    highlight_moved = highlight_moved or []
    for comp in components:
        x, y = comp.get("x", 0), comp.get("y", 0)
        w, h = comp.get("width", 5), comp.get("height", 5)
        name = comp.get("name", "UNK")
        package = comp.get("package", "Unknown")
        power = comp.get("power", 0)
        angle = comp.get("angle", 0)
        
        # Component color based on power and movement
        if name in highlight_moved:
            color = "rgba(0, 100, 255, 0.7)"  # Blue for moved
            edge_color = "blue"
        elif power > 1.0:
            color = "rgba(255, 100, 0, 0.7)"  # Orange for high power
            edge_color = "red"
        else:
            color = "rgba(100, 200, 100, 0.7)"  # Green for normal
            edge_color = "darkgreen"
        
        # Draw component footprint (rectangle with rotation)
        # For simplicity, we'll draw as rectangle (rotation can be added)
        fig.add_shape(
            type="rect",
            x0=x - w/2, y0=y - h/2, x1=x + w/2, y1=y + h/2,
            fillcolor=color,
            line=dict(color=edge_color, width=2),
            layer="above"
        )
        
        # Component label
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=[f"{name}<br>{package}"],
            textfont=dict(size=10, color="black", family="Arial Black"),
            textposition="middle center",
            showlegend=False,
            hoverinfo="text",
            hovertext=f"Component: {name}<br>Package: {package}<br>Power: {power}W<br>Position: ({x:.1f}, {y:.1f})mm"
        ))
    
    # Update layout for professional EDA appearance
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family="Arial", color="black")
        ),
        xaxis=dict(
            title="X (mm)",
            range=[-5, board_width + 5],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2
        ),
        yaxis=dict(
            title="Y (mm)",
            range=[-5, board_height + 5],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=600,
        hovermode="closest",
        showlegend=False
    )
    
    return fig

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
    # Auto-load LED circuit example
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

# Sidebar with natural language input
with st.sidebar:
    st.header("🎯 Natural Language Design")

    user_intent = st.text_area(
        "Describe Your PCB Design",
        value="Design a simple LED driver circuit with excellent thermal management - prioritize cooling over trace length",
        height=100,
        help="Describe your circuit requirements, optimization goals, and constraints in natural language"
    )

    st.header("🎛️ Board Configuration")

    col1, col2 = st.columns(2)
    with col1:
        board_width = st.slider("Board Width (mm)", 50, 300, 80, 10)
    with col2:
        board_height = st.slider("Board Height (mm)", 50, 300, 60, 10)

    # File upload for custom designs
    st.header("📤 Upload Your Design")
    
    # Template download
    try:
        import os
        template_path = os.path.join(os.path.dirname(__file__), "..", "examples", "template_design.json")
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template_json = f.read()
            st.download_button(
                label="📥 Download JSON Template",
                data=template_json,
                file_name="pcb_design_template.json",
                mime="application/json",
                help="Download a template JSON file to create your own PCB design"
            )
        else:
            # Fallback template
            template_json = json.dumps({
                "board": {"width": 100, "height": 100, "clearance": 0.5},
                "components": [
                    {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": True, "pins": []}
                ],
                "nets": []
            }, indent=2)
            st.download_button(
                label="📥 Download JSON Template",
                data=template_json,
                file_name="pcb_design_template.json",
                mime="application/json",
                help="Download a template JSON file to create your own PCB design"
            )
    except Exception:
        pass  # Skip template if file not found
    
    uploaded_file = st.file_uploader(
        "Upload PCB Design (JSON format)",
        type=["json"],
        help="Upload a JSON file with your PCB design. Format: {board: {...}, components: [...], nets: [...]}"
    )
    
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read()
            design_data = json.loads(file_contents.decode('utf-8'))
            
            # Validate structure
            if "board" in design_data and "components" in design_data:
                st.session_state.board_data = design_data
                st.session_state.uploaded_design = True  # Flag for preview
                st.success(f"✅ Design loaded! {len(design_data.get('components', []))} components, {len(design_data.get('nets', []))} nets")
                
                # Update board dimensions
                if "board" in design_data:
                    board_width = design_data["board"].get("width", 100)
                    board_height = design_data["board"].get("height", 100)
            else:
                st.error("❌ Invalid design format. Expected: {board: {...}, components: [...], nets: [...]}")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please check the format.")
        except Exception as e:
            st.error(f"❌ Error loading design: {str(e)}")
    
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
                    initial_components = st.session_state.board_data["components"] if st.session_state.board_data else [
                        {"name": "U1", "package": "BGA", "width": 10, "height": 10, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": True}
                    ]
                    initial_nets = st.session_state.board_data["nets"] if st.session_state.board_data else []

                    request_data = {
                        "board": {
                            "width": board_width,
                            "height": board_height
                        },
                        "components": initial_components,
                        "nets": initial_nets,
                        "intent": user_intent
                    }

                    # Save initial placement for comparison
                    st.session_state.initial_placement = {
                        "board": {"width": board_width, "height": board_height, "clearance": 0.5},
                        "components": [comp.copy() for comp in initial_components],
                        "nets": [net.copy() for net in initial_nets]
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
            
            # Show uploaded design preview if available
            if "uploaded_design" in st.session_state:
                st.markdown("### 📊 Uploaded Design Preview")
                preview_fig = create_professional_pcb_visualization(
                    st.session_state.board_data,
                    title="Your Uploaded Design",
                    show_nets=True,
                    show_thermal=True
                )
                st.plotly_chart(preview_fig, use_container_width=True)

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
        st.subheader("🤖 Multi-Agent Architecture")
        agents = results.get("agents_used", [])
        cols = st.columns(len(agents))
        for i, agent in enumerate(agents):
            with cols[i]:
                st.success(f"✅ {agent}")
        
        # Computational Geometry Analysis
        if results.get("geometry_data"):
            st.subheader("🔬 Computational Geometry Analysis (xAI Reasoning Input)")
            geometry = results["geometry_data"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MST Length", f"{geometry.get('mst_length', 0):.1f} mm", 
                         help="Minimum Spanning Tree - trace length approximation")
            with col2:
                st.metric("Voronoi Variance", f"{geometry.get('voronoi_variance', 0):.2f}",
                         help="Component distribution uniformity")
            with col3:
                st.metric("Thermal Hotspots", f"{geometry.get('thermal_hotspots', 0)}",
                         help="High-power component regions")
            with col4:
                st.metric("Net Crossings", f"{geometry.get('net_crossings', 0)}",
                         help="Potential trace routing conflicts")
            
            with st.expander("📊 Detailed Computational Geometry Metrics"):
                st.json(geometry)

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
                    # Filter to only numeric scalar values
                    numeric_stats = {}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            # Additional check to ensure it's not a numpy array or list
                            try:
                                # This will fail for lists/arrays
                                float(value)
                                numeric_stats[key] = value
                            except (TypeError, ValueError):
                                continue  # Skip non-numeric values

                    if numeric_stats:
                        for key, value in numeric_stats.items():
                            st.metric(key.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, float) else value)
                    else:
                        st.info("No numeric performance metrics available")
                else:
                    st.info("Detailed stats not available")

        # Show score improvement
        initial_score = results.get("stats", {}).get("initial_score", 0)
        final_score = results.get("score", 0)
        improvement = initial_score - final_score if initial_score > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Score", f"{initial_score:.3f}")
        with col2:
            st.metric("Final Score", f"{final_score:.3f}")
        with col3:
            st.metric("Improvement", f"{improvement:.3f}", delta=f"{improvement:.3f}")

        # Visual comparison
        st.subheader("📊 Before vs After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Before (Initial)")
            # Show original placement with professional visualization
            if "initial_placement" in st.session_state:
                initial_data = st.session_state.initial_placement
                fig = create_professional_pcb_visualization(
                    initial_data,
                    title="Initial Placement",
                    show_nets=True,
                    show_thermal=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No initial placement data available")

        with col2:
            st.markdown("### After (AI Optimized)")
            # Show optimized placement with professional visualization
            if results.get("placement"):
                # Check if components moved
                moved_components = []
                if "initial_placement" in st.session_state:
                    initial_comps = {c["name"]: (c["x"], c["y"]) for c in st.session_state.initial_placement["components"]}
                    for comp in results["placement"]["components"]:
                        initial_pos = initial_comps.get(comp["name"])
                        if initial_pos and (abs(comp["x"] - initial_pos[0]) > 1 or abs(comp["y"] - initial_pos[1]) > 1):
                            moved_components.append(comp["name"])

                fig = create_professional_pcb_visualization(
                    results["placement"],
                    title="AI Optimized Placement",
                    show_nets=True,
                    show_thermal=True,
                    highlight_moved=moved_components
                )
                st.plotly_chart(fig, use_container_width=True)

                if moved_components:
                    st.success(f"🔄 Components optimized: {', '.join(moved_components)}")
                else:
                    st.info("ℹ️ Components are already well-placed. Optimization focused on fine-tuning.")

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
                            use_container_width=True,
                            key="kicad_download"
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

