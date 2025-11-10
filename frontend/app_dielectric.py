"""
Dielectric - AI-Powered PCB Design Platform

Clean, professional interface with separate design generation and optimization workflows.
"""

import streamlit as st
import requests
import json
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_visualizer import (
    visualize_voronoi_diagram,
    visualize_minimum_spanning_tree,
    visualize_convex_hull,
    create_geometry_dashboard
)

API_BASE = "http://localhost:8000"

# Professional dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #252526;
        border-right: 1px solid #3e3e42;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #4ec9b0;
        font-size: 24px;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #005a9e;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #252526;
        border-bottom: 1px solid #3e3e42;
    }
    .stTabs [data-baseweb="tab"] {
        color: #cccccc;
    }
    .stTabs [aria-selected="true"] {
        color: #4ec9b0;
        border-bottom: 2px solid #4ec9b0;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Dielectric",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_pcb_plot(placement_data: Dict, title: str = "PCB Layout") -> go.Figure:
    """Create professional PCB visualization."""
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])
    nets = placement_data.get("nets", [])
    
    board_width = board.get("width", 100)
    board_height = board.get("height", 100)
    
    fig = go.Figure()
    
    # Board outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="#4ec9b0", width=2),
        fillcolor="rgba(78, 201, 176, 0.05)",
        layer="below"
    )
    
    # Thermal heatmap
    if components:
        grid_size = 40
        x_grid = np.linspace(0, board_width, grid_size)
        y_grid = np.linspace(0, board_height, grid_size)
        thermal_map = np.zeros((grid_size, grid_size))
        
        for comp in components:
            if comp.get("power", 0) > 0:
                x, y = comp.get("x", 0), comp.get("y", 0)
                power = comp.get("power", 0)
                for i, gx in enumerate(x_grid):
                    for j, gy in enumerate(y_grid):
                        dist = np.sqrt((gx - x)**2 + (gy - y)**2)
                        thermal_map[j, i] += power * np.exp(-(dist**2) / (2 * 15**2))
        
        fig.add_trace(go.Contour(
            x=x_grid, y=y_grid, z=thermal_map,
            colorscale="Hot",
            showscale=True,
            name="Thermal",
            opacity=0.2,
            hovertemplate="Temp: %{z:.2f}°C<extra></extra>"
        ))
    
    # Components
    for comp in components:
        name = comp.get("name", "UNK")
        x = comp.get("x", 0)
        y = comp.get("y", 0)
        width = comp.get("width", 5)
        height = comp.get("height", 5)
        power = comp.get("power", 0)
        
        # Color by power
        if power > 1.0:
            color = "#f48771"
        elif power > 0.1:
            color = "#4ec9b0"
        else:
            color = "#007acc"
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=max(width, height) * 3,
                color=color,
                line=dict(width=1, color="#1e1e1e"),
                opacity=0.8
            ),
            text=name,
            textposition="middle center",
            textfont=dict(size=8, color="#ffffff"),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Package: {comp.get('package', 'Unknown')}<br>Power: {power}W<extra></extra>"
        ))
    
    # Nets
    for net in nets:
        net_pins = net.get("pins", [])
        if len(net_pins) >= 2:
            positions = []
            for pin_ref in net_pins:
                comp_name = pin_ref[0] if isinstance(pin_ref, list) else pin_ref
                comp = next((c for c in components if c.get("name") == comp_name), None)
                if comp:
                    positions.append([comp.get("x", 0), comp.get("y", 0)])
            
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    x0, y0 = positions[i]
                    x1, y1 = positions[i+1]
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode="lines",
                        line=dict(color="#3e3e42", width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=16)),
        xaxis=dict(
            title=dict(text="X (mm)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Y (mm)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        width=900,
        height=600,
        showlegend=False,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc")
    )
    
    return fig


# Sidebar
with st.sidebar:
    st.markdown("## Dielectric")
    st.markdown("AI-Powered PCB Design")
    st.markdown("---")
    
    # Workflow selection
    st.markdown("### Workflow")
    workflow = st.radio(
        "Choose workflow",
        ["Generate Design", "Optimize Design"],
        help="Generate: Create new PCB from natural language\nOptimize: Optimize existing design"
    )

# Initialize session state
if "design_data" not in st.session_state:
    st.session_state.design_data = None
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

# Main content
st.markdown("# Dielectric")
st.markdown("Computational Geometry + Multi-Agent AI for PCB Design")
st.markdown("---")

if workflow == "Generate Design":
    st.markdown("## Generate PCB Design from Natural Language")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        design_description = st.text_area(
            "Describe your PCB design",
            height=150,
            placeholder="e.g., Design an audio amplifier circuit with power supply. Include op-amp, resistors, capacitors, and power management IC. Optimize for low noise and thermal efficiency.",
            help="Describe the PCB you want to create in natural language"
        )
        
        col_size1, col_size2 = st.columns(2)
        with col_size1:
            board_width = st.number_input("Board Width (mm)", 50, 500, 120, 10)
        with col_size2:
            board_height = st.number_input("Board Height (mm)", 50, 500, 80, 10)
        
        if st.button("Generate Design", type="primary", use_container_width=True):
            if not design_description:
                st.error("Please enter a design description")
            else:
                with st.spinner("Generating PCB design from natural language..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/generate",
                            json={
                                "description": design_description,
                                "board_size": {"width": board_width, "height": board_height, "clearance": 0.5}
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.design_data = result.get("placement")
                            st.success("Design generated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    with col2:
        st.markdown("### Design Examples")
        examples = {
            "Audio Amplifier": "Design an audio amplifier with op-amp, input/output capacitors, and power supply filtering",
            "Power Supply": "Create a switching power supply with buck converter IC, inductor, capacitors, and feedback resistors",
            "Sensor Module": "Design a sensor interface board with ADC, voltage reference, and signal conditioning",
            "MCU Board": "Create a microcontroller board with MCU, crystal oscillator, decoupling capacitors, and programming header"
        }
        
        for name, desc in examples.items():
            if st.button(name, use_container_width=True, key=f"example_{name}"):
                st.session_state.example_description = desc
                st.rerun()
        
        if "example_description" in st.session_state:
            design_description = st.session_state.example_description
    
    if st.session_state.design_data:
        st.markdown("---")
        st.markdown("### Generated Design")
        fig = create_pcb_plot(st.session_state.design_data, "Generated PCB Design")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Components", len(st.session_state.design_data.get("components", [])))
        with col2:
            st.metric("Nets", len(st.session_state.design_data.get("nets", [])))
        with col3:
            board = st.session_state.design_data.get("board", {})
            st.metric("Board Size", f"{board.get('width', 0)}×{board.get('height', 0)}mm")

else:  # Optimize Design
    st.markdown("## Optimize Existing Design")
    
    # Design input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Load Design")
        uploaded_file = st.file_uploader("Upload PCB Design (JSON)", type=["json"])
        
        if uploaded_file:
            try:
                design_data = json.loads(uploaded_file.read().decode('utf-8'))
                if "board" in design_data and "components" in design_data:
                    st.session_state.design_data = design_data
                    st.success(f"Loaded: {len(design_data.get('components', []))} components")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Examples
        st.markdown("### Examples")
        if st.button("iPhone Speaker & Siri", use_container_width=True):
            try:
                import os
                example_path = os.path.join(os.path.dirname(__file__), "..", "examples", "iphone_speaker_siri.json")
                if os.path.exists(example_path):
                    with open(example_path, "r") as f:
                        st.session_state.design_data = json.load(f)
                        st.success("Example loaded")
                        st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.session_state.design_data:
            fig = create_pcb_plot(st.session_state.design_data, "Current Design")
            st.plotly_chart(fig, use_container_width=True)
    
    # Optimization
    if st.session_state.design_data:
        st.markdown("---")
        st.markdown("### Optimization Settings")
        
        optimization_intent = st.text_input(
            "Optimization Intent",
            value="Optimize for thermal management and signal integrity",
            placeholder="e.g., Minimize trace length, prioritize thermal cooling, ensure design rule compliance",
            help="Describe how you want to optimize the design"
        )
        
        if st.button("Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Optimizing design with AI agents..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/optimize",
                        json={
                            "board": st.session_state.design_data["board"],
                            "components": st.session_state.design_data["components"],
                            "nets": st.session_state.design_data.get("nets", []),
                            "intent": optimization_intent
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        st.session_state.optimization_results = response.json()
                        st.success("Optimization complete!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        
        # Results
        if st.session_state.optimization_results:
            st.markdown("---")
            st.markdown("### Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Before")
                fig = create_pcb_plot(st.session_state.design_data, "Initial Design")
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("#### After")
                if st.session_state.optimization_results.get("placement"):
                    fig = create_pcb_plot(st.session_state.optimization_results["placement"], "Optimized Design")
                    st.plotly_chart(fig, width='stretch')
            
            # Multi-Agent Workflow Status
            agents_used = st.session_state.optimization_results.get("agents_used", [])
            if agents_used:
                st.markdown("---")
                st.markdown("### 🤖 Multi-Agent Workflow")
                st.markdown("**Why this matters:** Each agent specializes in one task, working together like a team of engineers.")
                
                agent_descriptions = {
                    "IntentAgent": "🧠 Understands your goals using computational geometry + xAI",
                    "LocalPlacerAgent": "⚡ Optimizes placement (deterministic, fast)",
                    "VerifierAgent": "✅ Validates design rules and constraints",
                    "ErrorFixerAgent": "🔧 Automatically fixes violations (agentic!)",
                    "DesignGeneratorAgent": "🎨 Creates designs from natural language"
                }
                
                cols = st.columns(len(agents_used))
                for i, agent in enumerate(agents_used):
                    with cols[i]:
                        st.markdown(f"**{agent}**")
                        if agent in agent_descriptions:
                            st.caption(agent_descriptions[agent])
                        st.success("✅ Active")
            
            # Computational Geometry Visualizations - THE MAIN DIFFERENTIATOR
            geometry_data = st.session_state.optimization_results.get("geometry_data")
            placement_data = st.session_state.optimization_results.get("placement", st.session_state.design_data)
            
            if placement_data and len(placement_data.get("components", [])) >= 2:
                st.markdown("---")
                st.markdown("### 🔬 Computational Geometry Analysis")
                st.markdown("**This is what makes Dielectric unique:** We use computational geometry to understand PCB layouts, then feed this structured data to xAI for reasoning.")
                
                # Geometry metrics
                if geometry_data:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MST Length", f"{geometry_data.get('mst_length', 0):.1f} mm",
                                 help="Minimum Spanning Tree - optimal trace length estimate")
                    with col2:
                        st.metric("Voronoi Variance", f"{geometry_data.get('voronoi_variance', 0):.2f}",
                                 help="Component distribution uniformity (lower = better)")
                    with col3:
                        st.metric("Thermal Hotspots", f"{geometry_data.get('thermal_hotspots', 0)}",
                                 help="High-power component regions")
                    with col4:
                        st.metric("Net Crossings", f"{geometry_data.get('net_crossings', 0)}",
                                 help="Potential routing conflicts")
                
                # Geometry visualizations
                viz_tabs = st.tabs(["Dashboard", "Voronoi", "MST", "Convex Hull"])
                
                with viz_tabs[0]:
                    st.markdown("#### Complete Geometry Dashboard")
                    
                    # Interpretation guide
                    with st.expander("📖 How to Read This Dashboard", expanded=False):
                        st.markdown("""
                        **What Each Visualization Shows:**
                        
                        **1. Voronoi Diagram (Top-Left)**
                        - **Purpose**: Shows component distribution and identifies modules
                        - **What to look for**: 
                          - Large regions = sparse areas (good for routing)
                          - Small regions = dense clusters (potential thermal issues)
                          - Uniform regions = balanced layout
                        - **Action**: If regions are very uneven, components may need repositioning
                        
                        **2. Minimum Spanning Tree (Top-Right)**
                        - **Purpose**: Shows optimal trace routing paths
                        - **What to look for**:
                          - Short lines = components are well-connected
                          - Long lines = components are far apart (increases trace length)
                          - Crossed lines = potential routing conflicts
                        - **Action**: Minimize total MST length for better signal integrity
                        
                        **3. Convex Hull (Bottom-Left)**
                        - **Purpose**: Shows board space utilization
                        - **What to look for**:
                          - Small hull = efficient space usage
                          - Large hull with empty center = wasted space
                        - **Action**: Optimize component placement to reduce wasted board area
                        
                        **4. Thermal Heatmap (Bottom-Right)**
                        - **Purpose**: Shows temperature distribution from power-dissipating components
                        - **Color scale**: 
                          - 🔴 **Red/Hot** = High temperature zones (thermal hotspots)
                          - 🟡 **Yellow** = Moderate temperature
                          - 🔵 **Blue/Cold** = Low temperature (good for sensitive components)
                        - **What to look for**:
                          - Concentrated red spots = thermal hotspots (needs cooling)
                          - Even distribution = good thermal management
                        - **Action**: 
                          - Move high-power components apart
                          - Add thermal vias or heat sinks in red zones
                          - Keep sensitive components in blue zones
                        """)
                    
                    fig = create_geometry_dashboard(placement_data, geometry_data)
                    st.plotly_chart(fig, width='stretch')
                
                with viz_tabs[1]:
                    st.markdown("#### Voronoi Diagram - Component Distribution")
                    st.caption("Shows component clustering and distribution uniformity. Used to identify modules automatically.")
                    fig = visualize_voronoi_diagram(
                        placement_data.get("components", []),
                        placement_data.get("board", {}).get("width", 100),
                        placement_data.get("board", {}).get("height", 100)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with viz_tabs[2]:
                    st.markdown("#### Minimum Spanning Tree - Optimal Routing")
                    st.caption("Shows the minimum trace length needed to connect all components. Used to optimize routing.")
                    fig = visualize_minimum_spanning_tree(
                        placement_data.get("components", []),
                        placement_data.get("nets", []),
                        placement_data.get("board", {}).get("width", 100),
                        placement_data.get("board", {}).get("height", 100)
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with viz_tabs[3]:
                    st.markdown("#### Convex Hull - Board Utilization")
                    st.caption("Shows how efficiently the board space is used. Helps identify wasted space.")
                    fig = visualize_convex_hull(
                        placement_data.get("components", []),
                        placement_data.get("board", {}).get("width", 100),
                        placement_data.get("board", {}).get("height", 100)
                    )
                    st.plotly_chart(fig, width='stretch')
            
            # Quality metrics
            quality = st.session_state.optimization_results.get("quality", {})
            if quality:
                st.markdown("---")
                st.markdown("### Quality Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    overall_score = quality.get("overall_score", 0)
                    score_color = "#4ec9b0" if overall_score >= 0.7 else "#f48771"
                    st.markdown(f'<div style="text-align: center;"><div style="font-size: 32px; color: {score_color}; font-weight: bold;">{overall_score:.2f}</div><div style="color: #cccccc;">Quality Score</div></div>', unsafe_allow_html=True)
                with col2:
                    drc = quality.get("categories", {}).get("design_rules", {})
                    st.metric("Design Rules", "PASS" if drc.get("pass") else "FAIL", delta=f"{drc.get('violation_count', 0)} violations")
                with col3:
                    thermal = quality.get("categories", {}).get("thermal", {})
                    st.metric("Thermal", "PASS" if thermal.get("pass") else "FAIL", delta=f"{thermal.get('hotspots', 0)} hotspots")
                with col4:
                    si = quality.get("categories", {}).get("signal_integrity", {})
                    st.metric("Signal Integrity", "PASS" if si.get("pass") else "FAIL", delta=f"{si.get('issue_count', 0)} issues")
                with col5:
                    dfm = quality.get("categories", {}).get("manufacturability", {})
                    st.metric("Manufacturability", "PASS" if dfm.get("pass") else "FAIL")
                
                # Performance
                performance = st.session_state.optimization_results.get("performance", {})
                if performance:
                    st.markdown("---")
                    st.markdown("### Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        opt_time = performance.get("optimization_time_seconds", 0)
                        st.metric("Optimization Time", f"{opt_time:.2f} seconds")
                    with col2:
                        trad_time = performance.get("traditional_time_weeks", 0)
                        st.metric("Traditional Time", f"{trad_time:.1f} weeks")
                    with col3:
                        savings = performance.get("time_savings_factor", 0)
                        st.metric("Time Savings", f"{savings:.0f}x faster")
                
                # Export
                st.markdown("---")
                st.markdown("### Export")
                if st.button("Export to KiCad", type="primary", use_container_width=True):
                    try:
                        placement_data = st.session_state.optimization_results.get("placement", {})
                        response = requests.post(
                            f"{API_BASE}/export/kicad",
                            json={"placement": placement_data},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.download_button(
                                "Download KiCad File",
                                data=data["content"],
                                file_name=data["filename"],
                                mime="text/plain",
                                use_container_width=True
                            )
                            st.success("KiCad file ready")
                        else:
                            st.error(f"Export failed: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

