"""
Dielectric - Professional PCB Design Interface

Clean, engineer-focused UI for computational geometry-driven PCB optimization.
"""

import streamlit as st
import requests
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import time

API_BASE = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Dielectric",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
    }
    .sub-header {
        font-size: 14px;
        color: #666;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        border-left: 3px solid #0066cc;
    }
    .section-divider {
        border-top: 1px solid #e0e0e0;
        margin: 24px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
</style>
""", unsafe_allow_html=True)


def create_pcb_visualization(
    placement_data: Dict,
    zoom_level: float = 1.0,
    show_layer: str = "all",
    highlight_modules: Optional[List[str]] = None
) -> go.Figure:
    """Create professional PCB visualization with zoom and layer support."""
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
        line=dict(color="#1a1a1a", width=2),
        fillcolor="rgba(250, 250, 250, 0.5)",
        layer="below"
    )
    
    # Thermal heatmap
    if components:
        grid_size = 50
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
                        thermal_map[j, i] += power * np.exp(-(dist**2) / (2 * 10**2))
        
        fig.add_trace(go.Contour(
            x=x_grid, y=y_grid, z=thermal_map,
            colorscale="Hot",
            showscale=True,
            name="Thermal",
            opacity=0.25,
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
        color = "#ff4444" if power > 1.0 else "#0066cc" if power > 0.1 else "#666666"
        
        if highlight_modules and name in highlight_modules:
            color = "#00cc00"
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=max(width, height) * 2,
                color=color,
                line=dict(width=1, color="#1a1a1a")
            ),
            text=name,
            textposition="middle center",
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
                        line=dict(color="#999", width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
    
    fig.update_layout(
        title="PCB Layout",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        width=800,
        height=600,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )
    
    return fig


# Sidebar
with st.sidebar:
    st.markdown('<div class="main-header">Dielectric</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered PCB Design</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Design Input
    st.markdown("### Design Input")
    user_intent = st.text_area(
        "Optimization Intent",
        value="Optimize for thermal management and minimize trace length",
        height=100,
        help="Describe your optimization goals in natural language"
    )
    
    # Board Configuration
    st.markdown("### Board Configuration")
    col1, col2 = st.columns(2)
    with col1:
        board_width = st.number_input("Width (mm)", 50, 500, 100, 10)
    with col2:
        board_height = st.number_input("Height (mm)", 50, 500, 100, 10)
    
    # File Upload
    st.markdown("### Upload Design")
    uploaded_file = st.file_uploader(
        "PCB Design (JSON)",
        type=["json"],
        help="Upload your PCB design file"
    )
    
    if uploaded_file:
        try:
            design_data = json.loads(uploaded_file.read().decode('utf-8'))
            if "board" in design_data and "components" in design_data:
                st.session_state.board_data = design_data
                st.success(f"Loaded: {len(design_data.get('components', []))} components")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Example Designs
    st.markdown("### Examples")
    example = st.selectbox("Load Example", ["", "Audio Amplifier", "Power Supply", "Sensor Module"])
    
    if example == "Audio Amplifier":
        st.session_state.board_data = {
            "board": {"width": 120, "height": 80, "clearance": 0.5},
            "components": [
                {"name": "U1", "package": "SOIC-8", "width": 5, "height": 4, "power": 0.5, "x": 30, "y": 25, "angle": 0, "placed": True},
                {"name": "U2", "package": "QFN-16", "width": 4, "height": 4, "power": 1.2, "x": 70, "y": 30, "angle": 0, "placed": True},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 20, "angle": 0, "placed": True},
                {"name": "C1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 40, "angle": 0, "placed": True},
            ],
            "nets": [
                {"name": "VCC", "pins": [["U1", "pin8"], ["U2", "pin1"]]},
                {"name": "GND", "pins": [["U1", "pin4"], ["U2", "pin2"]]},
            ]
        }


# Main Content
st.markdown('<div class="main-header">PCB Design Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Computational Geometry + Multi-Agent AI</div>', unsafe_allow_html=True)

# Initialize session state
if "board_data" not in st.session_state:
    st.session_state.board_data = None
if "results" not in st.session_state:
    st.session_state.results = None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Design", "Optimization", "Analysis", "Export"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.board_data:
            fig = create_pcb_visualization(st.session_state.board_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a design or load an example to begin")
    
    with col2:
        st.markdown("### Design Metrics")
        if st.session_state.board_data:
            comps = len(st.session_state.board_data.get("components", []))
            nets = len(st.session_state.board_data.get("nets", []))
            st.metric("Components", comps)
            st.metric("Nets", nets)
            st.metric("Board Size", f"{board_width}×{board_height}mm")
        
        if st.button("Run Optimization", type="primary", use_container_width=True):
            if not st.session_state.board_data:
                st.error("Please load a design first")
            else:
                with st.spinner("Optimizing..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/optimize",
                            json={
                                "board": st.session_state.board_data["board"],
                                "components": st.session_state.board_data["components"],
                                "nets": st.session_state.board_data.get("nets", []),
                                "intent": user_intent
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            st.session_state.results = response.json()
                            st.success("Optimization complete")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

with tab2:
    if st.session_state.results:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Before")
            if st.session_state.board_data:
                fig = create_pcb_visualization(st.session_state.board_data)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### After")
            if st.session_state.results.get("placement"):
                fig = create_pcb_visualization(st.session_state.results["placement"])
                st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{st.session_state.results.get('score', 0):.2f}")
        with col2:
            weights = st.session_state.results.get("weights_used", {})
            st.metric("Trace Weight", f"{weights.get('alpha', 0):.2f}")
        with col3:
            st.metric("Thermal Weight", f"{weights.get('beta', 0):.2f}")
        with col4:
            st.metric("Clearance Weight", f"{weights.get('gamma', 0):.2f}")
    else:
        st.info("Run optimization to see results")

with tab3:
    if st.session_state.results and st.session_state.results.get("geometry_data"):
        geometry = st.session_state.results["geometry_data"]
        
        st.markdown("### Computational Geometry Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MST Length", f"{geometry.get('mst_length', 0):.1f} mm")
        with col2:
            st.metric("Voronoi Variance", f"{geometry.get('voronoi_variance', 0):.2f}")
        with col3:
            st.metric("Thermal Hotspots", geometry.get('thermal_hotspots', 0))
        with col4:
            st.metric("Net Crossings", geometry.get('net_crossings', 0))
        
        with st.expander("Detailed Metrics"):
            st.json(geometry)
    else:
        st.info("Run optimization to see geometry analysis")

with tab4:
    if st.session_state.results:
        st.markdown("### Export to KiCad")
        
        if st.button("Generate KiCad File", type="primary", use_container_width=True):
            try:
                placement_data = st.session_state.results.get("placement", {})
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
                    st.error(f"Export failed: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Run optimization to export")

