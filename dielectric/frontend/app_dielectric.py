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
import logging

logger = logging.getLogger(__name__)

# Add frontend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import visualizers with error handling
_geometry_import_error = None
_circuit_import_error = None

try:
    from geometry_visualizer import (
        visualize_voronoi_diagram,
        visualize_minimum_spanning_tree,
        visualize_convex_hull,
        create_geometry_dashboard
    )
except ImportError as e:
    _geometry_import_error = str(e)
    # Define dummy functions to prevent crashes
    def visualize_voronoi_diagram(*args, **kwargs):
        return go.Figure()
    def visualize_minimum_spanning_tree(*args, **kwargs):
        return go.Figure()
    def visualize_convex_hull(*args, **kwargs):
        return go.Figure()
    def create_geometry_dashboard(*args, **kwargs):
        return go.Figure()

try:
    from circuit_visualizer import (
        create_circuit_visualization,
        create_pcb_layout_view,
        create_schematic_view
    )
except ImportError as e:
    _circuit_import_error = str(e)
    # Define dummy functions to prevent crashes
    def create_circuit_visualization(*args, **kwargs):
        return go.Figure()
    def create_pcb_layout_view(*args, **kwargs):
        return go.Figure()
    def create_schematic_view(*args, **kwargs):
        return go.Figure()

API_BASE = "http://localhost:8000"

# Apple-Inspired Sleek Design - Minimal & Refined
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .stApp {
        background: #000000;
        color: #f5f5f7;
    }
    
    [data-testid="stSidebar"] {
        background: #1d1d1f;
        border-right: none;
    }
    
    /* Apple Typography */
    h1 {
        color: #ffffff;
        font-weight: 600;
        font-size: 2.75rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 600;
        font-size: 2rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    h3 {
        color: #f5f5f7;
        font-weight: 600;
        font-size: 1.25rem;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    
    .main .block-container {
        padding-top: 4rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 1200px;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Apple-Style Buttons - Refined & Minimal */
    .stButton>button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.875rem 1.75rem;
        font-weight: 500;
        font-size: 0.9375rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: -0.01em;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        background: #f5f5f7;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        background: #e8e8ed;
    }
    
    [data-baseweb="button"][kind="primary"] {
        background: #007aff !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    [data-baseweb="button"][kind="primary"]:hover {
        background: #0051d5 !important;
    }
    
    /* Apple Input Fields - Refined */
    [data-testid="stTextInput"]>div>div>input,
    [data-testid="stTextArea"]>div>div>textarea,
    [data-testid="stNumberInput"]>div>div>input {
        background: #1d1d1f !important;
        color: #f5f5f7 !important;
        border: 1px solid #424245 !important;
        border-radius: 8px !important;
        padding: 0.875rem 1rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 0.9375rem !important;
    }
    
    [data-testid="stTextInput"]>div>div>input:focus,
    [data-testid="stTextArea"]>div>div>textarea:focus,
    [data-testid="stNumberInput"]>div>div>input:focus {
        border-color: #007aff !important;
        background: #2d2d2f !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
    }
    
    /* Apple Select Boxes */
    .stSelectbox>div>div>select {
        background: #1d1d1f !important;
        color: #f5f5f7 !important;
        border: 1px solid #424245 !important;
        border-radius: 8px !important;
        padding: 0.625rem !important;
    }
    
    /* Apple Expanders */
    .stExpander {
        background: transparent !important;
        border: none !important;
        margin: 1.5rem 0 !important;
    }
    
    .streamlit-expanderHeader {
        color: #f5f5f7 !important;
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
    }
    
    /* Radio Buttons - Apple Style */
    [data-testid="stRadio"] label {
        background: #1d1d1f !important;
        border: 1px solid #424245 !important;
        border-radius: 8px !important;
        padding: 0.875rem 1.5rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    [data-testid="stRadio"] label:hover {
        background: #2d2d2f !important;
        border-color: #545458 !important;
    }
    
    [data-testid="stRadio"] input[type="radio"]:checked + label {
        background: #007aff !important;
        border-color: #007aff !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* File Uploader - Apple Style */
    [data-testid="stFileUploader"] {
        background: #1d1d1f !important;
        border: 2px dashed #424245 !important;
        border-radius: 12px !important;
        padding: 3rem 2rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #007aff !important;
        background: #2d2d2f !important;
    }
    
    /* Alert Boxes - Apple Style */
    .stAlert {
        background: #1d1d1f !important;
        border: none !important;
        border-radius: 12px !important;
        border-left: 4px solid #007aff !important;
    }
    
    /* Progress Bars - Apple Style */
    [data-testid="stProgressBar"]>div>div {
        background: #007aff !important;
        border-radius: 2px !important;
    }
    
    /* Tabs - Apple Style */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #424245 !important;
        padding: 0 !important;
    }
    
    [data-baseweb="tab"] {
        color: #86868b !important;
        border-radius: 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0.75rem 1rem !important;
        margin-right: 2rem !important;
    }
    
    [data-baseweb="tab"]:hover {
        color: #f5f5f7 !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        background: transparent !important;
        color: #007aff !important;
        font-weight: 500 !important;
        border-bottom: 2px solid #007aff !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f5f5f7 !important;
        font-weight: 600 !important;
    }
    
    /* Smooth Scroll */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom Scrollbar - Apple Style */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #424245;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #545458;
    }
    
    /* Subtle Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main .block-container {
        animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Metric Cards - Apple Style */
    [data-testid="stMetricContainer"] {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    [data-testid="stMetricContainer"]:hover {
        background: #1d1d1f !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Dielectric",
    page_icon=None,
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


# Apple-Style Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: left; padding: 2rem 0 3rem 0;">
        <h1 style="font-size: 1.5rem; margin: 0; color: #ffffff; font-weight: 600; letter-spacing: -0.02em;">
            Dielectric
        </h1>
        <p style="color: #86868b; font-size: 0.8125rem; margin-top: 0.5rem; font-weight: 400;">
            AI-Powered PCB Design
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow selection
    st.markdown("### Workflow")
    workflow = st.radio(
        "Choose workflow",
        ["Generate Design", "Optimize Design"],
        help="Generate: Create new PCB from natural language\nOptimize: Optimize existing design",
        label_visibility="collapsed"
    )

def process_zip_file(zip_file):
    """Process a single zip file with progress indicators."""
    st.session_state.processing_status = "processing"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.info("**Step 1/4:** Uploading zip file to server...")
        progress_bar.progress(10)
        
        files = {'file': (zip_file.name, zip_file.getvalue(), zip_file.type)}
        
        status_text.info("**Step 2/4:** Extracting and scanning folder structure...")
        progress_bar.progress(30)
        
        response = requests.post(f"{API_BASE}/upload/pcb", files=files, timeout=180)
        
        status_text.info("**Step 3/4:** Parsing PCB design files...")
        progress_bar.progress(60)
        
        if response.status_code == 200:
            result = response.json()
            parsed_placement = result.get("parsed_placement", {})
            
            status_text.info("**Step 4/4:** Finalizing...")
            progress_bar.progress(90)
            
            if parsed_placement and parsed_placement.get("components"):
                st.session_state.design_data = parsed_placement
                st.session_state.processing_status = "complete"
                
                # Show success with details
                components_count = len(parsed_placement.get("components", []))
                nets_count = len(parsed_placement.get("nets", []))
                board = parsed_placement.get("board", {})
                
                progress_bar.progress(100)
                status_text.success("**Folder parsed successfully!**")
                
                st.info(f"""
                **Design Loaded:**
                - **{components_count}** components
                - **{nets_count}** nets
                - Board size: {board.get('width', 'N/A')}×{board.get('height', 'N/A')}mm
                
                *Scroll down to optimize this design with natural language.*
                """)
                
                if "files_found" in result:
                    with st.expander("Folder Analysis Details"):
                        st.json(result.get("folder_structure", {}))
                        st.json(result.get("files_found", {}))
                        st.json(result.get("files_parsed", []))
                
                st.rerun()
            else:
                st.session_state.processing_status = "error"
                progress_bar.progress(0)
                status_text.warning("Folder parsed but no design data found. Check folder contents.")
                if "files_found" in result:
                    with st.expander("What was found in the folder"):
                        st.json(result.get("files_found", {}))
        else:
            st.session_state.processing_status = "error"
            progress_bar.progress(0)
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text[:500])
            except:
                error_detail = response.text[:500] if hasattr(response, 'text') else f"HTTP {response.status_code}"
            status_text.error(f"**Error {response.status_code}:** {error_detail}")
            
    except requests.exceptions.Timeout:
        st.session_state.processing_status = "error"
        progress_bar.progress(0)
        status_text.error("**Timeout:** Processing took too long. Try with a smaller folder or check your connection.")
    except Exception as e:
        st.session_state.processing_status = "error"
        progress_bar.progress(0)
        status_text.error(f"**Error:** {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def process_multiple_files(files):
    """Process multiple files with progress indicators."""
    st.session_state.processing_status = "processing"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.info(f"**Step 1/4:** Uploading {len(files)} files to server...")
        progress_bar.progress(10)
        
        files_data = [('files', (f.name, f.getvalue(), f.type)) for f in files]
        
        status_text.info("**Step 2/4:** Scanning folder structure...")
        progress_bar.progress(30)
        
        response = requests.post(f"{API_BASE}/upload/folder", files=files_data, timeout=180)
        
        status_text.info("**Step 3/4:** Parsing PCB design files...")
        progress_bar.progress(60)
        
        if response.status_code == 200:
            result = response.json()
            parsed_placement = result.get("parsed_placement", {})
            
            status_text.info("**Step 4/4:** Finalizing...")
            progress_bar.progress(90)
            
            if parsed_placement and parsed_placement.get("components"):
                st.session_state.design_data = parsed_placement
                st.session_state.processing_status = "complete"
                
                components_count = len(parsed_placement.get("components", []))
                nets_count = len(parsed_placement.get("nets", []))
                board = parsed_placement.get("board", {})
                
                progress_bar.progress(100)
                status_text.success("**Folder parsed successfully!**")
                
                st.info(f"""
                **Design Loaded:**
                - **{components_count}** components
                - **{nets_count}** nets
                - Board size: {board.get('width', 'N/A')}×{board.get('height', 'N/A')}mm
                
                *Scroll down to optimize this design with natural language.*
                """)
                
                if "files_found" in result:
                    with st.expander("Folder Analysis Details"):
                        st.json(result.get("folder_structure", {}))
                        st.json(result.get("files_found", {}))
                        st.json(result.get("files_parsed", []))
                
                st.rerun()
            else:
                st.session_state.processing_status = "error"
                progress_bar.progress(0)
                status_text.warning("Folder parsed but no design data found. Check folder contents.")
        else:
            st.session_state.processing_status = "error"
            progress_bar.progress(0)
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text[:500])
            except:
                error_detail = response.text[:500] if hasattr(response, 'text') else f"HTTP {response.status_code}"
            status_text.error(f"**Error {response.status_code}:** {error_detail}")
            
    except requests.exceptions.Timeout:
        st.session_state.processing_status = "error"
        progress_bar.progress(0)
        status_text.error("**Timeout:** Processing took too long. Try with fewer files or check your connection.")
    except Exception as e:
        st.session_state.processing_status = "error"
        progress_bar.progress(0)
        status_text.error(f"**Error:** {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


# Initialize session state early
if "design_data" not in st.session_state:
    st.session_state.design_data = None
if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None
if "example_description" not in st.session_state:
    st.session_state.example_description = ""
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = "idle"

# Check backend connection
try:
    backend_check = requests.get(f"{API_BASE}/health", timeout=2)
    backend_online = backend_check.status_code == 200
except:
    backend_online = False

# Show import errors if any (after Streamlit is initialized)
if _geometry_import_error:
    st.warning(f"Geometry visualizer import failed: {_geometry_import_error}")
if _circuit_import_error:
    st.warning(f"Circuit visualizer import failed: {_circuit_import_error}")

# Check backend connection
if not backend_online:
    st.error("**Backend server is not running!**")
    st.info("Please start the backend server:\n```bash\ncd /Users/abiralshakya/Documents/hackprinceton2025/dielectric\nsource venv/bin/activate\nuvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload\n```")
    st.stop()

# Apple-Style Header
st.markdown("""
<div style="text-align: left; padding: 0 0 3rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.75rem; color: #ffffff; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1;">
        Dielectric
    </h1>
    <p style="font-size: 1.0625rem; color: #86868b; margin-top: 0; font-weight: 400; letter-spacing: -0.01em;">
        Computational Geometry + Multi-Agent AI for PCB Design
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if workflow == "Generate Design":
    st.markdown("""
    <div style="margin-bottom: 3rem;">
        <h2 style="margin-bottom: 0.75rem;">Generate PCB Design from Natural Language</h2>
        <p style="color: #86868b; font-size: 1.0625rem; margin-top: 0; font-weight: 400;">Describe your PCB design in plain English and let AI create it for you</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize example_description if not set
    if "example_description" not in st.session_state:
        st.session_state.example_description = ""
    
    # Examples Section
    st.markdown("""
    <div style="margin: 2rem 0 1.5rem 0;">
        <h3 style="color: #f5f5f7; margin-bottom: 1rem; font-weight: 600; font-size: 1.0625rem;">Quick Examples</h3>
    </div>
    """, unsafe_allow_html=True)
    
    examples = {
        "Audio Amplifier": "Design an audio amplifier with op-amp, input/output capacitors, and power supply filtering",
        "Power Supply": "Create a switching power supply with buck converter IC, inductor, capacitors, and feedback resistors",
        "Sensor Module": "Design a sensor interface board with ADC, voltage reference, and signal conditioning",
        "MCU Board": "Create a microcontroller board with MCU, crystal oscillator, decoupling capacitors, and programming header",
        "RF Module": "Design a 2.4GHz RF module with transceiver IC, matching network, antenna, and power management"
    }
    
    cols = st.columns(2)
    for idx, (name, desc) in enumerate(examples.items()):
        with cols[idx % 2]:
            if st.button(name, key=f"example_{idx}", use_container_width=True):
                st.session_state.design_description_input = desc
                st.rerun()
    
    # Main input area
    # Initialize if not exists
    if "design_description_input" not in st.session_state:
        st.session_state.design_description_input = ""
    
    st.markdown("""
    <div style="margin-bottom: 0.75rem;">
        <label style="color: #f5f5f7; font-weight: 500; font-size: 0.9375rem;">Describe your PCB design</label>
    </div>
    """, unsafe_allow_html=True)
    
    design_description = st.text_area(
        "",
        value=st.session_state.design_description_input,
        height=140,
        placeholder="Example: Design an audio amplifier circuit with power supply. Include op-amp, resistors, capacitors, and power management IC. Optimize for low noise and thermal efficiency.",
        help="Describe the PCB you want to create in natural language",
        key="design_description_input",
        label_visibility="collapsed"
    )
        
    # Board size inputs
    st.markdown("""
    <div style="margin: 2rem 0 0.75rem 0;">
        <label style="color: #f5f5f7; font-weight: 500; font-size: 0.9375rem;">Board Dimensions</label>
    </div>
    """, unsafe_allow_html=True)
    
    col_size1, col_size2, col_size3 = st.columns([2, 2, 1])
    with col_size1:
        board_width = st.number_input("Width (mm)", 50, 500, 120, 10, key="board_width", label_visibility="visible")
    with col_size2:
        board_height = st.number_input("Height (mm)", 50, 500, 80, 10, key="board_height", label_visibility="visible")
    with col_size3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
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
                            timeout=180  # Increased timeout for xAI calls with extensive reasoning (3 minutes)
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.design_data = result.get("placement")
                            st.success("Design generated successfully!")
                            st.rerun()
                        else:
                            try:
                                error_json = response.json()
                                error_detail = error_json.get("detail", response.text[:500])
                            except:
                                error_detail = response.text[:500] if hasattr(response, 'text') else f"HTTP {response.status_code}"
                            st.error(f"Error {response.status_code}: {error_detail}")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The design generation is taking longer than expected. Please try again.")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    if st.session_state.design_data:
        st.markdown("---")
        st.markdown("### Generated Design")
        
        # Circuit visualization tabs
        viz_tabs = st.tabs(["PCB Layout", "Schematic", "Thermal View"])
        
        with viz_tabs[0]:
            st.markdown("#### PCB Layout View")
            st.caption("Proper PCB layout showing components, pads, traces, and board outline")
            pcb_fig = create_pcb_layout_view(
                st.session_state.design_data.get("board", {}).get("width", 100),
                st.session_state.design_data.get("board", {}).get("height", 100),
                st.session_state.design_data.get("components", []),
                st.session_state.design_data.get("nets", [])
            )
            st.plotly_chart(pcb_fig, use_container_width=True)
        
        with viz_tabs[1]:
            st.markdown("#### Schematic View")
            st.caption("Schematic representation with component symbols and connections")
            schematic_fig = create_schematic_view(
                st.session_state.design_data.get("components", []),
                st.session_state.design_data.get("nets", [])
            )
            st.plotly_chart(schematic_fig, use_container_width=True)
        
        with viz_tabs[2]:
            st.markdown("#### Thermal View")
            st.caption("Thermal heatmap overlay")
            thermal_fig = create_pcb_plot(st.session_state.design_data, "Thermal Analysis")
            st.plotly_chart(thermal_fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Components", len(st.session_state.design_data.get("components", [])))
        with col2:
            st.metric("Nets", len(st.session_state.design_data.get("nets", [])))
        with col3:
            board = st.session_state.design_data.get("board", {})
            st.metric("Board Size", f"{board.get('width', 0)}×{board.get('height', 0)}mm")
        with col4:
            # Export button for generated designs
            if st.button("Export to KiCad", type="primary", use_container_width=True):
                try:
                    response = requests.post(
                        f"{API_BASE}/export/kicad",
                        json={"placement": st.session_state.design_data},
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

else:  # Optimize Design
    st.markdown("""
    <div style="margin-bottom: 3rem;">
        <h2 style="margin-bottom: 0.75rem;">Optimize Existing PCB Design</h2>
        <p style="color: #86868b; font-size: 1.0625rem; margin-top: 0; font-weight: 400;">Upload your PCB design and optimize it with AI-powered algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Design input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Load Design")
        
        upload_option = st.radio(
            "Upload Type",
            ["Single File", "Folder/Zip"],
            horizontal=True,
            help="Choose to upload a single PCB file or an entire folder/zip"
        )
        
        if upload_option == "Single File":
            uploaded_file = st.file_uploader("Upload PCB Design", type=["json", "kicad_pcb"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.json'):
                    design_data = json.loads(uploaded_file.read().decode('utf-8'))
                    if "board" in design_data and "components" in design_data:
                        st.session_state.design_data = design_data
                        components_count = len(design_data.get("components", []))
                        st.success(f"**File loaded successfully!** ({components_count} components)")
                        st.rerun()  # Refresh UI
                    else:
                        st.error("Invalid design format")
                else:
                    # KiCad file - upload to API
                    with st.spinner("Parsing PCB file..."):
                        try:
                            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            response = requests.post(f"{API_BASE}/upload/pcb", files=files, timeout=60)
                            if response.status_code == 200:
                                result = response.json()
                                parsed_placement = result.get("parsed_placement", {})
                                if parsed_placement and parsed_placement.get("components"):
                                    st.session_state.design_data = parsed_placement
                                    components_count = len(parsed_placement.get("components", []))
                                    st.success(f"**PCB file parsed successfully!** ({components_count} components)")
                                    st.rerun()  # Refresh UI
                                else:
                                    st.warning("File parsed but no design data found")
                            else:
                                st.error(f"Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:  # Folder/Zip
            uploaded_files = st.file_uploader(
                "Upload Folder/Zip",
                type=None,  # Accept all file types for folder uploads
                accept_multiple_files=True,
                help="Upload a zip file or multiple files. Supports .zip, .kicad_pcb, .json, and other PCB file formats"
            )
            
            # Store uploaded files in session state to detect changes
            if uploaded_files:
                # Check if files changed
                file_names = [f.name for f in uploaded_files]
                if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != file_names:
                    st.session_state.last_uploaded_files = file_names
                    st.session_state.files_to_process = uploaded_files
                    st.session_state.processing_status = "ready"
            
            # Show uploaded files
            if uploaded_files:
                st.markdown("**Uploaded Files:**")
                for f in uploaded_files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"{f.name} ({f.size / 1024 / 1024:.2f} MB)")
                    with col2:
                        if st.button("Remove", key=f"remove_{f.name}", use_container_width=True):
                            st.session_state.last_uploaded_files = []
                        st.rerun()
            
            # Process button - explicit trigger
            if uploaded_files and st.session_state.get('files_to_process'):
                st.markdown("---")
                if st.button("Process Folder", type="primary", use_container_width=True):
                    zip_files = [f for f in uploaded_files if f.name.endswith('.zip')]
                    other_files = [f for f in uploaded_files if not f.name.endswith('.zip')]
                    
                    if zip_files and len(zip_files) == 1 and len(other_files) == 0:
                        # Single zip file
                        process_zip_file(zip_files[0])
                    elif other_files:
                        # Multiple files
                        process_multiple_files(other_files)
                    else:
                        st.warning("Please upload either a single zip file or multiple PCB files")
            
            # Show processing status if in progress
            if st.session_state.get('processing_status') == 'processing':
                st.info("Processing folder... This may take a moment.")
                st.progress(0.5)  # Indeterminate progress
        
        # Examples in compact format
        with st.expander("Quick Examples", expanded=False):
            example_cols = st.columns(2)
            examples = [
                ("Audio Amplifier", "Design an audio amplifier with op-amp, input/output capacitors, and power supply filtering"),
                ("Power Supply", "Create a switching power supply with buck converter IC, inductor, capacitors, and feedback resistors"),
                ("Sensor Module", "Design a sensor interface board with ADC, voltage reference, and signal conditioning"),
                ("MCU Board", "Create a microcontroller board with MCU, crystal oscillator, decoupling capacitors, and programming header")
            ]
            
            for idx, (name, desc) in enumerate(examples):
                with example_cols[idx % 2]:
                    if st.button(name, key=f"opt_example_{idx}", use_container_width=True):
                        st.session_state.design_data = None  # Clear any existing design
                        st.session_state.example_description = desc
                        st.rerun()
    
    with col2:
        if st.session_state.design_data:
            fig = create_pcb_plot(st.session_state.design_data, "Current Design")
            st.plotly_chart(fig, use_container_width=True)
    
    # Optimization Section - Show prominently after upload
    if st.session_state.design_data:
        st.markdown("---")
        
        # Show design summary banner
        components_count = len(st.session_state.design_data.get("components", []))
        nets_count = len(st.session_state.design_data.get("nets", []))
        board = st.session_state.design_data.get("board", {})
        
        st.markdown(f"""
        ### Ready to Optimize
        
        **Current Design:** {components_count} components • {nets_count} nets • {board.get('width', 'N/A')}×{board.get('height', 'N/A')}mm
        """)
        
        st.markdown("### Optimization Settings")
        
        col_opt1, col_opt2 = st.columns([2, 1])
        
        with col_opt1:
            optimization_intent = st.text_area(
            "Optimization Intent",
            value="Optimize for thermal management and signal integrity",
                placeholder="Describe how you want to optimize the design. Examples:\n- Minimize trace length and improve thermal distribution\n- Prioritize thermal cooling for high-power components\n- Ensure design rule compliance and reduce crosstalk\n- Optimize for manufacturing cost and assembly efficiency",
                help="Describe your optimization goals in natural language",
                height=100
        )
        
        with col_opt2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Optimizing design with AI agents..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/optimize",
                        json={
                                "board": st.session_state.design_data.get("board", {}),
                                "components": st.session_state.design_data.get("components", []),
                            "nets": st.session_state.design_data.get("nets", []),
                            "intent": optimization_intent
                        },
                            timeout=120  # Longer timeout for complex optimizations
                    )
                    
                    if response.status_code == 200:
                        st.session_state.optimization_results = response.json()
                        st.success("Optimization complete!")
                        st.balloons()  # Celebration!
                        
                        # Quick export button right after success
                        placement_data = response.json().get("placement", {})
                        if placement_data:
                            try:
                                export_response = requests.post(
                                    f"{API_BASE}/export/kicad",
                                    json={"placement": placement_data},
                                    timeout=10
                                )
                                if export_response.status_code == 200:
                                    export_data = export_response.json()
                                    st.download_button(
                                        "Download KiCad File",
                                        data=export_data["content"],
                                        file_name=export_data["filename"],
                                        mime="text/plain",
                                        use_container_width=True,
                                        type="primary"
                                    )
                            except Exception as e:
                                logger.debug(f"Export preview failed: {e}")
                        
                        st.rerun()
                    else:
                        try:
                            error_json = response.json()
                            error_detail = error_json.get("detail", response.text[:500])
                        except:
                            error_detail = response.text[:500] if hasattr(response, 'text') else f"HTTP {response.status_code}"
                        st.error(f"Error {response.status_code}: {error_detail}")
                except requests.exceptions.Timeout:
                    st.error("Optimization timed out. Try simplifying your optimization intent or try again.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Results
        if st.session_state.optimization_results:
            st.markdown("---")
            st.markdown("### Optimization Results")
            
            # Prominent export section at the top
            placement_data = st.session_state.optimization_results.get("placement", {})
            if placement_data:
                export_col1, export_col2, export_col3 = st.columns([2, 2, 1])
                with export_col1:
                    st.markdown("#### Export Optimized Design")
                    st.info("Export your optimized PCB layout to KiCad format for further editing or manufacturing.")
                with export_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Export to KiCad", type="primary", use_container_width=True, key="export_main"):
                        try:
                            export_response = requests.post(
                                f"{API_BASE}/export/kicad",
                                json={"placement": placement_data},
                                timeout=10
                            )
                            
                            if export_response.status_code == 200:
                                export_data = export_response.json()
                                st.download_button(
                                    "⬇️ Download KiCad File",
                                    data=export_data["content"],
                                    file_name=export_data["filename"],
                                    mime="text/plain",
                                    use_container_width=True,
                                    key="download_kicad_main"
                                )
                                st.success("KiCad file ready for download!")
                            else:
                                st.error(f"Export failed: {export_response.status_code}")
                        except Exception as e:
                            st.error(f"Export error: {str(e)}")
                with export_col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="padding: 10px; background-color: #1e1e1e; border-radius: 5px;">
                    <small><strong>3D Support:</strong><br>
                    PCB design is primarily <strong>2D layout</strong> (component placement & routing), but:
                    <ul style="margin: 5px 0; padding-left: 20px;">
                    <li>KiCad can export <strong>3D STEP files</strong></li>
                    <li>JLCPCB supports <strong>3D visualization</strong></li>
                    <li>Components have <strong>3D models</strong></li>
                    </ul>
                    </small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            # Circuit visualization comparison
            comparison_tabs = st.tabs(["PCB Layout Comparison", "Schematic Comparison", "Thermal Comparison"])
            
            with comparison_tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before - PCB Layout")
                    board = st.session_state.design_data.get("board", {})
                    before_pcb = create_pcb_layout_view(
                        board.get("width", 100),
                        board.get("height", 100),
                        st.session_state.design_data.get("components", []),
                        st.session_state.design_data.get("nets", [])
                    )
                    st.plotly_chart(before_pcb, use_container_width=True)
                
                with col2:
                    st.markdown("#### After - PCB Layout")
                    if st.session_state.optimization_results.get("placement"):
                        opt_board = st.session_state.optimization_results["placement"].get("board", {})
                        after_pcb = create_pcb_layout_view(
                            opt_board.get("width", 100),
                            opt_board.get("height", 100),
                            st.session_state.optimization_results["placement"].get("components", []),
                            st.session_state.optimization_results["placement"].get("nets", [])
                        )
                        st.plotly_chart(after_pcb, use_container_width=True)
            
            with comparison_tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before - Schematic")
                    before_sch = create_schematic_view(
                        st.session_state.design_data.get("components", []),
                        st.session_state.design_data.get("nets", [])
                    )
                    st.plotly_chart(before_sch, use_container_width=True)
                
                with col2:
                    st.markdown("#### After - Schematic")
                    if st.session_state.optimization_results.get("placement"):
                        after_sch = create_schematic_view(
                            st.session_state.optimization_results["placement"].get("components", []),
                            st.session_state.optimization_results["placement"].get("nets", [])
                        )
                        st.plotly_chart(after_sch, use_container_width=True)
            
            with comparison_tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before - Thermal")
                    thermal_before = create_pcb_plot(st.session_state.design_data, "Initial Thermal")
                    st.plotly_chart(thermal_before, use_container_width=True)
                
                with col2:
                    st.markdown("#### After - Thermal")
                    if st.session_state.optimization_results.get("placement"):
                        thermal_after = create_pcb_plot(
                            st.session_state.optimization_results["placement"],
                            "Optimized Thermal"
                        )
                        st.plotly_chart(thermal_after, use_container_width=True)
            
            # Multi-Agent Workflow Status
            agents_used = st.session_state.optimization_results.get("agents_used", [])
            if agents_used:
                st.markdown("---")
                st.markdown("### Multi-Agent Workflow")
                st.markdown("**Why this matters:** Each agent specializes in one task, working together like a team of engineers.")
                
                agent_descriptions = {
                    "IntentAgent": "Understands your goals using computational geometry + xAI",
                    "LocalPlacerAgent": "Optimizes placement (deterministic, fast)",
                    "VerifierAgent": "Validates design rules and constraints",
                    "ErrorFixerAgent": "Automatically fixes violations",
                    "DesignGeneratorAgent": "Creates designs from natural language"
                }
                
                cols = st.columns(len(agents_used))
                for i, agent in enumerate(agents_used):
                    with cols[i]:
                        st.markdown(f"**{agent}**")
                        if agent in agent_descriptions:
                            st.caption(agent_descriptions[agent])
                        st.success("Active")
            
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

