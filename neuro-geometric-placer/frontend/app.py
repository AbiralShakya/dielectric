"""
Streamlit Frontend for Neuro-Geometric Placer

Interactive UI for PCB placement optimization.
"""

import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# Configuration
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Neuro-Geometric Placer",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 Neuro-Geometric Placer")
st.markdown("**AI-Powered PCB Component Placement Optimization**")
st.markdown("*Powered by xAI (Grok) + Dedalus Labs MCP Servers*")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    user_intent = st.text_area(
        "Optimization Intent",
        value="Optimize for minimal trace length, but keep high-power components cool",
        help="Describe your optimization goals in natural language"
    )
    
    optimization_type = st.selectbox(
        "Optimization Type",
        ["fast", "quality"],
        help="Fast: <200ms for interactive UI. Quality: Background optimization for best results."
    )
    
    if st.button("Load Example Board", use_container_width=True):
        # Load example
        example_data = {
            "components": [
                {"name": "U1", "package": "BGA256", "width": 15, "height": 15, "power": 2.0, "x": 20, "y": 20, "angle": 0, "placed": True},
                {"name": "R1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 50, "y": 30, "angle": 0, "placed": True},
                {"name": "R2", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 60, "y": 30, "angle": 0, "placed": True},
                {"name": "C1", "package": "0805", "width": 2, "height": 1.25, "power": 0.0, "x": 40, "y": 40, "angle": 0, "placed": True},
            ],
            "board": {"width": 100, "height": 100, "clearance": 0.5},
            "nets": [
                {"name": "net1", "pins": [["U1", "pin1"], ["R1", "pin1"]]},
                {"name": "net2", "pins": [["R1", "pin2"], ["R2", "pin1"]]},
            ]
        }
        st.session_state.example_data = example_data
        st.success("Example board loaded!")

# Main content
tab1, tab2, tab3 = st.tabs(["Placement", "Optimization", "Results"])

with tab1:
    st.header("Placement Editor")
    
    # Load or create placement
    if "placement_data" not in st.session_state:
        if "example_data" in st.session_state:
            st.session_state.placement_data = st.session_state.example_data
        else:
            st.session_state.placement_data = {
                "components": [],
                "board": {"width": 100, "height": 100, "clearance": 0.5},
                "nets": []
            }
    
    placement_data = st.session_state.placement_data
    
    # Board configuration
    col1, col2 = st.columns(2)
    with col1:
        board_width = st.number_input("Board Width (mm)", value=float(placement_data["board"]["width"]), min_value=10.0, max_value=500.0)
    with col2:
        board_height = st.number_input("Board Height (mm)", value=float(placement_data["board"]["height"]), min_value=10.0, max_value=500.0)
    
    placement_data["board"]["width"] = board_width
    placement_data["board"]["height"] = board_height
    
    # Visualize placement
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, board_width)
    ax.set_ylim(0, board_height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Component Placement")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    # Draw components
    for comp in placement_data["components"]:
        x, y = comp["x"], comp["y"]
        w, h = comp["width"], comp["height"]
        
        # Draw rectangle
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, 
                            fill=True, alpha=0.5, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Label
        ax.text(x, y, comp["name"], ha='center', va='center', fontsize=8, fontweight='bold')
    
    st.pyplot(fig)
    
    # Upload button
    if st.button("Upload Placement", use_container_width=True):
        try:
            response = requests.post(
                f"{API_BASE}/upload",
                json={
                    "placement_data": placement_data,
                    "user_intent": user_intent,
                    "optimization_type": optimization_type
                }
            )
            response.raise_for_status()
            result = response.json()
            st.session_state.task_id = result["task_id"]
            st.success(f"Placement uploaded! Task ID: {result['task_id']}")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

with tab2:
    st.header("Optimization")
    
    if "task_id" not in st.session_state:
        st.warning("Please upload a placement first.")
    else:
        task_id = st.session_state.task_id
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Optimize (Fast Path)", use_container_width=True):
                with st.spinner("Optimizing (fast path, <200ms)..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/optimize_fast",
                            json={"task_id": task_id, "user_intent": user_intent}
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.session_state.optimization_result = result
                        st.success("Optimization completed!")
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
        
        with col2:
            if st.button("Optimize (Quality Path)", use_container_width=True):
                with st.spinner("Optimizing (quality path, may take minutes)..."):
                    try:
                        response = requests.post(
                            f"{API_BASE}/optimize",
                            json={"task_id": task_id, "user_intent": user_intent, "optimization_type": "quality"}
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.session_state.optimization_result = result
                        st.success("Optimization completed!")
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
        
        # Show results if available
        if "optimization_result" in st.session_state:
            result = st.session_state.optimization_result
            st.json(result)

with tab3:
    st.header("Results")
    
    if "task_id" not in st.session_state:
        st.warning("No optimization results available.")
    else:
        task_id = st.session_state.task_id
        
        if st.button("Refresh Results", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE}/results/{task_id}")
                response.raise_for_status()
                result = response.json()
                st.session_state.final_results = result
            except Exception as e:
                st.error(f"Failed to fetch results: {str(e)}")
        
        if "final_results" in st.session_state:
            results = st.session_state.final_results
            
            if results.get("success"):
                st.success("✅ Optimization Successful!")
                
                # Score breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Score", f"{results.get('score', 0):.2f}")
                with col2:
                    weights = results.get("weights", {})
                    st.metric("Trace Priority", f"{weights.get('alpha', 0)*100:.0f}%")
                with col3:
                    st.metric("Thermal Priority", f"{weights.get('beta', 0)*100:.0f}%")
                
                # Intent explanation
                if results.get("intent_explanation"):
                    st.info(f"**Intent:** {results['intent_explanation']}")
                
                # Stats
                if results.get("stats"):
                    stats = results["stats"]
                    st.subheader("Optimization Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Iterations", stats.get("iterations", 0))
                    with col2:
                        st.metric("Improvements", stats.get("improvements", 0))
                    with col3:
                        if "time_ms" in stats:
                            st.metric("Time", f"{stats['time_ms']:.1f} ms")
                
                # Visualize optimized placement
                if results.get("placement"):
                    placement = results["placement"]
                    fig, ax = plt.subplots(figsize=(10, 10))
                    board = placement["board"]
                    ax.set_xlim(0, board["width"])
                    ax.set_ylim(0, board["height"])
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_title("Optimized Placement")
                    
                    for comp in placement["components"]:
                        x, y = comp["x"], comp["y"]
                        w, h = comp["width"], comp["height"]
                        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                                            fill=True, alpha=0.5, edgecolor='green', linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x, y, comp["name"], ha='center', va='center', fontsize=8, fontweight='bold')
                    
                    st.pyplot(fig)
                
                # Verification
                if results.get("verification"):
                    verification = results["verification"]
                    if verification.get("violations"):
                        st.error(f"⚠️ {len(verification['violations'])} violations found")
                    if verification.get("warnings"):
                        st.warning(f"⚠️ {len(verification['warnings'])} warnings")
            else:
                st.error("❌ Optimization failed")

# Footer
st.markdown("---")
st.markdown("**Neuro-Geometric Placer** - Built for HackPrinceton 2025")
st.markdown("*Multi-Agent AI • Computational Geometry • xAI Reasoning • Dedalus Labs MCP*")

