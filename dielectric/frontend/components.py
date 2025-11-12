"""
Enhanced Frontend Components
Visionary scientist + fast-moving startup design
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any


def create_sleek_file_upload():
    """
    Sleek, modern file upload component with drag-and-drop.
    Matches the dark, professional aesthetic.
    """
    st.markdown("""
    <style>
    .file-upload-container {
        background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
        border: 2px dashed #424245;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .file-upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 122, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .file-upload-container:hover {
        border-color: #007aff;
        background: linear-gradient(135deg, #2d2d2f 0%, #3d3d3f 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 122, 255, 0.2);
    }
    
    .file-upload-container:hover::before {
        left: 100%;
    }
    
    .upload-icon {
        font-size: 3rem;
        color: #007aff;
        margin-bottom: 1rem;
        display: block;
    }
    
    .upload-text {
        color: #f5f5f7;
        font-size: 1.125rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtext {
        color: #86868b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .file-type-badges {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .file-badge {
        background: rgba(0, 122, 255, 0.1);
        border: 1px solid rgba(0, 122, 255, 0.3);
        color: #007aff;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .uploaded-file-card {
        background: #1d1d1f;
        border: 1px solid #424245;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s;
    }
    
    .uploaded-file-card:hover {
        border-color: #007aff;
        background: #2d2d2f;
    }
    
    .file-info {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .file-icon {
        font-size: 1.5rem;
        color: #007aff;
    }
    
    .file-details {
        display: flex;
        flex-direction: column;
    }
    
    .file-name {
        color: #f5f5f7;
        font-weight: 500;
        font-size: 0.9375rem;
    }
    
    .file-size {
        color: #86868b;
        font-size: 0.8125rem;
    }
    </style>
    
    <div class="file-upload-container" onclick="document.getElementById('file-upload-input').click()">
        <span class="upload-icon">📤</span>
        <div class="upload-text">Drag & drop your PCB design here</div>
        <div class="upload-subtext">or click to browse</div>
        <div class="file-type-badges">
            <span class="file-badge">.kicad_pcb</span>
            <span class="file-badge">.json</span>
            <span class="file-badge">.zip</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_multi_agent_pipeline_viz(agents_used: List[str], agent_results: Dict = None):
    """
    Visualize multi-agent workflow as a pipeline/timeline.
    Shows agent collaboration and data flow.
    """
    agent_descriptions = {
        "IntentAgent": {
            "icon": "🧠",
            "description": "Understands goals using computational geometry + xAI",
            "color": "#007aff",
            "output": "Optimization weights"
        },
        "LocalPlacerAgent": {
            "icon": "⚙️",
            "description": "Optimizes placement (deterministic, fast)",
            "color": "#4ec9b0",
            "output": "Optimized placement"
        },
        "PhysicsSimulationAgent": {
            "icon": "🔬",
            "description": "Simulates thermal, SI, PDN physics",
            "color": "#f48771",
            "output": "Physics metrics"
        },
        "VerifierAgent": {
            "icon": "✅",
            "description": "Validates design rules and constraints",
            "color": "#ffd60a",
            "output": "DRC violations"
        },
        "ErrorFixerAgent": {
            "icon": "🔧",
            "description": "Automatically fixes violations",
            "color": "#ff6b6b",
            "output": "Fixed design"
        },
        "ExporterAgent": {
            "icon": "📤",
            "description": "Exports to KiCad format",
            "color": "#9d4edd",
            "output": "KiCad file"
        }
    }
    
    # Create pipeline visualization
    fig = go.Figure()
    
    x_positions = np.linspace(0, 100, len(agents_used))
    y_position = 50
    
    # Draw connecting lines
    for i in range(len(agents_used) - 1):
        fig.add_trace(go.Scatter(
            x=[x_positions[i] + 8, x_positions[i+1] - 8],
            y=[y_position, y_position],
            mode="lines",
            line=dict(color="#424245", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))
        # Arrow
        fig.add_annotation(
            x=x_positions[i+1] - 8,
            y=y_position,
            ax=x_positions[i] + 8,
            ay=y_position,
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#007aff"
        )
    
    # Draw agent nodes
    for i, agent_name in enumerate(agents_used):
        agent_info = agent_descriptions.get(agent_name, {
            "icon": "🤖",
            "description": "AI Agent",
            "color": "#86868b",
            "output": "Output"
        })
        
        # Agent circle
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_position],
            mode="markers+text",
            marker=dict(
                size=60,
                color=agent_info["color"],
                line=dict(width=3, color="#ffffff")
            ),
            text=[agent_info["icon"]],
            textfont=dict(size=24),
            name=agent_name,
            hovertemplate=f"<b>{agent_name}</b><br>{agent_info['description']}<br>Output: {agent_info['output']}<extra></extra>",
            showlegend=False
        ))
        
        # Agent label
        fig.add_annotation(
            x=x_positions[i],
            y=y_position - 15,
            text=agent_name.replace("Agent", ""),
            showarrow=False,
            font=dict(color="#f5f5f7", size=10, family="Inter"),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor=agent_info["color"],
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title=dict(
            text="Multi-Agent Pipeline: Collaborative AI Workflow",
            font=dict(color="#ffffff", size=18, family="Inter"),
            x=0.5
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 105]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        width=1000,
        height=200,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_physics_insights_dashboard(physics_results: Dict, geometry_data: Dict = None):
    """
    Create physics insights dashboard showing thermal, SI, PDN analysis.
    Aligned with research vision: real-time physics simulation.
    """
    if not physics_results:
        return None
    
    st.markdown("### 🔬 Physics Insights Dashboard")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 122, 255, 0.1) 0%, rgba(78, 201, 176, 0.1) 100%);
                border-left: 4px solid #007aff;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1.5rem;">
    <strong>Research Vision:</strong> Real-time physics simulation using neural fields (1000x faster than FDTD).
    This dashboard shows current physics analysis - future versions will use ML-accelerated simulation.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different physics domains
    physics_tabs = st.tabs(["🌡️ Thermal", "📡 Signal Integrity", "⚡ Power Integrity", "📊 Combined"])
    
    with physics_tabs[0]:
        thermal_data = physics_results.get("thermal", {})
        if thermal_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Thermal heatmap
                st.markdown("#### Thermal Distribution")
                # Create thermal visualization
                fig = create_thermal_visualization(thermal_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Thermal Metrics")
                max_temp = thermal_data.get("max_temperature", 25)
                hotspots = thermal_data.get("hotspots", [])
                thermal_gradient = thermal_data.get("thermal_gradient", 0)
                
                st.metric("Max Temperature", f"{max_temp:.1f}°C", 
                         delta=f"{max_temp - 25:.1f}°C above ambient")
                st.metric("Hotspots", len(hotspots), 
                         delta="Critical" if len(hotspots) > 3 else "OK")
                st.metric("Thermal Gradient", f"{thermal_gradient:.1f}°C",
                         delta="High" if thermal_gradient > 20 else "Low")
                
                # Recommendations
                if hotspots:
                    st.markdown("#### 🔍 Insights")
                    with st.expander("Thermal Recommendations"):
                        for rec in thermal_data.get("cooling_recommendations", []):
                            st.markdown(f"- {rec}")
    
    with physics_tabs[1]:
        si_data = physics_results.get("signal", {}).get("signal_integrity", {})
        if si_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Signal Integrity Analysis")
                # Create SI visualization
                fig = create_si_visualization(si_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### SI Metrics")
                impedance_nets = si_data.get("impedance_controlled_nets", [])
                crosstalk_risks = si_data.get("crosstalk_risks", [])
                emi_risks = si_data.get("emi_risks", [])
                
                st.metric("Impedance-Controlled Nets", len(impedance_nets))
                st.metric("Crosstalk Risks", len(crosstalk_risks),
                         delta="High" if len(crosstalk_risks) > 5 else "Low")
                st.metric("EMI Risks", len(emi_risks),
                         delta="High" if len(emi_risks) > 3 else "Low")
    
    with physics_tabs[2]:
        pdn_data = physics_results.get("pdn", {})
        if pdn_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Power Distribution Network")
                # Create PDN visualization
                fig = create_pdn_visualization(pdn_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### PDN Metrics")
                voltage_drops = pdn_data.get("voltage_drop", {})
                power_loss = pdn_data.get("power_loss", 0)
                decap_effectiveness = pdn_data.get("decoupling_effectiveness", {})
                
                max_v_drop = max(voltage_drops.values()) if voltage_drops else 0
                st.metric("Max Voltage Drop", f"{max_v_drop:.3f}V",
                         delta="High" if max_v_drop > 0.1 else "OK")
                st.metric("Power Loss", f"{power_loss:.2f}W")
                st.metric("Decoupling Caps", len(decap_effectiveness))
    
    with physics_tabs[3]:
        # Combined physics score
        physics_score = physics_results.get("physics_score", 1.0)
        
        st.markdown("#### Overall Physics Score")
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; font-weight: 700; color: {'#4ec9b0' if physics_score > 0.7 else '#f48771'};">
                {physics_score:.2f}
            </div>
            <div style="color: #86868b; font-size: 1.125rem; margin-top: 0.5rem;">
                Physics Quality Score
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Physics recommendations
        recommendations = physics_results.get("recommendations", [])
        if recommendations:
            st.markdown("#### 📋 Physics Recommendations")
            for rec in recommendations:
                st.info(f"💡 {rec}")


def create_geometry_insights_dashboard(geometry_data: Dict, placement_data: Dict):
    """
    Enhanced computational geometry dashboard with actionable insights.
    Aligned with research vision: geometric deep learning for generative design.
    """
    st.markdown("### 🔬 Computational Geometry Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(78, 201, 176, 0.1) 0%, rgba(0, 122, 255, 0.1) 100%);
                border-left: 4px solid #4ec9b0;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1.5rem;">
    <strong>Research Vision:</strong> Transform geometry from <em>analysis</em> to <em>generative design</em> using 
    geometric deep learning (GNNs for routing prediction, differentiable geometry optimization).
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics with insights
    col1, col2, col3, col4 = st.columns(4)
    
    mst_length = geometry_data.get("mst_length", 0)
    voronoi_variance = geometry_data.get("voronoi_variance", 0)
    thermal_hotspots = geometry_data.get("thermal_hotspots", 0)
    net_crossings = geometry_data.get("net_crossings", 0)
    
    with col1:
        mst_status = "Good" if mst_length < 150 else "Needs Optimization"
        st.metric(
            "MST Length",
            f"{mst_length:.1f} mm",
            delta=mst_status,
            delta_color="inverse",
            help="Minimum trace length estimate. Lower = better routing efficiency."
        )
    
    with col2:
        voronoi_status = "Uniform" if voronoi_variance < 50 else "Clustered"
        st.metric(
            "Voronoi Variance",
            f"{voronoi_variance:.1f}",
            delta=voronoi_status,
            delta_color="inverse",
            help="Component distribution uniformity. Lower = more uniform distribution."
        )
    
    with col3:
        thermal_status = "OK" if thermal_hotspots < 3 else "Critical"
        st.metric(
            "Thermal Hotspots",
            thermal_hotspots,
            delta=thermal_status,
            delta_color="inverse",
            help="High-power component regions. Lower = better thermal management."
        )
    
    with col4:
        crossings_status = "Low" if net_crossings < 10 else "High"
        st.metric(
            "Net Crossings",
            net_crossings,
            delta=crossings_status,
            delta_color="inverse",
            help="Potential routing conflicts. Lower = easier routing."
        )
    
    # Actionable insights
    st.markdown("#### 💡 Actionable Insights")
    
    insights = []
    
    if mst_length > 200:
        insights.append({
            "severity": "high",
            "title": "Long Trace Length Detected",
            "message": f"MST length is {mst_length:.1f}mm. Consider repositioning components to reduce trace length.",
            "action": "Optimize component placement for shorter connections"
        })
    
    if voronoi_variance > 100:
        insights.append({
            "severity": "medium",
            "title": "Uneven Component Distribution",
            "message": f"Voronoi variance is {voronoi_variance:.1f}. Components are clustered, which may cause thermal issues.",
            "action": "Redistribute components for better thermal spreading"
        })
    
    if thermal_hotspots > 3:
        insights.append({
            "severity": "high",
            "title": "Multiple Thermal Hotspots",
            "message": f"{thermal_hotspots} thermal hotspots detected. High-power components are too close together.",
            "action": "Increase spacing between high-power components or add thermal vias"
        })
    
    if net_crossings > 15:
        insights.append({
            "severity": "medium",
            "title": "High Routing Complexity",
            "message": f"{net_crossings} potential net crossings detected. Routing may be difficult.",
            "action": "Reposition components to reduce net crossings"
        })
    
    if insights:
        for insight in insights:
            severity_color = "#f48771" if insight["severity"] == "high" else "#ffd60a"
            st.markdown(f"""
            <div style="background: {severity_color}15;
                        border-left: 4px solid {severity_color};
                        padding: 1rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;">
                <strong style="color: {severity_color};">{insight['title']}</strong><br>
                <span style="color: #f5f5f7;">{insight['message']}</span><br>
                <span style="color: #86868b; font-size: 0.875rem;">💡 <em>{insight['action']}</em></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Geometry analysis shows good component distribution and routing potential!")
    
    # Research roadmap connection
    st.markdown("#### 🚀 Research Roadmap Connection")
    st.markdown("""
    <div style="background: #1d1d1f;
                border: 1px solid #424245;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;">
    <strong>Future Enhancements (from Research Roadmap):</strong>
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #86868b;">
        <li><strong>Graph Neural Networks:</strong> Predict optimal routing paths before autorouting (10x faster)</li>
        <li><strong>Differentiable Geometry:</strong> Gradient-based optimization through Voronoi/MST operations</li>
        <li><strong>Geometric Constraints:</strong> Predict manufacturability violations before DRC</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def create_thermal_visualization(thermal_data: Dict) -> go.Figure:
    """Create thermal heatmap visualization."""
    fig = go.Figure()
    
    # This would use actual thermal data
    # For now, create a placeholder
    fig.add_trace(go.Contour(
        z=[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
        colorscale="Hot",
        showscale=True
    ))
    
    fig.update_layout(
        title="Thermal Distribution",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc")
    )
    
    return fig


def create_si_visualization(si_data: Dict) -> go.Figure:
    """Create signal integrity visualization."""
    fig = go.Figure()
    
    # Placeholder
    fig.add_trace(go.Bar(
        x=["Impedance", "Crosstalk", "EMI"],
        y=[len(si_data.get("impedance_controlled_nets", [])),
           len(si_data.get("crosstalk_risks", [])),
           len(si_data.get("emi_risks", []))],
        marker_color="#007aff"
    ))
    
    fig.update_layout(
        title="Signal Integrity Metrics",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc")
    )
    
    return fig


def create_pdn_visualization(pdn_data: Dict) -> go.Figure:
    """Create power distribution network visualization."""
    fig = go.Figure()
    
    # Placeholder
    voltage_drops = pdn_data.get("voltage_drop", {})
    if voltage_drops:
        fig.add_trace(go.Bar(
            x=list(voltage_drops.keys())[:10],
            y=list(voltage_drops.values())[:10],
            marker_color="#4ec9b0"
        ))
    
    fig.update_layout(
        title="Voltage Drop Analysis",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc")
    )
    
    return fig

