"""
Circuit Visualization - Proper PCB and Schematic Views

Creates proper circuit visualizations showing:
- PCB layout with components, traces, and layers
- Schematic view with component symbols and connections
- Not just thermal maps!
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional


def create_circuit_visualization(placement_data: Dict, view_type: str = "pcb") -> go.Figure:
    """
    Create proper circuit visualization (PCB or schematic).
    
    Args:
        placement_data: Placement dictionary with board, components, nets
        view_type: "pcb" or "schematic"
    
    Returns:
        Plotly figure with circuit visualization
    """
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])
    nets = placement_data.get("nets", [])
    
    board_width = board.get("width", 100)
    board_height = board.get("height", 100)
    
    if view_type == "pcb":
        return create_pcb_layout_view(board_width, board_height, components, nets)
    else:
        return create_schematic_view(components, nets)


def create_pcb_layout_view(
    board_width: float,
    board_height: float,
    components: List[Dict],
    nets: List[Dict]
) -> go.Figure:
    """Create proper PCB layout visualization."""
    fig = go.Figure()
    
    # Board outline (Edge.Cuts layer)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="#00ff00", width=3),  # Green for Edge.Cuts
        fillcolor="rgba(0, 255, 0, 0.05)",
        layer="below"
    )
    
    # Add components as proper footprints
    for comp in components:
        name = comp.get("name", "UNK")
        x = comp.get("x", 0)
        y = comp.get("y", 0)
        width = comp.get("width", 5)
        height = comp.get("height", 5)
        package = comp.get("package", "Unknown")
        angle = comp.get("angle", 0)
        
        # Component body (F.Fab layer - fabrication outline)
        fig.add_shape(
            type="rect",
            x0=x - width/2, y0=y - height/2,
            x1=x + width/2, y1=y + height/2,
            line=dict(color="#808080", width=1),  # Gray for Fab layer
            fillcolor="rgba(128, 128, 128, 0.1)",
            layer="below"
        )
        
        # Component reference (F.SilkS layer)
        fig.add_trace(go.Scatter(
            x=[x], y=[y - height/2 - 1],
            mode="text",
            text=name,
            textfont=dict(size=10, color="#ffff00"),  # Yellow for SilkS
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Component value (F.Fab layer)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=package,
            textfont=dict(size=8, color="#808080"),  # Gray for Fab
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Pads (F.Cu layer - copper pads)
        pads = get_pads_for_component(comp)
        for pad_x, pad_y in pads:
            # Rotate pad position based on component angle
            if angle != 0:
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                pad_x_rot = (pad_x * cos_a - pad_y * sin_a) + x
                pad_y_rot = (pad_x * sin_a + pad_y * cos_a) + y
            else:
                pad_x_rot = pad_x + x
                pad_y_rot = pad_y + y
            
            fig.add_trace(go.Scatter(
                x=[pad_x_rot], y=[pad_y_rot],
                mode="markers",
                marker=dict(
                    size=8,
                    color="#ff8800",  # Orange for copper pads
                    line=dict(width=1, color="#000000")
                ),
                name=f"{name} pads",
                showlegend=False,
                hovertemplate=f"<b>{name}</b> Pad<br>Package: {package}<extra></extra>"
            ))
    
    # Add traces (F.Cu layer - copper traces)
    shown_nets = set()
    for net in nets:
        net_name = net.get("name", "")
        net_pins = net.get("pins", [])
        
        if len(net_pins) >= 2:
            positions = []
            for pin_ref in net_pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    comp_name = pin_ref[0]
                    comp = next((c for c in components if c.get("name") == comp_name), None)
                    if comp:
                        # Find pad position for this pin
                        pad_pos = get_pad_position_for_pin(comp, pin_ref[1])
                        if pad_pos:
                            positions.append(pad_pos)
            
            # Draw traces between pads
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    x0, y0 = positions[i]
                    x1, y1 = positions[i+1]
                    
                    # Determine net color based on type
                    net_color = get_net_color(net_name)
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode="lines",
                        line=dict(color=net_color, width=2),
                        name=net_name,
                        showlegend=(net_name not in shown_nets),
                        legendgroup=net_name,
                        hovertemplate=f"<b>{net_name}</b><br>Trace<extra></extra>"
                    ))
                shown_nets.add(net_name)
    
    # Update layout
    fig.update_layout(
        title=dict(text="PCB Layout View (F.Cu Layer)", font=dict(color="#ffffff", size=16)),
        xaxis=dict(
            title=dict(text="X (mm)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False,
            range=[-5, board_width + 5]
        ),
        yaxis=dict(
            title=dict(text="Y (mm)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            range=[-5, board_height + 5]
        ),
        width=1000,
        height=700,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc"),
        legend=dict(
            bgcolor="#252526",
            bordercolor="#3e3e42",
            borderwidth=1
        )
    )
    
    return fig


def create_schematic_view(components: List[Dict], nets: List[Dict]) -> go.Figure:
    """Create schematic view with component symbols and connections."""
    fig = go.Figure()
    
    # Add component symbols
    for comp in components:
        name = comp.get("name", "UNK")
        x = comp.get("x", 0) * 2.54  # Convert to schematic units
        y = comp.get("y", 0) * 2.54
        package = comp.get("package", "Unknown")
        
        # Component symbol (rectangle)
        fig.add_shape(
            type="rect",
            x0=x - 5, y0=y - 3,
            x1=x + 5, y1=y + 3,
            line=dict(color="#00ffff", width=2),  # Cyan for schematic
            fillcolor="rgba(0, 255, 255, 0.1)",
            layer="below"
        )
        
        # Reference designator
        fig.add_trace(go.Scatter(
            x=[x], y=[y - 4],
            mode="text",
            text=name,
            textfont=dict(size=12, color="#00ffff", family="monospace"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Value/package
        fig.add_trace(go.Scatter(
            x=[x], y=[y + 4],
            mode="text",
            text=package,
            textfont=dict(size=10, color="#888888", family="monospace"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Pins
        pins = get_schematic_pins(comp)
        for pin_x, pin_y, pin_name in pins:
            fig.add_trace(go.Scatter(
                x=[x + pin_x], y=[y + pin_y],
                mode="markers",
                marker=dict(size=6, color="#ffff00", symbol="square"),
                name=f"{name} pins",
                showlegend=False,
                hovertemplate=f"<b>{name}</b> Pin {pin_name}<extra></extra>"
            ))
    
    # Add wires (connections)
    for net in nets:
        net_name = net.get("name", "")
        net_pins = net.get("pins", [])
        
        if len(net_pins) >= 2:
            positions = []
            for pin_ref in net_pins:
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    comp_name = pin_ref[0]
                    comp = next((c for c in components if c.get("name") == comp_name), None)
                    if comp:
                        pin_pos = get_schematic_pin_position(comp, pin_ref[1])
                        if pin_pos:
                            positions.append(pin_pos)
            
            # Draw wires
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    x0, y0 = positions[i]
                    x1, y1 = positions[i+1]
                    
                    net_color = get_net_color(net_name)
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode="lines",
                        line=dict(color=net_color, width=2, dash="solid"),
                        name=net_name,
                        showlegend=True,
                        hovertemplate=f"<b>{net_name}</b><br>Wire<extra></extra>"
                    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text="Schematic View", font=dict(color="#ffffff", size=16)),
        xaxis=dict(
            title=dict(text="X (schematic units)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Y (schematic units)", font=dict(color="#cccccc")),
            tickfont=dict(color="#cccccc"),
            gridcolor="#3e3e42",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        width=1000,
        height=700,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#cccccc"),
        legend=dict(
            bgcolor="#252526",
            bordercolor="#3e3e42",
            borderwidth=1
        )
    )
    
    return fig


def get_pads_for_component(comp: Dict) -> List[tuple]:
    """Get pad positions for a component."""
    package = comp.get("package", "").lower()
    width = comp.get("width", 5)
    height = comp.get("height", 5)
    
    if "soic" in package or "so" in package:
        # SOIC: pads on sides
        pad_spacing = min(width, height) * 0.8
        return [
            (-pad_spacing/2, -height/2),
            (-pad_spacing/2, 0),
            (-pad_spacing/2, height/2),
            (pad_spacing/2, height/2),
            (pad_spacing/2, 0),
            (pad_spacing/2, -height/2)
        ]
    else:
        # Default: 2 pads
        return [(-width/4, 0), (width/4, 0)]


def get_pad_position_for_pin(comp: Dict, pin_name: str) -> Optional[tuple]:
    """Get absolute pad position for a pin."""
    pads = get_pads_for_component(comp)
    x = comp.get("x", 0)
    y = comp.get("y", 0)
    angle = comp.get("angle", 0)
    
    # Try to match pin name to pad index
    pin_num = None
    if isinstance(pin_name, str):
        # Extract number from pin name
        import re
        match = re.search(r'\d+', pin_name)
        if match:
            pin_num = int(match.group()) - 1
    
    if pin_num is not None and 0 <= pin_num < len(pads):
        pad_x, pad_y = pads[pin_num]
        
        # Rotate pad position
        if angle != 0:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            pad_x_rot = (pad_x * cos_a - pad_y * sin_a) + x
            pad_y_rot = (pad_x * sin_a + pad_y * cos_a) + y
            return (pad_x_rot, pad_y_rot)
        else:
            return (pad_x + x, pad_y + y)
    
    # Fallback: return component center
    return (x, y)


def get_schematic_pins(comp: Dict) -> List[tuple]:
    """Get schematic pin positions."""
    package = comp.get("package", "").lower()
    
    if "soic" in package or "so" in package:
        return [
            (-5, -2, "1"), (-5, -1, "2"), (-5, 0, "3"), (-5, 1, "4"),
            (5, 1, "5"), (5, 0, "6"), (5, -1, "7"), (5, -2, "8")
        ]
    else:
        return [(-5, 0, "1"), (5, 0, "2")]


def get_schematic_pin_position(comp: Dict, pin_name: str) -> Optional[tuple]:
    """Get absolute schematic pin position."""
    pins = get_schematic_pins(comp)
    x = comp.get("x", 0) * 2.54
    y = comp.get("y", 0) * 2.54
    
    pin_num = None
    if isinstance(pin_name, str):
        import re
        match = re.search(r'\d+', pin_name)
        if match:
            pin_num = int(match.group()) - 1
    
    if pin_num is not None and 0 <= pin_num < len(pins):
        pin_x, pin_y, _ = pins[pin_num]
        return (x + pin_x, y + pin_y)
    
    return None


def get_net_color(net_name: str) -> str:
    """Get color for net based on type."""
    name_lower = net_name.lower()
    
    if any(x in name_lower for x in ["vcc", "vdd", "power", "vin"]):
        return "#ff0000"  # Red for power
    elif any(x in name_lower for x in ["gnd", "ground", "vss"]):
        return "#0000ff"  # Blue for ground
    elif any(x in name_lower for x in ["clk", "clock"]):
        return "#00ff00"  # Green for clock
    else:
        return "#ffff00"  # Yellow for signal

