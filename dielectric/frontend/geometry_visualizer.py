"""
Computational Geometry Visualizations

Shows Voronoi diagrams, Minimum Spanning Trees, Convex Hulls, and other
geometric data structures that feed into xAI reasoning.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import Voronoi, ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def visualize_voronoi_diagram(
    components: List[Dict],
    board_width: float,
    board_height: float,
    title: str = "Voronoi Diagram - Component Distribution"
) -> go.Figure:
    """
    Visualize Voronoi diagram showing component distribution.
    
    Voronoi diagrams help identify:
    - Component clustering (modules)
    - Distribution uniformity
    - Placement density
    """
    if len(components) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 components for Voronoi diagram", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract component positions
    points = np.array([[comp.get("x", 0), comp.get("y", 0)] for comp in components])
    comp_names = [comp.get("name", f"C{i}") for i, comp in enumerate(components)]
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(points)
    except:
        fig = go.Figure()
        fig.add_annotation(text="Could not compute Voronoi diagram", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Board outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="#4ec9b0", width=2),
        fillcolor="rgba(78, 201, 176, 0.05)",
        layer="below"
    )
    
    # Draw Voronoi regions
    for i, region in enumerate(vor.regions):
        if len(region) == 0 or -1 in region:
            continue
        
        # Get vertices of Voronoi region
        vertices = vor.vertices[region]
        
        # Filter vertices within board bounds
        valid_vertices = []
        for v in vertices:
            if 0 <= v[0] <= board_width and 0 <= v[1] <= board_height:
                valid_vertices.append(v)
        
        if len(valid_vertices) >= 3:
            vertices = np.array(valid_vertices)
            # Close the polygon
            vertices = np.vstack([vertices, vertices[0]])
            
            fig.add_trace(go.Scatter(
                x=vertices[:, 0],
                y=vertices[:, 1],
                mode='lines',
                line=dict(color='rgba(78, 201, 176, 0.3)', width=1),
                fill='toself',
                fillcolor='rgba(78, 201, 176, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw Voronoi vertices
    if len(vor.vertices) > 0:
        fig.add_trace(go.Scatter(
            x=vor.vertices[:, 0],
            y=vor.vertices[:, 1],
            mode='markers',
            marker=dict(size=4, color='#4ec9b0', symbol='circle'),
            name='Voronoi Vertices',
            hovertemplate='Vertex: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
    
    # Draw components
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='#f48771', symbol='square', line=dict(width=2, color='white')),
        text=comp_names,
        textposition='top center',
        name='Components',
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=18)),
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_width + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_height + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#e0e0e0'),
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    )
    
    return fig


def visualize_minimum_spanning_tree(
    components: List[Dict],
    nets: List[Dict],
    board_width: float,
    board_height: float,
    title: str = "Minimum Spanning Tree - Optimal Trace Routing"
) -> go.Figure:
    """
    Visualize Minimum Spanning Tree showing optimal trace routing.
    
    MST helps:
    - Estimate minimum trace length
    - Identify critical connections
    - Optimize routing paths
    """
    if len(components) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 components for MST", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract component positions
    comp_dict = {comp.get("name"): comp for comp in components}
    points = np.array([[comp.get("x", 0), comp.get("y", 0)] for comp in components])
    comp_names = [comp.get("name", f"C{i}") for i, comp in enumerate(components)]
    
    # Build connectivity graph from nets
    n = len(components)
    distance_matrix = np.zeros((n, n))
    
    # Calculate distances between connected components
    for net in nets:
        net_pins = net.get("pins", [])
        connected_comps = []
        for pin in net_pins:
            if isinstance(pin, list) and len(pin) >= 1:
                comp_name = pin[0]
                if comp_name in comp_dict:
                    if comp_name not in connected_comps:
                        connected_comps.append(comp_name)
        
        # Connect all components in this net
        for i, comp1_name in enumerate(connected_comps):
            if comp1_name not in comp_names:
                continue
            idx1 = comp_names.index(comp1_name)
            for comp2_name in connected_comps[i+1:]:
                if comp2_name not in comp_names:
                    continue
                idx2 = comp_names.index(comp2_name)
                dist = np.sqrt((points[idx1][0] - points[idx2][0])**2 + 
                              (points[idx1][1] - points[idx2][1])**2)
                distance_matrix[idx1, idx2] = dist
                distance_matrix[idx2, idx1] = dist
    
    # If no net connections, use all-to-all
    if np.sum(distance_matrix) == 0:
        distance_matrix = squareform(pdist(points))
    
    # Compute MST
    try:
        mst = minimum_spanning_tree(distance_matrix)
        mst_dense = mst.toarray()
    except:
        fig = go.Figure()
        fig.add_annotation(text="Could not compute MST", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Board outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="#4ec9b0", width=2),
        fillcolor="rgba(78, 201, 176, 0.05)",
        layer="below"
    )
    
    # Draw MST edges
    total_length = 0
    for i in range(n):
        for j in range(i+1, n):
            if mst_dense[i, j] > 0:
                total_length += mst_dense[i, j]
                fig.add_trace(go.Scatter(
                    x=[points[i][0], points[j][0]],
                    y=[points[i][1], points[j][1]],
                    mode='lines',
                    line=dict(color='#4ec9b0', width=3),
                    showlegend=False,
                    hovertemplate=f'Distance: {mst_dense[i, j]:.2f}mm<extra></extra>'
                ))
    
    # Draw components
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='#f48771', symbol='square', line=dict(width=2, color='white')),
        text=comp_names,
        textposition='top center',
        name='Components',
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>Total MST Length: {total_length:.1f}mm</sub>", 
                  font=dict(color='white', size=18)),
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_width + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_height + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#e0e0e0'),
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    )
    
    return fig


def visualize_convex_hull(
    components: List[Dict],
    board_width: float,
    board_height: float,
    title: str = "Convex Hull - Board Utilization"
) -> go.Figure:
    """
    Visualize Convex Hull showing board utilization.
    
    Convex Hull helps:
    - Measure board utilization
    - Identify wasted space
    - Optimize component placement
    """
    if len(components) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 3 components for Convex Hull", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract component positions
    points = np.array([[comp.get("x", 0), comp.get("y", 0)] for comp in components])
    comp_names = [comp.get("name", f"C{i}") for i, comp in enumerate(components)]
    
    # Compute Convex Hull
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_area = hull.volume  # For 2D, volume is area
    except:
        fig = go.Figure()
        fig.add_annotation(text="Could not compute Convex Hull", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Board outline
    board_area = board_width * board_height
    utilization = (hull_area / board_area) * 100 if board_area > 0 else 0
    
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=board_width, y1=board_height,
        line=dict(color="#4ec9b0", width=2),
        fillcolor="rgba(78, 201, 176, 0.05)",
        layer="below"
    )
    
    # Draw Convex Hull
    hull_closed = np.vstack([hull_points, hull_points[0]])
    fig.add_trace(go.Scatter(
        x=hull_closed[:, 0],
        y=hull_closed[:, 1],
        mode='lines',
        line=dict(color='#4ec9b0', width=3, dash='dash'),
        fill='toself',
        fillcolor='rgba(78, 201, 176, 0.2)',
        name=f'Convex Hull (Area: {hull_area:.1f}mm²)',
        hovertemplate='Hull Point: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    # Draw components
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='#f48771', symbol='square', line=dict(width=2, color='white')),
        text=comp_names,
        textposition='top center',
        name='Components',
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>Utilization: {utilization:.1f}% | Board: {board_area:.0f}mm²</sub>", 
                  font=dict(color='white', size=18)),
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_width + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#cccccc")), 
                  range=[-10, board_height + 10], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#e0e0e0'),
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    )
    
    return fig


def create_geometry_dashboard(
    placement_data: Dict,
    geometry_data: Optional[Dict] = None
) -> go.Figure:
    """
    Create comprehensive computational geometry dashboard.
    
    Shows all geometric analyses in subplots:
    - Voronoi Diagram
    - Minimum Spanning Tree
    - Convex Hull
    - Thermal Heatmap
    """
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])
    nets = placement_data.get("nets", [])
    
    board_width = board.get("width", 100)
    board_height = board.get("height", 100)
    
    # Create subplots - use proper subplot types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Voronoi Diagram", "Minimum Spanning Tree", 
                       "Convex Hull", "Thermal Heatmap"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    points = np.array([[comp.get("x", 0), comp.get("y", 0)] for comp in components]) if components else np.array([])
    comp_names = [comp.get("name", f"C{i}") for i, comp in enumerate(components)]
    
    # 1. Voronoi Diagram (top-left) - row=1, col=1
    if len(components) >= 2:
        try:
            vor = Voronoi(points)
            
            # Draw Voronoi regions
            for region in vor.regions:
                if len(region) == 0 or -1 in region:
                    continue
                vertices = vor.vertices[region]
                valid_vertices = [v for v in vertices if 0 <= v[0] <= board_width and 0 <= v[1] <= board_height]
                if len(valid_vertices) >= 3:
                    vertices = np.array(valid_vertices)
                    vertices = np.vstack([vertices, vertices[0]])
                    fig.add_trace(
                        go.Scatter(
                            x=vertices[:, 0], y=vertices[:, 1],
                            mode='lines', line=dict(color='rgba(78, 201, 176, 0.3)', width=1),
                            fill='toself', fillcolor='rgba(78, 201, 176, 0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        except Exception as e:
            pass
    
    # 2. Minimum Spanning Tree (top-right) - row=1, col=2
    if len(components) >= 2:
        try:
            distance_matrix = squareform(pdist(points))
            mst = minimum_spanning_tree(distance_matrix)
            mst_dense = mst.toarray()
            
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    if mst_dense[i, j] > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=[points[i][0], points[j][0]],
                                y=[points[i][1], points[j][1]],
                                mode='lines', line=dict(color='#4ec9b0', width=2),
                                showlegend=False
                            ),
                            row=1, col=2
                        )
        except Exception as e:
            pass
    
    # 3. Convex Hull (bottom-left) - row=2, col=1
    if len(components) >= 3:
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_closed = np.vstack([hull_points, hull_points[0]])
            
            fig.add_trace(
                go.Scatter(
                    x=hull_closed[:, 0], y=hull_closed[:, 1],
                    mode='lines', line=dict(color='#4ec9b0', width=3, dash='dash'),
                    fill='toself', fillcolor='rgba(78, 201, 176, 0.2)',
                    showlegend=False
                ),
                row=2, col=1
            )
        except Exception as e:
            pass
    
    # 4. Thermal Heatmap (bottom-right) - row=2, col=2
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
        
        # Add contour to subplot correctly
        fig.add_trace(
            go.Contour(
                x=x_grid, y=y_grid, z=thermal_map,
                colorscale="Hot", showscale=True,
                name="Thermal", opacity=0.6
            ),
            row=2, col=2
        )
    
    # Add components to all subplots
    if len(components) > 0:
        for subplot_row, subplot_col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_trace(
                go.Scatter(
                    x=points[:, 0], y=points[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='#f48771', symbol='square', line=dict(width=1, color='white')),
                    text=comp_names,
                    showlegend=(subplot_row == 1 and subplot_col == 1),
                    name='Components'
                ),
                row=subplot_row, col=subplot_col
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text="Computational Geometry Analysis Dashboard", 
                  font=dict(color='white', size=20)),
        height=1200,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#e0e0e0'),
        showlegend=True
    )
    
    # Update axes for all subplots
    for subplot_row in [1, 2]:
        for subplot_col in [1, 2]:
            fig.update_xaxes(
                title_text="X (mm)", 
                range=[-10, board_width + 10], 
                row=subplot_row, col=subplot_col, 
                gridcolor='rgba(255,255,255,0.1)'
            )
            fig.update_yaxes(
                title_text="Y (mm)", 
                range=[-10, board_height + 10], 
                row=subplot_row, col=subplot_col, 
                gridcolor='rgba(255,255,255,0.1)'
            )
    
    return fig

