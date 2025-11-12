# Computational Geometry Usage in Dielectric

## Overview

Dielectric extensively uses computational geometry algorithms to analyze and optimize PCB layouts. These algorithms provide quantitative metrics that feed into AI reasoning (Grok/xAI) and guide optimization strategies.

## Algorithms Implemented

### 1. **Voronoi Diagrams** (`compute_voronoi_diagram`)
- **Purpose**: Analyze component distribution and identify clustering
- **Metrics**:
  - `voronoi_variance`: Variance in Voronoi cell areas (lower = more uniform distribution)
  - `cell_areas`: Area of each Voronoi cell
  - `num_regions`: Number of valid Voronoi regions
- **Use Case**: 
  - Detect thermal hotspots (clustered components = high variance)
  - Identify optimal component spacing
  - Guide thermal spreading optimization

### 2. **Minimum Spanning Tree (MST)** (`compute_minimum_spanning_tree`)
- **Purpose**: Optimize trace routing by minimizing total wire length
- **Metrics**:
  - `mst_length`: Total length of MST edges (shorter = better routing)
  - `edges`: List of MST edges connecting components
  - `avg_edge_length`: Average edge length
- **Use Case**:
  - Minimize trace length (signal integrity)
  - Reduce routing complexity
  - Guide component placement for shorter connections

### 3. **Convex Hull** (`compute_convex_hull`)
- **Purpose**: Measure board utilization and component spread
- **Metrics**:
  - `convex_hull_area`: Area covered by component placement
  - `hull_vertices`: Vertices of the convex hull
  - `utilization_ratio`: Ratio of hull area to board area
- **Use Case**:
  - Assess board space efficiency
  - Identify wasted space
  - Guide compact placement

### 4. **Delaunay Triangulation** (`compute_delaunay_triangulation`)
- **Purpose**: Analyze local connectivity and component relationships
- **Metrics**:
  - `num_triangles`: Number of Delaunay triangles
  - `avg_edge_length`: Average edge length in triangulation
  - `max_edge_length`: Maximum edge length
- **Use Case**:
  - Identify component neighborhoods
  - Analyze local density variations
  - Guide hierarchical placement

### 5. **Thermal Hotspot Analysis** (`compute_thermal_hotspots`)
- **Purpose**: Identify regions with high thermal density
- **Method**: Gaussian thermal diffusion model
- **Metrics**:
  - `thermal_hotspots`: Number of hotspot regions
  - `thermal_risk_score`: Overall thermal risk (0-1)
  - `hotspot_locations`: Coordinates of hotspots
- **Use Case**:
  - Prevent thermal issues
  - Guide component spacing for cooling
  - Optimize for thermal management

### 6. **Net Crossing Analysis** (`compute_net_crossings`)
- **Purpose**: Detect routing conflicts and complexity
- **Metrics**:
  - `net_crossings`: Number of potential net crossings
  - `routing_complexity`: Complexity score (0-1)
  - `max_fanout`: Maximum component fanout
  - `avg_fanout`: Average component fanout
- **Use Case**:
  - Minimize routing conflicts
  - Identify high-complexity nets
  - Guide placement to reduce crossings

### 7. **Force-Directed Layout Metrics** (`compute_force_directed_layout_metrics`)
- **Purpose**: Analyze force equilibrium in component placement
- **Metrics**:
  - `equilibrium_score`: How close to force equilibrium (1.0 = perfect)
  - `total_force`: Total force magnitude
  - `force_vectors`: Force vectors for each component
- **Use Case**:
  - Guide force-directed placement algorithms
  - Assess placement stability
  - Optimize for balanced forces

### 8. **Overlap Risk Analysis** (`compute_overlap_risk`)
- **Purpose**: Detect potential component collisions
- **Metrics**:
  - `overlap_risk`: Risk score (0-1)
  - `num_overlaps`: Number of overlapping pairs
  - `overlap_pairs`: List of overlapping component pairs
- **Use Case**:
  - Prevent design rule violations
  - Ensure manufacturability
  - Guide clearance optimization

## Integration with AI Reasoning

All computational geometry metrics are passed to Grok/xAI for reasoning:

1. **Design Generation**: Geometry metrics inform initial placement
2. **Intent Analysis**: Geometry data helps map user intent to optimization weights
3. **Optimization Strategy**: Real-time geometry analysis guides simulated annealing
4. **Post-Optimization**: Geometry comparison shows improvements

## Usage in Optimization

### During Simulated Annealing:
- Geometry analysis runs every **25 iterations** (configurable)
- Metrics feed into xAI reasoning calls
- Strategy adjustments based on geometry state:
  - High Voronoi variance → Focus on thermal spreading
  - Long MST → Focus on trace length minimization
  - Many net crossings → Focus on routing optimization
  - High overlap risk → Focus on clearance violations

### Metrics Returned:
```python
{
    "density": 0.003,  # components/mm²
    "voronoi_variance": 0.45,
    "mst_length": 234.5,  # mm
    "convex_hull_area": 8500.0,  # mm²
    "thermal_hotspots": 3,
    "thermal_risk_score": 0.65,
    "net_crossings": 12,
    "routing_complexity": 0.42,
    "overlap_risk": 0.08,
    "delaunay_triangles": 45,
    "force_equilibrium_score": 0.89,
    # ... plus detailed data structures
}
```

## Performance

- **Analysis Time**: ~50-200ms per analysis (depends on component count)
- **Frequency**: Every 25 iterations during optimization
- **Memory**: O(n²) for some algorithms (n = component count)

## Future Enhancements

- GPU acceleration for large designs
- Incremental geometry updates (only recompute changed regions)
- Advanced routing algorithms (A*, maze routing)
- 3D thermal analysis
- Multi-layer geometry analysis

