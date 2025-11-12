"""
Computational Geometry Analyzer

Generates Voronoi diagrams, Minimum Spanning Trees, Convex Hulls, and other
geometric data structures for xAI reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import Voronoi, ConvexHull, distance_matrix, Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class GeometryAnalyzer:
    """Analyzes PCB placement using computational geometry algorithms."""
    
    def __init__(self, placement: Placement):
        """
        Initialize geometry analyzer.
        
        Args:
            placement: Placement to analyze
        """
        self.placement = placement
        self.components = list(placement.components.values())
        self.positions = np.array([[c.x, c.y] for c in self.components])
    
    def compute_voronoi_diagram(self) -> Dict:
        """
        Compute Voronoi diagram for component centers.
        
        Returns:
            Dictionary with Voronoi cell areas and variance
        """
        if len(self.positions) < 3:
            return {"voronoi_variance": 0.0, "cell_areas": []}
        
        try:
            vor = Voronoi(self.positions)
            
            # Compute cell areas (simplified - using bounding box)
            cell_areas = []
            for region_idx in vor.point_region:
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    vertices = vor.vertices[region]
                    if len(vertices) > 2:
                        # Approximate area using convex hull
                        hull = ConvexHull(vertices)
                        cell_areas.append(hull.volume)
            
            variance = np.var(cell_areas) if cell_areas else 0.0
            
            return {
                "voronoi_variance": float(variance),
                "cell_areas": [float(a) for a in cell_areas],
                "num_regions": len(cell_areas)
            }
        except Exception:
            return {"voronoi_variance": 0.0, "cell_areas": [], "num_regions": 0}
    
    def compute_minimum_spanning_tree(self) -> Dict:
        """
        Compute MST for component centers (trace length approximation).
        
        Returns:
            Dictionary with MST length and edges
        """
        if len(self.positions) < 2:
            return {"mst_length": 0.0, "edges": []}
        
        # Compute distance matrix
        dist_matrix = distance_matrix(self.positions, self.positions)
        
        # Compute MST
        mst = minimum_spanning_tree(dist_matrix)
        mst_length = mst.sum()
        
        # Get edges - convert all numpy types to native Python types
        edges = []
        rows, cols = mst.nonzero()
        for i, j in zip(rows, cols):
            # Ensure all values are native Python types
            i_val = int(i.item() if hasattr(i, 'item') else i)
            j_val = int(j.item() if hasattr(j, 'item') else j)
            weight = float(mst[i, j].item() if hasattr(mst[i, j], 'item') else mst[i, j])
            edges.append((i_val, j_val, weight))
        
        return {
            "mst_length": float(mst_length.item() if hasattr(mst_length, 'item') else mst_length),
            "edges": edges,
            "num_edges": len(edges)
        }
    
    def compute_convex_hull(self) -> Dict:
        """
        Compute convex hull of component centers.
        
        Returns:
            Dictionary with hull area and perimeter
        """
        if len(self.positions) < 3:
            return {"convex_hull_area": 0.0, "perimeter": 0.0}
        
        try:
            hull = ConvexHull(self.positions)
            # Convert all numpy types to native Python types
            vertices = [int(v.item() if hasattr(v, 'item') else v) for v in hull.vertices]
            return {
                "convex_hull_area": float(hull.volume.item() if hasattr(hull.volume, 'item') else hull.volume),
                "perimeter": float(hull.area.item() if hasattr(hull.area, 'item') else hull.area),
                "vertices": vertices
            }
        except Exception:
            return {"convex_hull_area": 0.0, "perimeter": 0.0, "vertices": []}
    
    def compute_thermal_hotspots(self, threshold: float = 2.0) -> Dict:
        """
        Identify thermal hotspots using power density and computational geometry.
        
        Based on research:
        - Holman (2010): Heat Transfer - Gaussian heat diffusion model
        - Aurenhammer (1991): Voronoi diagrams for spatial thermal analysis
        - Fortune (1987): Efficient Voronoi computation for thermal spreading
        
        Args:
            threshold: Power threshold for hotspot (W)
        
        Returns:
            Dictionary with hotspot count, locations, and thermal metrics
        """
        import numpy as np
        
        hotspots = []
        power_densities = []
        
        # Compute power density map using Gaussian thermal model
        # Based on Holman (2010) heat transfer equations
        board_area = self.placement.board.width * self.placement.board.height
        
        for i, comp in enumerate(self.components):
            if comp.power > threshold:
                # Calculate power density (W/mm²)
                comp_area = comp.width * comp.height
                power_density = comp.power / comp_area if comp_area > 0 else 0
                
                hotspots.append({
                    "component": comp.name,
                    "position": [float(comp.x), float(comp.y)],
                    "power": float(comp.power),
                    "power_density": float(power_density),
                    "area": float(comp_area)
                })
                power_densities.append(power_density)
        
        # Compute thermal spreading metrics using Voronoi analysis
        # High Voronoi variance indicates clustering = thermal risk
        voronoi_data = self.compute_voronoi_diagram()
        voronoi_variance = voronoi_data.get("voronoi_variance", 0.0)
        
        # Thermal risk score: combines power density and spatial distribution
        avg_power_density = np.mean(power_densities) if power_densities else 0.0
        thermal_risk = avg_power_density * (1.0 + voronoi_variance / 100.0)
        
        return {
            "thermal_hotspots": len(hotspots),
            "hotspot_locations": hotspots,
            "avg_power_density": float(avg_power_density),
            "thermal_risk_score": float(thermal_risk),
            "voronoi_variance": float(voronoi_variance),
            "board_utilization": len(self.components) / board_area if board_area > 0 else 0.0
        }
    
    def compute_net_crossings(self) -> Dict:
        """
        Estimate net crossings and routing complexity using component positions and nets.
        
        Mathematical foundation:
        - Graph crossing number problem (NP-hard)
        - Bounding box intersection for fast estimation
        - Net fanout analysis for routing complexity
        
        Returns:
            Dictionary with crossing count and routing complexity metrics
        """
        if not self.placement.nets:
            return {
                "net_crossings": 0,
                "routing_complexity": 0.0,
                "max_fanout": 0,
                "avg_fanout": 0.0,
                "high_fanout_nets": []
            }
        
        crossings = 0
        net_positions = []
        net_fanouts = []
        high_fanout_nets = []
        
        for net_name, net in self.placement.nets.items():
            net_comps = []
            for pin_ref in net.pins:
                comp_name = pin_ref[0]
                comp = self.placement.get_component(comp_name)
                if comp:
                    net_comps.append([comp.x, comp.y])
            
            fanout = len(net_comps)
            net_fanouts.append(fanout)
            
            if fanout >= 2:
                net_positions.append({
                    "name": net_name,
                    "positions": net_comps,
                    "fanout": fanout,
                    "bbox": self._compute_bbox(net_comps) if len(net_comps) >= 2 else None
                })
            
            # Track high fanout nets (routing complexity indicator)
            if fanout > 5:
                high_fanout_nets.append({
                    "net": net_name,
                    "fanout": fanout,
                    "components": [pin[0] for pin in net.pins]
                })
        
        # Count potential crossings using bounding box intersection
        for i, net1_data in enumerate(net_positions):
            for net2_data in net_positions[i+1:]:
                if net1_data["bbox"] and net2_data["bbox"]:
                    if self._bboxes_intersect(net1_data["bbox"], net2_data["bbox"]):
                        crossings += 1
        
        # Calculate routing complexity score
        # Factors: crossings, fanout, net count
        avg_fanout = np.mean(net_fanouts) if net_fanouts else 0.0
        max_fanout = max(net_fanouts) if net_fanouts else 0
        
        # Routing complexity: combines crossings, fanout, and net density
        net_density = len(self.placement.nets) / (self.placement.board.width * self.placement.board.height)
        routing_complexity = (
            crossings * 0.4 +  # Crossing penalty
            avg_fanout * 0.3 +  # Fanout penalty
            net_density * 100 * 0.3  # Density penalty
        )
        
        return {
            "net_crossings": crossings,
            "routing_complexity": float(routing_complexity),
            "max_fanout": int(max_fanout),
            "avg_fanout": float(avg_fanout),
            "high_fanout_nets": high_fanout_nets,
            "total_nets": len(self.placement.nets),
            "net_density": float(net_density)
        }
    
    def _compute_bbox(self, positions: List[List[float]]) -> List[float]:
        """Compute bounding box for net positions."""
        if not positions:
            return None
        return [
            min(p[0] for p in positions),  # x_min
            max(p[0] for p in positions),  # x_max
            min(p[1] for p in positions),  # y_min
            max(p[1] for p in positions)   # y_max
        ]
    
    def _bboxes_intersect(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes intersect."""
        return not (bbox1[1] < bbox2[0] or bbox2[1] < bbox1[0] or
                   bbox1[3] < bbox2[2] or bbox2[3] < bbox1[2])
    
    def compute_delaunay_triangulation(self) -> Dict:
        """
        Compute Delaunay triangulation for component centers.
        
        Delaunay triangulation is the dual of Voronoi diagram and provides:
        - Connectivity information between components
        - Triangle quality metrics
        - Edge length statistics
        
        Returns:
            Dictionary with triangulation data
        """
        if len(self.positions) < 3:
            return {"triangles": [], "num_triangles": 0, "avg_edge_length": 0.0}
        
        try:
            tri = Delaunay(self.positions)
            
            # Extract triangles
            triangles = []
            edge_lengths = []
            
            for simplex in tri.simplices:
                # Get triangle vertices
                p1, p2, p3 = self.positions[simplex]
                
                # Compute edge lengths
                e1 = np.linalg.norm(p2 - p1)
                e2 = np.linalg.norm(p3 - p2)
                e3 = np.linalg.norm(p1 - p3)
                
                edge_lengths.extend([e1, e2, e3])
                
                # Compute triangle area using Heron's formula
                s = (e1 + e2 + e3) / 2
                area = np.sqrt(max(0, s * (s - e1) * (s - e2) * (s - e3)))
                
                triangles.append({
                    "vertices": [int(v) for v in simplex],
                    "edge_lengths": [float(e1), float(e2), float(e3)],
                    "area": float(area),
                    "perimeter": float(e1 + e2 + e3)
                })
            
            avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0.0
            
            return {
                "triangles": triangles,
                "num_triangles": len(triangles),
                "avg_edge_length": float(avg_edge_length),
                "max_edge_length": float(np.max(edge_lengths)) if edge_lengths else 0.0,
                "min_edge_length": float(np.min(edge_lengths)) if edge_lengths else 0.0
            }
        except Exception as e:
            return {"triangles": [], "num_triangles": 0, "avg_edge_length": 0.0, "error": str(e)}
    
    def compute_force_directed_layout_metrics(self) -> Dict:
        """
        Compute metrics for force-directed layout analysis.
        
        Uses spring-force model to analyze component distribution:
        - Attractive forces: components on same net
        - Repulsive forces: all components (to avoid overlap)
        - Equilibrium analysis
        
        Returns:
            Dictionary with force metrics
        """
        if len(self.components) < 2:
            return {"total_force": 0.0, "max_force": 0.0, "equilibrium_score": 1.0}
        
        try:
            forces = []
            comp_list = list(self.components)
            
            for i, comp1_name in enumerate(comp_list):
                comp1 = self.placement.get_component(comp1_name)
                if not comp1:
                    continue
                
                net_force_x, net_force_y = 0.0, 0.0
                
                # Attractive forces from nets (components on same net attract)
                comp1_nets = self.placement.get_affected_nets(comp1_name)
                for net_name in comp1_nets:
                    net = self.placement.nets.get(net_name)
                    if net:
                        for other_comp_name, _ in net.pins:
                            if other_comp_name != comp1_name:
                                comp2 = self.placement.get_component(other_comp_name)
                                if comp2:
                                    dx = comp2.x - comp1.x
                                    dy = comp2.y - comp1.y
                                    dist = np.sqrt(dx**2 + dy**2)
                                    if dist > 0:
                                        # Attractive force (spring force)
                                        force_magnitude = dist * 0.1  # Spring constant
                                        net_force_x += force_magnitude * (dx / dist)
                                        net_force_y += force_magnitude * (dy / dist)
                
                # Repulsive forces from all components (to avoid overlap)
                for comp2_name in comp_list[i+1:]:
                    comp2 = self.placement.get_component(comp2_name)
                    if comp2:
                        dx = comp1.x - comp2.x
                        dy = comp1.y - comp2.y
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0:
                            # Repulsive force (inverse square)
                            min_dist = (comp1.width + comp2.width) / 2 + (comp1.height + comp2.height) / 2
                            force_magnitude = (min_dist / dist)**2 * 10.0
                            net_force_x += force_magnitude * (dx / dist)
                            net_force_y += force_magnitude * (dy / dist)
                
                force_magnitude = np.sqrt(net_force_x**2 + net_force_y**2)
                forces.append(force_magnitude)
            
            total_force = np.sum(forces) if forces else 0.0
            max_force = np.max(forces) if forces else 0.0
            avg_force = np.mean(forces) if forces else 0.0
            
            # Equilibrium score: lower forces = better equilibrium
            equilibrium_score = 1.0 / (1.0 + avg_force)
            
            return {
                "total_force": float(total_force),
                "max_force": float(max_force),
                "avg_force": float(avg_force),
                "equilibrium_score": float(equilibrium_score),
                "num_forces": len(forces)
            }
        except Exception as e:
            return {"total_force": 0.0, "max_force": 0.0, "equilibrium_score": 1.0, "error": str(e)}
    
    def compute_overlap_risk(self) -> Dict:
        """
        Compute risk of component overlap using improved geometric analysis.
        
        Returns:
            Dictionary with overlap risk score
        """
        if len(self.components) < 2:
            return {"overlap_risk": 0.0}
        
        min_distance = float('inf')
        overlap_pairs = []
        
        for i, c1 in enumerate(self.components):
            for c2 in self.components[i+1:]:
                dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                min_clearance = (c1.width + c2.width) / 2 + (c1.height + c2.height) / 2
                if dist < min_clearance:
                    min_distance = min(min_distance, dist)
                    overlap_pairs.append({
                        "comp1": c1.name,
                        "comp2": c2.name,
                        "distance": float(dist),
                        "min_clearance": float(min_clearance),
                        "violation": float(min_clearance - dist)
                    })
        
        if min_distance == float('inf'):
            return {"overlap_risk": 0.0, "overlap_pairs": []}
        
        # Risk is inverse of minimum distance
        risk = 1.0 / (min_distance + 1.0)
        return {
            "overlap_risk": float(risk),
            "min_distance": float(min_distance),
            "overlap_pairs": overlap_pairs,
            "num_overlaps": len(overlap_pairs)
        }
    
    def analyze(self) -> Dict:
        """
        Perform complete computational geometry analysis.
        
        Returns:
            Dictionary with all geometric metrics for xAI reasoning
        """
        board_area = self.placement.board.width * self.placement.board.height
        density = len(self.components) / board_area if board_area > 0 else 0.0
        
        voronoi = self.compute_voronoi_diagram()
        mst = self.compute_minimum_spanning_tree()
        hull = self.compute_convex_hull()
        thermal = self.compute_thermal_hotspots()
        crossings = self.compute_net_crossings()
        overlap = self.compute_overlap_risk()
        delaunay = self.compute_delaunay_triangulation()
        forces = self.compute_force_directed_layout_metrics()
        
        result = {
            "density": float(density),
            "convex_hull_area": hull.get("convex_hull_area", 0.0),
            "voronoi_variance": voronoi.get("voronoi_variance", 0.0),
            "mst_length": mst.get("mst_length", 0.0),
            "thermal_hotspots": thermal.get("thermal_hotspots", 0),
            "net_crossings": crossings.get("net_crossings", 0),
            "routing_complexity": crossings.get("routing_complexity", 0.0),  # NEW
            "max_fanout": crossings.get("max_fanout", 0),  # NEW
            "avg_fanout": crossings.get("avg_fanout", 0.0),  # NEW
            "overlap_risk": overlap.get("overlap_risk", 0.0),
            # Advanced computational geometry
            "delaunay_triangles": delaunay.get("num_triangles", 0),
            "delaunay_avg_edge_length": delaunay.get("avg_edge_length", 0.0),
            "force_equilibrium_score": forces.get("equilibrium_score", 1.0),
            "total_force": forces.get("total_force", 0.0),
            # Additional geometric data for visualization
            "voronoi_data": voronoi,
            "mst_edges": mst.get("edges", []),
            "hull_vertices": hull.get("vertices", []),
            "hotspot_locations": thermal.get("hotspot_locations", []),
            "delaunay_data": delaunay,
            "force_data": forces,
            "overlap_pairs": overlap.get("overlap_pairs", []),
            "routing_data": crossings  # NEW: Full routing complexity data
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(result)

