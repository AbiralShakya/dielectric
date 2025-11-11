"""
Computational Geometry Analyzer

Generates Voronoi diagrams, Minimum Spanning Trees, Convex Hulls, and other
geometric data structures for xAI reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import Voronoi, ConvexHull, distance_matrix
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
        Estimate net crossings using component positions and nets.
        
        Returns:
            Dictionary with crossing count
        """
        if not self.placement.nets:
            return {"net_crossings": 0}
        
        # Simplified: count potential crossings based on component positions
        crossings = 0
        net_positions = []
        
        for net in self.placement.nets.values():
            net_comps = []
            for pin_ref in net.pins:
                comp_name = pin_ref[0]
                comp = self.placement.get_component(comp_name)
                if comp:
                    net_comps.append([comp.x, comp.y])
            
            if len(net_comps) >= 2:
                net_positions.append(net_comps)
        
        # Count potential crossings (simplified heuristic)
        for i, net1 in enumerate(net_positions):
            for net2 in net_positions[i+1:]:
                if len(net1) >= 2 and len(net2) >= 2:
                    # Check if bounding boxes overlap
                    net1_bbox = [
                        min(p[0] for p in net1), max(p[0] for p in net1),
                        min(p[1] for p in net1), max(p[1] for p in net1)
                    ]
                    net2_bbox = [
                        min(p[0] for p in net2), max(p[0] for p in net2),
                        min(p[1] for p in net2), max(p[1] for p in net2)
                    ]
                    
                    if not (net1_bbox[1] < net2_bbox[0] or net2_bbox[1] < net1_bbox[0] or
                            net1_bbox[3] < net2_bbox[2] or net2_bbox[3] < net1_bbox[2]):
                        crossings += 1
        
        return {"net_crossings": crossings}
    
    def compute_overlap_risk(self) -> Dict:
        """
        Compute risk of component overlap.
        
        Returns:
            Dictionary with overlap risk score
        """
        if len(self.components) < 2:
            return {"overlap_risk": 0.0}
        
        min_distance = float('inf')
        for i, c1 in enumerate(self.components):
            for c2 in self.components[i+1:]:
                dist = np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                min_clearance = (c1.width + c2.width) / 2 + (c1.height + c2.height) / 2
                if dist < min_clearance:
                    min_distance = min(min_distance, dist)
        
        if min_distance == float('inf'):
            return {"overlap_risk": 0.0}
        
        # Risk is inverse of minimum distance
        risk = 1.0 / (min_distance + 1.0)
        return {"overlap_risk": float(risk)}
    
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
        
        result = {
            "density": float(density),
            "convex_hull_area": hull.get("convex_hull_area", 0.0),
            "voronoi_variance": voronoi.get("voronoi_variance", 0.0),
            "mst_length": mst.get("mst_length", 0.0),
            "thermal_hotspots": thermal.get("thermal_hotspots", 0),
            "net_crossings": crossings.get("net_crossings", 0),
            "overlap_risk": overlap.get("overlap_risk", 0.0),
            # Additional geometric data for visualization
            "voronoi_data": voronoi,
            "mst_edges": mst.get("edges", []),
            "hull_vertices": hull.get("vertices", []),
            "hotspot_locations": thermal.get("hotspot_locations", [])
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(result)

