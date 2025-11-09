"""
Large PCB Design Handler

Handles multi-layer, hierarchical PCB designs with abstraction layers.
Supports module-based design, zoom/pan visualization, and computational geometry
analysis at different abstraction levels.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.geometry.placement import Placement


class DesignModule:
    """Represents a functional module in a large PCB design."""
    
    def __init__(self, name: str, components: List[Dict], bounds: Tuple[float, float, float, float]):
        """
        Initialize a design module.
        
        Args:
            name: Module name (e.g., "Power Supply", "MCU Section")
            components: List of component dicts in this module
            bounds: (x_min, y_min, x_max, y_max) bounding box
        """
        self.name = name
        self.components = components
        self.bounds = bounds
        self.geometry = None
    
    def analyze_geometry(self, placement: Placement):
        """Analyze computational geometry for this module."""
        # Extract module placement
        module_components = []
        for comp_name in [c.get("name") for c in self.components]:
            comp = placement.get_component(comp_name)
            if comp:
                module_components.append(comp)
        
        if module_components:
            # Create sub-placement for module
            from src.backend.geometry.component import Component
            from src.backend.geometry.board import Board
            from src.backend.geometry.net import Net
            
            module_board = Board(
                width=self.bounds[2] - self.bounds[0],
                height=self.bounds[3] - self.bounds[1],
                clearance=placement.board.clearance
            )
            
            module_placement = Placement(module_components, module_board, [])
            analyzer = GeometryAnalyzer(module_placement)
            self.geometry = analyzer.analyze()
        
        return self.geometry


class LargeDesignHandler:
    """Handles large, hierarchical PCB designs with multiple abstraction layers."""
    
    def __init__(self, placement: Placement):
        """Initialize handler for large design."""
        self.placement = placement
        self.modules: List[DesignModule] = []
        self.abstraction_layers = {
            "system": [],      # Top-level system view
            "module": [],      # Functional modules
            "component": []    # Individual components
        }
    
    def identify_modules(self, module_definitions: Optional[List[Dict]] = None) -> List[DesignModule]:
        """
        Identify functional modules in the design.
        
        Args:
            module_definitions: Optional list of module definitions with names and component lists
        
        Returns:
            List of identified modules
        """
        if module_definitions:
            # Use provided module definitions
            for mod_def in module_definitions:
                mod_name = mod_def.get("name", "Unknown")
                comp_names = mod_def.get("components", [])
                
                # Find components
                mod_components = []
                positions = []
                for comp_name in comp_names:
                    comp = self.placement.get_component(comp_name)
                    if comp:
                        mod_components.append({
                            "name": comp.name,
                            "package": comp.package,
                            "x": comp.x,
                            "y": comp.y,
                            "width": comp.width,
                            "height": comp.height,
                            "power": comp.power
                        })
                        positions.append([comp.x, comp.y])
                
                if positions:
                    positions = np.array(positions)
                    bounds = (
                        float(positions[:, 0].min() - 5),
                        float(positions[:, 1].min() - 5),
                        float(positions[:, 0].max() + 5),
                        float(positions[:, 1].max() + 5)
                    )
                    
                    module = DesignModule(mod_name, mod_components, bounds)
                    self.modules.append(module)
        else:
            # Auto-identify modules using clustering
            self._auto_identify_modules()
        
        return self.modules
    
    def _auto_identify_modules(self):
        """Automatically identify modules using computational geometry clustering."""
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
        except ImportError:
            # Fallback: simple distance-based clustering
            self._simple_cluster_modules()
            return
        
        components = list(self.placement.components.values())
        if len(components) < 2:
            return
        
        # Get component positions
        positions = np.array([[c.x, c.y] for c in components])
        
        # Cluster components by proximity
        if len(positions) > 1:
            distances = pdist(positions)
            linkage_matrix = linkage(distances, method='ward')
            
            # Determine number of clusters (aim for 3-5 modules for large designs)
            n_clusters = min(max(3, len(components) // 10), 5)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group components by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(components[i])
            
            # Create modules from clusters
            for cluster_id, cluster_comps in clusters.items():
                mod_components = [{
                    "name": c.name,
                    "package": c.package,
                    "x": c.x,
                    "y": c.y,
                    "width": c.width,
                    "height": c.height,
                    "power": c.power
                } for c in cluster_comps]
                
                positions = np.array([[c.x, c.y] for c in cluster_comps])
                bounds = (
                    float(positions[:, 0].min() - 5),
                    float(positions[:, 1].min() - 5),
                    float(positions[:, 0].max() + 5),
                    float(positions[:, 1].max() + 5)
                )
                
                module = DesignModule(f"Module {cluster_id}", mod_components, bounds)
                self.modules.append(module)
    
    def _simple_cluster_modules(self):
        """Simple distance-based clustering fallback."""
        components = list(self.placement.components.values())
        if len(components) < 2:
            return
        
        # Simple grid-based clustering
        positions = np.array([[c.x, c.y] for c in components])
        
        # Divide board into grid
        n_clusters = min(max(3, len(components) // 10), 5)
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        
        # Assign components to grid cells
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        clusters = {}
        for i, comp in enumerate(components):
            x, y = comp.x, comp.y
            cell_x = int((x - positions[:, 0].min()) / (x_range / grid_size))
            cell_y = int((y - positions[:, 1].min()) / (y_range / grid_size))
            cell_id = cell_x * grid_size + cell_y
            
            if cell_id not in clusters:
                clusters[cell_id] = []
            clusters[cell_id].append(comp)
        
        # Create modules from clusters
        for cluster_id, cluster_comps in clusters.items():
            if len(cluster_comps) > 0:
                mod_components = [{
                    "name": c.name,
                    "package": c.package,
                    "x": c.x,
                    "y": c.y,
                    "width": c.width,
                    "height": c.height,
                    "power": c.power
                } for c in cluster_comps]
                
                positions = np.array([[c.x, c.y] for c in cluster_comps])
                bounds = (
                    float(positions[:, 0].min() - 5),
                    float(positions[:, 1].min() - 5),
                    float(positions[:, 0].max() + 5),
                    float(positions[:, 1].max() + 5)
                )
                
                module = DesignModule(f"Module {cluster_id}", mod_components, bounds)
                self.modules.append(module)
    
    def analyze_hierarchical_geometry(self) -> Dict:
        """
        Analyze computational geometry at different abstraction levels.
        
        Returns:
            Dictionary with geometry analysis at system, module, and component levels
        """
        results = {
            "system": {},
            "modules": {},
            "component": {}
        }
        
        # System-level analysis
        system_analyzer = GeometryAnalyzer(self.placement)
        results["system"] = system_analyzer.analyze()
        
        # Module-level analysis
        for module in self.modules:
            module_geometry = module.analyze_geometry(self.placement)
            if module_geometry:
                results["modules"][module.name] = module_geometry
        
        # Component-level analysis (for critical components)
        high_power_comps = [c for c in self.placement.components.values() if c.power > 1.0]
        if high_power_comps:
            # Analyze thermal distribution around high-power components
            results["component"]["high_power"] = {
                "count": len(high_power_comps),
                "total_power": sum(c.power for c in high_power_comps),
                "components": [c.name for c in high_power_comps]
            }
        
        return results
    
    def get_viewport_data(self, x_min: float, y_min: float, x_max: float, y_max: float, zoom_level: float = 1.0) -> Dict:
        """
        Get data for a specific viewport (for zoom/pan visualization).
        
        Args:
            x_min, y_min, x_max, y_max: Viewport bounds
            zoom_level: Zoom level (1.0 = full board)
        
        Returns:
            Dictionary with components, nets, and geometry data in viewport
        """
        viewport_components = []
        viewport_nets = []
        
        # Filter components in viewport
        for comp in self.placement.components.values():
            if x_min <= comp.x <= x_max and y_min <= comp.y <= y_max:
                viewport_components.append({
                    "name": comp.name,
                    "package": comp.package,
                    "x": comp.x,
                    "y": comp.y,
                    "width": comp.width,
                    "height": comp.height,
                    "power": comp.power,
                    "angle": comp.angle
                })
        
        # Filter nets connected to viewport components
        viewport_comp_names = {c["name"] for c in viewport_components}
        for net in self.placement.nets.values():
            net_pins = net.pins
            if any(pin[0] in viewport_comp_names for pin in net_pins):
                viewport_nets.append({
                    "name": net.name,
                    "pins": net_pins
                })
        
        # Analyze geometry for viewport
        viewport_placement = Placement.from_dict({
            "board": {
                "width": x_max - x_min,
                "height": y_max - y_min,
                "clearance": self.placement.board.clearance
            },
            "components": viewport_components,
            "nets": viewport_nets
        })
        
        analyzer = GeometryAnalyzer(viewport_placement)
        viewport_geometry = analyzer.analyze()
        
        return {
            "components": viewport_components,
            "nets": viewport_nets,
            "geometry": viewport_geometry,
            "viewport": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            "zoom_level": zoom_level
        }
    
    def get_module_view(self, module_name: str) -> Optional[Dict]:
        """Get detailed view of a specific module."""
        module = next((m for m in self.modules if m.name == module_name), None)
        if not module:
            return None
        
        return {
            "name": module.name,
            "bounds": module.bounds,
            "components": module.components,
            "geometry": module.geometry,
            "component_count": len(module.components)
        }

