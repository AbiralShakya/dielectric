"""
PCB Design Database

Learns from industry PCB designs to improve optimization.
Stores design patterns, successful placements, and optimization strategies.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
try:
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.geometry.placement import Placement


class PCBDatabase:
    """Database for learning from industry PCB designs."""
    
    def __init__(self, db_path: str = "data/pcb_database.json"):
        """
        Initialize PCB database.
        
        Args:
            db_path: Path to database JSON file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.database = self._load_database()
    
    def _load_database(self) -> Dict:
        """Load database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "designs": [],
            "patterns": {},
            "statistics": {}
        }
    
    def _save_database(self):
        """Save database to file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    def add_design(self, placement_data: Dict, metadata: Dict = None):
        """
        Add a PCB design to the database for learning.
        
        Args:
            placement_data: Placement data
            metadata: Optional metadata (application, industry, etc.)
        """
        try:
            placement = Placement.from_dict(placement_data)
            analyzer = GeometryAnalyzer(placement)
            geometry = analyzer.analyze()
            
            from datetime import datetime
            design_entry = {
                "placement": placement_data,
                "geometry": geometry,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.database["designs"].append(design_entry)
            
            # Update statistics
            self._update_statistics(geometry)
            
            # Extract patterns
            self._extract_patterns(placement_data, geometry)
            
            self._save_database()
        except Exception as e:
            print(f"Error adding design to database: {e}")
    
    def _update_statistics(self, geometry: Dict):
        """Update database statistics."""
        stats = self.database.setdefault("statistics", {})
        
        # Component density distribution
        density = geometry.get("density", 0)
        if "density_distribution" not in stats:
            stats["density_distribution"] = []
        stats["density_distribution"].append(density)
        
        # MST length distribution
        mst_length = geometry.get("mst_length", 0)
        if "mst_length_distribution" not in stats:
            stats["mst_length_distribution"] = []
        stats["mst_length_distribution"].append(mst_length)
        
        # Thermal hotspot distribution
        hotspots = geometry.get("thermal_hotspots", 0)
        if "thermal_hotspot_distribution" not in stats:
            stats["thermal_hotspot_distribution"] = []
        stats["thermal_hotspot_distribution"].append(hotspots)
    
    def _extract_patterns(self, placement_data: Dict, geometry: Dict):
        """Extract design patterns for learning."""
        patterns = self.database.setdefault("patterns", {})
        
        # Pattern: High-power component spacing
        components = placement_data.get("components", [])
        high_power_comps = [c for c in components if c.get("power", 0) > 1.0]
        if len(high_power_comps) >= 2:
            if "high_power_spacing" not in patterns:
                patterns["high_power_spacing"] = []
            
            # Calculate average spacing between high-power components
            positions = [(c.get("x", 0), c.get("y", 0)) for c in high_power_comps]
            if len(positions) >= 2:
                distances = []
                for i, p1 in enumerate(positions):
                    for p2 in positions[i+1:]:
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        distances.append(dist)
                if distances:
                    patterns["high_power_spacing"].append(np.mean(distances))
        
        # Pattern: Component density vs board utilization
        density = geometry.get("density", 0)
        utilization = geometry.get("convex_hull_area", 0) / (
            placement_data.get("board", {}).get("width", 1) * 
            placement_data.get("board", {}).get("height", 1)
        )
        if "density_utilization" not in patterns:
            patterns["density_utilization"] = []
        patterns["density_utilization"].append({"density": density, "utilization": utilization})
    
    def get_similar_designs(self, geometry: Dict, top_k: int = 5) -> List[Dict]:
        """
        Find similar designs based on geometry metrics.
        
        Args:
            geometry: Geometry metrics to match
            top_k: Number of similar designs to return
        
        Returns:
            List of similar design entries
        """
        if not self.database["designs"]:
            return []
        
        similarities = []
        target_mst = geometry.get("mst_length", 0)
        target_density = geometry.get("density", 0)
        target_hotspots = geometry.get("thermal_hotspots", 0)
        
        for design in self.database["designs"]:
            design_geo = design.get("geometry", {})
            design_mst = design_geo.get("mst_length", 0)
            design_density = design_geo.get("density", 0)
            design_hotspots = design_geo.get("thermal_hotspots", 0)
            
            # Simple similarity metric
            similarity = (
                1.0 / (1.0 + abs(target_mst - design_mst) / max(target_mst, 1.0)) +
                1.0 / (1.0 + abs(target_density - design_density) / max(target_density, 0.01)) +
                1.0 / (1.0 + abs(target_hotspots - design_hotspots) + 1)
            ) / 3.0
            
            similarities.append((similarity, design))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [design for _, design in similarities[:top_k]]
    
    def get_optimization_hints(self, geometry: Dict, user_intent: str) -> Dict:
        """
        Get optimization hints based on learned patterns.
        
        Args:
            geometry: Current geometry metrics
            user_intent: User's optimization intent
        
        Returns:
            Dictionary with optimization hints
        """
        hints = {
            "recommended_weights": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34},
            "patterns": [],
            "warnings": []
        }
        
        # Analyze patterns
        patterns = self.database.get("patterns", {})
        
        # High-power spacing pattern
        if "high_power_spacing" in patterns and len(patterns["high_power_spacing"]) > 0:
            avg_spacing = np.mean(patterns["high_power_spacing"])
            current_hotspots = geometry.get("thermal_hotspots", 0)
            
            if current_hotspots > 0:
                hints["patterns"].append({
                    "type": "thermal_spacing",
                    "recommendation": f"Industry average spacing for high-power components: {avg_spacing:.1f}mm",
                    "current_hotspots": current_hotspots
                })
                
                if "thermal" in user_intent.lower() or "cool" in user_intent.lower():
                    hints["recommended_weights"]["beta"] = 0.6
                    hints["recommended_weights"]["alpha"] = 0.2
                    hints["recommended_weights"]["gamma"] = 0.2
        
        # Density utilization pattern
        if "density_utilization" in patterns:
            density_util = patterns["density_utilization"]
            if len(density_util) > 0:
                avg_util = np.mean([d["utilization"] for d in density_util])
                current_density = geometry.get("density", 0)
                
                if current_density > 0.1:  # High density
                    hints["warnings"].append({
                        "type": "high_density",
                        "message": f"High component density detected. Industry average utilization: {avg_util:.1%}",
                        "recommendation": "Consider increasing board size or reducing component count"
                    })
        
        return hints

