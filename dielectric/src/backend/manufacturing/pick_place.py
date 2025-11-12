"""
Pick-and-Place File Generator

Generates pick-and-place files for automated assembly:
- Component positions
- Rotations
- Side (top/bottom)
- Reference designators
"""

import logging
import csv
from typing import Dict, List, Optional
from pathlib import Path

try:
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component

logger = logging.getLogger(__name__)


class PickPlaceGenerator:
    """
    Generate pick-and-place files for automated assembly.
    
    Supports:
    - CSV format (most common)
    - Component positions and rotations
    - Top/bottom side identification
    - Reference designators
    """
    
    def generate_csv(
        self,
        placement: Placement,
        output_path: str,
        side: str = "top"  # "top" or "bottom"
    ) -> str:
        """
        Generate CSV pick-and-place file.
        
        Args:
            placement: Placement with components
            output_path: Output file path
            side: Component side ("top" or "bottom")
        
        Returns:
            Path to generated file
        """
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Designator",
                "Mid X",
                "Mid Y",
                "Layer",
                "Rotation",
                "Comment",
                "Package",
                "Value"
            ])
            
            # Component data
            for comp_name, comp in placement.components.items():
                # Determine layer based on component type
                # Simplified: SMT components on top, through-hole on both
                layer = "top" if comp.height < 2.0 else "both"
                
                # Only include components on requested side
                if side == "top" and layer not in ["top", "both"]:
                    continue
                if side == "bottom" and layer not in ["bottom", "both"]:
                    continue
                
                writer.writerow([
                    comp_name,  # Designator
                    f"{comp.x:.4f}",  # Mid X (mm)
                    f"{comp.y:.4f}",  # Mid Y (mm)
                    layer,  # Layer
                    f"{comp.angle:.2f}",  # Rotation (degrees)
                    comp_name,  # Comment
                    comp.package or "UNKNOWN",  # Package
                    comp.value or ""  # Value
                ])
        
        logger.info(f"Generated pick-place file: {filepath}")
        return str(filepath)
    
    def generate_json(
        self,
        placement: Placement,
        output_path: str
    ) -> str:
        """
        Generate JSON pick-and-place file.
        
        Args:
            placement: Placement with components
            output_path: Output file path
        
        Returns:
            Path to generated file
        """
        import json
        
        components = []
        
        for comp_name, comp in placement.components.items():
            layer = "top" if comp.height < 2.0 else "both"
            
            components.append({
                "designator": comp_name,
                "x": comp.x,
                "y": comp.y,
                "layer": layer,
                "rotation": comp.angle,
                "package": comp.package or "UNKNOWN",
                "value": comp.value or ""
            })
        
        data = {
            "board": {
                "width": placement.board.width,
                "height": placement.board.height
            },
            "components": components
        }
        
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(data, indent=2))
        
        logger.info(f"Generated JSON pick-place file: {filepath}")
        return str(filepath)

