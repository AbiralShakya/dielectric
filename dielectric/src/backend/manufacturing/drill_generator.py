"""
Drill File Generator

Generates Excellon drill files for PCB manufacturing:
- Through-hole vias
- Component mounting holes
- Tool change commands
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

try:
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.geometry.placement import Placement

logger = logging.getLogger(__name__)


class DrillGenerator:
    """
    Generate Excellon drill files for PCB manufacturing.
    
    Supports:
    - Through-hole vias
    - Component mounting holes
    - Tool definitions
    - Tool changes
    """
    
    def generate_drill_file(
        self,
        placement: Placement,
        output_path: str,
        board_name: str = "board"
    ) -> str:
        """
        Generate Excellon drill file.
        
        Args:
            placement: Placement with components and vias
            output_path: Output file path
            board_name: Board name
        
        Returns:
            Path to generated drill file
        """
        lines = []
        
        # Excellon header
        lines.append("M48")
        lines.append(";DRILL file {format} Excellon")
        lines.append(f";FILE_FORMAT={2:02d}:{3:02d}")
        lines.append(";#@! TF.CreationDate," + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append(";#@! TF.GenerationSoftware,Dielectric")
        lines.append(";#@! TF.FileFunction,Plated,1,2,P")
        lines.append(";#@! TF.FilePolarity,Positive")
        lines.append("")
        
        # Tool definitions
        tools = self._get_tools(placement)
        for tool_num, (diameter, count) in enumerate(tools.items(), start=1):
            lines.append(f"T{tool_num:02d}C{diameter:.3f}")
        
        lines.append("%")
        lines.append("G90")
        lines.append("G05")
        lines.append("M71")  # Metric units
        
        # Drill holes
        hole_count = 0
        for tool_num, (diameter, holes) in enumerate(tools.items(), start=1):
            if not holes:
                continue
            
            lines.append(f"T{tool_num:02d}")
            
            for x, y in holes:
                # Convert mm to format units (typically 1/10000 inch or mm)
                x_units = int(x * 10000)  # 0.0001mm units
                y_units = int(y * 10000)
                lines.append(f"X{x_units}Y{y_units}")
                hole_count += 1
        
        lines.append("M30")
        
        content = "\n".join(lines)
        
        # Write to file
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        
        logger.info(f"Generated drill file with {hole_count} holes")
        return str(filepath)
    
    def _get_tools(self, placement: Placement) -> Dict[float, List[Tuple[float, float]]]:
        """
        Get drill tools and their holes.
        
        Returns:
            Dictionary mapping drill diameter (mm) to list of (x, y) positions
        """
        tools = {}
        
        # Collect vias
        if hasattr(placement, 'vias') and placement.vias:
            for via in placement.vias:
                drill = via.get("drill", 0.2)  # Default 0.2mm
                pos = via["position"]
                
                if drill not in tools:
                    tools[drill] = []
                tools[drill].append((pos[0], pos[1]))
        
        # Collect component mounting holes
        for comp_name, comp in placement.components.items():
            # Check if component has mounting holes (simplified)
            # In production, would check footprint for mounting holes
            if "mounting" in comp_name.lower() or "hole" in comp_name.lower():
                # Assume 3mm mounting hole
                drill = 3.0
                if drill not in tools:
                    tools[drill] = []
                tools[drill].append((comp.x, comp.y))
        
        return tools

