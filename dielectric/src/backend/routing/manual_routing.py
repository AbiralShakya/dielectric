"""
Manual Routing Tools

Interactive routing tools for manual trace editing:
- Push-and-shove routing
- Trace editing
- Via placement
- Copper pour/fill zones
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

try:
    from backend.geometry.placement import Placement
    from backend.geometry.net import Net
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.net import Net

logger = logging.getLogger(__name__)


class ManualRouter:
    """
    Manual routing tools for interactive PCB editing.
    
    Features:
    - Push-and-shove routing
    - Trace editing (move, delete, reroute)
    - Via placement and editing
    - Copper pour/fill zones
    - Obstacle avoidance
    """
    
    def __init__(self, kicad_client=None):
        """
        Initialize manual router.
        
        Args:
            kicad_client: Optional KiCad client for direct editing
        """
        self.kicad_client = kicad_client
        self.traces: List[Dict] = []
        self.vias: List[Dict] = []
        self.copper_pours: List[Dict] = []
    
    def add_trace(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        net: str,
        width: float = 0.2,
        layer: str = "F.Cu"
    ) -> Dict:
        """
        Add a manual trace.
        
        Args:
            start: Start position (x, y) in mm
            end: End position (x, y) in mm
            net: Net name
            width: Trace width (mm)
            layer: Layer name
        
        Returns:
            Trace dictionary
        """
        length = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        trace = {
            "net": net,
            "start_position": start,
            "end_position": end,
            "width": width,
            "length": length,
            "layer": layer,
            "manual": True
        }
        
        self.traces.append(trace)
        
        # Place in KiCad if available
        if self.kicad_client and self.kicad_client.is_available():
            try:
                result = self.kicad_client.route_trace(
                    start={"x": start[0], "y": start[1], "unit": "mm"},
                    end={"x": end[0], "y": end[1], "unit": "mm"},
                    layer=layer,
                    width=width,
                    net=net
                )
                if result.get("success"):
                    trace["kicad_placed"] = True
            except Exception as e:
                logger.warning(f"Failed to place trace in KiCad: {e}")
        
        return trace
    
    def delete_trace(self, trace_id: int) -> bool:
        """Delete a trace by index."""
        if 0 <= trace_id < len(self.traces):
            del self.traces[trace_id]
            return True
        return False
    
    def move_trace(
        self,
        trace_id: int,
        new_start: Optional[Tuple[float, float]] = None,
        new_end: Optional[Tuple[float, float]] = None
    ) -> bool:
        """Move trace endpoints."""
        if not (0 <= trace_id < len(self.traces)):
            return False
        
        trace = self.traces[trace_id]
        
        if new_start:
            trace["start_position"] = new_start
        if new_end:
            trace["end_position"] = new_end
        
        # Recalculate length
        start = trace["start_position"]
        end = trace["end_position"]
        trace["length"] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        return True
    
    def add_via(
        self,
        position: Tuple[float, float],
        net: str,
        size: float = 0.5,
        drill: float = 0.2
    ) -> Dict:
        """
        Add a via.
        
        Args:
            position: Via position (x, y) in mm
            net: Net name
            size: Via diameter (mm)
            drill: Drill diameter (mm)
        
        Returns:
            Via dictionary
        """
        via = {
            "position": position,
            "net": net,
            "size": size,
            "drill": drill,
            "manual": True
        }
        
        self.vias.append(via)
        
        # Place in KiCad if available
        if self.kicad_client and self.kicad_client.is_available():
            try:
                result = self.kicad_client.add_via(
                    position={"x": position[0], "y": position[1], "unit": "mm"},
                    net=net,
                    size=size,
                    drill=drill
                )
                if result.get("success"):
                    via["kicad_placed"] = True
            except Exception as e:
                logger.warning(f"Failed to place via in KiCad: {e}")
        
        return via
    
    def add_copper_pour(
        self,
        outline: List[Tuple[float, float]],
        net: str,
        layer: str = "F.Cu",
        clearance: float = 0.2
    ) -> Dict:
        """
        Add copper pour/fill zone.
        
        Args:
            outline: List of (x, y) points defining zone outline
            net: Net name (usually GND or power)
            layer: Layer name
            clearance: Clearance from other objects (mm)
        
        Returns:
            Copper pour dictionary
        """
        pour = {
            "outline": outline,
            "net": net,
            "layer": layer,
            "clearance": clearance,
            "manual": True
        }
        
        self.copper_pours.append(pour)
        
        return pour
    
    def push_and_shove(
        self,
        new_trace: Dict,
        existing_traces: List[Dict]
    ) -> List[Dict]:
        """
        Push-and-shove routing: move existing traces to make room.
        
        Args:
            new_trace: New trace to add
            existing_traces: Existing traces that may conflict
        
        Returns:
            List of moved traces
        """
        moved_traces = []
        
        # Check for conflicts
        for trace in existing_traces:
            if self._traces_conflict(new_trace, trace):
                # Calculate push direction
                push_vector = self._calculate_push_vector(new_trace, trace)
                
                # Move trace
                moved_trace = trace.copy()
                moved_trace["start_position"] = (
                    trace["start_position"][0] + push_vector[0],
                    trace["start_position"][1] + push_vector[1]
                )
                moved_trace["end_position"] = (
                    trace["end_position"][0] + push_vector[0],
                    trace["end_position"][1] + push_vector[1]
                )
                
                moved_traces.append(moved_trace)
        
        return moved_traces
    
    def _traces_conflict(self, trace1: Dict, trace2: Dict) -> bool:
        """Check if two traces conflict."""
        # Simplified conflict detection
        # In production, would use proper line segment intersection
        
        # Check if traces are on same layer
        if trace1.get("layer") != trace2.get("layer"):
            return False
        
        # Check minimum spacing
        min_spacing = 0.2  # mm
        width1 = trace1.get("width", 0.2) / 2
        width2 = trace2.get("width", 0.2) / 2
        
        # Calculate distance between trace centerlines
        # Simplified: check distance between start/end points
        start1 = trace1["start_position"]
        end1 = trace1["end_position"]
        start2 = trace2["start_position"]
        end2 = trace2["end_position"]
        
        # Check if traces are too close
        dist = min(
            np.sqrt((start1[0]-start2[0])**2 + (start1[1]-start2[1])**2),
            np.sqrt((start1[0]-end2[0])**2 + (start1[1]-end2[1])**2),
            np.sqrt((end1[0]-start2[0])**2 + (end1[1]-start2[1])**2),
            np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2)
        )
        
        return dist < (width1 + width2 + min_spacing)
    
    def _calculate_push_vector(
        self,
        new_trace: Dict,
        existing_trace: Dict
    ) -> Tuple[float, float]:
        """Calculate vector to push existing trace away."""
        # Simplified: push perpendicular to new trace
        start = new_trace["start_position"]
        end = new_trace["end_position"]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return (0.0, 0.0)
        
        # Perpendicular direction
        push_distance = 0.3  # mm
        push_x = -dy / length * push_distance
        push_y = dx / length * push_distance
        
        return (push_x, push_y)
    
    def get_routing_statistics(self) -> Dict:
        """Get routing statistics."""
        return {
            "total_traces": len(self.traces),
            "total_vias": len(self.vias),
            "total_copper_pours": len(self.copper_pours),
            "total_trace_length": sum(t.get("length", 0) for t in self.traces),
            "manual_traces": sum(1 for t in self.traces if t.get("manual"))
        }

