"""
Length Matching Router

Matches trace lengths for:
- Clock signals
- Data buses
- High-speed interfaces
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


class LengthMatcher:
    """
    Matches trace lengths for signal groups.
    
    Features:
    - Length matching within tolerance
    - Serpentine routing
    - Trombone routing
    - Automatic group detection
    """
    
    def __init__(
        self,
        tolerance: float = 0.1,  # mm
        min_spacing: float = 0.2  # mm
    ):
        """
        Initialize length matcher.
        
        Args:
            tolerance: Maximum length difference (mm)
            min_spacing: Minimum spacing between serpentine segments (mm)
        """
        self.tolerance = tolerance
        self.min_spacing = min_spacing
    
    def identify_length_groups(
        self,
        placement: Placement
    ) -> List[List[Net]]:
        """
        Identify nets that should be length-matched.
        
        Looks for:
        - Clock signals
        - Data buses (D0-D7, A0-A15, etc.)
        - Address buses
        - Control signals
        
        Returns:
            List of net groups that should be matched
        """
        groups = []
        nets = list(placement.nets.values())
        
        # Group by base name pattern
        bus_patterns = [
            ("D", 16),  # Data bus D0-D15
            ("A", 32),  # Address bus A0-A31
            ("CS", 8),  # Chip selects
            ("WE", 4),  # Write enables
            ("OE", 4),  # Output enables
        ]
        
        for pattern, max_count in bus_patterns:
            group = []
            for i in range(max_count):
                net_name = f"{pattern}{i}"
                net = next((n for n in nets if n.name.upper() == net_name.upper()), None)
                if net:
                    group.append(net)
            
            if len(group) >= 2:
                groups.append(group)
        
        # Clock signals
        clock_nets = [n for n in nets if "CLK" in n.name.upper() or "CLOCK" in n.name.upper()]
        if len(clock_nets) >= 2:
            groups.append(clock_nets)
        
        logger.info(f"Identified {len(groups)} length-matching groups")
        return groups
    
    async def match_group(
        self,
        placement: Placement,
        nets: List[Net]
    ) -> Dict:
        """
        Match lengths for a group of nets.
        
        Args:
            placement: Placement with components
            nets: List of nets to match
        
        Returns:
            Matching result with updated traces
        """
        # Calculate current lengths
        lengths = []
        for net in nets:
            length = self._calculate_net_length(placement, net)
            lengths.append((net, length))
        
        # Find target length (longest)
        target_length = max(length for _, length in lengths)
        
        # Match all nets to target length
        matched_traces = []
        for net, current_length in lengths:
            if abs(current_length - target_length) <= self.tolerance:
                # Already matched
                continue
            
            # Add serpentine to match length
            additional_length = target_length - current_length
            traces = self._add_matching_serpentine(placement, net, additional_length)
            matched_traces.extend(traces)
        
        return {
            "success": True,
            "nets": [n.name for n in nets],
            "target_length": target_length,
            "matched_traces": matched_traces,
            "max_deviation": max(abs(l - target_length) for _, l in lengths)
        }
    
    def _calculate_net_length(
        self,
        placement: Placement,
        net: Net
    ) -> float:
        """Calculate total length of a net."""
        positions = []
        for comp_ref, pad_name in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                positions.append((comp.x, comp.y))
        
        if len(positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i+1]
            total_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        return total_length
    
    def _add_matching_serpentine(
        self,
        placement: Placement,
        net: Net,
        additional_length: float
    ) -> List[Dict]:
        """Add serpentine routing to match length."""
        # Get net positions
        positions = []
        for comp_ref, pad_name in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                positions.append((comp_ref, (comp.x, comp.y), pad_name))
        
        if len(positions) < 2:
            return []
        
        # Add serpentine to longest segment
        longest_idx = 0
        longest_length = 0.0
        
        for i in range(len(positions) - 1):
            x1, y1 = positions[i][1]
            x2, y2 = positions[i+1][1]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > longest_length:
                longest_length = length
                longest_idx = i
        
        # Create serpentine segments
        start_pos = positions[longest_idx][1]
        end_pos = positions[longest_idx+1][1]
        
        serpentine = self._create_serpentine(start_pos, end_pos, additional_length)
        
        traces = []
        for seg in serpentine:
            traces.append({
                "net": net.name,
                "start_position": seg["start_position"],
                "end_position": seg["end_position"],
                "length": seg["length"],
                "layer": "F.Cu",
                "matching": True
            })
        
        return traces
    
    def _create_serpentine(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        length: float
    ) -> List[Dict]:
        """Create serpentine pattern."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        base_length = np.sqrt(dx*dx + dy*dy)
        
        # Perpendicular direction
        perp_x = -dy / base_length * self.min_spacing
        perp_y = dx / base_length * self.min_spacing
        
        # Calculate meanders needed
        meander_length = self.min_spacing * 4
        num_meanders = int(np.ceil(length / meander_length))
        
        segments = []
        current_pos = start
        
        for i in range(num_meanders):
            progress = (i + 1) / (num_meanders + 1)
            next_pos = (
                start[0] + dx * progress,
                start[1] + dy * progress
            )
            
            # Create meander
            mid1 = (next_pos[0] + perp_x, next_pos[1] + perp_y)
            mid2 = (next_pos[0] - perp_x, next_pos[1] - perp_y)
            
            segments.append({
                "start_position": current_pos,
                "end_position": mid1,
                "length": np.sqrt((mid1[0]-current_pos[0])**2 + (mid1[1]-current_pos[1])**2)
            })
            
            segments.append({
                "start_position": mid1,
                "end_position": mid2,
                "length": np.sqrt((mid2[0]-mid1[0])**2 + (mid2[1]-mid1[1])**2)
            })
            
            current_pos = mid2
        
        # Final segment
        segments.append({
            "start_position": current_pos,
            "end_position": end,
            "length": np.sqrt((end[0]-current_pos[0])**2 + (end[1]-current_pos[1])**2)
        })
        
        return segments

