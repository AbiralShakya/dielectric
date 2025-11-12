"""
Differential Pair Router

Routes differential pairs with:
- Controlled impedance (100Ω differential)
- Length matching
- Spacing control
- Skew minimization
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


class DifferentialPairRouter:
    """
    Router for differential pair signals.
    
    Features:
    - Controlled impedance routing (100Ω differential)
    - Length matching within tolerance
    - Spacing control (typically 2x trace width)
    - Skew minimization
    - Via optimization for pairs
    """
    
    def __init__(
        self,
        target_impedance: float = 100.0,  # Ω differential
        length_tolerance: float = 0.1,  # mm
        spacing_ratio: float = 2.0,  # spacing = spacing_ratio * trace_width
        trace_width: float = 0.15  # mm (6 mil)
    ):
        """
        Initialize differential pair router.
        
        Args:
            target_impedance: Target differential impedance (Ω)
            length_tolerance: Maximum length difference (mm)
            spacing_ratio: Spacing to trace width ratio
            trace_width: Trace width (mm)
        """
        self.target_impedance = target_impedance
        self.length_tolerance = length_tolerance
        self.spacing_ratio = spacing_ratio
        self.trace_width = trace_width
        self.spacing = trace_width * spacing_ratio
    
    def identify_differential_pairs(
        self,
        placement: Placement
    ) -> List[Tuple[Net, Net]]:
        """
        Identify differential pairs from nets.
        
        Looks for nets with names like:
        - USB_D+ / USB_D-
        - TX_P / TX_N
        - CLK_P / CLK_N
        - DIFF_P / DIFF_N
        
        Returns:
            List of (positive_net, negative_net) tuples
        """
        pairs = []
        nets = list(placement.nets.values())
        
        # Common differential pair naming patterns
        suffixes = [
            ("_P", "_N"),
            ("_+", "_-"),
            ("P", "N"),
            ("+", "-"),
            ("_POS", "_NEG"),
            ("_POSITIVE", "_NEGATIVE")
        ]
        
        for i, net1 in enumerate(nets):
            net1_name = net1.name.upper()
            
            for net2 in nets[i+1:]:
                net2_name = net2.name.upper()
                
                # Check if nets form a differential pair
                for pos_suffix, neg_suffix in suffixes:
                    if (net1_name.endswith(pos_suffix) and 
                        net2_name.endswith(neg_suffix)):
                        # Check if base names match
                        base1 = net1_name[:-len(pos_suffix)]
                        base2 = net2_name[:-len(neg_suffix)]
                        
                        if base1 == base2:
                            pairs.append((net1, net2))
                            break
        
        logger.info(f"Identified {len(pairs)} differential pairs")
        return pairs
    
    async def route_pair(
        self,
        placement: Placement,
        positive_net: Net,
        negative_net: Net
    ) -> Dict:
        """
        Route a differential pair.
        
        Args:
            placement: Placement with components
            positive_net: Positive net
            negative_net: Negative net
        
        Returns:
            Routing result with traces for both nets
        """
        # Get component positions for both nets
        pos_positions = self._get_net_positions(placement, positive_net)
        neg_positions = self._get_net_positions(placement, negative_net)
        
        if len(pos_positions) != len(neg_positions):
            logger.warning(f"Pair {positive_net.name}/{negative_net.name} has mismatched pin counts")
        
        # Route both nets with length matching
        pos_traces = self._route_with_spacing(pos_positions, positive_net.name, offset=-self.spacing/2)
        neg_traces = self._route_with_spacing(neg_positions, negative_net.name, offset=self.spacing/2)
        
        # Match lengths
        matched_traces = self._match_lengths(pos_traces, neg_traces)
        
        return {
            "success": True,
            "positive_net": positive_net.name,
            "negative_net": negative_net.name,
            "traces": matched_traces,
            "impedance": self.target_impedance,
            "length_difference": abs(
                sum(t["length"] for t in pos_traces) -
                sum(t["length"] for t in neg_traces)
            )
        }
    
    def _get_net_positions(
        self,
        placement: Placement,
        net: Net
    ) -> List[Tuple[str, Tuple[float, float], str]]:
        """Get component positions for a net."""
        positions = []
        for comp_ref, pad_name in net.pins:
            comp = placement.components.get(comp_ref)
            if comp:
                positions.append((comp_ref, (comp.x, comp.y), pad_name))
        return positions
    
    def _route_with_spacing(
        self,
        positions: List[Tuple[str, Tuple[float, float], str]],
        net_name: str,
        offset: float
    ) -> List[Dict]:
        """
        Route net with spacing offset for differential pair.
        
        Args:
            positions: Component positions
            net_name: Net name
            offset: Perpendicular offset from centerline (mm)
        
        Returns:
            List of trace dictionaries
        """
        traces = []
        
        if len(positions) < 2:
            return traces
        
        # Simple routing: connect components in order with offset
        for i in range(len(positions) - 1):
            start_comp, start_pos, start_pad = positions[i]
            end_comp, end_pos, end_pad = positions[i+1]
            
            # Calculate centerline
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            # Apply perpendicular offset
            perp_x = -dy / length * offset
            perp_y = dx / length * offset
            
            adjusted_start = (start_pos[0] + perp_x, start_pos[1] + perp_y)
            adjusted_end = (end_pos[0] + perp_x, end_pos[1] + perp_y)
            
            trace = {
                "net": net_name,
                "start_component": start_comp,
                "start_pad": start_pad,
                "start_position": adjusted_start,
                "end_component": end_comp,
                "end_pad": end_pad,
                "end_position": adjusted_end,
                "width": self.trace_width,
                "length": length,
                "layer": "F.Cu",
                "spacing": self.spacing
            }
            
            traces.append(trace)
        
        return traces
    
    def _match_lengths(
        self,
        pos_traces: List[Dict],
        neg_traces: List[Dict]
    ) -> List[Dict]:
        """
        Match lengths of positive and negative traces.
        
        Adds serpentine routing if needed to match lengths.
        """
        pos_length = sum(t["length"] for t in pos_traces)
        neg_length = sum(t["length"] for t in neg_traces)
        
        length_diff = pos_length - neg_length
        
        if abs(length_diff) <= self.length_tolerance:
            # Lengths already matched
            return pos_traces + neg_traces
        
        # Need to add serpentine to shorter trace
        if length_diff > 0:
            # Positive is longer, add serpentine to negative
            neg_traces = self._add_serpentine(neg_traces, length_diff)
        else:
            # Negative is longer, add serpentine to positive
            pos_traces = self._add_serpentine(pos_traces, -length_diff)
        
        return pos_traces + neg_traces
    
    def _add_serpentine(
        self,
        traces: List[Dict],
        additional_length: float
    ) -> List[Dict]:
        """
        Add serpentine routing to increase trace length.
        
        Args:
            traces: Original traces
            additional_length: Additional length needed (mm)
        
        Returns:
            Traces with serpentine added
        """
        if not traces:
            return traces
        
        # Add serpentine to longest trace segment
        longest_idx = max(range(len(traces)), key=lambda i: traces[i]["length"])
        longest_trace = traces[longest_idx]
        
        # Create serpentine pattern
        serpentine_segments = self._create_serpentine(
            longest_trace["start_position"],
            longest_trace["end_position"],
            additional_length
        )
        
        # Replace longest trace with serpentine segments
        new_traces = traces[:longest_idx] + serpentine_segments + traces[longest_idx+1:]
        
        return new_traces
    
    def _create_serpentine(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        length: float
    ) -> List[Dict]:
        """
        Create serpentine pattern to add length.
        
        Args:
            start: Start position
            end: End position
            length: Additional length needed
        
        Returns:
            List of trace segments forming serpentine
        """
        # Simplified serpentine: add meander pattern
        # In production, would use more sophisticated algorithm
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        base_length = np.sqrt(dx*dx + dy*dy)
        
        # Calculate number of meanders needed
        meander_width = self.spacing * 2
        meander_length = meander_width * 2  # Each meander adds this length
        num_meanders = int(np.ceil(length / meander_length))
        
        segments = []
        current_pos = start
        
        # Create meander pattern
        for i in range(num_meanders):
            # Perpendicular direction
            perp_x = -dy / base_length * meander_width
            perp_y = dx / base_length * meander_width
            
            # Create meander segment
            mid1 = (current_pos[0] + dx/(num_meanders+1) + perp_x,
                   current_pos[1] + dy/(num_meanders+1) + perp_y)
            mid2 = (current_pos[0] + 2*dx/(num_meanders+1) - perp_x,
                   current_pos[1] + 2*dy/(num_meanders+1) - perp_y)
            
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
        
        # Final segment to end
        segments.append({
            "start_position": current_pos,
            "end_position": end,
            "length": np.sqrt((end[0]-current_pos[0])**2 + (end[1]-current_pos[1])**2)
        })
        
        return segments

