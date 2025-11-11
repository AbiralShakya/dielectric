"""
Smart PCB File Parser & Context Analyzer

Parses real PCB files (KiCad, Altium, etc.) and builds rich context
using knowledge graphs, computational geometry, and ML reasoning.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

try:
    from backend.knowledge.knowledge_graph import KnowledgeGraph, ModuleType
    from backend.geometry.placement import Placement
    from backend.geometry.component import Component
    from backend.geometry.board import Board
    from backend.geometry.net import Net
    from backend.geometry.geometry_analyzer import GeometryAnalyzer
    from backend.ai.enhanced_xai_client import EnhancedXAIClient
except ImportError:
    from src.backend.knowledge.knowledge_graph import KnowledgeGraph, ModuleType
    from src.backend.geometry.placement import Placement
    from src.backend.geometry.component import Component
    from src.backend.geometry.board import Board
    from src.backend.geometry.net import Net
    from src.backend.geometry.geometry_analyzer import GeometryAnalyzer
    from src.backend.ai.enhanced_xai_client import EnhancedXAIClient


class SmartPCBParser:
    """
    Intelligent PCB file parser that:
    1. Parses PCB files (KiCad, JSON, etc.)
    2. Builds knowledge graph with hierarchy
    3. Analyzes computational geometry
    4. Uses xAI to understand context and intent
    5. Provides rich context for optimization
    """
    
    def __init__(self):
        """Initialize parser with xAI client."""
        try:
            self.xai_client = EnhancedXAIClient()
        except Exception:
            from src.backend.ai.xai_client import XAIClient
            self.xai_client = XAIClient()
        
        self.geometry_analyzer = GeometryAnalyzer()
    
    def parse_pcb_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PCB file and return rich context.
        
        Supports:
        - KiCad (.kicad_pcb)
        - JSON placement files
        - Altium exports (future)
        """
        path = Path(file_path)
        
        if path.suffix == '.kicad_pcb':
            return self._parse_kicad_pcb(file_path)
        elif path.suffix == '.json':
            return self._parse_json_placement(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _parse_kicad_pcb(self, file_path: str) -> Dict[str, Any]:
        """Parse KiCad PCB file (.kicad_pcb format)."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract board info
        board_match = re.search(r'\(board\s+\(version\s+(\d+)\)\s+\(generator\s+"([^"]+)"\)', content)
        board_info = {
            "version": board_match.group(1) if board_match else "unknown",
            "generator": board_match.group(2) if board_match else "unknown"
        }
        
        # Extract board outline
        edge_cuts = re.findall(r'\(gr_line\s+\(start\s+([\d.]+)\s+([\d.]+)\)\s+\(end\s+([\d.]+)\s+([\d.]+)\)\s+\(layer\s+"Edge.Cuts"\)', content)
        
        # Extract components (footprints)
        components = []
        footprint_pattern = r'\(footprint\s+"([^"]+)"[^)]*\(at\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?\)'
        for match in re.finditer(footprint_pattern, content):
            name = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            angle = float(match.group(4)) if match.group(4) else 0.0
            
            # Extract package info
            package_match = re.search(rf'\(footprint\s+"{re.escape(name)}"[^)]*\(layer\s+"([^"]+)"\)', content)
            layer = package_match.group(1) if package_match else "F.Cu"
            
            # Estimate size from pads
            pad_pattern = rf'\(pad\s+"[^"]+"\s+\(at\s+[\d.]+\s+[\d.]+\)\s+\(size\s+([\d.]+)\s+([\d.]+)\)'
            pads = list(re.finditer(pad_pattern, content))
            
            # Estimate component size
            width = 5.0  # default
            height = 5.0  # default
            if pads:
                sizes = [(float(p.group(1)), float(p.group(2))) for p in pads]
                max_w = max(s[0] for s in sizes)
                max_h = max(s[1] for s in sizes)
                width = max_w * 3  # rough estimate
                height = max_h * 3
            
            components.append({
                "name": name,
                "package": name.split(':')[0] if ':' in name else "UNKNOWN",
                "x": x,
                "y": y,
                "angle": angle,
                "width": width,
                "height": height,
                "layer": layer,
                "placed": True,
                "power": 0.0  # Will be estimated later
            })
        
        # Extract nets
        nets = []
        net_pattern = r'\(net\s+(\d+)\s+"([^"]+)"\)'
        for match in re.finditer(net_pattern, content):
            net_id = match.group(1)
            net_name = match.group(2)
            
            # Find pads connected to this net
            pad_pattern = rf'\(pad\s+"([^"]+)"[^)]*\(net\s+{net_id}\)'
            pads = re.findall(pad_pattern, content)
            
            if pads:
                nets.append({
                    "name": net_name,
                    "pins": [[p.split(':')[0], p.split(':')[1] if ':' in p else "pin1"] for p in pads[:10]]  # Limit pins
                })
        
        # Calculate board bounds
        if edge_cuts:
            all_x = []
            all_y = []
            for edge in edge_cuts:
                all_x.extend([float(edge[0]), float(edge[2])])
                all_y.extend([float(edge[1]), float(edge[3])])
            board_width = max(all_x) - min(all_x) if all_x else 100.0
            board_height = max(all_y) - min(all_y) if all_y else 100.0
        else:
            # Estimate from components
            if components:
                x_coords = [c["x"] for c in components]
                y_coords = [c["y"] for c in components]
                board_width = (max(x_coords) - min(x_coords)) * 1.5 if x_coords else 100.0
                board_height = (max(y_coords) - min(y_coords)) * 1.5 if y_coords else 100.0
            else:
                board_width = 100.0
                board_height = 100.0
        
        placement_data = {
            "board": {
                "width": board_width,
                "height": board_height,
                "clearance": 0.5
            },
            "components": components,
            "nets": nets,
            "modules": []
        }
        
        return self._build_rich_context(placement_data, file_path)
    
    def _parse_json_placement(self, file_path: str) -> Dict[str, Any]:
        """Parse JSON placement file."""
        with open(file_path, 'r') as f:
            placement_data = json.load(f)
        
        return self._build_rich_context(placement_data, file_path)
    
    def _build_rich_context(
        self,
        placement_data: Dict[str, Any],
        source_file: str
    ) -> Dict[str, Any]:
        """
        Build rich context using:
        1. Knowledge graph (hierarchy, modules)
        2. Computational geometry analysis
        3. xAI reasoning about design intent
        """
        # Convert to Placement object
        placement = Placement.from_dict(placement_data)
        
        # 1. Build knowledge graph
        print("📊 Building knowledge graph...")
        kg = KnowledgeGraph(placement)
        
        # 2. Computational geometry analysis
        print("🔬 Analyzing computational geometry...")
        geometry_data = self.geometry_analyzer.analyze_placement(placement)
        
        # 3. Use xAI to understand design context
        print("🤖 Using xAI to understand design context...")
        design_context = self._analyze_design_with_xai(placement_data, geometry_data, kg)
        
        return {
            "placement": placement_data,
            "knowledge_graph": {
                "modules": {name: {
                    "type": mod.module_type.value,
                    "components": mod.components,
                    "bounds": mod.bounds,
                    "thermal_zone": mod.thermal_zone
                } for name, mod in kg.modules.items()},
                "hierarchy_levels": kg.hierarchy_levels,
                "thermal_zones": kg.thermal_zones
            },
            "geometry_analysis": geometry_data,
            "design_context": design_context,
            "source_file": source_file,
            "optimization_insights": self._generate_optimization_insights(kg, geometry_data, design_context)
        }
    
    def _analyze_design_with_xai(
        self,
        placement_data: Dict,
        geometry_data: Dict,
        kg: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Use xAI to understand design intent and context."""
        # Build context prompt
        components_summary = f"{len(placement_data.get('components', []))} components"
        modules_summary = ", ".join([f"{name} ({mod.module_type.value})" for name, mod in kg.modules.items()])
        
        prompt = f"""
        Analyze this PCB design and provide context:
        
        Design Summary:
        - {components_summary}
        - Modules: {modules_summary}
        - Board size: {placement_data.get('board', {}).get('width', 0)}mm x {placement_data.get('board', {}).get('height', 0)}mm
        
        Geometry Analysis:
        - Thermal hotspots: {len(geometry_data.get('thermal_hotspots', []))}
        - Component density: {geometry_data.get('component_density', 0):.2f}
        - Trace length estimate: {geometry_data.get('estimated_trace_length', 0):.2f}mm
        
        Provide:
        1. Design intent (what is this PCB for?)
        2. Key optimization opportunities
        3. Thermal management concerns
        4. Signal integrity considerations
        5. Manufacturing considerations
        
        Return JSON format.
        """
        
        try:
            response = self.xai_client._call_api([
                {"role": "system", "content": "You are a PCB design expert analyzing designs."},
                {"role": "user", "content": prompt}
            ], max_tokens=1500)
            
            if response.get("choices"):
                content = response["choices"][0]["message"]["content"]
                # Try to extract JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            # Fallback: return structured summary
            return {
                "design_intent": "PCB design",
                "optimization_opportunities": ["thermal_management", "component_placement"],
                "thermal_concerns": ["high_power_components"],
                "signal_integrity": ["trace_length", "noise_isolation"],
                "manufacturing": ["component_spacing", "assembly"]
            }
        except Exception as e:
            print(f"⚠️ xAI analysis failed: {e}")
            return {
                "design_intent": "PCB design",
                "error": str(e)
            }
    
    def _generate_optimization_insights(
        self,
        kg: KnowledgeGraph,
        geometry_data: Dict,
        design_context: Dict
    ) -> List[Dict[str, Any]]:
        """Generate optimization insights combining knowledge graph + geometry + context."""
        insights = []
        
        # Thermal insights
        hotspots = geometry_data.get('thermal_hotspots', [])
        if hotspots:
            insights.append({
                "type": "thermal",
                "priority": "high",
                "message": f"Found {len(hotspots)} thermal hotspots. Consider component spacing and thermal vias.",
                "components": [h.get('component') for h in hotspots[:5]]
            })
        
        # Module insights
        if kg.modules:
            insights.append({
                "type": "hierarchy",
                "priority": "medium",
                "message": f"Design has {len(kg.modules)} functional modules. Consider module-level optimization.",
                "modules": list(kg.modules.keys())[:5]
            })
        
        # Density insights
        density = geometry_data.get('component_density', 0)
        if density > 0.7:
            insights.append({
                "type": "density",
                "priority": "medium",
                "message": f"High component density ({density:.2f}). Consider spreading components.",
            })
        
        return insights


def parse_and_optimize(
    pcb_file: str,
    optimization_intent: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse PCB file, understand context, and optimize based on natural language intent.
    
    Args:
        pcb_file: Path to PCB file (.kicad_pcb or .json)
        optimization_intent: Natural language optimization request
        output_path: Optional path to save optimized design
    
    Returns:
        Dictionary with optimized design and analysis
    """
    parser = SmartPCBParser()
    
    # Parse and build context
    context = parser.parse_pcb_file(pcb_file)
    
    # Use orchestrator to optimize with intent
    try:
        from src.backend.agents.orchestrator import AgentOrchestrator
        
        placement = Placement.from_dict(context["placement"])
        orchestrator = AgentOrchestrator()
        
        # Optimize with natural language intent
        result = await orchestrator.optimize_fast(
            placement,
            optimization_intent,
            callback=None
        )
        
        return {
            "original_context": context,
            "optimized_placement": result.get("placement"),
            "optimization_metrics": result.get("metrics", {}),
            "insights": context.get("optimization_insights", [])
        }
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return {
            "original_context": context,
            "error": str(e)
        }

