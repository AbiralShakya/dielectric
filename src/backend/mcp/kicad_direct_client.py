"""
KiCad MCP Direct Client

Direct Python client for KiCad operations using the KiCad MCP server's Python commands.
This bypasses the MCP protocol and directly uses the Python command implementations.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile

# Add KiCad MCP server Python path
KICAD_MCP_PYTHON_PATH = Path(__file__).parent.parent.parent / "kicad-mcp-server" / "python"
if str(KICAD_MCP_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(KICAD_MCP_PYTHON_PATH))

logger = logging.getLogger(__name__)


class KiCadDirectClient:
    """
    Direct Python client for KiCad operations.
    
    Uses KiCad MCP server's Python commands directly without MCP protocol overhead.
    """
    
    def __init__(self, project_path: Optional[str] = None):
        """
        Initialize KiCad direct client.
        
        Args:
            project_path: Optional path to existing KiCad project (.kicad_pcb file)
        """
        self.project_path = project_path
        self.board = None
        self._initialize_kicad()
    
    def _initialize_kicad(self):
        """Initialize KiCad Python API."""
        try:
            import pcbnew
            
            # Try to load board if project path provided
            if self.project_path and os.path.exists(self.project_path):
                self.board = pcbnew.LoadBoard(self.project_path)
                logger.info(f"✅ Loaded KiCad board from {self.project_path}")
            else:
                # Create new board
                self.board = pcbnew.CreateEmptyBoard()
                logger.info("✅ Created new KiCad board")
            
            # Import command classes
            from commands.component import ComponentCommands
            from commands.routing import RoutingCommands
            from commands.design_rules import DesignRuleCommands
            from commands.board import BoardCommands
            
            self.component_cmds = ComponentCommands(self.board)
            self.routing_cmds = RoutingCommands(self.board)
            self.design_rule_cmds = DesignRuleCommands(self.board)
            self.board_cmds = BoardCommands(self.board)
            
            logger.info("✅ KiCad Direct Client initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️  KiCad Python API not available: {e}")
            self.board = None
            self.component_cmds = None
            self.routing_cmds = None
            self.design_rule_cmds = None
            self.board_cmds = None
    
    def is_available(self) -> bool:
        """Check if KiCad is available."""
        return self.board is not None
    
    async def route_trace(
        self,
        start: Dict[str, float],
        end: Dict[str, float],
        layer: str,
        width: float,
        net: str
    ) -> Dict[str, Any]:
        """
        Route a trace between two points.
        
        Args:
            start: {"x": float, "y": float, "unit": "mm"}
            end: {"x": float, "y": float, "unit": "mm"}
            layer: Layer name (e.g., "F.Cu")
            width: Trace width in mm
            net: Net name
        
        Returns:
            Result dictionary
        """
        if not self.routing_cmds:
            return {
                "success": False,
                "error": "KiCad routing commands not available"
            }
        
        try:
            params = {
                "start": start,
                "end": end,
                "layer": layer,
                "width": width,
                "net": net
            }
            
            result = self.routing_cmds.route_trace(params)
            logger.info(f"   KiCad: Routed trace {net} on {layer}")
            return result
            
        except Exception as e:
            logger.error(f"KiCad trace routing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_via(
        self,
        position: Dict[str, float],
        net: str,
        size: Optional[float] = None,
        drill: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add a via.
        
        Args:
            position: {"x": float, "y": float, "unit": "mm"}
            net: Net name
            size: Via size in mm (optional)
            drill: Drill size in mm (optional)
        
        Returns:
            Result dictionary
        """
        if not self.routing_cmds:
            return {
                "success": False,
                "error": "KiCad routing commands not available"
            }
        
        try:
            params = {
                "position": position,
                "net": net
            }
            if size:
                params["size"] = size
            if drill:
                params["drill"] = drill
            
            result = self.routing_cmds.add_via(params)
            logger.info(f"   KiCad: Added via for {net}")
            return result
            
        except Exception as e:
            logger.error(f"KiCad via addition failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_net(self, name: str, net_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a net to the board.
        
        Args:
            name: Net name
            net_class: Optional net class name
        
        Returns:
            Result dictionary
        """
        if not self.routing_cmds:
            return {
                "success": False,
                "error": "KiCad routing commands not available"
            }
        
        try:
            params = {"name": name}
            if net_class:
                params["class"] = net_class
            
            result = self.routing_cmds.add_net(params)
            logger.info(f"   KiCad: Added net {name}")
            return result
            
        except Exception as e:
            logger.error(f"KiCad net addition failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_drc(self, report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Design Rule Check.
        
        Args:
            report_path: Optional path to save DRC report
        
        Returns:
            DRC result with violations
        """
        if not self.design_rule_cmds:
            return {
                "success": False,
                "error": "KiCad DRC commands not available",
                "violations": []
            }
        
        try:
            params = {}
            if report_path:
                params["reportPath"] = report_path
            
            result = self.design_rule_cmds.run_drc(params)
            logger.info(f"   KiCad DRC: Found {len(result.get('violations', []))} violations")
            return result
            
        except Exception as e:
            logger.error(f"KiCad DRC failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "violations": []
            }
    
    async def get_drc_violations(self, severity: str = "all") -> Dict[str, Any]:
        """
        Get DRC violations.
        
        Args:
            severity: "all", "error", or "warning"
        
        Returns:
            Violations list
        """
        if not self.design_rule_cmds:
            return {
                "success": False,
                "violations": []
            }
        
        try:
            params = {"severity": severity}
            result = self.design_rule_cmds.get_drc_violations(params)
            return result
            
        except Exception as e:
            logger.error(f"Get DRC violations failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "violations": []
            }
    
    def save_board(self, output_path: Optional[str] = None) -> str:
        """
        Save board to file.
        
        Args:
            output_path: Optional output path (creates temp file if None)
        
        Returns:
            Path to saved file
        """
        if not self.board:
            raise RuntimeError("No board loaded")
        
        if not output_path:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix='.kicad_pcb', prefix='dielectric_')
            os.close(fd)
        
        import pcbnew
        pcbnew.SaveBoard(output_path, self.board)
        logger.info(f"✅ Saved board to {output_path}")
        return output_path
    
    def get_board_info(self) -> Dict[str, Any]:
        """Get board information."""
        if not self.board:
            return {
                "success": False,
                "error": "No board loaded"
            }
        
        try:
            return {
                "success": True,
                "width": self.board.GetBoardEdgesBoundingBox().GetWidth() / 1000000,  # nm to mm
                "height": self.board.GetBoardEdgesBoundingBox().GetHeight() / 1000000,
                "layer_count": self.board.GetCopperLayerCount(),
                "net_count": self.board.GetNetCount()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

