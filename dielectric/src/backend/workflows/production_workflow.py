"""
Production Workflow Orchestrator

Complete production-ready workflow: Design → Optimize → Route → Validate → Export
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

try:
    from backend.geometry.placement import Placement
    from backend.agents.orchestrator import AgentOrchestrator
    from backend.agents.routing_agent import RoutingAgent
    from backend.agents.verifier_agent import VerifierAgent
    from backend.agents.exporter_agent import ExporterAgent
    from backend.agents.error_fixer_agent import ErrorFixerAgent
    from backend.constraints.pcb_fabrication import FabricationConstraints
except ImportError:
    from src.backend.geometry.placement import Placement
    from src.backend.agents.orchestrator import AgentOrchestrator
    from src.backend.agents.routing_agent import RoutingAgent
    from src.backend.agents.verifier_agent import VerifierAgent
    from src.backend.agents.exporter_agent import ExporterAgent
    from src.backend.agents.error_fixer_agent import ErrorFixerAgent
    from src.backend.constraints.pcb_fabrication import FabricationConstraints

logger = logging.getLogger(__name__)


class ProductionWorkflow:
    """
    Production-ready workflow orchestrator.
    
    Complete workflow:
    1. Design generation/optimization
    2. Trace routing
    3. DFM validation
    4. Error fixing
    5. Production file export
    """
    
    def __init__(self, constraints: Optional[FabricationConstraints] = None):
        """
        Initialize production workflow.
        
        Args:
            constraints: Fabrication constraints (defaults to standard)
        """
        self.constraints = constraints or FabricationConstraints()
        self.orchestrator = AgentOrchestrator()
        self.routing_agent = RoutingAgent(constraints=self.constraints)
        self.verifier_agent = VerifierAgent(constraints=self.constraints)
        self.exporter_agent = ExporterAgent()
        self.error_fixer_agent = ErrorFixerAgent()
    
    async def optimize_for_production(
        self,
        placement: Placement,
        optimization_intent: str,
        auto_fix: bool = True,
        max_fix_iterations: int = 5
    ) -> Dict:
        """
        Complete production optimization workflow.
        
        Args:
            placement: Initial placement
            optimization_intent: Natural language optimization intent
            auto_fix: Whether to automatically fix violations
            max_fix_iterations: Maximum iterations for error fixing
        
        Returns:
            {
                "success": bool,
                "placement": Placement,
                "production_ready": bool,
                "dfm_score": float,
                "routing_stats": Dict,
                "verification": Dict,
                "export_files": Dict,
                "workflow_stats": Dict
            }
        """
        workflow_start = datetime.now()
        workflow_stats = {
            "stages": [],
            "total_time_ms": 0.0
        }
        
        try:
            logger.info(f"🚀 Production Workflow: Starting optimization")
            
            # Stage 1: Placement Optimization
            stage_start = datetime.now()
            logger.info("📐 Stage 1: Placement Optimization")
            optimization_result = await self.orchestrator.optimize_fast(
                placement,
                optimization_intent
            )
            
            if not optimization_result.get("success"):
                return {
                    "success": False,
                    "error": f"Placement optimization failed: {optimization_result.get('error')}",
                    "workflow_stats": workflow_stats
                }
            
            optimized_placement = optimization_result["placement"]
            stage_time = (datetime.now() - stage_start).total_seconds() * 1000
            workflow_stats["stages"].append({
                "stage": "placement_optimization",
                "time_ms": stage_time,
                "success": True
            })
            logger.info(f"✅ Stage 1 complete: {stage_time:.1f}ms")
            
            # Stage 2: Trace Routing
            stage_start = datetime.now()
            logger.info("🔌 Stage 2: Trace Routing")
            routing_result = await self.routing_agent.route_design(optimized_placement)
            
            if not routing_result.get("success"):
                logger.warning(f"Routing failed: {routing_result.get('error')}")
                # Continue without routing - can be added manually
            
            stage_time = (datetime.now() - stage_start).total_seconds() * 1000
            workflow_stats["stages"].append({
                "stage": "routing",
                "time_ms": stage_time,
                "success": routing_result.get("success", False),
                "routed_nets": routing_result.get("routed_nets", 0)
            })
            logger.info(f"✅ Stage 2 complete: {stage_time:.1f}ms")
            
            # Stage 3: DFM Validation
            stage_start = datetime.now()
            logger.info("✅ Stage 3: DFM Validation")
            verification_result = await self.verifier_agent.process(
                optimized_placement,
                include_dfm=True
            )
            
            dfm_score = verification_result.get("dfm_score", 0.0)
            dfm_ready = verification_result.get("dfm_ready", False)
            
            stage_time = (datetime.now() - stage_start).total_seconds() * 1000
            workflow_stats["stages"].append({
                "stage": "dfm_validation",
                "time_ms": stage_time,
                "success": verification_result.get("passed", False),
                "dfm_score": dfm_score,
                "violations": len(verification_result.get("violations", []))
            })
            logger.info(f"✅ Stage 3 complete: DFM score = {dfm_score:.2f}, {stage_time:.1f}ms")
            
            # Stage 4: Error Fixing (if needed)
            final_placement = optimized_placement
            if not verification_result.get("passed") and auto_fix:
                stage_start = datetime.now()
                logger.info("🔧 Stage 4: Auto-Fixing Violations")
                
                fix_result = await self.error_fixer_agent.fix_design(
                    optimized_placement,
                    max_iterations=max_fix_iterations
                )
                
                if fix_result.get("success") and fix_result.get("fixes_applied"):
                    final_placement = fix_result["placement"]
                    logger.info(f"✅ Fixed {len(fix_result.get('fixes_applied', []))} violations")
                    
                    # Re-verify after fixes
                    verification_result = await self.verifier_agent.process(
                        final_placement,
                        include_dfm=True
                    )
                    dfm_score = verification_result.get("dfm_score", 0.0)
                    dfm_ready = verification_result.get("dfm_ready", False)
                
                stage_time = (datetime.now() - stage_start).total_seconds() * 1000
                workflow_stats["stages"].append({
                    "stage": "error_fixing",
                    "time_ms": stage_time,
                    "success": fix_result.get("success", False),
                    "fixes_applied": len(fix_result.get("fixes_applied", []))
                })
            
            # Stage 5: Production File Export
            export_files = {}
            if dfm_ready or verification_result.get("passed"):
                stage_start = datetime.now()
                logger.info("📦 Stage 5: Production File Export")
                
                # Export KiCad PCB file
                kicad_result = await self.exporter_agent.process(final_placement, format="kicad")
                if kicad_result.get("success"):
                    export_files["kicad_pcb"] = kicad_result.get("output")
                
                # TODO: Add Gerber, drill, BOM, pick-place exports
                # export_files["gerber"] = await self._export_gerber(final_placement)
                # export_files["drill"] = await self._export_drill(final_placement)
                # export_files["bom"] = await self._generate_bom(final_placement)
                # export_files["pick_place"] = await self._generate_pick_place(final_placement)
                
                stage_time = (datetime.now() - stage_start).total_seconds() * 1000
                workflow_stats["stages"].append({
                    "stage": "export",
                    "time_ms": stage_time,
                    "success": len(export_files) > 0,
                    "files_generated": list(export_files.keys())
                })
                logger.info(f"✅ Stage 5 complete: {len(export_files)} files, {stage_time:.1f}ms")
            
            # Calculate total time
            total_time = (datetime.now() - workflow_start).total_seconds() * 1000
            workflow_stats["total_time_ms"] = total_time
            
            production_ready = dfm_ready and verification_result.get("passed", False)
            
            logger.info(f"🎉 Production Workflow Complete: "
                       f"Ready={production_ready}, DFM={dfm_score:.2f}, Time={total_time:.1f}ms")
            
            return {
                "success": True,
                "placement": final_placement,
                "production_ready": production_ready,
                "dfm_score": dfm_score,
                "routing_stats": routing_result.get("routing_stats", {}),
                "verification": verification_result,
                "export_files": export_files,
                "workflow_stats": workflow_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Production workflow failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "workflow_stats": workflow_stats
            }
    
    def calculate_production_readiness_score(self, result: Dict) -> float:
        """
        Calculate overall production readiness score.
        
        Args:
            result: Workflow result dictionary
        
        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0
        
        # DFM score (40%)
        dfm_score = result.get("dfm_score", 0.0)
        score += dfm_score * 0.4
        
        # Verification passed (30%)
        verification = result.get("verification", {})
        if verification.get("passed", False):
            score += 0.3
        
        # Routing complete (20%)
        routing_stats = result.get("routing_stats", {})
        total_nets = routing_stats.get("total_nets", 0)
        routed_nets = routing_stats.get("routed_nets", 0)
        if total_nets > 0:
            routing_completion = routed_nets / total_nets
            score += routing_completion * 0.2
        
        # Export files generated (10%)
        export_files = result.get("export_files", {})
        if len(export_files) > 0:
            score += 0.1
        
        return min(1.0, score)

