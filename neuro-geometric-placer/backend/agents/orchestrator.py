"""
Agent Orchestrator

Coordinates multi-agent system for PCB placement optimization.
"""

import asyncio
from typing import Dict, Optional, Callable
from backend.geometry.placement import Placement
from .intent_agent import IntentAgent
from .planner_agent import PlannerAgent
from .local_placer_agent import LocalPlacerAgent
from .global_optimizer_agent import GlobalOptimizerAgent
from .verifier_agent import VerifierAgent
from .exporter_agent import ExporterAgent


class AgentOrchestrator:
    """Orchestrates multi-agent placement optimization."""
    
    def __init__(self):
        """Initialize orchestrator."""
        self.intent_agent = IntentAgent()
        self.planner_agent = PlannerAgent()
        self.local_placer = LocalPlacerAgent()
        self.global_optimizer = GlobalOptimizerAgent()
        self.verifier = VerifierAgent()
        self.exporter = ExporterAgent()
    
    async def optimize_fast(
        self,
        placement: Placement,
        user_intent: str,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Fast path optimization (<200ms).
        
        Args:
            placement: Initial placement
            user_intent: Natural language optimization intent
            callback: Optional progress callback
        
        Returns:
            Complete optimization result
        """
        # Step 1: Intent Agent - Convert intent to weights
        context = {
            "num_components": len(placement.components),
            "board_area": placement.board.width * placement.board.height
        }
        intent_result = await self.intent_agent.process(user_intent, context)
        
        if not intent_result["success"]:
            return {
                "success": False,
                "error": "Intent processing failed",
                "details": intent_result
            }
        
        weights = intent_result["weights"]
        
        # Step 2: Local Placer - Fast optimization
        local_result = await self.local_placer.process(
            placement,
            weights,
            max_time_ms=200.0,
            callback=callback
        )
        
        if not local_result["success"]:
            return {
                "success": False,
                "error": "Local optimization failed",
                "details": local_result
            }
        
        optimized_placement = local_result["placement"]
        
        # Step 3: Verifier - Check design rules
        verify_result = await self.verifier.process(optimized_placement)
        
        return {
            "success": True,
            "placement": optimized_placement,
            "score": local_result["score"],
            "weights": weights,
            "intent_explanation": intent_result["explanation"],
            "stats": local_result["stats"],
            "verification": verify_result
        }
    
    async def optimize_quality(
        self,
        placement: Placement,
        user_intent: str,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None
    ) -> Dict:
        """
        Quality path optimization (background).
        
        Args:
            placement: Initial placement
            user_intent: Natural language optimization intent
            callback: Optional progress callback
            timeout: Optional timeout in seconds
        
        Returns:
            Complete optimization result
        """
        # Step 1: Intent Agent
        context = {
            "num_components": len(placement.components),
            "board_area": placement.board.width * placement.board.height
        }
        intent_result = await self.intent_agent.process(user_intent, context)
        
        if not intent_result["success"]:
            return {
                "success": False,
                "error": "Intent processing failed",
                "details": intent_result
            }
        
        weights = intent_result["weights"]
        
        # Step 2: Planner Agent
        placement_info = {
            "num_components": len(placement.components),
            "board_area": placement.board.width * placement.board.height
        }
        plan_result = await self.planner_agent.process(
            placement_info,
            weights,
            optimization_type="quality"
        )
        
        if not plan_result["success"]:
            return {
                "success": False,
                "error": "Planning failed",
                "details": plan_result
            }
        
        plan = plan_result["plan"]
        
        # Step 3: Global Optimizer
        global_result = await self.global_optimizer.process(
            placement,
            weights,
            plan,
            callback=callback,
            timeout=timeout
        )
        
        if not global_result["success"]:
            return {
                "success": False,
                "error": "Global optimization failed",
                "details": global_result
            }
        
        optimized_placement = global_result["placement"]
        
        # Step 4: Verifier
        verify_result = await self.verifier.process(optimized_placement)
        
        # Step 5: Exporter (optional)
        export_result = await self.exporter.process(optimized_placement, format="json")
        
        return {
            "success": True,
            "placement": optimized_placement,
            "score": global_result["score"],
            "weights": weights,
            "plan": plan,
            "intent_explanation": intent_result["explanation"],
            "stats": global_result["stats"],
            "verification": verify_result,
            "export": export_result
        }

