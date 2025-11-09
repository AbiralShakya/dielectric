#!/usr/bin/env python3
"""
Simple FastAPI server for hackathon demo - skips Dedalus MCP complexity
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from typing import Dict, Any

# Import our existing code
from src.backend.geometry.placement import Placement
from src.backend.ai.xai_client import XAIClient
from src.backend.agents.orchestrator import AgentOrchestrator
from src.backend.scoring.scorer import WorldModelScorer, ScoreWeights
from src.backend.optimization.local_placer import LocalPlacer

app = FastAPI(title="Neuro-Geometric Placer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Neuro-Geometric Placer API", "status": "running"}

@app.post("/optimize")
async def optimize_placement(request: Dict[str, Any]):
    """AI-agent driven optimization using IntentAgent + Orchestrator"""
    try:
        # Parse input
        board_data = request.get("board", {})
        components_data = request.get("components", [])
        nets_data = request.get("nets", [])
        intent = request.get("intent", "minimize trace length")

        # Create placement
        placement = Placement.from_dict({
            "board": board_data,
            "components": components_data,
            "nets": nets_data
        })

        # Try AI agents first, fallback to direct optimization
        try:
            orchestrator = AgentOrchestrator()
            result = await orchestrator.optimize_fast(placement, intent)

            if result.get("success", False):
                return {
                    "success": True,
                    "optimized_placement": placement.to_dict(),
                    "score": result.get("score", 0.0),
                    "weights_used": result.get("weights", {}),
                    "intent": result.get("intent_explanation", intent),
                    "stats": result.get("stats", {}),
                    "agents_used": ["IntentAgent", "LocalPlacerAgent", "VerifierAgent"],
                    "ai_driven": True,
                    "method": "multi_agent"
                }
        except Exception as agent_error:
            print(f"Agent optimization failed: {agent_error}, falling back to direct optimization")

        # Fallback: Direct optimization with simple intent parsing
        if "cool" in intent.lower() or "thermal" in intent.lower():
            weights = ScoreWeights(alpha=0.3, beta=0.5, gamma=0.2)  # Favor thermal
        elif "trace" in intent.lower() or "wire" in intent.lower():
            weights = ScoreWeights(alpha=0.6, beta=0.2, gamma=0.2)  # Favor trace length
        else:
            weights = ScoreWeights(alpha=0.5, beta=0.3, gamma=0.2)  # Balanced

        # Run direct optimization
        scorer = WorldModelScorer(weights)
        optimizer = LocalPlacer(scorer)
        direct_result = optimizer.optimize_fast(placement, max_iterations=50)

        return {
            "success": True,
            "optimized_placement": placement.to_dict(),
            "score": direct_result.get("score", 0.0),
            "weights_used": {"alpha": weights.alpha, "beta": weights.beta, "gamma": weights.gamma},
            "intent": f"Direct optimization for: {intent}",
            "stats": {"method": "direct", "iterations": 50},
            "agents_used": ["DirectOptimizer"],
            "ai_driven": False,
            "method": "direct_fallback"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
