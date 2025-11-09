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
from src.backend.agents.orchestrator import AgentOrchestrator

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

        # Use Dedalus AI agents - NO FALLBACK
        orchestrator = AgentOrchestrator()
        result = await orchestrator.optimize_fast(placement, intent)

        if not result.get("success", False):
            error_msg = result.get("error", "Unknown Dedalus error")
            raise HTTPException(status_code=500, detail=f"Dedalus optimization failed: {error_msg}")

        return {
            "success": True,
            "optimized_placement": placement.to_dict(),
            "score": result.get("score", 0.0),
            "weights_used": result.get("weights", {}),
            "intent": result.get("intent_explanation", intent),
            "stats": result.get("stats", {}),
            "agents_used": ["IntentAgent", "LocalPlacerAgent", "VerifierAgent"],
            "ai_driven": True,
            "method": "dedalus_multi_agent"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
