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
from src.backend.geometry.geometry_analyzer import convert_numpy_types
from src.backend.export.kicad_exporter import KiCadExporter

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

@app.post("/export/kicad")
async def export_kicad(request: Dict[str, Any]):
    """Export optimized placement to KiCad format"""
    try:
        placement_data = request.get("placement", {})
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")

        # Ensure placement_data has the correct structure
        if not isinstance(placement_data, dict):
            raise HTTPException(status_code=400, detail=f"Placement data must be a dict, got {type(placement_data)}")

        # Validate required fields
        if "board" not in placement_data:
            raise HTTPException(status_code=400, detail="Placement data missing 'board' field")
        if "components" not in placement_data:
            raise HTTPException(status_code=400, detail="Placement data missing 'components' field")
        
        # Ensure components is a list
        if not isinstance(placement_data.get("components"), list):
            raise HTTPException(status_code=400, detail="Components must be a list")

        # Generate KiCad PCB file content
        try:
            kicad_content = generate_kicad_pcb(placement_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"KiCad generation failed: {str(e)}")

        return {
            "success": True,
            "format": "kicad_pcb",
            "filename": "optimized_layout.kicad_pcb",
            "content": kicad_content,
            "size_bytes": len(kicad_content)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)

def generate_kicad_pcb(placement_data: Dict[str, Any]) -> str:
    """Generate professional KiCad PCB file from placement data"""
    exporter = KiCadExporter()
    return exporter.export(placement_data, include_nets=True)

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

        response_data = {
            "success": True,
            "placement": result.get("placement").to_dict() if hasattr(result.get("placement"), "to_dict") else result.get("placement"),
            "optimized_placement": result.get("placement").to_dict() if hasattr(result.get("placement"), "to_dict") else result.get("placement"),  # Backward compatibility
            "score": result.get("score", 0.0),
            "weights_used": result.get("weights", {}),
            "intent": result.get("intent_explanation", intent),
            "geometry_data": result.get("geometry_data", {}),  # Computational geometry analysis
            "stats": result.get("stats", {}),
            "verification": result.get("verification", {}),
            "agents_used": result.get("agents_used", ["IntentAgent", "LocalPlacerAgent", "VerifierAgent"]),
            "ai_driven": True,
            "method": result.get("method", "direct_ai_agents")
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
