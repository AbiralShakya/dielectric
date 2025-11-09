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

@app.post("/export/kicad")
async def export_kicad(request: Dict[str, Any]):
    """Export optimized placement to KiCad format"""
    try:
        placement_data = request.get("placement", {})
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")

        # Generate KiCad PCB file content
        kicad_content = generate_kicad_pcb(placement_data)

        return {
            "success": True,
            "format": "kicad_pcb",
            "filename": "optimized_layout.kicad_pcb",
            "content": kicad_content,
            "size_bytes": len(kicad_content)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_kicad_pcb(placement_data: Dict[str, Any]) -> str:
    """Generate KiCad PCB file from placement data"""
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])

    lines = [
        "(kicad_pcb (version 20221018) (generator \"neuro-geometric-placer\")",
        "",
        "  (general",
        "    (thickness 1.6)",
        "  )",
        "",
        "  (layers",
        '    (0 "F.Cu" signal)',
        '    (31 "B.Cu" signal)',
        "  )",
        "",
        "  (net_class \"Default\" \"\"",
        "    (clearance 0.2)",
        "    (trace_width 0.25)",
        "    (via_dia 0.8)",
        "    (via_drill 0.4)",
        "    (uvia_dia 0.3)",
        "    (uvia_drill 0.1)",
        "  )",
        ""
    ]

    # Add components as footprints
    for comp in components:
        name = comp.get("name", "UNK")
        package = comp.get("package", "Unknown")
        x = comp.get("x", 0)
        y = comp.get("y", 0)
        angle = comp.get("angle", 0)
        width = comp.get("width", 5)
        height = comp.get("height", 5)

        lines.extend([
            f'  (footprint "{package}" (layer "F.Cu")',
            "    (tedit 0) (tstamp 0)",
            f"    (at {x:.3f} {y:.3f} {angle})",
            f'    (descr "{package} footprint - AI optimized")',
            "    (tags \"ai-optimized\")",
            f"    (pad \"\" thru_hole circle (at 0 0) (size {min(width, height)*0.8:.2f} {min(width, height)*0.8:.2f}) (drill {min(width, height)*0.4:.2f})",
            '      (layers *.Cu *.Mask))',
            "  )",
            ""
        ])

    lines.append(")")

    return "\n".join(lines)

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
            "method": "direct_ai_agents"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
