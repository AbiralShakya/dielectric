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
from src.backend.advanced.large_design_handler import LargeDesignHandler
from src.backend.simulation.simulation_automation import SimulationAutomation
from src.backend.quality.design_validator import DesignQualityValidator
from src.backend.agents.design_generator_agent import DesignGeneratorAgent

app = FastAPI(title="Dielectric", version="2.0.0", description="AI-Powered PCB Design Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Dielectric API", "status": "running", "version": "2.0.0"}

@app.post("/generate")
async def generate_design(request: Dict[str, Any]):
    """Generate PCB design from natural language description."""
    try:
        description = request.get("description", "")
        board_size = request.get("board_size", None)
        
        if not description:
            raise HTTPException(status_code=400, detail="No description provided")
        
        generator = DesignGeneratorAgent()
        result = await generator.generate_from_natural_language(description, board_size)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Design generation failed")
        
        return convert_numpy_types({
            "success": True,
            "placement": result.get("placement"),
            "description": description,
            "agent": result.get("agent")
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
            # Ensure all required fields exist
            if not isinstance(placement_data.get("components"), list):
                placement_data["components"] = []
            if not isinstance(placement_data.get("nets"), list):
                placement_data["nets"] = []
            if not isinstance(placement_data.get("board"), dict):
                placement_data["board"] = {"width": 100, "height": 100, "clearance": 0.5}
            
            kicad_content = generate_kicad_pcb(placement_data)
        except Exception as e:
            import traceback
            error_detail = f"KiCad generation failed: {str(e)}\n{traceback.format_exc()}"
            raise HTTPException(status_code=500, detail=error_detail)

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
    """Generate professional KiCad PCB file from placement data.
    
    Tries to use KiCAD Python API (via KiCAD-MCP-Server) first,
    falls back to manual file generation if KiCAD is not available.
    """
    # Try KiCAD MCP exporter first (uses pcbnew API)
    try:
        from src.backend.export.kicad_mcp_exporter import KiCadMCPExporter
        mcp_exporter = KiCadMCPExporter()
        output_path = mcp_exporter.export(placement_data)
        content = mcp_exporter.get_file_content(output_path)
        mcp_exporter.cleanup()
        return content
    except ImportError as e:
        print(f"KiCAD Python API not available, using manual exporter: {e}")
        # Fallback to manual exporter
        exporter = KiCadExporter()
        return exporter.export(placement_data, include_nets=True)
    except Exception as e:
        print(f"KiCAD MCP export failed, using manual exporter: {e}")
        # Fallback to manual exporter
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
        import time
        start_time = time.time()
        
        orchestrator = AgentOrchestrator()
        result = await orchestrator.optimize_fast(placement, intent)
        
        optimization_time = time.time() - start_time

        if not result.get("success", False):
            error_msg = result.get("error", "Unknown Dedalus error")
            raise HTTPException(status_code=500, detail=f"Dedalus optimization failed: {error_msg}")

        # Quality validation
        optimized_placement = result.get("placement")
        validator = DesignQualityValidator()
        quality_results = validator.validate_design(optimized_placement)
        
        # Calculate time savings
        component_count = len(placement.components)
        traditional_time_hours = max(4, component_count * 0.5)  # Rough estimate: 0.5 hours per component
        traditional_time_weeks = traditional_time_hours / (40 * 5)  # 40 hours/week
        time_savings_factor = (traditional_time_hours * 3600) / optimization_time if optimization_time > 0 else 0
        
        response_data = {
            "success": True,
            "placement": optimized_placement.to_dict() if hasattr(optimized_placement, "to_dict") else optimized_placement,
            "optimized_placement": optimized_placement.to_dict() if hasattr(optimized_placement, "to_dict") else optimized_placement,  # Backward compatibility
            "score": result.get("score", 0.0),
            "weights_used": result.get("weights", {}),
            "intent": result.get("intent_explanation", intent),
            "geometry_data": result.get("geometry_data", {}),  # Computational geometry analysis
            "quality": convert_numpy_types(quality_results),  # Quality validation
            "stats": result.get("stats", {}),
            "verification": result.get("verification", {}),
            "agents_used": result.get("agents_used", ["IntentAgent", "LocalPlacerAgent", "VerifierAgent"]),
            "ai_driven": True,
            "method": result.get("method", "direct_ai_agents"),
            "performance": {
                "optimization_time_seconds": optimization_time,
                "traditional_time_hours": traditional_time_hours,
                "traditional_time_weeks": traditional_time_weeks,
                "time_savings_factor": time_savings_factor
            }
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulate")
async def run_simulations(request: Dict[str, Any]):
    """Run simulation suite on PCB design."""
    try:
        placement_data = request.get("placement", {})
        design_intent = request.get("intent", "Standard PCB design")
        
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        simulator = SimulationAutomation()
        results = simulator.run_full_simulation_suite(placement_data, design_intent)
        
        return convert_numpy_types(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/large-design/analyze")
async def analyze_large_design(request: Dict[str, Any]):
    """Analyze large design with module identification."""
    try:
        placement_data = request.get("placement", {})
        module_definitions = request.get("modules", None)
        
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        placement = Placement.from_dict(placement_data)
        handler = LargeDesignHandler(placement)
        
        # Identify modules
        modules = handler.identify_modules(module_definitions)
        
        # Analyze hierarchical geometry
        geometry = handler.analyze_hierarchical_geometry()
        
        return convert_numpy_types({
            "modules": [{"name": m.name, "bounds": m.bounds, "component_count": len(m.components)} for m in modules],
            "geometry": geometry,
            "module_count": len(modules)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/large-design/viewport")
async def get_viewport_data(request: Dict[str, Any]):
    """Get viewport data for zoom/pan visualization."""
    try:
        placement_data = request.get("placement", {})
        x_min = request.get("x_min", 0)
        y_min = request.get("y_min", 0)
        x_max = request.get("x_max", 100)
        y_max = request.get("y_max", 100)
        zoom_level = request.get("zoom_level", 1.0)
        
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        placement = Placement.from_dict(placement_data)
        handler = LargeDesignHandler(placement)
        
        viewport_data = handler.get_viewport_data(x_min, y_min, x_max, y_max, zoom_level)
        
        return convert_numpy_types(viewport_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
