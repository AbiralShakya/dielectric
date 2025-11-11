"""
FastAPI Backend Server

Low-latency API for Dielectric.
"""

import os
import uuid
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.agents.orchestrator import AgentOrchestrator
    from backend.geometry.placement import Placement
    from backend.agents.design_generator_agent import DesignGeneratorAgent
    from backend.geometry.geometry_analyzer import convert_numpy_types
except ImportError:
    from src.backend.agents.orchestrator import AgentOrchestrator
    from src.backend.geometry.placement import Placement
    from src.backend.agents.design_generator_agent import DesignGeneratorAgent
    from src.backend.geometry.geometry_analyzer import convert_numpy_types

app = FastAPI(title="Dielectric API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
placements_store: Dict[str, Any] = {}
results_store: Dict[str, Any] = {}


class PlacementRequest(BaseModel):
    """Request model for placement."""
    placement_data: Dict[str, Any]
    user_intent: str
    optimization_type: Optional[str] = "fast"  # "fast" or "quality"


class OptimizeRequest(BaseModel):
    """Request model for optimization."""
    task_id: str
    user_intent: Optional[str] = None
    optimization_type: Optional[str] = "fast"


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dielectric API",
        "version": "0.1.0",
        "endpoints": {
            "generate": "/generate",
            "upload": "/upload",
            "optimize": "/optimize",
            "optimize_fast": "/optimize_fast",
            "export_kicad": "/export/kicad",
            "results": "/results/{task_id}",
            "health": "/health"
        }
    }


@app.post("/upload")
async def upload_placement(request: PlacementRequest):
    """
    Upload placement for optimization.
    
    Returns:
        Task ID for tracking
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Parse placement
        placement = Placement.from_dict(request.placement_data)
        
        # Store
        placements_store[task_id] = {
            "placement": placement,
            "user_intent": request.user_intent,
            "optimization_type": request.optimization_type,
            "status": "uploaded"
        }
        
        return {
            "task_id": task_id,
            "status": "uploaded",
            "message": "Placement uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize_fast")
async def optimize_fast(request: OptimizeRequest):
    """
    Fast path optimization (<200ms).
    
    Returns:
        Optimization result
    """
    try:
        task_id = request.task_id
        
        if task_id not in placements_store:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = placements_store[task_id]
        placement = task_data["placement"]
        user_intent = request.user_intent or task_data.get("user_intent", "Optimize placement")
        
        # Update status
        placements_store[task_id]["status"] = "optimizing"
        
        # Run optimization
        orchestrator = AgentOrchestrator()
        result = await orchestrator.optimize_fast(placement, user_intent)
        
        # Store result
        results_store[task_id] = result
        placements_store[task_id]["status"] = "completed" if result["success"] else "failed"
        
        return {
            "task_id": task_id,
            "status": placements_store[task_id]["status"],
            "success": result["success"],
            "score": result.get("score"),
            "weights": result.get("weights"),
            "stats": result.get("stats", {})
        }
    except Exception as e:
        if task_id in placements_store:
            placements_store[task_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_placement(request: Dict[str, Any]):
    """
    Optimize PCB placement directly (frontend-compatible).
    
    Accepts direct placement data or task_id.
    """
    try:
        # Check if request has direct placement data (frontend format)
        if "board" in request and "components" in request:
            # Direct placement data from frontend
            board_data = request.get("board", {})
            components_data = request.get("components", [])
            nets_data = request.get("nets", [])
            intent = request.get("intent", "Optimize placement")
            
            # Create placement
            placement = Placement.from_dict({
                "board": board_data,
                "components": components_data,
                "nets": nets_data
            })
            
            # Run optimization
            orchestrator = AgentOrchestrator()
            result = await orchestrator.optimize_fast(placement, intent)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=result.get("error", "Optimization failed"))
            
            # Return frontend-compatible format
            return convert_numpy_types({
                "success": True,
                "placement": result.get("placement").to_dict() if result.get("placement") else None,
                "score": result.get("score", 0.0),
                "weights_used": result.get("weights", {}),
                "intent": result.get("intent_explanation", intent),
                "geometry_data": result.get("geometry_data", {}),
                "stats": result.get("stats", {}),
                "verification": result.get("verification", {}),
                "agents_used": result.get("agents_used", ["IntentAgent", "LocalPlacerAgent", "VerifierAgent"]),
                "method": result.get("method", "direct_ai_agents")
            })
        else:
            # Legacy task_id format
            task_id = request.get("task_id")
            if not task_id or task_id not in placements_store:
                raise HTTPException(status_code=404, detail="Task not found or invalid request")
            
            task_data = placements_store[task_id]
            placement = task_data["placement"]
            user_intent = request.get("user_intent") or task_data.get("user_intent", "Optimize placement")
            opt_type = request.get("optimization_type") or task_data.get("optimization_type", "fast")
            
            # Update status
            placements_store[task_id]["status"] = "optimizing"
            
            # Run optimization
            orchestrator = AgentOrchestrator()
            
            if opt_type == "fast":
                result = await orchestrator.optimize_fast(placement, user_intent)
            else:
                result = await orchestrator.optimize_quality(placement, user_intent, timeout=300.0)
            
            # Store result
            results_store[task_id] = result
            placements_store[task_id]["status"] = "completed" if result["success"] else "failed"
            
            return {
                "task_id": task_id,
                "status": placements_store[task_id]["status"],
                "success": result["success"],
                "score": result.get("score"),
                "message": "Optimization completed" if result["success"] else "Optimization failed"
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """
    Get optimization results.
    
    Args:
        task_id: Task ID
    
    Returns:
        Complete results
    """
    if task_id not in results_store:
        if task_id in placements_store:
            return {
                "task_id": task_id,
                "status": placements_store[task_id]["status"],
                "message": "Optimization in progress or not started"
            }
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = results_store[task_id]
    
    return {
        "task_id": task_id,
        "status": placements_store.get(task_id, {}).get("status", "completed"),
        "success": result.get("success", False),
        "placement": result.get("placement").to_dict() if result.get("placement") else None,
        "score": result.get("score"),
        "weights": result.get("weights"),
        "stats": result.get("stats", {}),
        "verification": result.get("verification", {}),
        "intent_explanation": result.get("intent_explanation")
    }


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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)


@app.post("/export/kicad")
async def export_kicad(request: Dict[str, Any]):
    """Export optimized placement to KiCad format"""
    try:
        placement_data = request.get("placement", {})
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")

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
            # Try KiCad MCP exporter first
            try:
                from src.backend.export.kicad_mcp_exporter import KiCadMCPExporter
                mcp_exporter = KiCadMCPExporter()
                output_path = mcp_exporter.export(placement_data)
                content = mcp_exporter.get_file_content(output_path)
                mcp_exporter.cleanup()
                kicad_content = content
            except ImportError:
                # Fallback to manual exporter
                try:
                    from src.backend.export.kicad_exporter import KiCadExporter
                    exporter = KiCadExporter()
                    kicad_content = exporter.export(placement_data, include_nets=True)
                except ImportError:
                    # Final fallback: basic KiCad format
                    kicad_content = generate_basic_kicad(placement_data)
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


def generate_basic_kicad(placement_data: Dict[str, Any]) -> str:
    """Generate basic KiCad PCB file format."""
    board = placement_data.get("board", {})
    components = placement_data.get("components", [])
    
    lines = [
        "(kicad_pcb (version 20211014) (generator pcbnew)",
        "  (general",
        f"    (thickness {board.get('thickness', 1.6)})",
        "  )",
        "  (paper \"A4\")",
        "  (layers",
        "    (0 \"F.Cu\" signal)",
        "    (31 \"B.Cu\" signal)",
        "    (32 \"B.Adhes\" user \"B.Adhesive\")",
        "    (33 \"F.Adhes\" user \"F.Adhesive\")",
        "  )"
    ]
    
    # Add components
    for comp in components:
        name = comp.get("name", "UNK")
        x = comp.get("x", 0) * 1000000  # Convert mm to nanometers
        y = comp.get("y", 0) * 1000000
        angle = comp.get("angle", 0)
        
        lines.extend([
            f"  (footprint \"{comp.get('package', 'SOIC-8')}\" (version 20211014)",
            f"    (layer \"F.Cu\")",
            f"    (tedit {int(time.time() * 1000000)})",
            f"    (tstamp {uuid.uuid4().hex[:8]})",
            f"    (at {x} {y} {angle})",
            f"    (descr \"{name}\")",
            f"    (tags \"{comp.get('package', '')}\")",
            "  )"
        ])
    
    lines.append(")")
    return "\n".join(lines)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Dielectric API"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

