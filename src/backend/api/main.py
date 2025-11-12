"""
FastAPI Backend Server

Low-latency API for Dielectric.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import uuid
import time
import logging
from typing import Optional, Dict, Any

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

# Logger
logger = logging.getLogger(__name__)

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
            "upload_pcb": "/upload/pcb",  # New: Smart PCB file upload
            "optimize": "/optimize",
            "optimize_fast": "/optimize_fast",
            "optimize_production": "/optimize/production",  # New: Production workflow
            "export_kicad": "/export/kicad",
            "simulate": "/simulate",  # New: Simulation endpoints
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
            
            # Validate input data
            if not board_data:
                raise HTTPException(status_code=400, detail="Missing 'board' data in request")
            if not components_data or not isinstance(components_data, list):
                raise HTTPException(status_code=400, detail="Missing or invalid 'components' data in request")
            if not isinstance(nets_data, list):
                nets_data = []
            
            # Ensure board has required fields
            if "width" not in board_data or "height" not in board_data:
                raise HTTPException(status_code=400, detail="Board data missing 'width' or 'height'")
            
            # Validate components
            for i, comp in enumerate(components_data):
                if not isinstance(comp, dict):
                    raise HTTPException(status_code=400, detail=f"Component {i} is not a valid dictionary")
                if "name" not in comp or "package" not in comp:
                    raise HTTPException(status_code=400, detail=f"Component {i} missing 'name' or 'package'")
                # Ensure required numeric fields have defaults
                comp.setdefault("width", 5.0)
                comp.setdefault("height", 5.0)
                comp.setdefault("power", 0.0)
                comp.setdefault("x", 0.0)
                comp.setdefault("y", 0.0)
                comp.setdefault("angle", 0.0)
                comp.setdefault("placed", True)
            
            # Create placement with error handling
            try:
                placement = Placement.from_dict({
                    "board": board_data,
                    "components": components_data,
                    "nets": nets_data
                })
            except Exception as e:
                import traceback
                error_detail = f"Failed to create Placement object: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"❌ Placement creation failed: {error_detail}")
                raise HTTPException(status_code=400, detail=f"Invalid placement data: {str(e)}")
            
            # Run optimization
            try:
                orchestrator = AgentOrchestrator()
                result = await orchestrator.optimize_fast(placement, intent)
            except Exception as e:
                import traceback
                error_detail = f"Optimization failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"❌ Optimization error: {error_detail}")
                raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
            
            if not result.get("success", False):
                error_msg = result.get("error", "Optimization failed")
                logger.error(f"❌ Optimization returned failure: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
            
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
        logger.error(f"❌ Unexpected error in optimize endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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


@app.post("/upload/pcb")
async def upload_pcb_file(
    file: UploadFile = File(...),
    optimization_intent: Optional[str] = None
):
    """
    Upload PCB file (.kicad_pcb, .json, or .zip folder) and build rich context.
    
    Supports:
    - Single files: .kicad_pcb, .json
    - Folders: .zip, .tgz (will be extracted and parsed)
    
    Uses:
    - Smart parser to understand design
    - Knowledge graph for hierarchy
    - Computational geometry analysis
    - xAI to understand design context
    
    Optionally optimizes if optimization_intent is provided.
    """
    try:
        from src.backend.api.pcb_upload import upload_and_analyze_pcb
        return await upload_and_analyze_pcb(file, optimization_intent)
    except ImportError:
        from backend.api.pcb_upload import upload_and_analyze_pcb
        return await upload_and_analyze_pcb(file, optimization_intent)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"❌ Upload endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@app.post("/upload/folder")
async def upload_pcb_folder(
    files: list[UploadFile] = File(...),
    optimization_intent: Optional[str] = None
):
    """
    Upload a folder containing PCB design files.
    
    Intelligently parses through any folder structure to find:
    - KiCad files (.kicad_pcb, .kicad_sch)
    - Altium files (.PcbDoc, .SchDoc)
    - JSON placement files
    - Other PCB-related files
    
    Automatically:
    - Finds the most relevant files
    - Merges data from multiple sources
    - Builds complete design context
    
    Optionally optimizes if optimization_intent is provided.
    """
    try:
        from src.backend.api.pcb_upload import upload_folder
        return await upload_folder(files, optimization_intent)
    except ImportError:
        from backend.api.pcb_upload import upload_folder
        return await upload_folder(files, optimization_intent)


@app.post("/simulate/thermal")
async def simulate_thermal(request: Dict[str, Any]):
    """Run thermal simulation on PCB design."""
    try:
        from src.backend.simulation.pcb_simulator import PCBSimulator
        from src.backend.geometry.placement import Placement
    except ImportError:
        from backend.simulation.pcb_simulator import PCBSimulator
        from backend.geometry.placement import Placement
    
    try:
        placement_data = request.get("placement")
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        placement = Placement.from_dict(placement_data)
        simulator = PCBSimulator()
        
        ambient_temp = request.get("ambient_temp", 25.0)
        board_material = request.get("board_material", "FR4")
        
        result = simulator.simulate_thermal(placement, ambient_temp, board_material)
        
        return {
            "success": True,
            "component_temperatures": result.component_temperatures,
            "max_temperature": result.max_temperature,
            "thermal_gradient": result.thermal_gradient,
            "hotspots": result.hotspots,
            "recommendations": result.cooling_recommendations
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/simulate/signal-integrity")
async def simulate_signal_integrity(request: Dict[str, Any]):
    """Run signal integrity analysis."""
    try:
        from src.backend.simulation.pcb_simulator import PCBSimulator
        from src.backend.geometry.placement import Placement
    except ImportError:
        from backend.simulation.pcb_simulator import PCBSimulator
        from backend.geometry.placement import Placement
    
    try:
        placement_data = request.get("placement")
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        placement = Placement.from_dict(placement_data)
        simulator = PCBSimulator()
        
        frequency = request.get("frequency", 100e6)  # 100 MHz default
        
        result = simulator.analyze_signal_integrity(placement, frequency)
        
        return {
            "success": True,
            "net_impedance": result.net_impedance,
            "crosstalk_risks": result.crosstalk_risks,
            "reflection_risks": result.reflection_risks,
            "timing_violations": result.timing_violations,
            "recommendations": result.recommendations
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/simulate/pdn")
async def simulate_pdn(request: Dict[str, Any]):
    """Run Power Distribution Network analysis."""
    try:
        from src.backend.simulation.pcb_simulator import PCBSimulator
        from src.backend.geometry.placement import Placement
    except ImportError:
        from backend.simulation.pcb_simulator import PCBSimulator
        from backend.geometry.placement import Placement
    
    try:
        placement_data = request.get("placement")
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        placement = Placement.from_dict(placement_data)
        simulator = PCBSimulator()
        
        supply_voltage = request.get("supply_voltage", 5.0)
        
        result = simulator.analyze_pdn(placement, supply_voltage)
        
        return {
            "success": True,
            "voltage_drop": result.voltage_drop,
            "power_loss": result.power_loss,
            "decoupling_effectiveness": result.decoupling_effectiveness,
            "recommendations": result.recommendations
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/optimize/production")
async def optimize_for_production(request: Dict[str, Any]):
    """
    Complete production optimization workflow.
    
    Workflow:
    1. Placement optimization
    2. Trace routing
    3. DFM validation
    4. Error fixing (if needed)
    5. Production file export
    
    Args:
        request: {
            "placement": Dict - Placement data
            "optimization_intent": str - Natural language optimization intent
            "auto_fix": bool - Auto-fix violations (default: True)
            "fabrication_constraints": Dict - Optional custom constraints
        }
    
    Returns:
        {
            "success": bool,
            "production_ready": bool,
            "dfm_score": float,
            "placement": Dict,
            "routing_stats": Dict,
            "verification": Dict,
            "export_files": Dict,
            "workflow_stats": Dict
        }
    """
    try:
        from src.backend.workflows.production_workflow import ProductionWorkflow
        from src.backend.constraints.pcb_fabrication import FabricationConstraints
        from src.backend.geometry.placement import Placement
    except ImportError:
        from backend.workflows.production_workflow import ProductionWorkflow
        from backend.constraints.pcb_fabrication import FabricationConstraints
        from backend.geometry.placement import Placement
    
    try:
        placement_data = request.get("placement", {})
        if not placement_data:
            raise HTTPException(status_code=400, detail="No placement data provided")
        
        # Parse placement
        placement = Placement.from_dict(placement_data)
        
        # Get optimization intent
        optimization_intent = request.get(
            "optimization_intent",
            "Optimize for production: ensure manufacturing constraints, proper routing, and DFM compliance"
        )
        
        # Get fabrication constraints (optional)
        constraints = None
        if "fabrication_constraints" in request:
            constraint_data = request["fabrication_constraints"]
            constraints = FabricationConstraints(**constraint_data)
        
        # Create production workflow
        workflow = ProductionWorkflow(constraints=constraints)
        
        # Run production optimization
        auto_fix = request.get("auto_fix", True)
        max_fix_iterations = request.get("max_fix_iterations", 5)
        
        result = await workflow.optimize_for_production(
            placement,
            optimization_intent,
            auto_fix=auto_fix,
            max_fix_iterations=max_fix_iterations
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Production workflow failed: {result.get('error')}"
            )
        
        # Convert placement to dict for JSON response
        placement_dict = result["placement"].to_dict() if result.get("placement") else None
        
        return convert_numpy_types({
            "success": True,
            "production_ready": result.get("production_ready", False),
            "dfm_score": result.get("dfm_score", 0.0),
            "production_readiness_score": workflow.calculate_production_readiness_score(result),
            "placement": placement_dict,
            "routing_stats": result.get("routing_stats", {}),
            "verification": result.get("verification", {}),
            "export_files": result.get("export_files", {}),
            "workflow_stats": result.get("workflow_stats", {})
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Production workflow failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


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

