"""
FastAPI Backend Server

Low-latency API for Neuro-Geometric Placer.
"""

import os
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.agents.orchestrator import AgentOrchestrator
from backend.geometry.placement import Placement

app = FastAPI(title="Neuro-Geometric Placer API", version="0.1.0")

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
        "message": "Neuro-Geometric Placer API",
        "version": "0.1.0",
        "endpoints": {
            "upload": "/upload",
            "optimize": "/optimize",
            "optimize_fast": "/optimize_fast",
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
async def optimize_quality(request: OptimizeRequest):
    """
    Quality path optimization (background).
    
    Returns:
        Optimization status
    """
    try:
        task_id = request.task_id
        
        if task_id not in placements_store:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = placements_store[task_id]
        placement = task_data["placement"]
        user_intent = request.user_intent or task_data.get("user_intent", "Optimize placement")
        opt_type = request.optimization_type or task_data.get("optimization_type", "quality")
        
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
    except Exception as e:
        if task_id in placements_store:
            placements_store[task_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Neuro-Geometric Placer API"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

