"""
Smart PCB File Upload & Context-Aware Optimization

Integrates:
1. Smart PCB file parser (KiCad, JSON)
2. Knowledge graph building
3. Computational geometry analysis
4. xAI reasoning about design context
5. Context-aware optimization
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

try:
    from backend.parsers.smart_pcb_parser import SmartPCBParser
    from backend.agents.orchestrator import AgentOrchestrator
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.parsers.smart_pcb_parser import SmartPCBParser
    from src.backend.agents.orchestrator import AgentOrchestrator
    from src.backend.geometry.placement import Placement


async def upload_and_analyze_pcb(
    file: UploadFile,
    optimization_intent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload PCB file, build rich context, and optionally optimize.
    
    Args:
        file: PCB file (.kicad_pcb or .json)
        optimization_intent: Optional natural language optimization request
    
    Returns:
        Dictionary with:
        - parsed_placement: Original placement data
        - knowledge_graph: Hierarchical module structure
        - geometry_analysis: Computational geometry metrics
        - design_context: xAI-understood design intent
        - optimization_insights: Recommendations
        - optimized_placement: If optimization_intent provided
    """
    # Save uploaded file temporarily
    temp_dir = "/tmp/dielectric_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Parse PCB file with smart parser
        parser = SmartPCBParser()
        context = parser.parse_pcb_file(file_path)
        
        result = {
            "success": True,
            "parsed_placement": context["placement"],
            "knowledge_graph": context["knowledge_graph"],
            "geometry_analysis": context["geometry_analysis"],
            "design_context": context["design_context"],
            "optimization_insights": context["optimization_insights"],
            "source_file": file.filename
        }
        
        # If optimization intent provided, optimize
        if optimization_intent:
            placement = Placement.from_dict(context["placement"])
            orchestrator = AgentOrchestrator()
            
            optimization_result = await orchestrator.optimize_fast(
                placement,
                optimization_intent,
                callback=None
            )
            
            result["optimized_placement"] = optimization_result.get("placement")
            result["optimization_metrics"] = optimization_result.get("metrics", {})
        
        return result
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse PCB file: {str(e)}\n{traceback.format_exc()}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

