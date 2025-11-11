"""
Smart PCB File Upload & Context-Aware Optimization

Integrates:
1. Smart PCB file parser (KiCad, JSON)
2. Smart folder parser (any folder structure)
3. Knowledge graph building
4. Computational geometry analysis
5. xAI reasoning about design context
6. Context-aware optimization
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path

try:
    from backend.parsers.smart_pcb_parser import SmartPCBParser
    from backend.parsers.folder_parser import FolderParser
    from backend.agents.orchestrator import AgentOrchestrator
    from backend.geometry.placement import Placement
except ImportError:
    from src.backend.parsers.smart_pcb_parser import SmartPCBParser
    from src.backend.parsers.folder_parser import FolderParser
    from src.backend.agents.orchestrator import AgentOrchestrator
    from src.backend.geometry.placement import Placement

logger = logging.getLogger(__name__)


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
    
    try:
        logger.info(f"Received file upload: {file.filename} ({file.size} bytes)")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to {file_path}")
        
        # Check if it's a zip/folder file (including Altium .PcbDoc which are zip archives)
        file_ext = Path(file.filename).suffix.lower()
        logger.info(f"File extension: {file_ext}")
        
        # Handle zip archives and Altium files (which are zip archives)
        if file_ext in ['.zip', '.tgz', '.tar.gz', '.pcbdoc', '.schdoc', '.prjpcb']:
            logger.info(f"Detected archive/folder file ({file_ext}), using folder parser")
            # Use folder parser
            try:
                folder_parser = FolderParser()
                result = folder_parser.parse_folder(file_path, optimization_intent)
                logger.info(f"Folder parsed successfully: {result.get('success', False)}")
            except Exception as e:
                logger.error(f"Folder parsing failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse folder/archive ({file_ext}): {str(e)}"
                )
        else:
            logger.info("Detected single file, using file parser")
            # Use single file parser
            try:
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
                logger.info(f"File parsed successfully")
            except Exception as e:
                logger.error(f"File parsing failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse PCB file: {str(e)}"
                )
        
        # Validate result
        if not result.get("parsed_placement"):
            logger.warning("No placement data in result")
            raise HTTPException(
                status_code=400,
                detail="Parsed file but no placement data found. Make sure the file contains PCB design data."
            )
        
        # If optimization intent provided, optimize
        if optimization_intent and result.get("parsed_placement"):
            try:
                placement = Placement.from_dict(result["parsed_placement"])
                orchestrator = AgentOrchestrator()
                
                optimization_result = await orchestrator.optimize_fast(
                    placement,
                    optimization_intent,
                    callback=None
                )
                
                result["optimized_placement"] = optimization_result.get("placement")
                result["optimization_metrics"] = optimization_result.get("metrics", {})
            except Exception as e:
                logger.error(f"Optimization failed: {str(e)}", exc_info=True)
                # Don't fail the whole request if optimization fails
                result["optimization_error"] = str(e)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path, ignore_errors=True)
                else:
                    os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {str(e)}")


async def upload_folder(
    files: list[UploadFile],
    optimization_intent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload a folder (as multiple files or zip) and parse intelligently.
    
    Args:
        files: List of uploaded files (or single zip file)
        optimization_intent: Optional natural language optimization request
    
    Returns:
        Dictionary with parsed design data
    """
    temp_dir = "/tmp/dielectric_folder_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    folder_path = os.path.join(temp_dir, "uploaded_folder")
    os.makedirs(folder_path, exist_ok=True)
    
    try:
        # Save all uploaded files
        for file in files:
            # Handle filename with path separators
            safe_filename = file.filename.replace('\\', '/')  # Normalize separators
            file_path = os.path.join(folder_path, safe_filename)
            # Create subdirectories if filename contains path separators
            file_dir = os.path.dirname(file_path)
            if file_dir != folder_path:
                os.makedirs(file_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        # Parse folder
        folder_parser = FolderParser()
        result = folder_parser.parse_folder(folder_path, optimization_intent)
        
        # If optimization intent provided, optimize
        if optimization_intent and result.get("parsed_placement"):
            placement = Placement.from_dict(result["parsed_placement"])
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
            detail=f"Failed to parse folder: {str(e)}\n{traceback.format_exc()}"
        )
    finally:
        # Clean up temp folder
        if os.path.exists(folder_path):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

