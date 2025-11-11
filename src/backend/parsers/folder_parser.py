"""
Smart Folder Parser for PCB Design Projects

Intelligently parses through any folder structure to find and extract PCB design files.
Supports:
- KiCad projects (.kicad_pcb, .kicad_sch, .PrjPcb)
- Altium projects (.PcbDoc, .SchDoc, .PrjPcb)
- JSON placement files
- Mixed folder structures
"""

import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    from backend.parsers.smart_pcb_parser import SmartPCBParser
except ImportError:
    from src.backend.parsers.smart_pcb_parser import SmartPCBParser

logger = logging.getLogger(__name__)


class FolderParser:
    """
    Intelligent folder parser that finds and parses PCB design files
    regardless of folder structure.
    """
    
    # File extensions prioritized by importance
    PCB_FILE_PATTERNS = {
        # KiCad files (highest priority)
        'kicad_pcb': ['.kicad_pcb'],
        'kicad_sch': ['.kicad_sch'],
        'kicad_pro': ['.kicad_pro'],
        
        # Altium files
        'altium_pcb': ['.PcbDoc', '.pcbdoc'],
        'altium_sch': ['.SchDoc', '.schdoc'],
        'altium_prj': ['.PrjPcb', '.prjpcb'],
        
        # JSON placement files
        'json_placement': ['.json'],
        
        # Other formats
        'gerber': ['.gbr', '.gerber', '.gko', '.gbl', '.gbs', '.gto', '.gtp', '.gts'],
        'drill': ['.drl', '.nc', '.txt'],
        'odb': ['.tgz', '.tar.gz'],
    }
    
    # Keywords that indicate important files
    IMPORTANT_KEYWORDS = [
        'main', 'board', 'layout', 'schematic', 'design', 'pcb',
        'rev', 'revision', 'final', 'production', 'assembly'
    ]
    
    def __init__(self):
        """Initialize folder parser."""
        self.smart_parser = SmartPCBParser()
    
    def parse_folder(
        self,
        folder_path: str,
        optimization_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a folder containing PCB design files.
        
        Args:
            folder_path: Path to folder or zip file
            optimization_intent: Optional natural language optimization request
        
        Returns:
            Dictionary with parsed design data and metadata
        """
        folder_path = Path(folder_path)
        
        # Handle zip files (including Altium .PcbDoc which are zip archives)
        if folder_path.suffix.lower() in ['.zip', '.tgz', '.tar.gz', '.pcbdoc', '.schdoc', '.prjpcb']:
            return self._parse_zip(folder_path, optimization_intent)
        
        # Handle directory
        if folder_path.is_dir():
            return self._parse_directory(folder_path, optimization_intent)
        
        raise ValueError(f"Invalid folder path: {folder_path}")
    
    def _parse_zip(
        self,
        zip_path: Path,
        optimization_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract and parse zip file."""
        temp_dir = tempfile.mkdtemp(prefix="dielectric_folder_")
        
        try:
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            logger.info(f"Extracted zip to {temp_dir}")
            return self._parse_directory(Path(temp_dir), optimization_intent)
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _parse_directory(
        self,
        dir_path: Path,
        optimization_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse directory structure to find PCB files."""
        logger.info(f"Scanning folder: {dir_path}")
        
        # Find all PCB-related files
        found_files = self._find_pcb_files(dir_path)
        
        if not found_files:
            logger.warning(f"No PCB design files found in {dir_path}")
            # List what files were actually found
            all_files = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            logger.info(f"   Found {len(all_files)} total files (none matched PCB patterns)")
            if all_files[:10]:  # Show first 10 files
                logger.info(f"   Sample files: {all_files[:10]}")
            raise ValueError(f"No PCB design files found in {dir_path}. Found {len(all_files)} files but none matched known PCB file patterns.")
        
        logger.info(f"Found {len(found_files)} PCB-related file types")
        for file_type, files in found_files.items():
            logger.info(f"   - {file_type}: {len(files)} file(s)")
        
        # Prioritize and select best files to parse
        primary_files = self._prioritize_files(found_files)
        
        if not primary_files:
            raise ValueError("No suitable PCB files found to parse")
        
        # Parse primary files
        parsed_data = {}
        errors = []
        
        for file_type, file_path in primary_files.items():
            try:
                logger.info(f"Parsing {file_type}: {file_path.name}")
                parsed_data[file_type] = self.smart_parser.parse_pcb_file(str(file_path))
                logger.info(f"   Successfully parsed {file_type}")
            except Exception as e:
                logger.warning(f"Failed to parse {file_path.name}: {str(e)}")
                errors.append(f"{file_path.name}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        if not parsed_data:
            raise ValueError(f"Failed to parse any files. Errors: {errors}")
        
        # Merge parsed data
        merged_placement = self._merge_parsed_data(parsed_data)
        
        if not merged_placement.get("placement"):
            raise ValueError("Parsed files but no placement data extracted")
        
        result = {
            "success": True,
            "source_folder": str(dir_path),
            "files_found": {k: [str(f) for f in v] for k, v in found_files.items()},
            "files_parsed": list(primary_files.keys()),
            "parsed_placement": merged_placement.get("placement", {}),
            "knowledge_graph": merged_placement.get("knowledge_graph", {}),
            "geometry_analysis": merged_placement.get("geometry_analysis", {}),
            "design_context": merged_placement.get("design_context", {}),
            "optimization_insights": merged_placement.get("optimization_insights", {}),
            "parsing_errors": errors,
            "folder_structure": self._analyze_folder_structure(dir_path)
        }
        
        logger.info(f"Folder parsing complete: {len(merged_placement.get('placement', {}).get('components', []))} components found")
        
        return result
    
    def _find_pcb_files(self, dir_path: Path) -> Dict[str, List[Path]]:
        """
        Recursively find all PCB-related files in directory.
        
        Returns:
            Dictionary mapping file types to lists of file paths
        """
        found_files = {file_type: [] for file_type in self.PCB_FILE_PATTERNS.keys()}
        found_files['other'] = []
        
        # Walk through directory recursively
        for root, dirs, files in os.walk(dir_path):
            root_path = Path(root)
            
            # Skip common non-design directories
            skip_dirs = ['node_modules', '.git', '__pycache__', '.venv', 'venv']
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                file_path = root_path / file
                file_ext = file_path.suffix.lower()
                
                # Check against known patterns
                matched = False
                for file_type, extensions in self.PCB_FILE_PATTERNS.items():
                    if file_ext in extensions:
                        found_files[file_type].append(file_path)
                        matched = True
                        break
                
                if not matched:
                    # Check if filename suggests PCB file
                    file_lower = file.lower()
                    if any(keyword in file_lower for keyword in ['pcb', 'board', 'layout', 'schematic']):
                        found_files['other'].append(file_path)
        
        # Remove empty lists
        return {k: v for k, v in found_files.items() if v}
    
    def _prioritize_files(self, found_files: Dict[str, List[Path]]) -> Dict[str, Path]:
        """
        Prioritize files and select the best ones to parse.
        
        Priority order:
        1. KiCad PCB files (.kicad_pcb)
        2. Altium PCB files (.PcbDoc)
        3. JSON placement files
        4. KiCad schematics
        5. Altium schematics
        
        Returns:
            Dictionary mapping file types to selected file paths
        """
        selected = {}
        
        # Priority order
        priority_order = [
            'kicad_pcb',
            'altium_pcb',
            'json_placement',
            'kicad_sch',
            'altium_sch',
            'kicad_pro',
            'altium_prj'
        ]
        
        for file_type in priority_order:
            if file_type in found_files and found_files[file_type]:
                files = found_files[file_type]
                
                # Score files by importance
                scored_files = [(self._score_file(f), f) for f in files]
                scored_files.sort(reverse=True, key=lambda x: x[0])
                
                # Select best file
                best_file = scored_files[0][1]
                selected[file_type] = best_file
                
                logger.info(f"   Selected {file_type}: {best_file.name} (score: {scored_files[0][0]})")
        
        return selected
    
    def _score_file(self, file_path: Path) -> float:
        """
        Score file by importance (higher = more important).
        
        Factors:
        - Filename contains important keywords
        - File is in root or common directories
        - File size (reasonable size preferred)
        """
        score = 0.0
        file_name_lower = file_path.name.lower()
        file_stem_lower = file_path.stem.lower()
        
        # Check for important keywords
        for keyword in self.IMPORTANT_KEYWORDS:
            if keyword in file_name_lower or keyword in file_stem_lower:
                score += 10.0
        
        # Prefer files in root or common directories
        depth = len(file_path.parts)
        # Subtract 1 if it's an absolute path (to account for root)
        if file_path.is_absolute():
            depth -= 1
        score += max(0, 20 - depth * 2)
        
        # Prefer reasonable file sizes (not too small, not too large)
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if 0.01 < size_mb < 50:  # Between 10KB and 50MB
                score += 5.0
            elif size_mb > 100:  # Very large files might be problematic
                score -= 10.0
        except:
            pass
        
        # Prefer files without version numbers in name (more likely to be main file)
        if not any(char.isdigit() for char in file_stem_lower[-5:]):
            score += 3.0
        
        return score
    
    def _merge_parsed_data(self, parsed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge data from multiple parsed files.
        
        Priority:
        1. PCB layout files (most complete)
        2. Schematic files (for net information)
        3. JSON files (for metadata)
        """
        if not parsed_data:
            return {}
        
        # Priority order for merging
        merge_order = ['kicad_pcb', 'altium_pcb', 'json_placement', 'kicad_sch', 'altium_sch']
        
        merged = {}
        
        for file_type in merge_order:
            if file_type in parsed_data:
                data = parsed_data[file_type]
                
                # Merge placement data (prefer PCB files)
                if 'placement' in data and not merged.get('placement'):
                    merged['placement'] = data['placement']
                elif 'placement' in data:
                    # Merge components and nets
                    existing = merged['placement']
                    new_data = data['placement']
                    
                    # Merge components (avoid duplicates)
                    existing_comps = {c.get('name'): c for c in existing.get('components', [])}
                    for comp in new_data.get('components', []):
                        if comp.get('name') not in existing_comps:
                            existing['components'].append(comp)
                    
                    # Merge nets (avoid duplicates)
                    existing_nets = {n.get('name'): n for n in existing.get('nets', [])}
                    for net in new_data.get('nets', []):
                        if net.get('name') not in existing_nets:
                            existing['nets'].append(net)
                
                # Merge other data (prefer first non-empty)
                for key in ['knowledge_graph', 'geometry_analysis', 'design_context', 'optimization_insights']:
                    if key in data and not merged.get(key):
                        merged[key] = data[key]
        
        return merged
    
    def _analyze_folder_structure(self, dir_path: Path) -> Dict[str, Any]:
        """Analyze folder structure to understand project organization."""
        structure = {
            "root_files": [],
            "subdirectories": [],
            "total_files": 0,
            "file_types": {}
        }
        
        for root, dirs, files in os.walk(dir_path):
            root_path = Path(root)
            relative_path = root_path.relative_to(dir_path)
            
            if relative_path == Path('.'):
                # Root level
                structure["root_files"] = files
            else:
                structure["subdirectories"].append(str(relative_path))
            
            structure["total_files"] += len(files)
            
            # Count file types
            for file in files:
                ext = Path(file).suffix.lower()
                structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
        
        return structure

