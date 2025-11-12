#!/usr/bin/env python3
"""
Dataset Collection Script

Collects PCB designs from various sources for training/evaluation.
Moved from dielectric/scripts/ to dielectric_ml_research/data_collection/
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import argparse

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Try to import from dielectric if available
    sys.path.insert(0, str(project_root / "dielectric" / "src"))
    from backend.geometry.placement import Placement
    from backend.geometry.board import Board
    from backend.geometry.component import Component
    from backend.geometry.net import Net
except ImportError:
    print("⚠️  Could not import Dielectric modules - some features may not work")
    # Define minimal classes for standalone operation
    class Placement:
        def __init__(self, **kwargs):
            self.board = kwargs.get('board')
            self.components = kwargs.get('components', {})
            self.nets = kwargs.get('nets', {})
        def to_dict(self):
            return {"board": self.board.__dict__ if self.board else {}, 
                   "components": {k: v.__dict__ for k, v in self.components.items()},
                   "nets": {k: v.__dict__ for k, v in self.nets.items()}}
    class Board:
        def __init__(self, width=100, height=100):
            self.width = width
            self.height = height
    class Component:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class Net:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


def collect_kicad_from_github(output_dir: Path, max_repos: int = 10):
    """
    Collect KiCad designs from GitHub repositories.
    
    Searches GitHub for repositories containing .kicad_pcb files.
    """
    print(f"🔍 Collecting KiCad designs from GitHub...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pcb_dir = output_dir / "pcb_files"
    pcb_dir.mkdir(exist_ok=True)
    
    # Known repositories with KiCad designs
    known_repos = [
        "adafruit/Adafruit-PCB-Library",
        "sparkfun/SparkFun-KiCad-Libraries",
        "raspberrypi/pico-examples",
        "kicad/kicad-footprints",
        "kicad/kicad-symbols",
    ]
    
    collected_count = 0
    
    for repo in known_repos[:max_repos]:
        repo_name = repo.split("/")[-1]
        repo_path = output_dir / repo_name
        
        print(f"  📦 Cloning {repo}...")
        
        try:
            # Clone repository
            if repo_path.exists():
                print(f"    ⚠️  Repository already exists, skipping...")
                continue
            
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
                capture_output=True,
                timeout=60,
                check=False
            )
            
            # Find all .kicad_pcb files
            pcb_files = list(repo_path.rglob("*.kicad_pcb"))
            print(f"    ✅ Found {len(pcb_files)} PCB files")
            
            # Copy to dataset directory
            for pcb_file in pcb_files:
                dest = pcb_dir / f"{repo_name}_{pcb_file.name}"
                if not dest.exists():
                    dest.write_bytes(pcb_file.read_bytes())
                    collected_count += 1
            
        except Exception as e:
            print(f"    ❌ Error cloning {repo}: {e}")
            continue
    
    print(f"✅ Collected {collected_count} PCB files")
    return collected_count


def generate_synthetic_designs(output_dir: Path, num_designs: int = 100):
    """
    Generate synthetic PCB designs for training.
    """
    print(f"🎨 Generating {num_designs} synthetic designs...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    
    designs = []
    
    for i in range(num_designs):
        # Vary complexity
        complexity = np.random.choice(["simple", "medium", "complex"])
        num_components = {
            "simple": np.random.randint(5, 15),
            "medium": np.random.randint(15, 50),
            "complex": np.random.randint(50, 100)
        }[complexity]
        
        # Generate design
        try:
            board = Board(
                width=np.random.uniform(50, 150),
                height=np.random.uniform(50, 150)
            )
            
            components = {}
            for j in range(num_components):
                comp = Component(
                    name=f"U{j+1}",
                    package=np.random.choice(["SOIC-8", "0805", "QFN-16", "BGA"]),
                    x=np.random.uniform(10, board.width - 10),
                    y=np.random.uniform(10, board.height - 10),
                    angle=np.random.choice([0, 90, 180, 270]),
                    power=np.random.uniform(0, 2.0)
                )
                components[comp.name] = comp
            
            # Generate nets
            nets = {}
            net_id = 1
            comp_names = list(components.keys())
            for j in range(len(comp_names) - 1):
                net = Net(
                    name=f"Net{net_id}",
                    pins=[(comp_names[j], "pin1"), (comp_names[j+1], "pin1")]
                )
                nets[net.name] = net
                net_id += 1
            
            placement = Placement(board=board, components=components, nets=nets)
            
            # Save design
            design_data = {
                "id": i,
                "complexity": complexity,
                "num_components": num_components,
                "placement": placement.to_dict()
            }
            
            design_file = output_dir / f"design_{i:05d}.json"
            design_file.write_text(json.dumps(design_data, indent=2))
            
            designs.append(design_data)
            
        except Exception as e:
            print(f"  ⚠️  Error generating design {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  ✅ Generated {i + 1}/{num_designs} designs")
    
    print(f"✅ Generated {len(designs)} synthetic designs")
    return designs


def validate_kicad_export(kicad_file_path: Path) -> Dict:
    """
    Validate a KiCad export file.
    
    Returns validation results.
    """
    try:
        content = kicad_file_path.read_text()
        
        errors = []
        warnings = []
        
        # Check format
        if "(kicad_pcb" not in content:
            errors.append("Invalid KiCad format - missing (kicad_pcb)")
        
        if "(version" not in content:
            errors.append("Missing version")
        
        if "(layers" not in content:
            errors.append("Missing layers definition")
        
        # Check footprints
        footprint_count = content.count("(footprint")
        if footprint_count == 0:
            warnings.append("No footprints found")
        
        # Check pads
        pad_count = content.count("(pad")
        if pad_count == 0:
            warnings.append("No pads found")
        elif pad_count < footprint_count * 2:
            warnings.append(f"Few pads ({pad_count}) compared to footprints ({footprint_count})")
        
        # Check nets
        net_count = content.count("(net ")
        if net_count == 0:
            warnings.append("No nets defined")
        
        # Check board outline
        if 'layer "Edge.Cuts"' not in content:
            errors.append("Missing board outline (Edge.Cuts)")
        
        # Check for common issues
        if "generator \"dielectric\"" not in content:
            warnings.append("Not generated by Dielectric (may be from another tool)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "footprint_count": footprint_count,
            "pad_count": pad_count,
            "net_count": net_count
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Error reading file: {e}"],
            "warnings": [],
            "footprint_count": 0,
            "pad_count": 0,
            "net_count": 0
        }


def validate_all_exports(exports_dir: Path) -> Dict:
    """
    Validate all KiCad exports in a directory.
    """
    print(f"🔍 Validating KiCad exports in {exports_dir}...")
    
    if not exports_dir.exists():
        print(f"  ⚠️  Directory does not exist: {exports_dir}")
        return {"valid": 0, "invalid": 0, "total": 0}
    
    kicad_files = list(exports_dir.rglob("*.kicad_pcb"))
    
    if not kicad_files:
        print(f"  ⚠️  No KiCad files found in {exports_dir}")
        return {"valid": 0, "invalid": 0, "total": 0}
    
    valid_count = 0
    invalid_count = 0
    
    for kicad_file in kicad_files:
        result = validate_kicad_export(kicad_file)
        
        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  ❌ {kicad_file.name}: {result['errors']}")
    
    print(f"✅ Validation complete: {valid_count} valid, {invalid_count} invalid out of {len(kicad_files)} files")
    
    return {
        "valid": valid_count,
        "invalid": invalid_count,
        "total": len(kicad_files)
    }


def main():
    parser = argparse.ArgumentParser(description="Collect PCB designs for training/evaluation")
    parser.add_argument("--output", type=str, default="datasets", help="Output directory")
    parser.add_argument("--synthetic", type=int, default=100, help="Number of synthetic designs to generate")
    parser.add_argument("--github", action="store_true", help="Collect from GitHub")
    parser.add_argument("--validate", type=str, help="Validate KiCad exports in directory")
    parser.add_argument("--max-repos", type=int, default=5, help="Maximum GitHub repositories to clone")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📊 Dielectric ML Research - Dataset Collection")
    print("=" * 60)
    
    # Collect from GitHub
    if args.github:
        github_dir = output_dir / "kicad_designs"
        collect_kicad_from_github(github_dir, max_repos=args.max_repos)
    
    # Generate synthetic designs
    if args.synthetic > 0:
        synthetic_dir = output_dir / "synthetic_designs"
        generate_synthetic_designs(synthetic_dir, num_designs=args.synthetic)
    
    # Validate exports
    if args.validate:
        validate_dir = Path(args.validate)
        validate_all_exports(validate_dir)
    
    print("\n" + "=" * 60)
    print("✅ Dataset collection complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
