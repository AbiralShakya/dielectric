#!/usr/bin/env python3
"""
Download Real PCB Designs for Testing

Downloads industry-scale PCB designs from GitHub and other sources
for testing optimization on large, real-world designs.
"""

import os
import sys
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Real PCB design repositories (GitHub)
PCB_REPOSITORIES = [
    {
        "name": "Raspberry Pi Pico",
        "url": "https://github.com/raspberrypi/pico-sdk",
        "description": "Raspberry Pi Pico microcontroller board",
        "files": ["hardware/boards/pico/*.kicad_pcb"],
        "complexity": "medium",
        "components": "~50"
    },
    {
        "name": "Arduino Uno",
        "url": "https://github.com/arduino/ArduinoCore-avr",
        "description": "Arduino Uno reference design",
        "files": ["hardware/arduino/avr/boards.txt"],
        "complexity": "medium",
        "components": "~30"
    },
    {
        "name": "ESP32 DevKit",
        "url": "https://github.com/espressif/esp-idf",
        "description": "ESP32 development board",
        "files": ["components/**/*.kicad_pcb"],
        "complexity": "medium-high",
        "components": "~60"
    },
    {
        "name": "KiCad Example Projects",
        "url": "https://github.com/KiCad/kicad-examples",
        "description": "Official KiCad example projects",
        "files": ["**/*.kicad_pcb"],
        "complexity": "low-high",
        "components": "10-200"
    },
    {
        "name": "Open Hardware Projects",
        "url": "https://github.com/search?q=extension:kicad_pcb+stars:%3E10",
        "description": "Popular open-source hardware projects",
        "files": ["**/*.kicad_pcb"],
        "complexity": "varies",
        "components": "varies"
    }
]

# Direct download URLs for PCB files
DIRECT_PCB_DOWNLOADS = [
    {
        "name": "STM32 Nucleo Board",
        "url": "https://github.com/STMicroelectronics/STM32CubeNucleo/archive/refs/heads/main.zip",
        "description": "STM32 Nucleo development board",
        "complexity": "medium",
        "components": "~40"
    },
    {
        "name": "BeagleBone Black",
        "url": "https://github.com/beagleboard/beaglebone-black/archive/refs/heads/master.zip",
        "description": "BeagleBone Black single-board computer",
        "complexity": "high",
        "components": "~150"
    }
]


def download_github_repo(repo_url: str, output_dir: Path) -> Path:
    """
    Download a GitHub repository as zip.
    
    Args:
        repo_url: GitHub repository URL
        output_dir: Directory to save the zip
        
    Returns:
        Path to downloaded zip file
    """
    # Convert GitHub URL to zip download URL
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    if '/archive/' not in repo_url:
        # Add /archive/refs/heads/main.zip
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        repo_url = f"{repo_url}/archive/refs/heads/main.zip"
    
    print(f"📥 Downloading: {repo_url}")
    
    try:
        response = requests.get(repo_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Get filename from URL or use default
        filename = repo_url.split('/')[-1] or "repo.zip"
        zip_path = output_dir / filename
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"❌ Failed to download {repo_url}: {e}")
        return None


def extract_kicad_files(zip_path: Path, output_dir: Path) -> List[Path]:
    """
    Extract KiCad PCB files from zip archive.
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract files to
        
    Returns:
        List of extracted .kicad_pcb file paths
    """
    kicad_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files
            file_list = zip_ref.namelist()
            
            # Find all .kicad_pcb files
            pcb_files = [f for f in file_list if f.endswith('.kicad_pcb')]
            
            if not pcb_files:
                print(f"   ⚠️  No .kicad_pcb files found in {zip_path.name}")
                return []
            
            print(f"   📁 Found {len(pcb_files)} KiCad PCB file(s)")
            
            # Extract PCB files
            for pcb_file in pcb_files:
                # Create safe filename
                safe_name = pcb_file.replace('/', '_').replace('\\', '_')
                extract_path = output_dir / safe_name
                
                # Extract file
                with zip_ref.open(pcb_file) as source:
                    with open(extract_path, 'wb') as target:
                        target.write(source.read())
                
                kicad_files.append(extract_path)
                print(f"   ✅ Extracted: {safe_name}")
        
        return kicad_files
    except Exception as e:
        print(f"   ❌ Failed to extract {zip_path}: {e}")
        return []


def download_pcb_examples(output_dir: str = "examples/real_pcbs") -> Dict:
    """
    Download real PCB designs for testing.
    
    Args:
        output_dir: Directory to save downloaded PCBs
        
    Returns:
        Dictionary with download results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "downloaded": [],
        "failed": [],
        "total_files": 0
    }
    
    print("🚀 Downloading Real PCB Designs")
    print("=" * 60)
    
    # Download from direct URLs
    print("\n📦 Direct Downloads:")
    for pcb_info in DIRECT_PCB_DOWNLOADS:
        print(f"\n📥 {pcb_info['name']}")
        print(f"   Description: {pcb_info['description']}")
        print(f"   Complexity: {pcb_info['complexity']}, Components: {pcb_info['components']}")
        
        zip_path = download_github_repo(pcb_info['url'], output_path)
        
        if zip_path:
            # Extract KiCad files
            kicad_files = extract_kicad_files(zip_path, output_path)
            
            if kicad_files:
                results["downloaded"].append({
                    "name": pcb_info['name'],
                    "files": [str(f) for f in kicad_files],
                    "complexity": pcb_info['complexity'],
                    "components": pcb_info['components']
                })
                results["total_files"] += len(kicad_files)
            else:
                results["failed"].append({
                    "name": pcb_info['name'],
                    "reason": "No KiCad files found"
                })
        else:
            results["failed"].append({
                "name": pcb_info['name'],
                "reason": "Download failed"
            })
    
    # Save results
    results_file = output_path / "download_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Download Complete!")
    print(f"   Downloaded: {len(results['downloaded'])} projects")
    print(f"   Total files: {results['total_files']} PCB files")
    print(f"   Failed: {len(results['failed'])} projects")
    print(f"\n📁 Files saved to: {output_path.absolute()}")
    
    return results


def create_test_manifest(output_dir: str = "examples/real_pcbs") -> Dict:
    """
    Create a manifest of downloaded PCB files for easy testing.
    
    Args:
        output_dir: Directory containing downloaded PCBs
        
    Returns:
        Manifest dictionary
    """
    output_path = Path(output_dir)
    manifest = {
        "pcb_files": [],
        "total_count": 0
    }
    
    # Find all .kicad_pcb files
    pcb_files = list(output_path.glob("*.kicad_pcb"))
    
    for pcb_file in pcb_files:
        # Get file size
        file_size = pcb_file.stat().st_size
        
        manifest["pcb_files"].append({
            "filename": pcb_file.name,
            "path": str(pcb_file),
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        })
    
    manifest["total_count"] = len(manifest["pcb_files"])
    
    # Save manifest
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Created manifest: {manifest_file}")
    print(f"   Found {manifest['total_count']} PCB files")
    
    return manifest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real PCB designs for testing")
    parser.add_argument(
        "--output-dir",
        default="examples/real_pcbs",
        help="Directory to save downloaded PCBs"
    )
    parser.add_argument(
        "--create-manifest",
        action="store_true",
        help="Create manifest of downloaded files"
    )
    
    args = parser.parse_args()
    
    # Download PCBs
    results = download_pcb_examples(args.output_dir)
    
    # Create manifest if requested
    if args.create_manifest:
        create_test_manifest(args.output_dir)
    
    print("\n💡 Next Steps:")
    print("   1. Upload these PCB files using the frontend")
    print("   2. Test optimization on industry-scale designs")
    print("   3. Compare before/after metrics")

