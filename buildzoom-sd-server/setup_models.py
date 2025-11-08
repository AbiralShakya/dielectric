#!/usr/bin/env python3
"""
Setup script for BuildZoom AI Stable Diffusion Server
Downloads required models and sets up the environment
"""

import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs and paths
MODELS = {
    "sdxl_base": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "path": "models/checkpoints/sd_xl_base_1.0.safetensors"
    },
    "controlnet_canny": {
        "url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
        "path": "models/controlnet/controlnet-canny-sdxl-1.0.safetensors"
    }
}

def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file with progress"""
    logger.info(f"Downloading {url} to {destination}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(".1f", end='', flush=True)

    print()  # New line after progress
    logger.info(f"Downloaded {destination}")

def create_directories():
    """Create necessary directories"""
    dirs = [
        "models/checkpoints",
        "models/controlnet",
        "models/loras",
        "outputs"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_models():
    """Download all required models"""
    logger.info("Starting model downloads...")

    for model_name, model_info in MODELS.items():
        url = model_info["url"]
        path = model_info["path"]

        if os.path.exists(path):
            logger.info(f"{model_name} already exists at {path}")
            continue

        try:
            download_file(url, path)
            logger.info(f"Successfully downloaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise

def main():
    """Main setup function"""
    logger.info("🚀 Setting up BuildZoom AI Stable Diffusion Server")

    try:
        # Create directories
        create_directories()

        # Download models
        download_models()

        logger.info("✅ Setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Run server: python sd_server.py")
        logger.info("3. Test: curl http://localhost:8000/health")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
