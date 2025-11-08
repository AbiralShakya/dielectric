#!/usr/bin/env python3
"""
BuildZoom AI - Stable Diffusion Server
Simple Python-based SDXL Turbo server for hackathon
Uses SDXL Turbo for ultra-fast generation (4 steps!)
"""

import base64
import io
import logging
from typing import Optional
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import AutoPipelineForImage2Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BuildZoom AI - Stable Diffusion Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipe = None
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_model():
    """Load Stable Diffusion XL Turbo model"""
    global pipe

    try:
        logger.info("🚀 Loading Stable Diffusion XL Turbo model...")
        logger.info("(First time: downloads ~7GB, takes 2-3 minutes)")

        # Use SDXL Turbo for fast generation (4 steps!)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        logger.info("✅ Model loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.info("💡 Make sure you ran: pip install diffusers transformers torch accelerate")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": pipe is not None,
        "message": "Ready for image generation!"
    }

@app.post("/generate")
async def generate_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Generate remodeled image using Stable Diffusion"""
    try:
        if pipe is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Read and process input image
        img_bytes = await image.read()
        init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize to SDXL size (maintaining aspect ratio)
        init_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        logger.info(f"🎨 Generating image with prompt: {prompt}")

        # Generate with SDXL Turbo (super fast!)
        result = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=4,  # Turbo uses only 4 steps!
            guidance_scale=0.0,     # Turbo doesn't need guidance
            strength=0.75
        ).images[0]

        # Convert to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        logger.info("✅ Image generated successfully!")

        return {
            "success": True,
            "image": img_str,
            "prompt": prompt,
            "format": "PNG",
            "generation_time": "fast"
        }

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-multi-angle")
async def generate_multi_angle(
    image: UploadFile = File(...),
    base_prompt: str = Form(...),
    num_angles: int = Form(3)
):
    """Generate images from multiple angles"""
    try:
        if pipe is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Read and process input image
        img_bytes = await image.read()
        init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        init_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # Define angle prompts
        angle_prompts = [
            f"{base_prompt}, front view, eye level perspective, architectural photography",
            f"{base_prompt}, view from left side, 45 degree angle, interior design photography",
            f"{base_prompt}, view from right side, 45 degree angle, professional architectural shot",
            f"{base_prompt}, wide angle view from doorway entrance, comprehensive room view",
            f"{base_prompt}, close up detail view of main renovation area, focused composition"
        ]

        results = []
        angles_to_generate = angle_prompts[:num_angles]

        for i, angle_prompt in enumerate(angles_to_generate):
            logger.info(f"🎨 Generating angle {i+1}/{len(angles_to_generate)}")

            # Generate image for this angle
            result = pipe(
                prompt=angle_prompt,
                image=init_image,
                num_inference_steps=4,
                guidance_scale=0.0,
                strength=0.75
            ).images[0]

            # Convert to base64
            buffered = io.BytesIO()
            result.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            results.append({
                "angle": i + 1,
                "description": angle_prompt.split(", ")[-2],  # Extract angle description
                "image": img_str,
                "prompt": angle_prompt
            })

        logger.info(f"✅ Generated {len(results)} angle images!")

        return {
            "success": True,
            "images": results,
            "total_angles": len(results),
            "base_prompt": base_prompt
        }

    except Exception as e:
        logger.error(f"❌ Multi-angle generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-angle generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    print("🚀 BuildZoom AI Stable Diffusion Server")
    print("=====================================")
    print("📝 This server uses SDXL Turbo for fast image generation")
    print("🔗 API Endpoints:")
    print("   GET  /health")
    print("   POST /generate (image + prompt)")
    print("   POST /generate-multi-angle")
    print("")
    print("⚡ SDXL Turbo: Only 4 inference steps = SUPER FAST!")
    print("")

    uvicorn.run(
        "sd_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
